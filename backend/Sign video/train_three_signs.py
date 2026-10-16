"""Train a 3-class ISL LSTM (COLD, DAYS, THREE) matching ai-service inference.

Feature layout matches ai-service/main.py:
  126 dims = left hand (21 x 3) + right hand (21 x 3)
  60-frame resampled sequences, wrist-centered normalization
"""
from __future__ import annotations

import json
import os
from collections import Counter
from datetime import datetime

import cv2
import mediapipe as mp
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Bidirectional, Dense, Dropout, Input, LSTM
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.utils import to_categorical

SEED = 42
np.random.seed(SEED)

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
AI_SERVICE_DIR = os.path.abspath(os.path.join(ROOT_DIR, "..", "..", "ai-service"))
DATASET_DIR = os.path.join(ROOT_DIR, "dataset")
MODEL_OUTPUT_PATH = os.path.join(AI_SERVICE_DIR, "sign_lstm_model.h5")
LABEL_MAP_OUTPUT_PATH = os.path.join(AI_SERVICE_DIR, "label_map.json")
LOCAL_LABEL_MAP_PATH = os.path.join(ROOT_DIR, "label_map.json")

CLASSES = ["COLD", "DAYS", "THREE"]
VIDEO_DIR_BY_CLASS = {
    "COLD": os.path.join(ROOT_DIR, "Cold"),
    "DAYS": os.path.join(ROOT_DIR, "Days"),
    "THREE": os.path.join(ROOT_DIR, "Three"),
}
UNSEEN_TEST_DIR = os.path.join(ROOT_DIR, "unseen_test")

SEQUENCE_LENGTH = 60
AUGMENT_FACTOR = 6
TEST_SIZE = 0.2
BATCH_SIZE = 16
EPOCHS = 60


def extract_keypoints(results) -> np.ndarray:
    left_hand = np.zeros(21 * 3)
    right_hand = np.zeros(21 * 3)

    if not results.multi_hand_landmarks:
        return np.concatenate([left_hand, right_hand])

    for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
        hand_points = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()
        label = handedness.classification[0].label
        if label == "Left":
            left_hand = hand_points
        elif label == "Right":
            right_hand = hand_points

    return np.concatenate([left_hand, right_hand])


def has_valid_landmarks(keypoints: np.ndarray) -> bool:
    return np.any(np.abs(keypoints) > 1e-6)


def resample_to_length(frames, target_length: int) -> np.ndarray:
    sequence = np.array(frames)
    original_len = sequence.shape[0]
    old_idx = np.linspace(0, original_len - 1, original_len)
    new_idx = np.linspace(0, original_len - 1, target_length)
    resampled = np.zeros((target_length, sequence.shape[1]))
    for feat in range(sequence.shape[1]):
        resampled[:, feat] = np.interp(new_idx, old_idx, sequence[:, feat])
    return resampled


def normalize_sequence(sequence: np.ndarray) -> np.ndarray:
    sequence = sequence.copy()
    for i in range(sequence.shape[0]):
        left_hand = sequence[i, :63].reshape(21, 3)
        if np.any(np.abs(left_hand) > 1e-6):
            left_hand = left_hand - left_hand[0]
        sequence[i, :63] = left_hand.flatten()

        right_hand = sequence[i, 63:126].reshape(21, 3)
        if np.any(np.abs(right_hand) > 1e-6):
            right_hand = right_hand - right_hand[0]
        sequence[i, 63:126] = right_hand.flatten()
    return sequence


def process_video(video_path: str, hands) -> np.ndarray | None:
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        keypoints = extract_keypoints(results)
        if has_valid_landmarks(keypoints):
            frames.append(keypoints)
    cap.release()
    if len(frames) < 4:
        return None
    return resample_to_length(frames, SEQUENCE_LENGTH)


def jitter(sequence: np.ndarray, sigma: float = 0.01) -> np.ndarray:
    return sequence + np.random.normal(0, sigma, sequence.shape)


def scale(sequence: np.ndarray, factor_range=(0.95, 1.05)) -> np.ndarray:
    return sequence * np.random.uniform(*factor_range)


def time_warp(sequence: np.ndarray, target_length: int) -> np.ndarray:
    num_frames, num_features = sequence.shape
    warp_factor = np.random.uniform(0.85, 1.15)
    new_len = max(2, int(num_frames * warp_factor))

    old_idx = np.linspace(0, num_frames - 1, num_frames)
    new_idx = np.linspace(0, num_frames - 1, new_len)
    warped = np.zeros((new_len, num_features))
    for feat in range(num_features):
        warped[:, feat] = np.interp(new_idx, old_idx, sequence[:, feat])

    final_idx = np.linspace(0, new_len - 1, target_length)
    orig_idx = np.linspace(0, new_len - 1, new_len)
    resampled = np.zeros((target_length, num_features))
    for feat in range(num_features):
        resampled[:, feat] = np.interp(final_idx, orig_idx, warped[:, feat])
    return resampled


def augment_sequence(sequence: np.ndarray) -> np.ndarray:
    seq = time_warp(sequence, target_length=sequence.shape[0])
    seq = scale(seq)
    seq = jitter(seq)
    return seq


def extract_class_videos(class_name: str, video_dir: str, hands) -> list[np.ndarray]:
    out_dir = os.path.join(DATASET_DIR, class_name)
    os.makedirs(out_dir, exist_ok=True)
    sequences = []

    if not os.path.isdir(video_dir):
        print(f"WARNING: missing video folder {video_dir}")
        return sequences

    for fname in sorted(os.listdir(video_dir)):
        if not fname.lower().endswith((".mp4", ".mov", ".avi")):
            continue
        npy_name = os.path.splitext(fname)[0] + ".npy"
        npy_path = os.path.join(out_dir, npy_name)

        if os.path.exists(npy_path):
            cached_sequence = np.load(npy_path)
            if cached_sequence.shape == (SEQUENCE_LENGTH, 126):
                sequences.append(cached_sequence)
                print(f"Loaded cached hand points {npy_path}")
                continue

        video_path = os.path.join(video_dir, fname)
        sequence = process_video(video_path, hands)
        if sequence is None:
            print(f"WARNING: no hand frames in {video_path}")
            continue
        np.save(npy_path, sequence)
        sequences.append(sequence)
        print(f"Extracted {video_path} -> {npy_path}")

    print(f"{class_name}: {len(sequences)} source sequences")
    return sequences


def extract_unseen_tests(hands) -> list[tuple[str, np.ndarray]]:
    samples = []
    if not os.path.isdir(UNSEEN_TEST_DIR):
        return samples

    for class_name in CLASSES:
        class_dir = os.path.join(UNSEEN_TEST_DIR, class_name.title())
        if not os.path.isdir(class_dir):
            class_dir = os.path.join(UNSEEN_TEST_DIR, class_name)
        if not os.path.isdir(class_dir):
            continue
        for fname in sorted(os.listdir(class_dir)):
            if not fname.lower().endswith((".mp4", ".mov", ".avi")):
                continue
            sequence = process_video(os.path.join(class_dir, fname), hands)
            if sequence is None:
                print(f"WARNING: no hand frames in unseen test {class_name}/{fname}")
                continue
            samples.append((class_name, sequence))
            print(f"Unseen test extracted: {class_name}/{fname}")
    return samples


def build_xy(class_sequences: dict[str, list[np.ndarray]]):
    x_data = []
    y_data = []
    for class_idx, class_name in enumerate(CLASSES):
        for sequence in class_sequences[class_name]:
            x_data.append(normalize_sequence(sequence))
            y_data.append(class_idx)
            for _ in range(AUGMENT_FACTOR):
                x_data.append(normalize_sequence(augment_sequence(sequence)))
                y_data.append(class_idx)
    return np.array(x_data, dtype=np.float32), np.array(y_data, dtype=np.int32)


def build_model(sequence_len: int, num_features: int, num_classes: int):
    model = Sequential(
        [
            Input(shape=(sequence_len, num_features)),
            Bidirectional(LSTM(128, return_sequences=True)),
            Dropout(0.3),
            Bidirectional(LSTM(96, return_sequences=True)),
            Dropout(0.3),
            Bidirectional(LSTM(64)),
            Dropout(0.3),
            Dense(96, activation="relu"),
            Dropout(0.3),
            Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def backup_existing_model(path: str):
    if not os.path.exists(path):
        return
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = path.replace(".h5", f"_backup_{timestamp}.h5")
    old_model = load_model(path, compile=False)
    old_model.save(backup_path)
    print(f"Backed up previous model to: {backup_path}")


def save_label_map(label_map: dict[int, str]):
    payload = {str(k): v for k, v in label_map.items()}
    os.makedirs(AI_SERVICE_DIR, exist_ok=True)
    for path in (LABEL_MAP_OUTPUT_PATH, LOCAL_LABEL_MAP_PATH):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)


def main():
    print("Training 3-sign model: COLD, DAYS, THREE")
    print(f"Videos: {ROOT_DIR}")
    print(f"Model output: {MODEL_OUTPUT_PATH}")

    mp_hands = mp.solutions.hands
    class_sequences: dict[str, list[np.ndarray]] = {}
    with mp_hands.Hands(
        max_num_hands=2,
        min_detection_confidence=0.35,
        min_tracking_confidence=0.35,
    ) as hands:
        for class_name in CLASSES:
            class_sequences[class_name] = extract_class_videos(
                class_name, VIDEO_DIR_BY_CLASS[class_name], hands
            )
        unseen = extract_unseen_tests(hands)

    missing = [name for name, seqs in class_sequences.items() if not seqs]
    if missing:
        raise RuntimeError(f"No usable videos for classes: {missing}")

    x, y = build_xy(class_sequences)
    print(f"Loaded samples (with augmentation): {len(x)}")
    print(f"Class distribution: { {CLASSES[k]: v for k, v in Counter(y).items()} }")

    x_train, x_val, y_train, y_val = train_test_split(
        x, y, test_size=TEST_SIZE, random_state=SEED, stratify=y
    )
    y_train_oh = to_categorical(y_train, num_classes=len(CLASSES))
    y_val_oh = to_categorical(y_val, num_classes=len(CLASSES))

    class_weights_np = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_train),
        y=y_train,
    )
    class_weights = {int(c): float(w) for c, w in zip(np.unique(y_train), class_weights_np)}

    model = build_model(x.shape[1], x.shape[2], len(CLASSES))
    model.fit(
        x_train,
        y_train_oh,
        validation_data=(x_val, y_val_oh),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weights,
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6),
        ],
        verbose=1,
    )

    val_pred = np.argmax(model.predict(x_val, verbose=0), axis=1)
    print("\nValidation classification report:")
    print(classification_report(y_val, val_pred, target_names=CLASSES, digits=4))
    print("Confusion matrix (rows=true, cols=pred):")
    print(confusion_matrix(y_val, val_pred))

    if unseen:
        true_labels = []
        pred_labels = []
        print("\nUnseen test videos:")
        for class_name, sequence in unseen:
            probs = model.predict(np.expand_dims(normalize_sequence(sequence), axis=0), verbose=0)[0]
            pred_idx = int(np.argmax(probs))
            pred_name = CLASSES[pred_idx]
            true_labels.append(class_name)
            pred_labels.append(pred_name)
            print(f"  expected={class_name} predicted={pred_name} confidence={probs[pred_idx]:.2%}")
        print("\nUnseen test classification report:")
        print(classification_report(true_labels, pred_labels, labels=CLASSES, zero_division=0, digits=4))
        print(confusion_matrix(true_labels, pred_labels, labels=CLASSES))

    backup_existing_model(MODEL_OUTPUT_PATH)
    model.save(MODEL_OUTPUT_PATH)
    label_map = {idx: name for idx, name in enumerate(CLASSES)}
    save_label_map(label_map)
    print("\nTraining complete.")
    print(f"Saved model: {MODEL_OUTPUT_PATH}")
    print(f"Saved label map: {LABEL_MAP_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
