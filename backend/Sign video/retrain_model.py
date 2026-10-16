import json
import os
from collections import Counter
from datetime import datetime

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import (  # type: ignore[reportMissingImports]
    EarlyStopping,
    ReduceLROnPlateau,
)
from tensorflow.keras.layers import (  # type: ignore[reportMissingImports]
    Bidirectional,
    Dense,
    Dropout,
    Input,
    LSTM,
)
from tensorflow.keras.models import Sequential, load_model  # type: ignore[reportMissingImports]
from tensorflow.keras.utils import to_categorical  # type: ignore[reportMissingImports]

SEED = 42
np.random.seed(SEED)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(CURRENT_DIR, "dataset")
AI_SERVICE_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", "..", "ai-service"))
MODEL_OUTPUT_PATH = os.path.join(AI_SERVICE_DIR, "sign_lstm_model.h5")
LABEL_MAP_OUTPUT_PATH = os.path.join(AI_SERVICE_DIR, "label_map.json")

TEST_SIZE = 0.2
BATCH_SIZE = 16
EPOCHS = 60


def normalize_sequence(sequence: np.ndarray) -> np.ndarray:
    """Match preprocessing used by realtime inference in ai-service/main.py."""
    sequence = sequence.copy()
    for i in range(sequence.shape[0]):
        pose = sequence[i, :132].reshape(33, 4)
        center = (pose[11, :3] + pose[12, :3]) / 2
        pose[:, :3] -= center
        sequence[i, :132] = pose.flatten()

        left_hand = sequence[i, 132:195].reshape(21, 3)
        left_hand -= center
        sequence[i, 132:195] = left_hand.flatten()

        right_hand = sequence[i, 195:258].reshape(21, 3)
        right_hand -= center
        sequence[i, 195:258] = right_hand.flatten()
    return sequence


def discover_classes(dataset_dir: str):
    classes = []
    for name in sorted(os.listdir(dataset_dir)):
        class_dir = os.path.join(dataset_dir, name)
        if not os.path.isdir(class_dir):
            continue
        npy_files = [f for f in os.listdir(class_dir) if f.lower().endswith(".npy")]
        if npy_files:
            classes.append(name.upper())
    if not classes:
        raise RuntimeError(f"No class folders with .npy files found in {dataset_dir}")
    return classes


def load_dataset(dataset_dir: str, class_names):
    x_data = []
    y_data = []

    name_map = {name.upper(): name for name in os.listdir(dataset_dir)}

    for class_idx, class_name in enumerate(class_names):
        raw_name = name_map.get(class_name, class_name)
        class_dir = os.path.join(dataset_dir, raw_name)
        files = [f for f in os.listdir(class_dir) if f.lower().endswith(".npy")]
        for fname in files:
            sample = np.load(os.path.join(class_dir, fname))
            if sample.ndim != 2:
                continue
            if sample.shape[1] != 258:
                continue

            x_data.append(normalize_sequence(sample))
            y_data.append(class_idx)

    if not x_data:
        raise RuntimeError("No valid .npy samples were loaded. Check dataset format.")

    x = np.array(x_data, dtype=np.float32)
    y = np.array(y_data, dtype=np.int32)
    return x, y


def build_model(sequence_len: int, num_features: int, num_classes: int):
    model = Sequential(
        [
            Input(shape=(sequence_len, num_features)),
            Bidirectional(LSTM(128, return_sequences=True)),
            Dropout(0.3),
            Bidirectional(LSTM(64)),
            Dropout(0.3),
            Dense(64, activation="relu"),
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
    old_model = load_model(path)
    old_model.save(backup_path)
    print(f"Backed up previous model to: {backup_path}")


def main():
    print(f"Dataset: {DATASET_DIR}")
    print(f"Model output: {MODEL_OUTPUT_PATH}")
    print(f"Label map output: {LABEL_MAP_OUTPUT_PATH}")

    class_names = discover_classes(DATASET_DIR)
    label_map = {idx: name for idx, name in enumerate(class_names)}

    x, y = load_dataset(DATASET_DIR, class_names)
    print(f"Loaded samples: {len(x)}")
    print(f"Class distribution: {Counter(y)}")

    sequence_len, num_features = x.shape[1], x.shape[2]
    num_classes = len(class_names)

    x_train, x_val, y_train, y_val = train_test_split(
        x,
        y,
        test_size=TEST_SIZE,
        random_state=SEED,
        stratify=y,
    )

    y_train_oh = to_categorical(y_train, num_classes=num_classes)
    y_val_oh = to_categorical(y_val, num_classes=num_classes)

    class_weights_np = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_train),
        y=y_train,
    )
    class_weights = {int(c): float(w) for c, w in zip(np.unique(y_train), class_weights_np)}

    model = build_model(sequence_len, num_features, num_classes)

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6),
    ]

    model.fit(
        x_train,
        y_train_oh,
        validation_data=(x_val, y_val_oh),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,
    )

    val_probs = model.predict(x_val, verbose=0)
    val_pred = np.argmax(val_probs, axis=1)

    print("\nValidation classification report:")
    print(classification_report(y_val, val_pred, target_names=class_names, digits=4))

    print("Confusion matrix (rows=true, cols=pred):")
    print(confusion_matrix(y_val, val_pred))

    os.makedirs(AI_SERVICE_DIR, exist_ok=True)
    backup_existing_model(MODEL_OUTPUT_PATH)
    model.save(MODEL_OUTPUT_PATH)

    with open(LABEL_MAP_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in label_map.items()}, f, indent=2)

    print("\nTraining complete.")
    print(f"Saved model: {MODEL_OUTPUT_PATH}")
    print(f"Saved label map: {LABEL_MAP_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
