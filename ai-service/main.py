from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import json
import os
import cv2
import mediapipe as mp
from tensorflow.keras.models import load_model
import requests

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(
    min_detection_confidence=0.35,
    min_tracking_confidence=0.35,
    model_complexity=1,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "sign_lstm_model.h5")
LABEL_MAP_PATH = os.path.join(BASE_DIR, "label_map.json")

model = load_model(MODEL_PATH)
with open(LABEL_MAP_PATH, "r", encoding="utf-8") as f:
    label_map = {int(k): v for k, v in json.load(f).items()}

SEQUENCE_LENGTH = 60
CONFIDENCE_THRESHOLD = 0.72
NEUTRAL_LABEL = "NEUTRAL"
NO_HAND_FRAMES_TO_END_SIGN = 6   # shorter end-gap to reduce stale predictions
MIN_SIGN_FRAMES = 8              # require a more stable live sequence before accepting a sign

# Per-session state (single patient at a time, matching your current architecture)
buffer_frames = []
is_signing = False
no_hand_counter = 0
committed_sequence = []
last_predicted_label = None


def has_valid_landmarks(keypoints):
    return np.any(np.abs(keypoints) > 1e-6)


def predict_buffer_label(frame_sequence):
    valid_buffer = [frame for frame in frame_sequence if has_valid_landmarks(frame)]
    if len(valid_buffer) < MIN_SIGN_FRAMES:
        return None

    sequence = resample_to_length(valid_buffer, SEQUENCE_LENGTH)
    sequence = normalize_sequence(sequence)
    input_data = np.expand_dims(sequence, axis=0)
    prediction = model.predict(input_data, verbose=0)[0]
    pred_idx = int(np.argmax(prediction))
    confidence = float(prediction[pred_idx])
    pred_label = label_map[pred_idx]

    if confidence >= CONFIDENCE_THRESHOLD and pred_label != NEUTRAL_LABEL:
        return pred_label, confidence
    return None


def extract_keypoints(results):
    pose = (
        np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in results.pose_landmarks.landmark]).flatten()
        if results.pose_landmarks else np.zeros(33 * 4)
    )
    left_hand = (
        np.array([[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark]).flatten()
        if results.left_hand_landmarks else np.zeros(21 * 3)
    )
    right_hand = (
        np.array([[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark]).flatten()
        if results.right_hand_landmarks else np.zeros(21 * 3)
    )
    return np.concatenate([pose, left_hand, right_hand])


def resample_to_length(frames, target_length):
    sequence = np.array(frames)
    original_len = sequence.shape[0]
    old_idx = np.linspace(0, original_len - 1, original_len)
    new_idx = np.linspace(0, original_len - 1, target_length)
    resampled = np.zeros((target_length, sequence.shape[1]))
    for feat in range(sequence.shape[1]):
        resampled[:, feat] = np.interp(new_idx, old_idx, sequence[:, feat])
    return resampled


def normalize_sequence(sequence):
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


def glosses_to_sentence(glosses):
    if not glosses:
        return ""
    prompt = f"""Convert this sequence of sign-language glosses into ONE short, natural, grammatically
correct sentence as if a patient said it to a doctor. If two consecutive numbers appear (e.g. TWO THREE),
interpret them as a range (e.g. "two to three"). NOT before a word means negation.
Do not add symptoms or details not present in the glosses. Return ONLY the sentence.

Glosses: {", ".join(glosses)}
Sentence:"""

    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": "llama3.2", "prompt": prompt, "stream": False},
            timeout=15,
        )
        return response.json().get("response", "").strip()
    except requests.exceptions.RequestException:
        return " ".join(glosses)


@app.get("/")
def health_check():
    return {"status": "AI service is running"}


@app.post("/predict-frame")
async def predict_frame(file: UploadFile = File(...)):
    """Receives ONE webcam frame. Buffers frames while hands are visible
    (a sign in progress); once hands disappear for a few frames, treats
    that as the sign ending, resamples the buffered segment to 60 frames
    (matching how training data was built), and predicts once."""
    global is_signing, no_hand_counter, buffer_frames, committed_sequence, last_predicted_label

    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if frame is None:
        return {"glosses": committed_sequence, "sentence": "", "ready": False}

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = holistic.process(rgb)
    hands_present = bool(results.left_hand_landmarks or results.right_hand_landmarks)
    keypoints = extract_keypoints(results)

    if hands_present and has_valid_landmarks(keypoints):
        is_signing = True
        no_hand_counter = 0
        buffer_frames.append(keypoints)
    elif is_signing:
        no_hand_counter += 1
        if no_hand_counter >= NO_HAND_FRAMES_TO_END_SIGN:
            prediction = predict_buffer_label(buffer_frames)
            if prediction:
                pred_label, confidence = prediction
                print(f"Segment finished ({len(buffer_frames)} valid frames) -> {pred_label} ({confidence:.2%})")
                if pred_label != last_predicted_label:
                    if not committed_sequence or committed_sequence[-1] != pred_label:
                        committed_sequence.append(pred_label)
                    last_predicted_label = pred_label

            buffer_frames = []
            is_signing = False
            no_hand_counter = 0

    return {"glosses": committed_sequence, "sentence": "", "ready": True}


@app.post("/generate-sentence")
def generate_sentence():
    sentence = glosses_to_sentence(committed_sequence)
    return {"glosses": committed_sequence, "sentence": sentence}


@app.post("/reset-session")
def reset_session():
    global is_signing, no_hand_counter, buffer_frames, committed_sequence, last_predicted_label
    buffer_frames = []
    is_signing = False
    no_hand_counter = 0
    committed_sequence = []
    last_predicted_label = None
    return {"message": "Session reset."}