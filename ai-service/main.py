from fastapi import FastAPI, UploadFile, File  # type: ignore[import-not-found]
from fastapi.middleware.cors import CORSMiddleware  # type: ignore[import-not-found]
import numpy as np  # type: ignore[import-not-found]
import json
import os
try:
    import cv2  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - handled at runtime when dependency is missing
    cv2 = None
try:
    import mediapipe as mp  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - handled at runtime when dependency is missing
    mp = None
try:
    from tensorflow.keras.models import load_model  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - handled at runtime when dependency is missing
    load_model = None
import requests  # type: ignore[import-not-found]

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

if mp is None:
    raise RuntimeError("mediapipe is required to run the AI service. Install it with: pip install mediapipe")
if load_model is None:
    raise RuntimeError("tensorflow is required to run the AI service. Install it with: pip install tensorflow")

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.35,
    min_tracking_confidence=0.35,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "sign_lstm_model.h5")
LABEL_MAP_PATH = os.path.join(BASE_DIR, "label_map.json")

model = load_model(MODEL_PATH)
with open(LABEL_MAP_PATH, "r", encoding="utf-8") as f:
    label_map = {int(k): v for k, v in json.load(f).items()}

SEQUENCE_LENGTH = 60
CONFIDENCE_THRESHOLD = 0.45
FEVER_CONFIDENCE_THRESHOLD = 0.40
NEUTRAL_LABEL = "NEUTRAL"
NO_HAND_FRAMES_TO_END_SIGN = 2   # commit sooner after the signer lowers their hands
MIN_SIGN_FRAMES = 4              # detect earlier while the sign is still forming
LIVE_PREDICTION_INTERVAL = 2     # retry more frequently while a gesture remains visible

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

    windows = [valid_buffer]
    if len(valid_buffer) >= MIN_SIGN_FRAMES * 2:
        windows.extend([
            valid_buffer[: max(MIN_SIGN_FRAMES, int(len(valid_buffer) * 0.8))],
            valid_buffer[-max(MIN_SIGN_FRAMES, int(len(valid_buffer) * 0.8)):],
        ])

    predictions = []
    for window in windows:
        sequence = resample_to_length(window, SEQUENCE_LENGTH)
        sequence = normalize_sequence(sequence)
        input_data = np.expand_dims(sequence, axis=0)
        predictions.append(model.predict(input_data, verbose=0)[0])

    prediction = np.mean(predictions, axis=0)
    pred_idx = int(np.argmax(prediction))
    confidence = float(prediction[pred_idx])
    pred_label = label_map[pred_idx]

    required_confidence = FEVER_CONFIDENCE_THRESHOLD if pred_label == "FEVER" else CONFIDENCE_THRESHOLD
    if confidence >= required_confidence and pred_label != NEUTRAL_LABEL:
        return pred_label, confidence
    return None


def extract_keypoints(results):
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
        left_hand = sequence[i, :63].reshape(21, 3)
        if np.any(np.abs(left_hand) > 1e-6):
            left_center = left_hand[0]
            left_hand = left_hand - left_center
        sequence[i, :63] = left_hand.flatten()

        right_hand = sequence[i, 63:126].reshape(21, 3)
        if np.any(np.abs(right_hand) > 1e-6):
            right_center = right_hand[0]
            right_hand = right_hand - right_center
        sequence[i, 63:126] = right_hand.flatten()
    return sequence


def glosses_to_sentence(glosses):
    if not glosses:
        return ""
    allowed_glosses = {
        "COLD",
        "DAYS",
        "EAT",
        "FEVER",
        "HEADACHE",
        "MEDICINE",
        "NEUTRAL",
        "NOT",
        "THREE",
        "TWO",
    }
    detected = [label for label in glosses if label in allowed_glosses]
    if not detected:
        return ""

    prompt = f"""Create one short, natural sentence from these sign-language glosses.
Use ONLY the information in the glosses. Do not add, remove, repeat, or guess any symptom,
number, duration, or action. Preserve every number exactly. If the glosses contain COLD,
a number, and DAYS, use the form: I have a cold for [number] days.
Return ONLY the sentence, with no explanation.

Glosses: {", ".join(detected)}
Sentence:"""

    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": "llama3.2:latest", "prompt": prompt, "stream": False},
            timeout=15,
        )
        sentence = response.json().get("response", "").strip()
        if sentence:
            replacements = {
                "COLD": "cold",
                "DAYS": "days",
                "EAT": "eat",
                "FEVER": "fever",
                "HEADACHE": "headache",
                "MEDICINE": "medicine",
                "NEUTRAL": "neutral",
                "NOT": "not",
                "THREE": "three",
                "TWO": "two",
            }
            for gloss, word in replacements.items():
                sentence = sentence.replace(gloss, word)
            return sentence
    except (requests.exceptions.RequestException, ValueError):
        pass

    words = {
        "COLD": "cold",
        "DAYS": "days",
        "EAT": "eat",
        "FEVER": "fever",
        "HEADACHE": "headache",
        "MEDICINE": "medicine",
        "NEUTRAL": "neutral",
        "NOT": "not",
        "THREE": "three",
        "TWO": "two",
    }
    numbers = [words[label] for label in detected if label in {"TWO", "THREE"}]
    symptoms = [words[label] for label in detected if label in {"COLD", "FEVER"}]
    has_days = "DAYS" in detected

    if symptoms and numbers and has_days:
        duration = " to ".join(numbers) if len(numbers) > 1 else numbers[0]
        article = "an" if symptoms[0][0] in "aeiou" else "a"
        return f"I have {article} {symptoms[0]} for {duration} days."
    if symptoms:
        article = "an" if symptoms[0][0] in "aeiou" else "a"
        return f"I have {article} {symptoms[0]}."
    if "EAT" in detected:
        return "I eat."
    if "NOT" in detected:
        return "I do not."
    return " ".join(words[label] for label in detected) + "."


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
    results = hands.process(rgb)
    hands_present = bool(results.multi_hand_landmarks)
    keypoints = extract_keypoints(results)

    if hands_present and has_valid_landmarks(keypoints):
        is_signing = True
        no_hand_counter = 0
        buffer_frames.append(keypoints)

        # Show a stable sign before the user lowers their hands.
        if (
            len(buffer_frames) >= MIN_SIGN_FRAMES
            and len(buffer_frames) % LIVE_PREDICTION_INTERVAL == 0
        ):
            prediction = predict_buffer_label(buffer_frames)
            if prediction:
                pred_label, confidence = prediction
                print(f"Stable sign ({len(buffer_frames)} valid frames) -> {pred_label} ({confidence:.2%})")
                if pred_label != last_predicted_label:
                    if not committed_sequence or committed_sequence[-1] != pred_label:
                        committed_sequence.append(pred_label)
                    last_predicted_label = pred_label
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