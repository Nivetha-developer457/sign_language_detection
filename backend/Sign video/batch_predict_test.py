import os
import json
import numpy as np
from tensorflow.keras.models import load_model  # type: ignore[reportMissingModuleSource]

MODEL_PATH = "sign_lstm_model.h5"
LABEL_MAP_PATH = "label_map.json"
BATCH_DIR = "batch_test_sequences"


def normalize_sequence(sequence):
    sequence = sequence.copy()
    for i in range(sequence.shape[0]):
        pose = sequence[i, :132].reshape(33, 4)
        left_shoulder = pose[11, :3]
        right_shoulder = pose[12, :3]
        center = (left_shoulder + right_shoulder) / 2

        pose[:, :3] -= center
        sequence[i, :132] = pose.flatten()

        left_hand = sequence[i, 132:132 + 63].reshape(21, 3)
        left_hand -= center
        sequence[i, 132:132 + 63] = left_hand.flatten()

        right_hand = sequence[i, 195:195 + 63].reshape(21, 3)
        right_hand -= center
        sequence[i, 195:195 + 63] = right_hand.flatten()
    return sequence


model = load_model(MODEL_PATH)
with open(LABEL_MAP_PATH, "r") as f:
    label_map = {int(k): v for k, v in json.load(f).items()}

with open(os.path.join(BATCH_DIR, "manifest.json"), "r") as f:
    manifest = json.load(f)

correct_count = 0
print(f"{'Source video':35} {'Correct':12} {'Predicted':12} {'Confidence':10} Result")
print("-" * 90)

for entry in manifest:
    sequence = np.load(os.path.join(BATCH_DIR, entry["file"]))
    sequence = normalize_sequence(sequence)
    input_data = np.expand_dims(sequence, axis=0)

    prediction = model.predict(input_data, verbose=0)[0]
    predicted_idx = np.argmax(prediction)
    predicted_label = label_map[predicted_idx]
    confidence = prediction[predicted_idx]

    is_correct = predicted_label == entry["correct_label"]
    if is_correct:
        correct_count += 1

    result = "CORRECT" if is_correct else "WRONG"
    print(f"{entry['source']:35} {entry['correct_label']:12} {predicted_label:12} {confidence:.2%}      {result}")

print("-" * 90)
print(f"\nOverall accuracy: {correct_count}/{len(manifest)} = {correct_count/len(manifest):.1%}")