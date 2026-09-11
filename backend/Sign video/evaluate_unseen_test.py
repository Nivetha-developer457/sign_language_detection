import json
import os

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import load_model  # type: ignore[reportMissingModuleSource]


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_DIR = os.path.join(ROOT_DIR, "unseen_test_sequences")
MODEL_PATH = os.path.abspath(os.path.join(ROOT_DIR, "..", "..", "ai-service", "sign_lstm_model.h5"))
LABEL_MAP_PATH = os.path.abspath(os.path.join(ROOT_DIR, "..", "..", "ai-service", "label_map.json"))


def normalize_sequence(sequence):
    sequence = sequence.copy()
    for frame_index in range(sequence.shape[0]):
        pose = sequence[frame_index, :132].reshape(33, 4)
        center = (pose[11, :3] + pose[12, :3]) / 2
        pose[:, :3] -= center
        sequence[frame_index, :132] = pose.flatten()

        left_hand = sequence[frame_index, 132:195].reshape(21, 3)
        left_hand -= center
        sequence[frame_index, 132:195] = left_hand.flatten()

        right_hand = sequence[frame_index, 195:258].reshape(21, 3)
        right_hand -= center
        sequence[frame_index, 195:258] = right_hand.flatten()
    return sequence


def main():
    model = load_model(MODEL_PATH, compile=False)
    with open(LABEL_MAP_PATH, "r", encoding="utf-8") as file:
        label_map = {int(key): value for key, value in json.load(file).items()}

    class_names = [label_map[index] for index in sorted(label_map)]
    true_labels = []
    predicted_labels = []

    for filename in sorted(os.listdir(TEST_DIR)):
        if not filename.endswith(".npy"):
            continue
        class_name = filename.split("_", 1)[0].upper()
        sequence = normalize_sequence(np.load(os.path.join(TEST_DIR, filename)))
        probabilities = model.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]
        predicted_index = int(np.argmax(probabilities))
        confidence = float(probabilities[predicted_index])
        predicted_name = label_map[predicted_index]
        true_labels.append(class_name)
        predicted_labels.append(predicted_name)
        print(f"{filename}: expected={class_name}, predicted={predicted_name}, confidence={confidence:.2%}")

    if not true_labels:
        raise RuntimeError(f"No extracted unseen tests found in {TEST_DIR}")

    print("\nUnseen test classification report:")
    print(classification_report(true_labels, predicted_labels, labels=class_names, zero_division=0, digits=4))
    print("Confusion matrix (rows=true, cols=predicted):")
    print(confusion_matrix(true_labels, predicted_labels, labels=class_names))


if __name__ == "__main__":
    main()