import os
import importlib

import numpy as np

from extract_dataset import process_video, SEQUENCE_LENGTH


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(ROOT_DIR, "unseen_test")
OUTPUT_DIR = os.path.join(ROOT_DIR, "unseen_test_sequences")


def main():
    try:
        mp = importlib.import_module("mediapipe")
    except ImportError as exc:
        raise RuntimeError(
            "MediaPipe is required. Install it with 'pip install mediapipe'."
        ) from exc

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    mp_holistic = mp.solutions.holistic
    processed = 0

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        for class_name in sorted(os.listdir(INPUT_DIR)):
            class_dir = os.path.join(INPUT_DIR, class_name)
            if not os.path.isdir(class_dir):
                continue

            for filename in sorted(os.listdir(class_dir)):
                if not filename.lower().endswith((".mp4", ".mov", ".avi")):
                    continue
                sequence = process_video(os.path.join(class_dir, filename), holistic)
                if sequence is None:
                    print(f"WARNING: no frames extracted from {class_name}/{filename}")
                    continue
                output_name = f"{class_name}_{os.path.splitext(filename)[0]}.npy"
                np.save(os.path.join(OUTPUT_DIR, output_name), sequence)
                processed += 1
                print(f"Processed {class_name}/{filename} -> {output_name}")

    print(f"Extracted {processed} unseen test sequences of {SEQUENCE_LENGTH} frames each.")


if __name__ == "__main__":
    main()