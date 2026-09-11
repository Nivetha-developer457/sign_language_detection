import os
from importlib import import_module

import numpy as np

from extract_dataset import process_video


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(ROOT_DIR, "new_recordings")
OUTPUT_DIR = os.path.join(ROOT_DIR, "dataset")


def load_mediapipe():
    try:
        return import_module("mediapipe")
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "MediaPipe is not installed in the active Python environment. "
            "Install it with: python -m pip install mediapipe"
        ) from error


def main():
    processed = 0
    mp = load_mediapipe()
    with mp.solutions.holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as holistic:
        for class_name in sorted(os.listdir(INPUT_DIR)):
            class_dir = os.path.join(INPUT_DIR, class_name)
            if not os.path.isdir(class_dir):
                continue

            output_class_dir = os.path.join(OUTPUT_DIR, class_name.upper())
            os.makedirs(output_class_dir, exist_ok=True)
            for filename in sorted(os.listdir(class_dir)):
                if not filename.lower().endswith((".mp4", ".mov", ".avi")):
                    continue

                output_name = f"new_{os.path.splitext(filename)[0]}.npy"
                output_path = os.path.join(output_class_dir, output_name)
                if os.path.exists(output_path):
                    continue

                sequence = process_video(os.path.join(class_dir, filename), holistic)
                if sequence is None:
                    print(f"WARNING: no frames extracted from {class_name}/{filename}")
                    continue

                np.save(output_path, sequence)
                processed += 1
                print(f"Processed {class_name}/{filename} -> {output_path}")

    print(f"Extracted {processed} new training sequences.")


if __name__ == "__main__":
    main()