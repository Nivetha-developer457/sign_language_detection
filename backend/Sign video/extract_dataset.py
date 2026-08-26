import os
import cv2
import numpy as np
import mediapipe as mp

mp_holistic = mp.solutions.holistic

SEQUENCE_LENGTH = 60  # fixed frame count after resampling -- works regardless of source video length
SOURCE_DIR = "."       # current folder -- run this script from inside "Sign video"
OUTPUT_DIR = "dataset"  # where the extracted .npy files will be saved
NEW_RECORDINGS_DIR = "new_recordings"
UNSEEN_TEST_DIR = "unseen_test"


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


def resample_sequence(sequence, target_length):
    sequence = np.array(sequence)
    original_len = sequence.shape[0]
    if original_len == target_length:
        return sequence

    old_idx = np.linspace(0, original_len - 1, original_len)
    new_idx = np.linspace(0, original_len - 1, target_length)
    resampled = np.zeros((target_length, sequence.shape[1]))
    for feat in range(sequence.shape[1]):
        resampled[:, feat] = np.interp(new_idx, old_idx, sequence[:, feat])
    return resampled


def process_video(video_path, holistic):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(rgb)
        if results.left_hand_landmarks or results.right_hand_landmarks:
            frames.append(extract_keypoints(results))
    cap.release()

    if len(frames) == 0:
        return None
    return resample_sequence(frames, SEQUENCE_LENGTH)


def main():
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        source_dirs = [SOURCE_DIR]
        if os.path.isdir(NEW_RECORDINGS_DIR):
            source_dirs.append(NEW_RECORDINGS_DIR)

        for source_dir in source_dirs:
            for class_name in os.listdir(source_dir):
                class_path = os.path.join(source_dir, class_name)
                if not os.path.isdir(class_path) or class_name.lower() in {"dataset", "new_recordings"}:
                    continue

                out_class_dir = os.path.join(OUTPUT_DIR, class_name.upper())
                os.makedirs(out_class_dir, exist_ok=True)

                for fname in os.listdir(class_path):
                    if not fname.lower().endswith((".mp4", ".mov", ".avi")):
                        continue

                    video_path = os.path.join(class_path, fname)
                    sequence = process_video(video_path, holistic)

                    if sequence is not None:
                        out_name = os.path.splitext(fname)[0] + ".npy"
                        out_path = os.path.join(out_class_dir, out_name)
                        np.save(out_path, sequence)
                        print(f"Processed {video_path} -> {out_path}")
                    else:
                        print(f"WARNING: no frames extracted from {video_path}")


def extract_unseen_test():
    output_dir = "unseen_test_sequences"
    os.makedirs(output_dir, exist_ok=True)
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        for class_name in os.listdir(UNSEEN_TEST_DIR):
            class_path = os.path.join(UNSEEN_TEST_DIR, class_name)
            if not os.path.isdir(class_path):
                continue

            for fname in os.listdir(class_path):
                if not fname.lower().endswith((".mp4", ".mov", ".avi")):
                    continue
                sequence = process_video(os.path.join(class_path, fname), holistic)
                if sequence is None:
                    print(f"WARNING: no frames extracted from {class_path}/{fname}")
                    continue

                output_name = f"{class_name}_{os.path.splitext(fname)[0]}.npy"
                np.save(os.path.join(output_dir, output_name), sequence)
                print(f"Processed unseen test: {class_path}/{fname}")


if __name__ == "__main__":
    main()
    if os.path.isdir(UNSEEN_TEST_DIR):
        extract_unseen_test()