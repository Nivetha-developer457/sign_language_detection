import os
import time

import cv2


CLASSES = [
    "Cold",
    "Days",
    "Three",
]
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAINING_DIR = ROOT_DIR
TEST_DIR = os.path.join(ROOT_DIR, "unseen_test")


def open_camera():
    requested_index = os.getenv("SIGN_CAMERA_INDEX")
    indexes = [int(requested_index)] if requested_index else range(6)
    backends = [cv2.CAP_DSHOW, cv2.CAP_ANY]

    for index in indexes:
        for backend in backends:
            camera = cv2.VideoCapture(index, backend)
            if camera.isOpened():
                return camera, index
            camera.release()

    return None, None


def record_video(output_path, camera):
    width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        20.0,
        (width, height),
    )
    if not writer.isOpened():
        return False

    print("Press SPACE to start recording, then SPACE to stop. Press Q to cancel.")
    recording = False
    frame_count = 0
    while True:
        ok, frame = camera.read()
        if not ok:
            writer.release()
            if os.path.exists(output_path):
                os.remove(output_path)
            break

        mirrored_frame = cv2.flip(frame, 1)
        display = mirrored_frame.copy()
        message = "RECORDING - SPACE stops" if recording else "SPACE starts - Q cancels"
        cv2.putText(display, message, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("Sign recording", display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            writer.release()
            if os.path.exists(output_path):
                os.remove(output_path)
            return False
        if key == 32:
            if recording:
                writer.release()
                return True
            recording = True

        if recording:
            writer.write(frame)
            frame_count += 1

    writer.release()
    if frame_count == 0 and os.path.exists(output_path):
        os.remove(output_path)
    return frame_count > 0


def main():
    mode = input("Record training videos or unseen test videos? [train/test]: ").strip().lower()
    if mode not in {"train", "test"}:
        raise SystemExit("Choose train or test.")

    output_root = TRAINING_DIR if mode == "train" else TEST_DIR
    os.makedirs(output_root, exist_ok=True)
    camera, camera_index = open_camera()
    if camera is None:
        raise SystemExit(
            "Could not open a webcam. Enable camera access for VS Code/Python, "
            "close other camera apps, or set SIGN_CAMERA_INDEX to the correct index."
        )
    print(f"Using camera index {camera_index}.")

    try:
        for class_name in CLASSES:
            class_dir = os.path.join(output_root, class_name)
            os.makedirs(class_dir, exist_ok=True)
            next_number = len([name for name in os.listdir(class_dir) if name.endswith(".mp4")]) + 1
            output_path = os.path.join(class_dir, f"webcam_{next_number:03d}.mp4")
            print(f"\nClass: {class_name}. Sign naturally with the current lighting and angle.")
            time.sleep(2)
            if not record_video(output_path, camera):
                print(f"Skipped {class_name}.")
            else:
                print(f"Saved {output_path}")
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()