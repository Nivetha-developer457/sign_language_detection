import os
import numpy as np

DATASET_DIR = "dataset"
AUGMENT_FACTOR = 6  # each original sample becomes 1 + 6 = 7 total samples


def jitter(sequence, sigma=0.01):
    return sequence + np.random.normal(0, sigma, sequence.shape)


def scale(sequence, factor_range=(0.95, 1.05)):
    return sequence * np.random.uniform(*factor_range)


def time_warp(sequence, target_length):
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


def augment_sequence(sequence):
    seq = time_warp(sequence, target_length=sequence.shape[0])
    seq = scale(seq)
    seq = jitter(seq)
    return seq


for class_name in os.listdir(DATASET_DIR):
    class_dir = os.path.join(DATASET_DIR, class_name)
    if not os.path.isdir(class_dir):
        continue

    originals = [f for f in os.listdir(class_dir) if "_aug" not in f]
    for fname in originals:
        path = os.path.join(class_dir, fname)
        sequence = np.load(path)
        base_name = fname.replace(".npy", "")

        for i in range(AUGMENT_FACTOR):
            aug_seq = augment_sequence(sequence)
            aug_path = os.path.join(class_dir, f"{base_name}_aug{i}.npy")
            np.save(aug_path, aug_seq)

    print(f"{class_name}: {len(originals)} originals -> {len(originals) * (AUGMENT_FACTOR + 1)} total")