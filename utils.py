'''
File name: utils.py
Author: ABx_Lab

This file contains all the shared functions and methods to support
associated scripts across the entire pipeline.
'''

import os
import cv2
import numpy as np

def get_sequence_id(filepath):
    """
    Extracts the base video ID from a filepath.
    Handles both raw format (video123.01.jpg) and processed format (video123_01.jpg).
    """
    filename = os.path.basename(filepath)
    name_without_ext = os.path.splitext(filename)[0]

    if "_" in name_without_ext:
        return name_without_ext.split("_")[0]
    return name_without_ext.split(".")[0]


def get_frame_number(filepath):
    """Extracts the frame number to ensure strict chronological sorting."""
    filename = os.path.basename(filepath)
    name_without_ext = os.path.splitext(filename)[0]

    if "_" in name_without_ext:
        return int(name_without_ext.split("_")[-1])
    return int(name_without_ext.split(".")[-1])


def get_sorted_frames(frame_paths, n_frames):
    """Sorts frames chronologically and uniformly samples/pads to n_frames."""
    frame_paths = sorted(frame_paths, key=get_frame_number)
    if not frame_paths:
        return []

    total = len(frame_paths)
    if total >= n_frames:
        # Uniform temporal sampling — matches how the model was trained
        indices = np.linspace(0, total - 1, n_frames, dtype=int)
        return [frame_paths[i] for i in indices]

    # Pad by repeating the last frame if the sequence is too short
    return frame_paths + [frame_paths[-1]] * (n_frames - total)

# Extract Mediapipe for skeleton extraction
def extract_keypoints_from_file(image_path, holistic_model, image_size):
    image = cv2.imread(image_path)
    if image is None:
        return np.zeros(225, dtype=np.float32)

    image = cv2.resize(image, image_size)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = holistic_model.process(image_rgb)

    return _keypoints_from_results(results)


def extract_keypoints_from_frame(results):
    """
    Extracts a 225-dimensional keypoint vector directly from a MediaPipe holistic results object (already processed frame).
    """
    return _keypoints_from_results(results)


def _keypoints_from_results(results):
    """
    Converts MediaPipe Holistic results to a flat 225-dimensional vector.
    Pose(33x3=99) + Left Hand(21x3=63) + Right Hand(21x3=63) = 225
    """
    parts = [
        (results.pose_landmarks,       33),
        (results.left_hand_landmarks,  21),
        (results.right_hand_landmarks, 21),
    ]

    keypoints = [
        np.array([[p.x, p.y, p.z] for p in landmarks.landmark], dtype=np.float32).flatten()
        if landmarks else
        np.zeros(n * 3, dtype=np.float32)
        for landmarks, n in parts
    ]

    return np.concatenate(keypoints).astype(np.float32)

# Sequence normalization
def normalize_sequence(seq_data, frames_per_sequence, expected_features):
    data = seq_data.reshape(frames_per_sequence, 75, 3).copy()

    # Pose: normalize to nose (index 0)
    data[:, :33, :]   -= data[:, 0:1,   :]

    # Left hand: normalize to left wrist (index 33)
    data[:, 33:54, :] -= data[:, 33:34, :]

    # Right hand: normalize to right wrist (index 54)
    data[:, 54:75, :] -= data[:, 54:55, :]

    return data.reshape(frames_per_sequence, expected_features).astype(np.float32)

def load_data_paths(root_dir):
    # Build class list from UNION of both splits
    all_classes = set()
    for split in ("train", "test"):
        split_dir = os.path.join(root_dir, split)
        if os.path.exists(split_dir):
            all_classes.update(
                c for c in os.listdir(split_dir)
                if os.path.isdir(os.path.join(split_dir, c))
            )

    class_names  = sorted(all_classes)
    class_to_idx = {c: i for i, c in enumerate(class_names)}

    train_paths, train_labels = _collect_split(root_dir, "train", class_to_idx)
    test_paths,  test_labels  = _collect_split(root_dir, "test",  class_to_idx)

    return train_paths, train_labels, test_paths, test_labels, class_names


def _collect_split(root_dir, split, class_to_idx):
    """
    Internal helper — collects all .npy paths and integer labels for one split.
    """
    paths, labels = [], []
    split_dir = os.path.join(root_dir, split)

    if not os.path.exists(split_dir):
        return paths, labels

    for class_folder, label in class_to_idx.items():
        class_path = os.path.join(split_dir, class_folder)
        if not os.path.isdir(class_path):
            continue
        for file in sorted(os.listdir(class_path)):
            if file.endswith(".npy"):
                paths.append(os.path.join(class_path, file))
                labels.append(label)

    return paths, labels