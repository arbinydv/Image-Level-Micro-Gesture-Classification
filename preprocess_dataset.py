'''
File: preprocess_dataset.py
Author: ABx_Lab

This script preprocesses the training data for multi gestures before
feeding into MEDIAPIPE solutions library.

Train- Test ==> 80-20
'''

import os
import shutil
import random
from collections import defaultdict
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from config import (
    GM_DATA_DIR,
    PROCESSED_GM_DIR,
    FRAMES_PER_SEQUENCE,
    MAX_VIDEOS_PER_CLASS,
)
from utils import get_sequence_id, get_sorted_frames

TRAIN_RATIO = 0.80
TEST_RATIO  = 0.20
RANDOM_SEED = 42   # reproducibility


def preprocess_dataset():
    if not os.path.exists(GM_DATA_DIR):
        raise FileNotFoundError(
            f"Could not find the '{GM_DATA_DIR}' folder. Please place the training data in the root folder"
        )
    random.seed(RANDOM_SEED)

    # Create output split directories
    for split in ("train", "test"):
        os.makedirs(os.path.join(PROCESSED_GM_DIR, split), exist_ok=True)

    for class_folder in tqdm(sorted(os.listdir(GM_DATA_DIR)), desc="Preprocessing Classes"):
        class_path = os.path.join(GM_DATA_DIR, class_folder)
        if not os.path.isdir(class_path):
            continue

        # Group frames by VIDEO ID
        sequence_map = defaultdict(list)
        for file in os.listdir(class_path):
            if file.lower().endswith((".jpg", ".jpeg", ".png")):
                seq_id = get_sequence_id(file)
                sequence_map[seq_id].append(os.path.join(class_path, file))

        # Shorten the large classes to prevent class imbalances across the dataset
        unique_video_ids = list(sequence_map.keys())
        if len(unique_video_ids) > MAX_VIDEOS_PER_CLASS:
            unique_video_ids = random.sample(unique_video_ids, MAX_VIDEOS_PER_CLASS)

        # Split at VIDEO level and need at least 2  videos from a class to operate
        if len(unique_video_ids) < 2:
             # pass to training set
            train_ids, test_ids = unique_video_ids, []
        else:
            train_ids, test_ids = train_test_split(
                unique_video_ids,
                test_size=TEST_RATIO,
                random_state=RANDOM_SEED,
            )

        split_assignment = (
            [("train", sid) for sid in train_ids]
            + [("test",  sid) for sid in test_ids]
        )

        # ── 4. Sample exactly FRAMES_PER_SEQUENCE frames
        for split, seq_id in split_assignment:
            raw_frames     = sequence_map[seq_id]
            selected_frames = get_sorted_frames(raw_frames, FRAMES_PER_SEQUENCE)

            dest_class_path = os.path.join(PROCESSED_GM_DIR, split, class_folder)
            os.makedirs(dest_class_path, exist_ok=True)

            for i, src_frame in enumerate(selected_frames):
                new_filename = f"{seq_id}_{i:02d}.jpg"
                dest_frame   = os.path.join(dest_class_path, new_filename)
                shutil.copy(src_frame, dest_frame)

    # Preprocessing summary
    for split in ("train", "test"):
        split_path = os.path.join(PROCESSED_GM_DIR, split)
        if not os.path.exists(split_path):
            continue
        total_frames = sum(
            len(os.listdir(os.path.join(split_path, c)))
            for c in os.listdir(split_path)
            if os.path.isdir(os.path.join(split_path, c))
        )
        n_classes = len(os.listdir(split_path))
        print(f"  {split:<6}: {n_classes} classes | {total_frames} frames")


if __name__ == "__main__":
    print("--- Starting Data Preprocessing for training ---")
    preprocess_dataset()
    print(f"\n--- Preprocessing Complete! Data saved to '{PROCESSED_GM_DIR}' ---")