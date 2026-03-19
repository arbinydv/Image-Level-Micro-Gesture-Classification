'''
File: extract_skeletons.py
Author: ABx_Lab

Extracts 225-dimensional skeletal keypoint vectors from preprocessed
frames and saves them as .npy files — one per video sequence.

'''

import os
import tqdm
import numpy as np
import mediapipe as mp
from collections import defaultdict

# internal imports
from config import PROCESSED_GM_DIR, OUTPUT_DIR, IMAGE_SIZE, FRAMES_PER_SEQUENCE,EXPECTED_FEATURES, ZERO_FRAME_THRESHOLD
from utils import get_sequence_id, extract_keypoints_from_file

def process_split(split, holistic_model):
    split_input_dir  = os.path.join(PROCESSED_GM_DIR, split)
    split_output_dir = os.path.join(OUTPUT_DIR, split)

    if not os.path.exists(split_input_dir):
        print(f"[WARNING] Split folder not found, skipping: {split_input_dir}")
        return 0, 0

    os.makedirs(split_output_dir, exist_ok=True)

    # 1. Map frames to sequences
    sequence_map = defaultdict(list)
    for class_folder in sorted(os.listdir(split_input_dir)):
        class_path = os.path.join(split_input_dir, class_folder)
        if not os.path.isdir(class_path):
            continue
        for file in os.listdir(class_path):
            if file.lower().endswith((".jpg", ".jpeg", ".png")):
                seq_id = get_sequence_id(file)
                sequence_map[(class_path, seq_id)].append(
                    os.path.join(class_path, file)
                )

    total_saved   = 0
    total_skipped = 0

    #  Extract and save
    for (class_path, seq_id), frames in tqdm(sequence_map.items(), desc=f"Extracting [{split}]"):
        frames = sorted(frames)

        sequence_array = np.array(
            [extract_keypoints_from_file(img, holistic_model, IMAGE_SIZE) for img in frames],
            dtype=np.float32,
        )

        # ── Validate shape ────────────────────────────────────────────────
        if sequence_array.shape != (FRAMES_PER_SEQUENCE, EXPECTED_FEATURES):
            print(
                f"[WARNING] Shape mismatch — skipping {seq_id}: "
                f"got {sequence_array.shape}, "
                f"expected ({FRAMES_PER_SEQUENCE}, {EXPECTED_FEATURES})"
            )
            total_skipped += 1
            continue

        # Skip near-empty sequences (MediaPipe detected nothing)
        zero_frames = np.all(sequence_array == 0, axis=1).sum()
        if zero_frames / FRAMES_PER_SEQUENCE > ZERO_FRAME_THRESHOLD:
            print(
                f"[WARNING] Mostly empty keypoints — skipping {seq_id} "
                f"({zero_frames}/{FRAMES_PER_SEQUENCE} zero frames)"
            )
            total_skipped += 1
            continue

        #  Save .npy format
        class_label      = os.path.basename(class_path)
        output_class_dir = os.path.join(split_output_dir, class_label)
        os.makedirs(output_class_dir, exist_ok=True)

        save_path = os.path.join(output_class_dir, f"{seq_id}.npy")
        np.save(save_path, sequence_array)
        total_saved += 1

    return total_saved, total_skipped


def process_skeletons():
    if not os.path.exists(PROCESSED_GM_DIR):
        raise FileNotFoundError(
            f"Could not find '{PROCESSED_GM_DIR}'. Please run preprocess_dataset.py first "
        )

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    mp_holistic    = mp.solutions.holistic
    holistic_model = mp_holistic.Holistic(
        static_image_mode=True,
        min_detection_confidence=0.5,
    )

    print("MediaPipe initialized. Starting extraction...\n")

    try:
        summary = {}
        for split in ("train", "test"):
            saved, skipped = process_split(split, holistic_model)
            summary[split] = {"saved": saved, "skipped": skipped}
    finally:
        holistic_model.close()

    print("\n--- Extraction Summary ---")
    for split, counts in summary.items():
        print(
            f"  {split:<6}: {counts['saved']} saved "
            f"| {counts['skipped']} skipped"
        )

if __name__ == "__main__":
    print("--- Starting Skeleton Extraction ---")
    process_skeletons()
    print(f"\n--- Extraction Complete! .npy files saved to '{OUTPUT_DIR}' ---")