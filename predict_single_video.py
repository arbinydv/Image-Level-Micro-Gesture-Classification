'''
File: predict_single.py
Author: ABx_Lab
Tests the trained model on a single raw video sequence (folder of frames).
'''

import os
import cv2
import numpy as np
import torch
import mediapipe as mp

from config import FRAMES_PER_SEQUENCE, IMAGE_SIZE
from utils import get_sorted_frames, extract_keypoints_from_file, normalize_sequence

MODEL_SAVE_PATH   = "mediapipe_lstm_imigue.pth"
EXPECTED_FEATURES = 225


def predict_sequence(frame_folder):
    """
    Given a folder of frames, runs the full pipeline:
    frames → keypoints → normalize → model → prediction
    """

    # --- 1. Load model ---
    checkpoint  = torch.load(MODEL_SAVE_PATH, map_location="cpu")
    model       = checkpoint["model"]
    class_names = checkpoint["class_names"]
    model.eval()
    print(f"Model loaded — {len(class_names)} classes")

    # --- 2. Collect and sample frames ---
    all_frames = [
        os.path.join(frame_folder, f)
        for f in os.listdir(frame_folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    if not all_frames:
        print(f"No frames found in: {frame_folder}")
        return

    selected_frames = get_sorted_frames(all_frames, FRAMES_PER_SEQUENCE)
    print(f"Found {len(all_frames)} frames → sampled {len(selected_frames)}")

    # --- 3. Extract keypoints ---
    mp_holistic    = mp.solutions.holistic
    holistic_model = mp_holistic.Holistic(
        static_image_mode=True,
        min_detection_confidence=0.5
    )

    keypoints = []
    for frame_path in selected_frames:
        kp = extract_keypoints_from_file(frame_path, holistic_model, IMAGE_SIZE)
        keypoints.append(kp)
        print(f"  Extracted: {os.path.basename(frame_path)} — "
              f"{'OK' if np.any(kp != 0) else 'EMPTY (no detection)'}")

    holistic_model.close()

    sequence = np.array(keypoints, dtype=np.float32)  # (16, 225)

    # --- 4. Check detection quality ---
    zero_frames = np.all(sequence == 0, axis=1).sum()
    print(f"\nDetection: {FRAMES_PER_SEQUENCE - zero_frames}/{FRAMES_PER_SEQUENCE} frames detected")

    if zero_frames / FRAMES_PER_SEQUENCE > 0.8:
        print("WARNING: Too many empty frames — MediaPipe detected nothing.")
        print("Check that the person is clearly visible in the frames.")
        return

    # --- 5. Normalize ---
    sequence = normalize_sequence(sequence, FRAMES_PER_SEQUENCE, EXPECTED_FEATURES)

    # --- 6. Run model ---
    tensor = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0)  # (1, 16, 225)

    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1).squeeze().numpy()

    # --- 7. Print results ---
    top5_idx   = np.argsort(probs)[::-1][:5]
    top5_probs = probs[top5_idx]
    top5_names = [class_names[i] for i in top5_idx]

    print("\n--- Prediction Results ---")
    print(f"  Top Prediction : Class {top5_names[0]} ({top5_probs[0]*100:.1f}% confidence)")
    print(f"\n  Top 5:")
    for i, (name, prob) in enumerate(zip(top5_names, top5_probs)):
        bar = "█" * int(prob * 40)
        print(f"  {i+1}. Class {name:>4} | {prob*100:5.1f}% | {bar}")

    return top5_names[0], top5_probs[0]


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python predict_single.py <path_to_frame_folder>")
        print("Example: python predict_single.py training/5/video001")
        sys.exit(1)

    folder = sys.argv[1]
    predict_sequence(folder)