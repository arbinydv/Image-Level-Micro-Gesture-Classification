'''
File: test_model.py
Author: ABx_Lab
Evaluates the trained BiLSTM and generates all report visualizations.
'''

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report
from tqdm import tqdm

from config import OUTPUT_DIR, FRAMES_PER_SEQUENCE, BATCH_SIZE, EXPECTED_FEATURES
from utils import (
    load_data_paths,
    normalize_sequence,
)
from generate_plot import (
    plot_confusion_matrix,
    plot_f1_per_class, plot_summary_card, )

MODEL_SAVE_PATH = "mediapipe_lstm_imigue.pth"

class SkeletonDataset(Dataset):
    def __init__(self, file_paths, labels):
        self.file_paths = file_paths
        self.labels     = labels

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        data = normalize_sequence(
            np.load(self.file_paths[idx]), FRAMES_PER_SEQUENCE, EXPECTED_FEATURES
        )
        return (
            torch.tensor(data,             dtype=torch.float32),
            torch.tensor(self.labels[idx], dtype=torch.long),
        )

class SkeletonLSTM(nn.Module):
    def __init__(self, input_size=EXPECTED_FEATURES, hidden_size=128,
                 num_layers=2, num_classes=32):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, bidirectional=True,
                            dropout=0.3 if num_layers > 1 else 0.0)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        return self.fc(torch.cat((hn[-2], hn[-1]), dim=1))


# Test/Reference
def run_inference(model, loader, device):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for x, y in tqdm(loader, desc="Testing"):
            x, y  = x.to(device), y.to(device)
            logits = model(x)
            probs  = torch.softmax(logits, dim=1)
            all_preds.extend(logits.argmax(dim=1).cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    return np.array(all_preds), np.array(all_labels), np.array(all_probs)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on: {device}")

    # Load checkpoint
    checkpoint  = torch.load(MODEL_SAVE_PATH, map_location=device)
    class_names = checkpoint["class_names"]
    num_classes = len(class_names)
    print(f"Model loaded — {num_classes} classes")

    # Build model
    model = SkeletonLSTM(num_classes=num_classes).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    # Load test data
    _, _, test_paths, test_labels, _ = load_data_paths(OUTPUT_DIR)
    print(f"Test samples: {len(test_paths)}")

    # Warn about missing classes
    missing = set(range(num_classes)) - set(np.unique(test_labels))
    if missing:
        print(f"WARNING: Missing from test set: {[class_names[i] for i in missing]}")

    loader = DataLoader(
        SkeletonDataset(test_paths, test_labels),
        batch_size=BATCH_SIZE, shuffle=False
    )

    # Run inference
    preds, labels, probs = run_inference(model, loader, device)

    # Print metrics
    accuracy = np.mean(preds == labels) * 100
    print(f"\n{'='*50}")
    print(f"  Test Accuracy : {accuracy:.2f}%")
    print(f"{'='*50}\n")
    print(classification_report(
        labels, preds,
        labels=list(range(num_classes)),
        target_names=class_names,
        zero_division=0
    ))

    # Generate plots — all from utils
    print("Generating plots...")
    plot_confusion_matrix(preds, labels, class_names)
    plot_f1_per_class(preds,     labels, class_names)
    plot_summary_card(preds,     labels, class_names)
    print("\nDone. All plots saved.")


if __name__ == "__main__":
    main()