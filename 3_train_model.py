'''
File: 3_train_model.py
Author: ABx_Lab
Trains a Bidirectional LSTM on extracted skeleton sequences.
'''

import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from config import (
    OUTPUT_DIR, FRAMES_PER_SEQUENCE, NUM_CLASSES,
    BATCH_SIZE, EPOCHS, LEARNING_RATE,EXPECTED_FEATURES,WEIGHT_DECAY
)
from utils import load_data_paths, normalize_sequence
MODEL_SAVE_PATH   = "mediapipe_lstm_imigue.pth"

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
            torch.tensor(data,                  dtype=torch.float32),
            torch.tensor(self.labels[idx],      dtype=torch.long),
        )

# model LSTM
class SkeletonLSTM(nn.Module):
    def __init__(self, input_size=EXPECTED_FEATURES, hidden_size=128,
                 num_layers=2, num_classes=NUM_CLASSES):
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


# Loss
class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0):
        super().__init__()
        self.alpha, self.gamma = alpha, gamma

    def forward(self, inputs, targets):
        ce = F.cross_entropy(inputs, targets, reduction="none")
        return (self.alpha * (1 - torch.exp(-ce)) ** self.gamma * ce).mean()


# Evaluate
def evaluate(model, loader, device):
    model.eval()
    total_loss, correct, seen = 0.0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y     = x.to(device), y.to(device)
            out      = model(x)
            total_loss += F.cross_entropy(out, y).item()
            correct  += (out.argmax(1) == y).sum().item()
            seen     += y.size(0)
    return total_loss / max(len(loader), 1), correct / max(seen, 1)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    # load_data_paths from utils handles train/test split correctly
    train_paths, train_labels, _, _, class_names = load_data_paths(OUTPUT_DIR)
    num_classes = len(class_names)
    print(f"Found {len(train_paths)} train samples across {num_classes} classes")

    train_loader = DataLoader(
        SkeletonDataset(train_paths, train_labels),
        batch_size=BATCH_SIZE, shuffle=True, drop_last=True
    )

    # Use test split as validation during training
    _, _, val_paths, val_labels, _ = load_data_paths(OUTPUT_DIR)
    val_loader = DataLoader(
        SkeletonDataset(val_paths, val_labels),
        batch_size=BATCH_SIZE, shuffle=False
    )

    model     = SkeletonLSTM(num_classes=num_classes).to(device)
    criterion = FocalLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    best_val_acc = 0.0
    for epoch in range(EPOCHS):
        model.train()
        train_loss, correct, seen = 0.0, 0, 0

        for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out  = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            correct    += (out.argmax(1) == y).sum().item()
            seen       += y.size(0)

        val_loss, val_acc = evaluate(model, val_loader, device)
        print(f"Epoch {epoch+1}/{EPOCHS} | "
              f"Train Loss: {train_loss/len(train_loader):.4f} | "
              f"Train Acc: {correct/seen:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "class_names":      class_names,
                "input_size":       EXPECTED_FEATURES,
                "sequence_length":  FRAMES_PER_SEQUENCE,
            }, MODEL_SAVE_PATH)
            print(f"*** Best model saved — Val Acc: {val_acc:.4f} ***")

    print("--- Training complete! ---")


if __name__ == "__main__":
    main()