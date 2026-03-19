import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix


def plot_confusion_matrix(preds, labels, class_names):
    cm = confusion_matrix(
        labels, preds,
        labels=list(range(len(class_names)))
    )
    cm_pct = cm.astype("float") / np.where(
        cm.sum(axis=1, keepdims=True) > 0,
        cm.sum(axis=1, keepdims=True), 1
    )

    plt.figure(figsize=(20, 16))
    sns.heatmap(cm_pct, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names,
                linewidths=0.5)
    plt.xlabel("Predicted Label", fontsize=13)
    plt.ylabel("True Label",      fontsize=13)
    plt.title("Confusion Matrix — BiLSTM Micro Gesture", fontsize=15)
    plt.tight_layout()
    plt.savefig("plot_confusion_matrix.png", dpi=150)
    plt.close()
    print("Saved: plot_confusion_matrix.png")


def plot_f1_per_class(preds, labels, class_names):
    report = classification_report(
        labels, preds,
        labels=list(range(len(class_names))),
        target_names=class_names,
        zero_division=0,
        output_dict=True,
    )
    f1     = np.array([report[c]["f1-score"] for c in class_names])
    colors = ["#2ecc71" if v >= 0.5 else "#e74c3c" for v in f1]
    mean   = np.mean(f1)

    plt.figure(figsize=(18, 6))
    bars = plt.bar(class_names, f1, color=colors,
                   edgecolor="black", linewidth=0.5)
    plt.axhline(y=mean, color="navy", linestyle="--",
                linewidth=1.5, label=f"Mean F1: {mean:.2f}")
    plt.xlabel("Gesture Class", fontsize=12)
    plt.ylabel("F1 Score",      fontsize=12)
    plt.title("F1 Score per Class — BiLSTM", fontsize=14)
    plt.xticks(rotation=45, ha="right")
    plt.ylim(0, 1.15)
    plt.legend()

    for bar, val in zip(bars, f1):
        plt.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.02,
                 f"{val:.2f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig("plot_f1_per_class.png", dpi=150)
    plt.close()
    print("Saved: plot_f1_per_class.png")


def plot_summary_card(preds, labels, class_names):
    report   = classification_report(
        labels, preds,
        labels=list(range(len(class_names))),
        target_names=class_names,
        zero_division=0,
        output_dict=True,
    )
    accuracy = np.mean(preds == labels) * 100
    macro_f1 = report["macro avg"]["f1-score"]  * 100
    macro_p  = report["macro avg"]["precision"] * 100
    macro_r  = report["macro avg"]["recall"]    * 100

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    metrics   = [
        ("Accuracy",  f"{accuracy:.1f}%", "#2ecc71"),
        ("Precision", f"{macro_p:.1f}%",  "#3498db"),
        ("Recall",    f"{macro_r:.1f}%",  "#e67e22"),
        ("F1-Score",  f"{macro_f1:.1f}%", "#9b59b6"),
    ]

    for ax, (title, value, color) in zip(axes, metrics):
        ax.set_facecolor(color)
        ax.text(0.5, 0.6,  value, ha="center", va="center",
                fontsize=36, fontweight="bold", color="white",
                transform=ax.transAxes)
        ax.text(0.5, 0.25, title, ha="center", va="center",
                fontsize=14, color="white", transform=ax.transAxes)
        ax.axis("off")

    plt.suptitle("BiLSTM Micro Gesture — Overall Results",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("plot_summary_card.png", dpi=150)
    plt.close()
    print("Saved: plot_summary_card.png")