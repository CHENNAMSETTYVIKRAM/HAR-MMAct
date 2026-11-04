"""
plots.py - Visualization utilities for training curves, confusion matrices, and ROC/PR curves
"""

import os
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Union
import numpy as np

# Ensure headless matplotlib without cache warning
os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    output_path: Union[str, Path],
    title: str = "Confusion Matrix",
    normalize: bool = False,
) -> None:
    """
    Renders and saves a confusion matrix heatmap.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    matrix = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] if normalize and cm.sum(axis=1).all() else cm

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(matrix, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        title=title,
        ylabel="True Label",
        xlabel="Predicted Label",
    )

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    fmt = ".2f" if normalize else "d"
    thresh = matrix.max() / 2.0
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(
                j,
                i,
                format(matrix[i, j], fmt),
                ha="center",
                va="center",
                color="white" if matrix[i, j] > thresh else "black",
                fontsize=8,
            )

    fig.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    train_accs: List[float],
    val_accs: List[float],
    output_path: Union[str, Path],
) -> None:
    """
    Plots training and validation loss and accuracy trajectories.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    epochs = range(1, len(train_losses) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Loss curve
    ax1.plot(epochs, train_losses, label="Train Loss", color="royalblue", lw=2)
    ax1.plot(epochs, val_losses, label="Val Loss", color="crimson", lw=2)
    ax1.set_title("Cross-Entropy Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.6)

    # Accuracy curve
    ax2.plot(epochs, train_accs, label="Train Accuracy", color="royalblue", lw=2)
    ax2.plot(epochs, val_accs, label="Val Accuracy", color="crimson", lw=2)
    ax2.set_title("Classification Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.6)

    fig.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_loss_curve_standalone(
    train_losses: List[float],
    val_losses: List[float],
    output_path: Union[str, Path],
    num_layers: int = 4,
) -> None:
    """
    Plots standalone 'Train vs Validation Loss ({num_layers} Layers, {epochs} Epochs)'.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    num_epochs = len(train_losses)
    epochs = list(range(num_epochs))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(epochs, train_losses, label="Train Loss", color="#2b6cb0", lw=1.8)
    ax.plot(epochs, val_losses, label="Validation Loss", color="#dd6b20", lw=1.8)
    ax.set_title(f"Train vs Validation Loss ({num_layers} Layers, {num_epochs} Epochs)", fontsize=11)
    ax.set_xlabel("Epochs", fontsize=10)
    ax.set_ylabel("Loss", fontsize=10)
    ax.legend(loc="upper right", frameon=True)
    ax.grid(True, color="#cbd5e0", linestyle="-", linewidth=0.7, alpha=0.7)

    fig.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_accuracy_curve_standalone(
    train_accs: List[float],
    eval_accs: List[float],
    output_path: Union[str, Path],
    eval_label: str = "Validation",
    num_layers: int = 4,
) -> None:
    """
    Plots standalone 'Train vs {Validation/Test} Accuracy ({num_layers} Layers, {epochs} Epochs)'.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    num_epochs = len(train_accs)
    epochs = list(range(num_epochs))

    # Normalize to 0-1 if in percent (0-100)
    t_acc = [a / 100.0 if a > 1.0 else a for a in train_accs]
    e_acc = [a / 100.0 if a > 1.0 else a for a in eval_accs]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(epochs, t_acc, label="Train Accuracy", color="#2b6cb0", lw=1.8)
    ax.plot(epochs, e_acc, label=f"{eval_label} Accuracy", color="#dd6b20", lw=1.8)
    ax.set_title(f"Train vs {eval_label} Accuracy ({num_layers} Layers, {num_epochs} Epochs)", fontsize=11)
    ax.set_xlabel("Epochs" if eval_label == "Validation" else "Epoch", fontsize=10)
    ax.set_ylabel("Accuracy", fontsize=10)
    ax.legend(loc="lower right", frameon=True)
    ax.grid(True, color="#cbd5e0", linestyle="-", linewidth=0.7, alpha=0.7)

    fig.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_annotated_train_test_accuracy(
    train_accs: List[float],
    test_accs: List[float],
    output_path: Union[str, Path],
) -> None:
    """
    Plots 'Train Accuracy vs Test Accuracy ({epochs} Epochs)' styled with
    square markers and final value callout arrows.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    num_epochs = len(train_accs)
    epochs = list(range(1, num_epochs + 1))

    # Scale to percentage (0 - 100)
    t_acc = [a * 100.0 if a <= 1.0 else a for a in train_accs]
    te_acc = [a * 100.0 if a <= 1.0 else a for a in test_accs]

    fig, ax = plt.subplots(figsize=(8, 4.8))
    # Sample markers periodically or all
    mark_every = max(1, num_epochs // 10)
    ax.plot(epochs, t_acc, label="Train Accuracy", color="#0000cd", lw=2, marker="s", markersize=4, markevery=mark_every)
    ax.plot(epochs, te_acc, label="Test Accuracy", color="#cc1100", lw=2, marker="s", markersize=4, markevery=mark_every)

    final_train = t_acc[-1]
    final_test = te_acc[-1]

    # Annotations
    ax.annotate(
        f"Final Train: {final_train:.1f}%",
        xy=(num_epochs, final_train),
        xytext=(num_epochs - max(3, num_epochs // 8), final_train - 4),
        arrowprops=dict(arrowstyle="->", color="#0000cd", lw=1.2),
        fontsize=8.5,
        ha="center",
    )
    ax.annotate(
        f"Final Test: {final_test:.1f}%",
        xy=(num_epochs, final_test),
        xytext=(num_epochs - max(3, num_epochs // 8), final_test - 8),
        arrowprops=dict(arrowstyle="->", color="#cc1100", lw=1.2),
        fontsize=8.5,
        ha="center",
    )

    ax.set_title(f"Train Accuracy vs Test Accuracy ({num_epochs} Epochs)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Epochs", fontsize=10, fontweight="bold")
    ax.set_ylabel("Accuracy (%)", fontsize=10, fontweight="bold")
    ax.set_xlim([1, num_epochs + 1])
    ax.set_ylim([min(min(t_acc), min(te_acc)) - 5, max(max(t_acc), max(te_acc)) + 5])
    ax.legend(loc="lower right", shadow=True)
    ax.grid(True, color="#e2e8f0", linestyle="-", linewidth=0.8)

    fig.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_multiclass_roc(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: List[str],
    output_path: Union[str, Path],
) -> None:
    """
    Plots One-vs-Rest ROC curves for all classes.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    num_classes = len(class_names)

    # Binarize
    y_onehot = np.zeros((len(y_true), num_classes))
    for i, c in enumerate(y_true):
        if 0 <= c < num_classes:
            y_onehot[i, c] = 1.0

    has_curve = False
    for i in range(num_classes):
        if y_onehot[:, i].sum() > 0:
            fpr, tpr, _ = roc_curve(y_onehot[:, i], y_prob[:, i])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, lw=1.5, label=f"{class_names[i]} (AUC = {roc_auc:.2f})")
            has_curve = True

    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("One-vs-Rest ROC Curves")
    if has_curve:
        ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.6)

    fig.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_multiclass_pr(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: List[str],
    output_path: Union[str, Path],
) -> None:
    """
    Plots One-vs-Rest Precision-Recall curves for all classes.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    num_classes = len(class_names)

    y_onehot = np.zeros((len(y_true), num_classes))
    for i, c in enumerate(y_true):
        if 0 <= c < num_classes:
            y_onehot[i, c] = 1.0

    has_curve = False
    for i in range(num_classes):
        if y_onehot[:, i].sum() > 0:
            precision, recall, _ = precision_recall_curve(y_onehot[:, i], y_prob[:, i])
            ap = average_precision_score(y_onehot[:, i], y_prob[:, i])
            ax.plot(recall, precision, lw=1.5, label=f"{class_names[i]} (AP = {ap:.2f})")
            has_curve = True

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves")
    if has_curve:
        ax.legend(loc="lower left", fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.6)

    fig.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
