"""
metrics.py - Evaluation Metrics for Multimodal HAR Classification

Calculates accuracy, macro F1, confusion matrix, OvR ROC AUC, and Average Precision.
Provides transparent error handling and logging if classes are unrepresented in small splits.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
)

from src.config import VERIFIED_TARGET_CLASSES


def compute_classification_metrics(
    y_true: Union[np.ndarray, List[int]],
    y_pred: Union[np.ndarray, List[int]],
    y_prob: Optional[np.ndarray] = None,
    classes: Optional[List[str]] = None,
) -> Dict[str, Union[float, np.ndarray, str]]:
    """
    Computes standard multiclass metrics.
    y_true : 1D array of ground truth integer labels
    y_pred : 1D array of predicted integer labels
    y_prob : (N, num_classes) predicted class probabilities
    classes: list of class names
    """
    target_classes = classes or VERIFIED_TARGET_CLASSES
    num_classes = len(target_classes)

    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(
        f1_score(y_true, y_pred, average="macro", zero_division=0)
    )
    macro_prec = float(
        precision_score(y_true, y_pred, average="macro", zero_division=0)
    )
    macro_rec = float(
        recall_score(y_true, y_pred, average="macro", zero_division=0)
    )

    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))

    metrics = {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "macro_precision": macro_prec,
        "macro_recall": macro_rec,
        "confusion_matrix": cm.tolist(),
        "total_samples": len(y_true),
    }

    # Multiclass ROC AUC and PR AUC (One-vs-Rest)
    if y_prob is not None:
        y_prob = np.asarray(y_prob, dtype=np.float32)
        unique_classes = set(y_true)

        if len(unique_classes) > 1:
            try:
                # One-vs-Rest ROC AUC
                roc_ovr = float(
                    roc_auc_score(
                        y_true,
                        y_prob,
                        multi_class="ovr",
                        average="macro",
                        labels=list(range(num_classes)),
                    )
                )
                metrics["roc_auc_ovr"] = roc_ovr
            except Exception as e:
                metrics["roc_auc_ovr_note"] = (
                    f"ROC AUC uncomputable: {str(e)}"
                )

            try:
                # Convert to one-hot for AP
                y_onehot = np.zeros((len(y_true), num_classes))
                for i, c in enumerate(y_true):
                    if 0 <= c < num_classes:
                        y_onehot[i, c] = 1.0

                ap_macro = float(
                    average_precision_score(
                        y_onehot, y_prob, average="macro"
                    )
                )
                metrics["average_precision_macro"] = ap_macro
            except Exception as e:
                metrics["average_precision_macro_note"] = (
                    f"AP uncomputable: {str(e)}"
                )
        else:
            metrics["roc_auc_ovr_note"] = (
                "Only 1 class present in true labels; multiclass ROC undefined."
            )
            metrics["average_precision_macro_note"] = (
                "Only 1 class present in true labels; multiclass PR undefined."
            )

    return metrics
