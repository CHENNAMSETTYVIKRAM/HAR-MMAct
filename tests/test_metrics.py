"""
test_metrics.py - Unit tests for classification metrics, probability validation, and ROC/PR
"""

import numpy as np
import pytest

from src.utils.metrics import compute_classification_metrics


def test_metrics_basic_perfect_predictions():
    classes = ["class_0", "class_1", "class_2"]
    y_true = [0, 1, 2, 0, 1, 2]
    y_pred = [0, 1, 2, 0, 1, 2]
    y_prob = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.9, 0.05, 0.05],
        [0.05, 0.9, 0.05],
        [0.05, 0.05, 0.9],
    ])

    m = compute_classification_metrics(y_true, y_pred, y_prob, classes)
    assert m["accuracy"] == 1.0
    assert m["macro_f1"] == 1.0
    assert m["macro_precision"] == 1.0
    assert m["macro_recall"] == 1.0
    assert "roc_auc_ovr" in m
    assert m["roc_auc_ovr"] == 1.0
    assert "average_precision_macro" in m
    assert m["average_precision_macro"] == 1.0


def test_probability_validation_rejects_nan_and_inf():
    classes = ["c0", "c1"]
    y_true = [0, 1]
    y_pred = [0, 1]

    # NaN probability
    y_prob_nan = np.array([[np.nan, 0.5], [0.5, 0.5]])
    with pytest.raises(ValueError, match="invalid probabilities"):
        compute_classification_metrics(y_true, y_pred, y_prob_nan, classes)

    # Inf probability
    y_prob_inf = np.array([[np.inf, 0.5], [0.5, 0.5]])
    with pytest.raises(ValueError, match="invalid probabilities"):
        compute_classification_metrics(y_true, y_pred, y_prob_inf, classes)


def test_probability_validation_rejects_negative():
    classes = ["c0", "c1"]
    y_true = [0, 1]
    y_pred = [0, 1]

    y_prob_neg = np.array([[-0.1, 1.1], [0.5, 0.5]])
    with pytest.raises(ValueError, match="invalid probabilities"):
        compute_classification_metrics(y_true, y_pred, y_prob_neg, classes)


def test_probability_validation_rejects_wrong_shape():
    classes = ["c0", "c1", "c2"]
    y_true = [0, 1]
    y_pred = [0, 1]

    # Wrong class dimension (2 instead of 3)
    y_prob_wrong = np.array([[0.5, 0.5], [0.5, 0.5]])
    with pytest.raises(ValueError, match="y_prob must have shape"):
        compute_classification_metrics(y_true, y_pred, y_prob_wrong, classes)


def test_probability_validation_rejects_zero_row_sum():
    classes = ["c0", "c1"]
    y_true = [0, 1]
    y_pred = [0, 1]

    y_prob_zero = np.array([[0.0, 0.0], [0.5, 0.5]])
    with pytest.raises(ValueError, match="zero sum"):
        compute_classification_metrics(y_true, y_pred, y_prob_zero, classes)
