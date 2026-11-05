"""
test_plots.py - Unit tests for plotting utilities and publication visualization suite
"""

import numpy as np
from pathlib import Path

from src.utils.plots import (
    plot_training_curves,
    plot_loss_curve_standalone,
    plot_accuracy_curve_standalone,
    plot_train_eval_annotated,
    plot_confusion_matrix,
    plot_multiclass_roc,
    plot_multiclass_pr,
    plot_per_class_metrics,
    plot_per_class_metric_bars,
    plot_cross_model_comparison,
    plot_modality_ablation,
    plot_hyperparameter_trials,
)


def test_train_eval_annotated_plot(tmp_path):
    train_acc = [60.0 + i * 0.5 for i in range(50)]
    val_acc = [58.0 + i * 0.45 for i in range(50)]
    out_path = tmp_path / "train_vs_val_annotated.png"

    plot_train_eval_annotated(train_acc, val_acc, out_path, metric_name="Accuracy", eval_label="Validation", epochs_label=50)
    assert out_path.exists()
    assert out_path.stat().st_size > 1000


def test_per_class_metric_bars(tmp_path):
    classes = ["walking", "jumping", "running", "standing"]
    per_class_dict = {
        c: {"precision": 0.85, "recall": 0.80, "f1": 0.82} for c in classes
    }

    p_path = tmp_path / "precision.png"
    r_path = tmp_path / "recall.png"
    f_path = tmp_path / "f1.png"

    plot_per_class_metric_bars(per_class_dict, "Precision", p_path)
    plot_per_class_metric_bars(per_class_dict, "Recall", r_path)
    plot_per_class_metric_bars(per_class_dict, "F1-Score", f_path)

    assert p_path.exists() and p_path.stat().st_size > 1000
    assert r_path.exists() and r_path.stat().st_size > 1000
    assert f_path.exists() and f_path.stat().st_size > 1000


def test_cross_model_comparison_plot(tmp_path):
    models = {
        "Custom CNN": {"accuracy": 0.82, "macro_f1": 0.80, "macro_precision": 0.81, "macro_recall": 0.80},
        "Tuned CNN": {"accuracy": 0.84, "macro_f1": 0.83, "macro_precision": 0.84, "macro_recall": 0.82},
        "MobileNetV2": {"accuracy": 0.86, "macro_f1": 0.85, "macro_precision": 0.86, "macro_recall": 0.85},
    }
    out_p = tmp_path / "model_comparison.png"
    plot_cross_model_comparison(models, out_p)
    assert out_p.exists() and out_p.stat().st_size > 1000


def test_modality_ablation_plot(tmp_path):
    ablation = {
        "Video-Only": {"accuracy": 0.85, "macro_f1": 0.84},
        "Sensor-Only": {"accuracy": 0.72, "macro_f1": 0.70},
        "Multimodal": {"accuracy": 0.89, "macro_f1": 0.88},
    }
    out_p = tmp_path / "ablation.png"
    plot_modality_ablation(ablation, out_p)
    assert out_p.exists() and out_p.stat().st_size > 1000


def test_hyperparameter_trials_plot(tmp_path):
    trials = [
        {"trial_id": 1, "lr": 1e-4, "weight_decay": 1e-3, "dropout": 0.35, "val_accuracy": 82.5},
        {"trial_id": 2, "lr": 3e-4, "weight_decay": 1e-3, "dropout": 0.35, "val_accuracy": 79.1},
        {"trial_id": 3, "lr": 1e-4, "weight_decay": 5e-3, "dropout": 0.45, "val_accuracy": 84.2},
    ]
    out_p = tmp_path / "trials.png"
    plot_hyperparameter_trials(trials, out_p)
    assert out_p.exists() and out_p.stat().st_size > 1000


def test_roc_and_pr_with_micro_macro(tmp_path):
    classes = ["walking", "running", "standing"]
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_prob = np.array([
        [0.8, 0.1, 0.1],
        [0.1, 0.7, 0.2],
        [0.2, 0.2, 0.6],
        [0.7, 0.2, 0.1],
        [0.2, 0.6, 0.2],
        [0.1, 0.3, 0.6],
    ])

    roc_p = tmp_path / "roc.png"
    pr_p = tmp_path / "pr.png"

    plot_multiclass_roc(y_true, y_prob, classes, roc_p)
    plot_multiclass_pr(y_true, y_prob, classes, pr_p)

    assert roc_p.exists() and roc_p.stat().st_size > 1000
    assert pr_p.exists() and pr_p.stat().st_size > 1000
