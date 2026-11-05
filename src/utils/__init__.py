"""
Utilities package for HAR_MMAct.
"""

from src.utils.metrics import compute_classification_metrics
from src.utils.plots import (
    plot_confusion_matrix,
    plot_training_curves,
    plot_loss_curve_standalone,
    plot_accuracy_curve_standalone,
    plot_annotated_train_test_accuracy,
    plot_train_eval_annotated,
    plot_multiclass_roc,
    plot_multiclass_pr,
    plot_cnn_vs_mobilenet_comparison,
    plot_per_class_metrics,
    plot_per_class_metric_bars,
    plot_cross_model_comparison,
    plot_modality_ablation,
    plot_hyperparameter_trials,
)

__all__ = [
    "compute_classification_metrics",
    "plot_confusion_matrix",
    "plot_training_curves",
    "plot_loss_curve_standalone",
    "plot_accuracy_curve_standalone",
    "plot_annotated_train_test_accuracy",
    "plot_train_eval_annotated",
    "plot_multiclass_roc",
    "plot_multiclass_pr",
    "plot_cnn_vs_mobilenet_comparison",
    "plot_per_class_metrics",
    "plot_per_class_metric_bars",
    "plot_cross_model_comparison",
    "plot_modality_ablation",
    "plot_hyperparameter_trials",
]
