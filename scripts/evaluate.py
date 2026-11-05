"""
evaluate.py - Standalone Evaluation Script for Multimodal and Unimodal HAR_MMAct Baselines

Evaluates:
1. Multimodal Fused Model
2. Video-Only Unimodal Baseline
3. Sensor-Only Unimodal Baseline
Computes complete metrics (Accuracy, Macro F1, Precision, Recall, Confusion Matrix, OvR ROC AUC, AP)
and generates publication-ready visualization artifacts.
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.config import SplitConfig, VERIFIED_TARGET_CLASSES
from src.data.indexer import MMActIndexer
from src.data.splits import DatasetSplitter, verify_split_isolation
from src.data.dataset import MMActMultimodalDataset
from src.models.fusion_model import MultimodalHARModel
from src.utils.metrics import compute_classification_metrics
from src.utils.plots import (
    plot_confusion_matrix,
    plot_multiclass_roc,
    plot_multiclass_pr,
    plot_per_class_metrics,
    plot_per_class_metric_bars,
    plot_modality_ablation,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate HAR_MMAct Models")
    parser.add_argument("--checkpoint", type=str, default="outputs/checkpoints/best_model.pth", help="Checkpoint path")
    parser.add_argument("--data_root", type=str, default=None, help="Dataset root (if None, read from checkpoint)")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--cache_dir", type=str, default=None, help="Cache directory for preprocessed tensors")
    parser.add_argument("--output_file", type=str, default="outputs/metrics.json", help="Path to save evaluation metrics")
    parser.add_argument("--plots_dir", type=str, default="outputs/plots", help="Directory for evaluation plots")
    parser.add_argument("--smoke_test", action="store_true", help="Force evaluate on synthetic data split")
    return parser.parse_args()


@torch.no_grad()
def run_evaluation_for_mode(model, loader, device, mode, class_names, use_amp=False):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    for video, sensor, label, _ in loader:
        video = video.to(device, non_blocking=True)
        sensor = sensor.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)

        if use_amp and device.type == "cuda":
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(video=video, sensor=sensor, override_mode=mode)
        else:
            logits = model(video=video, sensor=sensor, override_mode=mode)

        probs = torch.softmax(logits, dim=-1)

        all_preds.extend(logits.argmax(dim=-1).cpu().numpy())
        all_labels.extend(label.cpu().numpy())
        all_probs.append(probs.cpu().numpy())

    probs_array = np.concatenate(all_probs, axis=0) if all_probs else None
    metrics = compute_classification_metrics(
        y_true=all_labels,
        y_pred=all_preds,
        y_prob=probs_array,
        classes=class_names,
    )
    return metrics, np.array(all_labels), probs_array


def main():
    args = parse_args()
    ckpt_path = Path(args.checkpoint)

    if not ckpt_path.exists():
        print(f"[ERROR] Checkpoint not found at: {ckpt_path.resolve()}")
        print("Please train a model first with 'python scripts/train.py'")
        sys.exit(1)

    print(f"Loading checkpoint from: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    saved_cfg = checkpoint.get("config", {})
    is_synthetic = checkpoint.get("is_synthetic", False)
    classes = checkpoint.get("classes", VERIFIED_TARGET_CLASSES)

    # Determine dataset root
    if args.smoke_test:
        data_root = Path("data/synthetic_smoke_test")
        is_synthetic = True
    elif args.data_root:
        data_root = Path(args.data_root)
    else:
        data_root = Path(saved_cfg.get("data_root", "data/genuine_mmact"))
        if is_synthetic:
            data_root = Path("data/synthetic_smoke_test")

    if not data_root.exists():
        print(f"[ERROR] Dataset directory not found at: {data_root.resolve()}")
        sys.exit(1)

    print("=" * 80)
    print(" HAR_MMAct: Standalone Model Evaluation & Benchmark")
    print(f" Checkpoint   : {ckpt_path} (Epoch {checkpoint.get('epoch', '?')})")
    print(f" Dataset Root : {data_root} ({'SYNTHETIC' if is_synthetic else 'GENUINE'})")
    print("=" * 80)

    # Index and split
    indexer = MMActIndexer(dataset_root=data_root)
    report = indexer.index()
    if report.paired_samples_count == 0:
        print(f"[ERROR] No paired samples found in {data_root}")
        sys.exit(1)

    split_strategy = saved_cfg.get("split_strategy", "subject")
    splitter = DatasetSplitter(SplitConfig(split_strategy=split_strategy))
    train_samples, val_samples, test_samples = splitter.split(report.paired_samples)

    eval_samples = test_samples if len(test_samples) > 0 else val_samples
    eval_split_name = "test" if len(test_samples) > 0 else "val"
    print(f"Evaluating on held-out {eval_split_name} split ({len(eval_samples)} samples)...")

    # DataLoader
    c_eval = (Path(args.cache_dir) / eval_split_name) if args.cache_dir else None
    eval_ds = MMActMultimodalDataset(eval_samples, cache_dir=c_eval, enable_cache=(c_eval is not None))
    eval_loader = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=(device.type == "cuda"))

    # Initialize model and load weights
    model = MultimodalHARModel(
        num_classes=len(classes),
        video_channels=3,
        sensor_channels=9,
        video_embed_dim=128,
        sensor_embed_dim=128,
        fusion_hidden_dim=128,
        video_backbone=saved_cfg.get("video_backbone", "resnet18"),
        sensor_backbone=saved_cfg.get("sensor_backbone", "inception_time"),
        pretrained_video=False,  # Loaded from checkpoint
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    use_amp = (device.type == "cuda")

    # Run evaluations
    print("\n[1/3] Evaluating Multimodal Fusion Model...")
    mm_metrics, y_true, mm_probs = run_evaluation_for_mode(model, eval_loader, device, "multimodal", classes, use_amp=use_amp)

    print("[2/3] Evaluating Video-Only Baseline...")
    vo_metrics, _, vo_probs = run_evaluation_for_mode(model, eval_loader, device, "video_only", classes, use_amp=use_amp)

    print("[3/3] Evaluating Sensor-Only Baseline...")
    so_metrics, _, so_probs = run_evaluation_for_mode(model, eval_loader, device, "sensor_only", classes, use_amp=use_amp)

    # Display comparison table
    print("\n" + "=" * 80)
    print(" EVALUATION RESULTS SUMMARY")
    print("=" * 80)
    print(f"{'Modality':<18} | {'Accuracy':<10} | {'Macro F1':<10} | {'Precision':<10} | {'Recall':<10}")
    print("-" * 80)
    print(f"{'Multimodal (Fused)':<18} | {mm_metrics['accuracy']*100:6.2f}%    | {mm_metrics['macro_f1']:8.4f}   | {mm_metrics['macro_precision']:8.4f}   | {mm_metrics['macro_recall']:8.4f}")
    print(f"{'Video-Only':<18} | {vo_metrics['accuracy']*100:6.2f}%    | {vo_metrics['macro_f1']:8.4f}   | {vo_metrics['macro_precision']:8.4f}   | {vo_metrics['macro_recall']:8.4f}")
    print(f"{'Sensor-Only':<18} | {so_metrics['accuracy']*100:6.2f}%    | {so_metrics['macro_f1']:8.4f}   | {so_metrics['macro_precision']:8.4f}   | {so_metrics['macro_recall']:8.4f}")
    print("=" * 80)

    # Save metrics JSON
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results = {
        "checkpoint": str(ckpt_path),
        "dataset_root": str(data_root),
        "is_synthetic": is_synthetic,
        "eval_split": eval_split_name,
        "eval_samples_count": len(eval_samples),
        "multimodal": mm_metrics,
        "video_only": vo_metrics,
        "sensor_only": so_metrics,
    }
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved complete metrics to: {output_path}")

    # Save metrics CSV summary table
    csv_metrics_path = output_path.parent / "metrics.csv"
    with open(csv_metrics_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Model", "Modality", "Split", "Accuracy (%)", "Macro F1", "Macro Precision", "Macro Recall"])
        for mod_name, m_dict in [("Multimodal (Fused)", mm_metrics), ("Video-Only", vo_metrics), ("Sensor-Only", so_metrics)]:
            writer.writerow([
                saved_cfg.get("video_backbone", "model"),
                mod_name,
                eval_split_name,
                round(m_dict.get("accuracy", 0.0) * 100.0, 2),
                round(m_dict.get("macro_f1", 0.0), 4),
                round(m_dict.get("macro_precision", 0.0), 4),
                round(m_dict.get("macro_recall", 0.0), 4),
            ])
    print(f"Saved metrics CSV table to: {csv_metrics_path}")

    # Save test predictions CSV (sample_idx, ground_truth, predicted, confidence, probabilities)
    if y_true is not None and mm_probs is not None:
        preds_path = output_path.parent / "test_predictions.csv"
        pred_labels = np.argmax(mm_probs, axis=-1)
        confidences = np.max(mm_probs, axis=-1)
        with open(preds_path, "w", newline="") as f:
            writer = csv.writer(f)
            header = ["sample_idx", "ground_truth", "predicted", "confidence"] + [f"prob_{c}" for c in classes]
            writer.writerow(header)
            for idx in range(len(y_true)):
                gt_idx = int(y_true[idx])
                pl_idx = int(pred_labels[idx])
                row = [
                    idx,
                    classes[gt_idx] if gt_idx < len(classes) else str(gt_idx),
                    classes[pl_idx] if pl_idx < len(classes) else str(pl_idx),
                    round(float(confidences[idx]), 4),
                ] + [round(float(p), 4) for p in mm_probs[idx]]
                writer.writerow(row)
        print(f"Saved test predictions table to: {preds_path}")

    # Generate plots
    plots_dir = Path(args.plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    video_bb = saved_cfg.get("video_backbone", "model")

    if "confusion_matrix" in mm_metrics:
        cm_array = np.array(mm_metrics["confusion_matrix"])
        title = f"Test Confusion Matrix - {video_bb.upper()} ({split_strategy} split)"
        plot_confusion_matrix(cm_array, classes, plots_dir / "test_confusion_matrix.png", title=title)
        plot_confusion_matrix(cm_array, classes, plots_dir / f"test_confusion_matrix_{video_bb}.png", title=title)
        print(f"Saved confusion matrix plot to: {plots_dir / 'test_confusion_matrix.png'}")

    if y_true is not None and mm_probs is not None:
        plot_multiclass_roc(y_true, mm_probs, classes, plots_dir / "test_roc_ovr.png")
        plot_multiclass_pr(y_true, mm_probs, classes, plots_dir / "test_pr_curves.png")
        plot_multiclass_roc(y_true, mm_probs, classes, plots_dir / f"test_roc_ovr_{video_bb}.png")
        plot_multiclass_pr(y_true, mm_probs, classes, plots_dir / f"test_pr_curves_{video_bb}.png")
        print(f"Saved ROC/PR curves to: {plots_dir / 'test_roc_ovr.png'}")

    if "per_class" in mm_metrics:
        plot_per_class_metrics(
            mm_metrics["per_class"],
            plots_dir / f"test_per_class_{video_bb}.png",
            model_name=video_bb.upper(),
            split_name=f"Test ({split_strategy} split)",
        )
        # Standalone per-class Precision, Recall, and F1-score bar charts
        plot_per_class_metric_bars(
            mm_metrics["per_class"],
            "Precision",
            plots_dir / f"test_precision_bars_{video_bb}.png",
            model_name=video_bb.upper(),
            split_name=f"Test ({split_strategy} split)",
        )
        plot_per_class_metric_bars(
            mm_metrics["per_class"],
            "Recall",
            plots_dir / f"test_recall_bars_{video_bb}.png",
            model_name=video_bb.upper(),
            split_name=f"Test ({split_strategy} split)",
        )
        plot_per_class_metric_bars(
            mm_metrics["per_class"],
            "F1-Score",
            plots_dir / f"test_f1_bars_{video_bb}.png",
            model_name=video_bb.upper(),
            split_name=f"Test ({split_strategy} split)",
        )
        print(f"Saved per-class metrics bar charts to: {plots_dir}")

    # Modality ablation comparison plot
    ablation_dict = {
        "Video-Only": vo_metrics,
        "Sensor-Only": so_metrics,
        "Multimodal": mm_metrics,
    }
    plot_modality_ablation(ablation_dict, plots_dir / f"test_modality_ablation_{video_bb}.png", split_name=f"Test ({split_strategy} split)")
    plot_modality_ablation(ablation_dict, plots_dir / "test_modality_ablation.png", split_name=f"Test ({split_strategy} split)")
    print(f"Saved modality ablation plot to: {plots_dir / 'test_modality_ablation.png'}")


if __name__ == "__main__":
    main()
