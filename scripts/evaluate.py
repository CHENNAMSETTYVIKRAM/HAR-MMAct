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
    parser.add_argument("--mode", type=str, default=None, choices=["multimodal", "video_only", "sensor_only", "auto"], help="Evaluation mode (if auto/None, infer from checkpoint)")
    parser.add_argument("--video_checkpoint", type=str, default=None, help="Independent video-only checkpoint for modality ablation comparison")
    parser.add_argument("--sensor_checkpoint", type=str, default=None, help="Independent sensor-only checkpoint for modality ablation comparison")
    parser.add_argument("--data_root", type=str, default=None, help="Dataset root (if None, read from checkpoint)")
    parser.add_argument(
        "--split_strategy",
        type=str,
        default=None,
        choices=[
            "subject",
            "cross_view",
            "cross_scene",
            "session",
            "official_x_scene",
            "official_x_view",
            "official_cross_scene",
            "official_cross_view",
        ],
        help="Split strategy (if None, read from checkpoint config)",
    )
    parser.add_argument("--split_dir", type=str, default=None, help="Directory containing official MMAct split files")
    parser.add_argument("--partition_val_for_test", action="store_true", help="Opt-in: Partition validation recordings into disjoint val and test sets (default: False)")
    parser.add_argument("--eval_split", type=str, default="auto", choices=["auto", "val", "test"], help="Evaluate validation candidates or held-out test winner")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--cache_dir", type=str, default=None, help="Cache directory for preprocessed tensors")
    parser.add_argument("--output_file", type=str, default="outputs/metrics.json", help="Path to save evaluation metrics")
    parser.add_argument("--plots_dir", type=str, default="outputs/plots", help="Directory for evaluation plots")
    parser.add_argument("--skip_if_completed", action="store_true", help="Skip evaluation if output_file already exists")
    parser.add_argument("--smoke_test", action="store_true", help="Force evaluate on synthetic data split")
    return parser.parse_args()


def load_model_from_checkpoint(ckpt_path, device, classes):
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    saved_cfg = checkpoint.get("config", {})
    ckpt_classes = checkpoint.get("classes", VERIFIED_TARGET_CLASSES)
    if ckpt_classes != classes:
        raise ValueError(
            f"Checkpoint classes ({ckpt_classes}) do not match expected target classes ({classes})."
        )
    ckpt_mode = saved_cfg.get("mode", "multimodal")
    valid_modes = {"multimodal", "video_only", "sensor_only"}
    if ckpt_mode not in valid_modes:
        raise ValueError(
            f"Checkpoint mode '{ckpt_mode}' is invalid; must be one of {valid_modes}."
        )

    model = MultimodalHARModel(
        num_classes=len(classes),
        video_channels=3,
        sensor_channels=9,
        video_embed_dim=128,
        sensor_embed_dim=128,
        fusion_hidden_dim=128,
        mode=ckpt_mode,
        video_backbone=saved_cfg.get("video_backbone", "resnet18"),
        sensor_backbone=saved_cfg.get("sensor_backbone", "inception_time"),
        pretrained_video=False,
        fusion_strategy=saved_cfg.get("fusion_strategy", "concat"),
    )

    if saved_cfg.get("video_backbone") in ("mobilenet_v2", "mobilenet"):
        try:
            import torchvision.models as tv_models
            import torch.nn as nn
            model.video_branch.backbone = tv_models.mobilenet_v2(weights=None).features
            model.video_branch.backbone_pool = nn.AdaptiveAvgPool2d((1, 1))
        except Exception:
            pass

    try:
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load state dict from {ckpt_path} strictly: {e}"
        ) from e
    model.to(device)
    return model, checkpoint, saved_cfg


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
    out_file = Path(args.output_file)
    if args.skip_if_completed and out_file.exists():
        try:
            with open(out_file) as f:
                data = json.load(f)
            if any(k in data for k in ["multimodal", "video_only", "sensor_only"]):
                print(f"[SKIP] Evaluation already completed! Reusing existing metrics at: {out_file}")
                return
        except Exception:
            pass

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

    if is_synthetic and not args.smoke_test and "synthetic" not in str(data_root).lower():
        raise ValueError(
            f"Cannot evaluate synthetic smoke-test checkpoint '{ckpt_path}' on genuine dataset '{data_root}'. "
            "Pass --smoke_test to evaluate on synthetic data or pass a genuine trained checkpoint."
        )

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

    split_strategy = args.split_strategy or saved_cfg.get("split_strategy", "subject")
    partition_test = args.partition_val_for_test or saved_cfg.get("is_custom_test_partition", False)
    split_cfg = SplitConfig(
        split_strategy=split_strategy,
        partition_val_for_test=partition_test,
        split_dir=Path(args.split_dir) if args.split_dir else None,
        dataset_root=data_root,
    )
    # Match the training notebook's required-modality filter exactly.
    required_modalities = ("acc_phone", "acc_watch", "gyro")
    valid_samples = []
    rejected_samples = []
    for sample in report.paired_samples:
        valid = True
        for modality in required_modalities:
            sensor_path = sample.sensor_paths.get(modality)
            if not sensor_path:
                valid = False
                break
            sensor_file = Path(sensor_path)
            if not sensor_file.is_file() or sensor_file.stat().st_size == 0:
                valid = False
                break
        (valid_samples if valid else rejected_samples).append(sample)

    if not valid_samples:
        raise RuntimeError("No samples remain after required sensor-file validation.")

    splitter = DatasetSplitter(split_cfg, dataset_root=data_root)
    train_samples, val_samples, test_samples = splitter.split(valid_samples)

    requested_split = args.eval_split
    if requested_split == "auto":
        requested_split = "test" if len(test_samples) > 0 else "val"

    if requested_split == "test":
        if not test_samples:
            raise RuntimeError("Test split requested but splitter produced no test samples.")
        eval_samples = test_samples
        eval_split_name = "test"
        eval_split_desc = (
            "Custom Held-Out Test Partition (Session-Disjoint from Val)"
            if partition_test else "Held-Out Test Split"
        )
    else:
        if not val_samples:
            raise RuntimeError("Validation split requested but splitter produced no validation samples.")
        eval_samples = val_samples
        eval_split_name = "val"
        eval_split_desc = "Validation Split (used for model selection)"
    print(f"Evaluating on {eval_split_desc} ({eval_split_name}) ({len(eval_samples)} samples)...")

    # Select evaluation device before creating the DataLoader
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluation device: {device}")

    # DataLoader
    c_eval = (Path(args.cache_dir) / eval_split_name) if args.cache_dir else None
    eval_ds = MMActMultimodalDataset(eval_samples, cache_dir=c_eval, enable_cache=(c_eval is not None))
    eval_loader = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=(device.type == "cuda"))

    # Load primary model
    model, checkpoint, saved_cfg = load_model_from_checkpoint(ckpt_path, device, classes)
    use_amp = (device.type == "cuda")

    ckpt_mode = saved_cfg.get("mode", "multimodal")
    if args.mode and args.mode != "auto":
        if args.mode != ckpt_mode:
            raise ValueError(
                f"Requested evaluation mode '{args.mode}' conflicts with checkpoint mode '{ckpt_mode}'."
            )
        active_mode = args.mode
    else:
        active_mode = ckpt_mode

    video_bb = saved_cfg.get("video_backbone", "model")
    sensor_bb = saved_cfg.get("sensor_backbone", "model")

    print(f"\nModel architecture mode: {active_mode} (Video: {video_bb}, Sensor: {sensor_bb})")

    results = {
        "checkpoint": str(ckpt_path),
        "dataset_root": str(data_root),
        "is_synthetic": is_synthetic,
        "split_strategy": split_strategy,
        "is_official_split": splitter.last_audit.is_official if splitter.last_audit else False,
        "is_custom_test": (eval_split_name == "test"),
        "is_official_validation": (eval_split_name == "val"),
        "train_source_file": splitter.last_audit.train_file if splitter.last_audit else None,
        "val_source_file": splitter.last_audit.val_file if splitter.last_audit else None,
        "eval_split": eval_split_name,
        "eval_split_description": eval_split_desc,
        "eval_samples_count": len(eval_samples),
        "rejected_missing_required_sensor_samples": len(rejected_samples),
        "requested_eval_split": args.eval_split,
        "train_samples_count": len(train_samples),
        "val_samples_count": len(val_samples),
        "test_samples_count": len(test_samples),
        "train_recordings_count": len({(s.subject, s.cam, s.scene, s.session) for s in train_samples}),
        "val_recordings_count": len({(s.subject, s.cam, s.scene, s.session) for s in val_samples}),
        "test_recordings_count": len({(s.subject, s.cam, s.scene, s.session) for s in test_samples}),
        "mode": active_mode,
        "video_backbone": video_bb,
        "sensor_backbone": sensor_bb,
    }

    # Prepare outputs directory
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plots_dir = Path(args.plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    primary_metrics = None
    y_true_out = None
    probs_out = None

    if active_mode == "video_only":
        print("\nEvaluating Video-Only Model...")
        vo_metrics, y_true_out, probs_out = run_evaluation_for_mode(model, eval_loader, device, "video_only", classes, use_amp=use_amp)
        primary_metrics = vo_metrics
        results["video_only"] = vo_metrics
        results["metrics"] = vo_metrics

        print("\n" + "=" * 80)
        print(" EVALUATION RESULTS SUMMARY (Video-Only)")
        print("=" * 80)
        print(f"{'Backbone':<18} | {'Accuracy':<10} | {'Macro F1':<10} | {'Precision':<10} | {'Recall':<10}")
        print("-" * 80)
        print(f"{video_bb:<18} | {vo_metrics['accuracy']*100:6.2f}%    | {vo_metrics['macro_f1']:8.4f}   | {vo_metrics['macro_precision']:8.4f}   | {vo_metrics['macro_recall']:8.4f}")
        print("=" * 80)

    elif active_mode == "sensor_only":
        print("\nEvaluating Sensor-Only Model...")
        so_metrics, y_true_out, probs_out = run_evaluation_for_mode(model, eval_loader, device, "sensor_only", classes, use_amp=use_amp)
        primary_metrics = so_metrics
        results["sensor_only"] = so_metrics
        results["metrics"] = so_metrics

        print("\n" + "=" * 80)
        print(" EVALUATION RESULTS SUMMARY (Sensor-Only)")
        print("=" * 80)
        print(f"{'Backbone':<18} | {'Accuracy':<10} | {'Macro F1':<10} | {'Precision':<10} | {'Recall':<10}")
        print("-" * 80)
        print(f"{sensor_bb:<18} | {so_metrics['accuracy']*100:6.2f}%    | {so_metrics['macro_f1']:8.4f}   | {so_metrics['macro_precision']:8.4f}   | {so_metrics['macro_recall']:8.4f}")
        print("=" * 80)

    else:  # Multimodal mode
        print("\nEvaluating Multimodal Fusion Model...")
        mm_metrics, y_true_out, probs_out = run_evaluation_for_mode(model, eval_loader, device, "multimodal", classes, use_amp=use_amp)
        primary_metrics = mm_metrics
        results["multimodal"] = mm_metrics
        results["metrics"] = mm_metrics

        # If independently trained video and sensor checkpoints are passed, evaluate them
        vo_metrics = None
        so_metrics = None
        if args.video_checkpoint:
            v_path = Path(args.video_checkpoint)
            if not v_path.exists():
                raise FileNotFoundError(f"Independent video checkpoint not found: {args.video_checkpoint}")
            print(f"\nEvaluating Independent Video Baseline from: {args.video_checkpoint}...")
            v_model, _, v_cfg = load_model_from_checkpoint(v_path, device, classes)
            if v_cfg.get("mode") != "video_only":
                raise ValueError(
                    f"Expected 'video_only' mode in {args.video_checkpoint}, got '{v_cfg.get('mode')}'."
                )
            vo_metrics, _, _ = run_evaluation_for_mode(v_model, eval_loader, device, "video_only", classes, use_amp=use_amp)
            results["video_only"] = vo_metrics

        if args.sensor_checkpoint:
            s_path = Path(args.sensor_checkpoint)
            if not s_path.exists():
                raise FileNotFoundError(f"Independent sensor checkpoint not found: {args.sensor_checkpoint}")
            print(f"Evaluating Independent Sensor Baseline from: {args.sensor_checkpoint}...")
            s_model, _, s_cfg = load_model_from_checkpoint(s_path, device, classes)
            if s_cfg.get("mode") != "sensor_only":
                raise ValueError(
                    f"Expected 'sensor_only' mode in {args.sensor_checkpoint}, got '{s_cfg.get('mode')}'."
                )
            so_metrics, _, _ = run_evaluation_for_mode(s_model, eval_loader, device, "sensor_only", classes, use_amp=use_amp)
            results["sensor_only"] = so_metrics

        print("\n" + "=" * 80)
        print(" EVALUATION RESULTS SUMMARY")
        print("=" * 80)
        print(f"{'Modality':<18} | {'Accuracy':<10} | {'Macro F1':<10} | {'Precision':<10} | {'Recall':<10}")
        print("-" * 80)
        print(f"{'Multimodal (Fused)':<18} | {mm_metrics['accuracy']*100:6.2f}%    | {mm_metrics['macro_f1']:8.4f}   | {mm_metrics['macro_precision']:8.4f}   | {mm_metrics['macro_recall']:8.4f}")
        if vo_metrics:
            print(f"{'Video-Only':<18} | {vo_metrics['accuracy']*100:6.2f}%    | {vo_metrics['macro_f1']:8.4f}   | {vo_metrics['macro_precision']:8.4f}   | {vo_metrics['macro_recall']:8.4f}")
        if so_metrics:
            print(f"{'Sensor-Only':<18} | {so_metrics['accuracy']*100:6.2f}%    | {so_metrics['macro_f1']:8.4f}   | {so_metrics['macro_precision']:8.4f}   | {so_metrics['macro_recall']:8.4f}")
        print("=" * 80)

        # Modality ablation comparison plot only if genuine independent models are available
        if vo_metrics and so_metrics:
            ablation_dict = {
                "Video-Only": vo_metrics,
                "Sensor-Only": so_metrics,
                "Multimodal": mm_metrics,
            }
            plot_modality_ablation(ablation_dict, plots_dir / f"test_modality_ablation_{video_bb}.png", split_name=f"Test ({split_strategy} split)")
            plot_modality_ablation(ablation_dict, plots_dir / "test_modality_ablation.png", split_name=f"Test ({split_strategy} split)")
            print(f"Saved modality ablation plot to: {plots_dir / 'test_modality_ablation.png'}")

    # Save metrics JSON
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved complete metrics to: {output_path}")

    # Save metrics CSV summary table
    csv_metrics_path = output_path.parent / "metrics.csv"
    with open(csv_metrics_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Model", "Modality", "Split", "Accuracy (%)", "Macro F1", "Macro Precision", "Macro Recall"])
        for mod_key, mod_name in [("multimodal", "Multimodal (Fused)"), ("video_only", "Video-Only"), ("sensor_only", "Sensor-Only")]:
            if mod_key in results:
                m_dict = results[mod_key]
                writer.writerow([
                    video_bb if mod_key != "sensor_only" else sensor_bb,
                    mod_name,
                    eval_split_name,
                    round(m_dict.get("accuracy", 0.0) * 100.0, 2),
                    round(m_dict.get("macro_f1", 0.0), 4),
                    round(m_dict.get("macro_precision", 0.0), 4),
                    round(m_dict.get("macro_recall", 0.0), 4),
                ])
    print(f"Saved metrics CSV table to: {csv_metrics_path}")

    # Save test predictions CSV
    if y_true_out is not None and probs_out is not None:
        preds_path = output_path.parent / "test_predictions.csv"
        pred_labels = np.argmax(probs_out, axis=-1)
        confidences = np.max(probs_out, axis=-1)
        with open(preds_path, "w", newline="") as f:
            writer = csv.writer(f)
            header = ["sample_idx", "ground_truth", "predicted", "confidence"] + [f"prob_{c}" for c in classes]
            writer.writerow(header)
            for idx in range(len(y_true_out)):
                gt_idx = int(y_true_out[idx])
                pl_idx = int(pred_labels[idx])
                row = [
                    idx,
                    classes[gt_idx] if gt_idx < len(classes) else str(gt_idx),
                    classes[pl_idx] if pl_idx < len(classes) else str(pl_idx),
                    round(float(confidences[idx]), 4),
                ] + [round(float(p), 4) for p in probs_out[idx]]
                writer.writerow(row)
        print(f"Saved test predictions table to: {preds_path}")

    # Generate model-specific plots
    model_tag = video_bb if active_mode != "sensor_only" else sensor_bb
    if primary_metrics and "confusion_matrix" in primary_metrics:
        cm_array = np.array(primary_metrics["confusion_matrix"])
        title = f"{eval_split_desc} Confusion Matrix - {model_tag.upper()} ({active_mode})"
        plot_confusion_matrix(cm_array, classes, plots_dir / "test_confusion_matrix.png", title=title)
        plot_confusion_matrix(cm_array, classes, plots_dir / f"test_confusion_matrix_{model_tag}.png", title=title)
        print(f"Saved confusion matrix plot to: {plots_dir / 'test_confusion_matrix.png'}")

    if y_true_out is not None and probs_out is not None:
        plot_multiclass_roc(y_true_out, probs_out, classes, plots_dir / "test_roc_ovr.png")
        plot_multiclass_pr(y_true_out, probs_out, classes, plots_dir / "test_pr_curves.png")
        plot_multiclass_roc(y_true_out, probs_out, classes, plots_dir / f"test_roc_ovr_{model_tag}.png")
        plot_multiclass_pr(y_true_out, probs_out, classes, plots_dir / f"test_pr_curves_{model_tag}.png")
        print(f"Saved ROC/PR curves to: {plots_dir / 'test_roc_ovr.png'}")

    if primary_metrics and "per_class" in primary_metrics:
        plot_per_class_metrics(
            primary_metrics["per_class"],
            plots_dir / f"test_per_class_{model_tag}.png",
            model_name=model_tag.upper(),
            split_name=f"{eval_split_desc} ({split_strategy} split)",
        )
        plot_per_class_metric_bars(
            primary_metrics["per_class"],
            "Precision",
            plots_dir / f"test_precision_bars_{model_tag}.png",
            model_name=model_tag.upper(),
            split_name=f"{eval_split_desc} ({split_strategy} split)",
        )
        plot_per_class_metric_bars(
            primary_metrics["per_class"],
            "Recall",
            plots_dir / f"test_recall_bars_{model_tag}.png",
            model_name=model_tag.upper(),
            split_name=f"{eval_split_desc} ({split_strategy} split)",
        )
        plot_per_class_metric_bars(
            primary_metrics["per_class"],
            "F1-Score",
            plots_dir / f"test_f1_bars_{model_tag}.png",
            model_name=model_tag.upper(),
            split_name=f"{eval_split_desc} ({split_strategy} split)",
        )
        print(f"Saved per-class metrics bar charts to: {plots_dir}")


if __name__ == "__main__":
    main()
