"""
train.py - Configurable Training Pipeline for Multimodal HAR_MMAct

Supports multimodal fusion, video-only, and sensor-only training with:
- Pretrained ResNet-18 video backbone and InceptionTime sensor backbone
- Leakage-free subject / view / scene / session split verification
- On-disk tensor caching for fast training iterations
- Learning rate scheduling and best-checkpoint persistence
- Comprehensive metrics and publication-ready visualization artifacts
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.config import ProjectConfig, SplitConfig, set_seed, VERIFIED_TARGET_CLASSES
from src.data.indexer import MMActIndexer
from src.data.splits import DatasetSplitter, verify_split_isolation
from src.data.dataset import MMActMultimodalDataset
from src.models.fusion_model import MultimodalHARModel
from src.utils.metrics import compute_classification_metrics
from src.utils.plots import (
    plot_training_curves,
    plot_confusion_matrix,
    plot_multiclass_roc,
    plot_multiclass_pr,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train Multimodal HAR Model on MMAct")
    parser.add_argument("--data_root", type=str, default="data/genuine_mmact", help="Root directory of dataset")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--split_strategy", type=str, default="subject", choices=["subject", "cross_view", "cross_scene", "session"], help="Split strategy")
    parser.add_argument("--mode", type=str, default="multimodal", choices=["multimodal", "video_only", "sensor_only"], help="Model mode")
    parser.add_argument("--video_backbone", type=str, default="resnet18", choices=["resnet18", "lightweight"], help="Video backbone")
    parser.add_argument("--sensor_backbone", type=str, default="inception_time", choices=["inception_time", "cnn1d"], help="Sensor backbone")
    parser.add_argument("--freeze_video", action="store_true", help="Freeze video backbone feature extractor")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Directory for artifacts")
    parser.add_argument("--cache_dir", type=str, default="outputs/cache", help="Cache directory for preprocessed tensors")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader workers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--smoke_test", action="store_true", help="Force run on synthetic data for end-to-end pipeline verification")
    return parser.parse_args()


def train_one_epoch(model, loader, criterion, optimizer, device, mode):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for video, sensor, label, _ in loader:
        video = video.to(device)
        sensor = sensor.to(device)
        label = label.to(device)

        optimizer.zero_grad()
        logits = model(video=video, sensor=sensor, override_mode=mode)
        loss = criterion(logits, label)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(label)
        preds = logits.argmax(dim=-1)
        correct += (preds == label).sum().item()
        total += len(label)

    avg_loss = total_loss / total if total > 0 else 0.0
    accuracy = (correct / total * 100.0) if total > 0 else 0.0
    return avg_loss, accuracy


@torch.no_grad()
def evaluate(model, loader, criterion, device, mode, class_names):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []

    for video, sensor, label, _ in loader:
        video = video.to(device)
        sensor = sensor.to(device)
        label = label.to(device)

        logits = model(video=video, sensor=sensor, override_mode=mode)
        loss = criterion(logits, label)
        probs = torch.softmax(logits, dim=-1)

        total_loss += loss.item() * len(label)
        all_preds.extend(logits.argmax(dim=-1).cpu().numpy())
        all_labels.extend(label.cpu().numpy())
        all_probs.append(probs.cpu().numpy())

    avg_loss = total_loss / len(all_labels) if len(all_labels) > 0 else 0.0
    probs_array = np.concatenate(all_probs, axis=0) if all_probs else None

    metrics = compute_classification_metrics(
        y_true=all_labels,
        y_pred=all_preds,
        y_prob=probs_array,
        classes=class_names,
    )
    metrics["loss"] = avg_loss
    return metrics, np.array(all_labels), probs_array


def main():
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    ckpt_dir = output_dir / "checkpoints"
    plots_dir = output_dir / "plots"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    data_path = Path(args.data_root)
    is_synthetic = False

    print("=" * 80)
    print(" HAR_MMAct: Multimodal Human Activity Recognition Training Pipeline")
    print("=" * 80)
    print(f" Mode            : {args.mode}")
    print(f" Video Backbone  : {args.video_backbone}")
    print(f" Sensor Backbone : {args.sensor_backbone}")
    print(f" Split Strategy  : {args.split_strategy}")
    print(f" Epochs          : {args.epochs} | Batch Size: {args.batch_size} | LR: {args.lr}")
    print(f" Random Seed     : {args.seed}")
    print("=" * 80)

    # 1. Dataset verification & indexing
    if args.smoke_test:
        data_path = Path("data/synthetic_smoke_test")
        is_synthetic = True
        print(f"\n[INFO] --smoke_test active. Using synthetic fixture at: {data_path}")
    elif not data_path.exists():
        print(f"\n[ERROR] Dataset root directory not found: {data_path.resolve()}")
        print("[GUIDANCE] Please place genuine MMAct trimmed data at 'data/genuine_mmact'")
        print("           or run with --smoke_test to verify pipeline execution on synthetic data.")
        sys.exit(1)

    print(f"\n[Step 1] Indexing dataset at: {data_path}")
    indexer = MMActIndexer(dataset_root=data_path)
    report = indexer.index()

    print(f"  - Total Videos Scanned   : {report.total_video_files_found}")
    print(f"  - Total Sensors Scanned  : {report.total_sensor_files_found}")
    print(f"  - Paired Samples Created : {report.paired_samples_count}")
    print(f"  - Unknown Labels Filtered: {len(report.unknown_label_files)}")

    if report.paired_samples_count == 0:
        print(f"\n[ERROR] No paired samples found in {data_path} for verified classes: {VERIFIED_TARGET_CLASSES}")
        if not is_synthetic:
            print("[NOTE] Genuine MMAct paired train/val archives have not yet been placed in the data directory.")
            print("       To test the complete pipeline forward/backward pass, use: python scripts/train.py --smoke_test")
        sys.exit(1)

    # 2. Partition into Train / Val / Test
    print(f"\n[Step 2] Partitioning into {args.split_strategy}-independent splits...")
    split_cfg = SplitConfig(split_strategy=args.split_strategy)
    splitter = DatasetSplitter(split_cfg)
    train_samples, val_samples, test_samples = splitter.split(report.paired_samples)

    print(f"  - Train Set : {len(train_samples)} samples")
    print(f"  - Val Set   : {len(val_samples)} samples")
    print(f"  - Test Set  : {len(test_samples)} samples")

    isolation = verify_split_isolation(train_samples, val_samples, test_samples, strategy=args.split_strategy)
    for check_name, passed in isolation.items():
        status = "PASSED" if passed else "FAILED"
        print(f"      * {check_name:<30}: {status}")
        assert passed, f"CRITICAL: Leakage detected across splits ({check_name})!"

    # 3. Create Datasets & Loaders
    print("\n[Step 3] Initializing PyTorch Datasets and Caching...")
    cache_path = Path(args.cache_dir)
    train_ds = MMActMultimodalDataset(train_samples, cache_dir=cache_path / "train", enable_cache=True)
    val_ds = MMActMultimodalDataset(val_samples, cache_dir=cache_path / "val", enable_cache=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # 4. Model Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[Step 4] Initializing MultimodalHARModel on device: {device}")
    model = MultimodalHARModel(
        num_classes=len(VERIFIED_TARGET_CLASSES),
        video_channels=3,
        sensor_channels=9,
        video_embed_dim=128,
        sensor_embed_dim=128,
        fusion_hidden_dim=128,
        dropout=0.3,
        mode=args.mode,
        video_backbone=args.video_backbone,
        sensor_backbone=args.sensor_backbone,
        pretrained_video=True,
        pretrained_path="outputs/pretrained/resnet18.pth",
        freeze_video_backbone=args.freeze_video,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  - Total Parameters     : {total_params:,}")
    print(f"  - Trainable Parameters : {trainable_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # 5. Training Loop
    print(f"\n[Step 5] Training for {args.epochs} epochs...")
    best_val_f1 = -1.0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    start_time = time.time()

    last_val_labels = None
    last_val_probs = None
    best_metrics = None

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, args.mode)
        val_metrics, val_labels, val_probs = evaluate(model, val_loader, criterion, device, args.mode, VERIFIED_TARGET_CLASSES)
        scheduler.step()

        val_loss = val_metrics["loss"]
        val_acc = val_metrics["accuracy"] * 100.0
        val_f1 = val_metrics["macro_f1"]
        epoch_dur = time.time() - epoch_start

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch:03d}/{args.epochs:03d} [{epoch_dur:.1f}s] - "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, Val Macro F1: {val_f1:.4f}"
        )

        # Save checkpoint
        ckpt_state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_f1": val_f1,
            "classes": VERIFIED_TARGET_CLASSES,
            "config": vars(args),
            "is_synthetic": is_synthetic,
        }

        # Latest
        torch.save(ckpt_state, ckpt_dir / "latest_model.pth")

        # Best
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_metrics = val_metrics
            last_val_labels = val_labels
            last_val_probs = val_probs
            torch.save(ckpt_state, ckpt_dir / "best_model.pth")
            print(f"  --> Saved new best checkpoint (Macro F1: {best_val_f1:.4f})")

    total_training_time = time.time() - start_time
    best_ckpt_path = ckpt_dir / "best_model.pth"
    best_ckpt_size_mb = best_ckpt_path.stat().st_size / (1024 * 1024) if best_ckpt_path.exists() else 0.0

    print("\n" + "=" * 80)
    print(" TRAINING COMPLETE")
    print("=" * 80)
    print(f" Total Runtime        : {total_training_time:.2f} seconds")
    print(f" Best Validation F1   : {best_val_f1:.4f}")
    print(f" Best Checkpoint Saved: {best_ckpt_path} ({best_ckpt_size_mb:.2f} MB)")
    print("=" * 80)

    # 6. Generate and save plots
    print("\n[Step 6] Generating visualization plots in outputs/plots/ ...")
    plot_training_curves(
        history["train_loss"],
        history["val_loss"],
        history["train_acc"],
        history["val_acc"],
        plots_dir / "training_curves.png",
    )
    print(f"  - Saved training curves to: {plots_dir / 'training_curves.png'}")

    if best_metrics and "confusion_matrix" in best_metrics:
        cm_array = np.array(best_metrics["confusion_matrix"])
        title = f"Validation Confusion Matrix ({'SYNTHETIC' if is_synthetic else 'GENUINE'})"
        plot_confusion_matrix(cm_array, VERIFIED_TARGET_CLASSES, plots_dir / "val_confusion_matrix.png", title=title)
        print(f"  - Saved confusion matrix to: {plots_dir / 'val_confusion_matrix.png'}")

    if last_val_labels is not None and last_val_probs is not None:
        plot_multiclass_roc(last_val_labels, last_val_probs, VERIFIED_TARGET_CLASSES, plots_dir / "val_roc_ovr.png")
        plot_multiclass_pr(last_val_labels, last_val_probs, VERIFIED_TARGET_CLASSES, plots_dir / "val_pr_curves.png")
        print(f"  - Saved ROC curves to: {plots_dir / 'val_roc_ovr.png'}")
        print(f"  - Saved PR curves to: {plots_dir / 'val_pr_curves.png'}")

    # 7. Save summary JSON
    summary = {
        "dataset_root": str(data_path),
        "is_synthetic": is_synthetic,
        "mode": args.mode,
        "video_backbone": args.video_backbone,
        "sensor_backbone": args.sensor_backbone,
        "split_strategy": args.split_strategy,
        "epochs": args.epochs,
        "runtime_seconds": round(total_training_time, 2),
        "best_checkpoint_mb": round(best_ckpt_size_mb, 2),
        "best_metrics": best_metrics,
        "history": history,
    }
    with open(output_dir / "training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  - Saved training summary to: {output_dir / 'training_summary.json'}")


if __name__ == "__main__":
    main()
