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
import csv
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
    plot_loss_curve_standalone,
    plot_accuracy_curve_standalone,
    plot_train_eval_annotated,
    plot_confusion_matrix,
    plot_multiclass_roc,
    plot_multiclass_pr,
    plot_per_class_metrics,
    plot_per_class_metric_bars,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train Multimodal HAR Model on MMAct")
    parser.add_argument("--data_root", type=str, default="data/genuine_mmact", help="Root directory of dataset")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--dropout", type=float, default=0.35, help="Dropout probability")
    parser.add_argument("--split_strategy", type=str, default="subject", choices=["subject", "cross_view", "cross_scene", "session"], help="Split strategy")
    parser.add_argument("--mode", type=str, default="multimodal", choices=["multimodal", "video_only", "sensor_only"], help="Model mode")
    parser.add_argument("--video_backbone", type=str, default="cnn", choices=["cnn", "custom_cnn", "mobilenet_v2", "resnet18", "lightweight"], help="Video backbone")
    parser.add_argument("--sensor_backbone", type=str, default="inception_time", choices=["inception_time", "cnn1d"], help="Sensor backbone")
    parser.add_argument("--freeze_video", action="store_true", help="Freeze video backbone feature extractor")
    parser.add_argument("--staged_finetune", action="store_true", help="Enable staged fine-tuning for transfer learning (frozen head first, then unfreeze backbone)")
    parser.add_argument("--unfreeze_epoch", type=int, default=15, help="Epoch to unfreeze video backbone in staged fine-tuning")
    parser.add_argument("--unfreeze_lr", type=float, default=1e-5, help="Learning rate for unfrozen video backbone")
    parser.add_argument("--early_stopping_patience", type=int, default=0, help="Epochs without improvement before early stopping (0=disabled)")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume training from")
    parser.add_argument("--checkpoint_name", type=str, default=None, help="Custom filename for saved checkpoint")
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
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        gpu_cap = torch.cuda.get_device_capability(0)
        print(f"\n[Step 4] Initializing MultimodalHARModel on device: {device} ({gpu_name}, Compute Capability {gpu_cap[0]}.{gpu_cap[1]})")
        print(f"  - CUDA Device Count    : {torch.cuda.device_count()}")
        print(f"  - Initial Allocated Mem: {torch.cuda.memory_allocated(0) / (1024**2):.2f} MB")
    else:
        print(f"\n[Step 4] Initializing MultimodalHARModel on device: {device} (CUDA not available, running on CPU)")

    freeze_video = args.freeze_video or args.staged_finetune
    if args.staged_finetune:
        print(f"\n[Staged Fine-Tuning Active] Backbone frozen for epochs 1 to {args.unfreeze_epoch - 1}. Will unfreeze at epoch {args.unfreeze_epoch} with LR={args.unfreeze_lr}.")

    model = MultimodalHARModel(
        num_classes=len(VERIFIED_TARGET_CLASSES),
        video_channels=3,
        sensor_channels=9,
        video_embed_dim=128,
        sensor_embed_dim=128,
        fusion_hidden_dim=128,
        dropout=args.dropout,
        mode=args.mode,
        video_backbone=args.video_backbone,
        sensor_backbone=args.sensor_backbone,
        pretrained_video=True,
        pretrained_path="outputs/pretrained/resnet18.pth",
        freeze_video_backbone=freeze_video,
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
    start_epoch = 1
    best_val_f1 = -1.0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    patience_counter = 0

    if args.resume and Path(args.resume).exists():
        print(f"\n[Resume] Loading checkpoint from: {args.resume}")
        resume_ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(resume_ckpt["model_state_dict"])
        if "optimizer_state_dict" in resume_ckpt:
            optimizer.load_state_dict(resume_ckpt["optimizer_state_dict"])
        start_epoch = resume_ckpt.get("epoch", 0) + 1
        best_val_f1 = resume_ckpt.get("val_f1", -1.0)
        history = resume_ckpt.get("history", history)
        print(f"  --> Resuming from Epoch {start_epoch} (Prior Best Val F1: {best_val_f1:.4f})")

    print(f"\n[Step 5] Training for {args.epochs} epochs (Starting at epoch {start_epoch})...")
    start_time = time.time()

    last_val_labels = None
    last_val_probs = None
    best_metrics = None

    for epoch in range(start_epoch, args.epochs + 1):
        # Handle staged fine-tuning transition
        if args.staged_finetune and epoch == args.unfreeze_epoch:
            print(f"\n>>> [Staged Fine-Tuning Transition] Epoch {epoch}: Unfreezing video backbone with LR={args.unfreeze_lr}...")
            if hasattr(model, "video_branch") and hasattr(model.video_branch, "backbone"):
                for p in model.video_branch.backbone.parameters():
                    p.requires_grad = True
                backbone_params = list(model.video_branch.backbone.parameters())
                backbone_ids = set(id(p) for p in backbone_params)
                head_params = [p for p in model.parameters() if id(p) not in backbone_ids and p.requires_grad]
                optimizer = torch.optim.AdamW([
                    {"params": backbone_params, "lr": args.unfreeze_lr, "weight_decay": args.weight_decay},
                    {"params": head_params, "lr": args.lr, "weight_decay": args.weight_decay},
                ])
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs - epoch + 1), eta_min=1e-6)
                print(f"    Backbone unfrozen ({sum(p.numel() for p in backbone_params):,} params @ lr={args.unfreeze_lr}), Heads @ lr={args.lr}")

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

        # Checkpoint state
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
            "history": history,
            "is_synthetic": is_synthetic,
        }

        # Save latest
        latest_filename = f"latest_model_{args.video_backbone}.pth"
        torch.save(ckpt_state, ckpt_dir / latest_filename)
        torch.save(ckpt_state, ckpt_dir / "latest_model.pth")

        # Save best
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_metrics = val_metrics
            last_val_labels = val_labels
            last_val_probs = val_probs
            patience_counter = 0

            best_filename = args.checkpoint_name or f"best_model_{args.video_backbone}.pth"
            torch.save(ckpt_state, ckpt_dir / best_filename)
            torch.save(ckpt_state, ckpt_dir / "best_model.pth")
            print(f"  --> Saved new best checkpoint: {best_filename} (Macro F1: {best_val_f1:.4f})")
        else:
            patience_counter += 1
            if args.early_stopping_patience > 0 and patience_counter >= args.early_stopping_patience:
                print(f"\n[Early Stopping] No improvement in validation F1 for {patience_counter} consecutive epochs. Stopping early at epoch {epoch}.")
                break

    total_training_time = time.time() - start_time
    chosen_ckpt = args.checkpoint_name or f"best_model_{args.video_backbone}.pth"
    best_ckpt_path = ckpt_dir / chosen_ckpt
    best_ckpt_size_mb = best_ckpt_path.stat().st_size / (1024 * 1024) if best_ckpt_path.exists() else 0.0

    print("\n" + "=" * 80)
    print(" TRAINING COMPLETE")
    print("=" * 80)
    print(f" Model Architecture   : {args.video_backbone} + {args.sensor_backbone}")
    print(f" Total Runtime        : {total_training_time:.2f} seconds")
    print(f" Best Validation F1   : {best_val_f1:.4f}")
    print(f" Best Checkpoint Saved: {best_ckpt_path} ({best_ckpt_size_mb:.2f} MB)")
    print("=" * 80)

    # 6. Generate and save plots (stating model, split, and configuration)
    print(f"\n[Step 6] Generating visualization plots for {args.video_backbone} in outputs/plots/ ...")
    plot_training_curves(
        history["train_loss"],
        history["val_loss"],
        history["train_acc"],
        history["val_acc"],
        plots_dir / f"training_curves_{args.video_backbone}.png",
    )
    plot_training_curves(
        history["train_loss"],
        history["val_loss"],
        history["train_acc"],
        history["val_acc"],
        plots_dir / "training_curves.png",
    )

    plot_loss_curve_standalone(
        history["train_loss"],
        history["val_loss"],
        plots_dir / f"train_vs_val_loss_{args.video_backbone}.png",
        num_layers=4,
    )
    plot_loss_curve_standalone(
        history["train_loss"],
        history["val_loss"],
        plots_dir / "train_vs_val_loss.png",
        num_layers=4,
    )

    plot_accuracy_curve_standalone(
        history["train_acc"],
        history["val_acc"],
        plots_dir / f"train_vs_val_accuracy_{args.video_backbone}.png",
        eval_label="Validation",
        num_layers=4,
    )
    plot_accuracy_curve_standalone(
        history["train_acc"],
        history["val_acc"],
        plots_dir / "train_vs_val_accuracy.png",
        eval_label="Validation",
        num_layers=4,
    )

    # Annotated curves matching publication standards (with final values)
    plot_train_eval_annotated(
        history["train_acc"],
        history["val_acc"],
        plots_dir / f"train_vs_val_accuracy_annotated_{args.video_backbone}.png",
        metric_name="Accuracy",
        eval_label="Validation",
        epochs_label=args.epochs,
        as_percent=True,
    )
    plot_train_eval_annotated(
        history["train_loss"],
        history["val_loss"],
        plots_dir / f"train_vs_val_loss_annotated_{args.video_backbone}.png",
        metric_name="Loss",
        eval_label="Validation",
        epochs_label=args.epochs,
        as_percent=False,
    )

    if best_metrics and "confusion_matrix" in best_metrics:
        cm_array = np.array(best_metrics["confusion_matrix"])
        title = f"Val Confusion Matrix - {args.video_backbone.upper()} ({args.split_strategy} split)"
        plot_confusion_matrix(cm_array, VERIFIED_TARGET_CLASSES, plots_dir / f"val_confusion_matrix_{args.video_backbone}.png", title=title)
        plot_confusion_matrix(cm_array, VERIFIED_TARGET_CLASSES, plots_dir / "val_confusion_matrix.png", title=title)

    if last_val_labels is not None and last_val_probs is not None:
        plot_multiclass_roc(last_val_labels, last_val_probs, VERIFIED_TARGET_CLASSES, plots_dir / f"val_roc_ovr_{args.video_backbone}.png")
        plot_multiclass_pr(last_val_labels, last_val_probs, VERIFIED_TARGET_CLASSES, plots_dir / f"val_pr_curves_{args.video_backbone}.png")
        plot_multiclass_roc(last_val_labels, last_val_probs, VERIFIED_TARGET_CLASSES, plots_dir / "val_roc_ovr.png")
        plot_multiclass_pr(last_val_labels, last_val_probs, VERIFIED_TARGET_CLASSES, plots_dir / "val_pr_curves.png")

    if best_metrics and "per_class" in best_metrics:
        plot_per_class_metrics(
            best_metrics["per_class"],
            plots_dir / f"val_per_class_{args.video_backbone}.png",
            model_name=args.video_backbone.upper(),
            split_name=f"Validation ({args.split_strategy})",
        )
        # Standalone per-class bar charts
        plot_per_class_metric_bars(
            best_metrics["per_class"],
            "Precision",
            plots_dir / f"val_precision_bars_{args.video_backbone}.png",
            model_name=args.video_backbone.upper(),
            split_name=f"Val ({args.split_strategy})",
        )
        plot_per_class_metric_bars(
            best_metrics["per_class"],
            "Recall",
            plots_dir / f"val_recall_bars_{args.video_backbone}.png",
            model_name=args.video_backbone.upper(),
            split_name=f"Val ({args.split_strategy})",
        )
        plot_per_class_metric_bars(
            best_metrics["per_class"],
            "F1-Score",
            plots_dir / f"val_f1_bars_{args.video_backbone}.png",
            model_name=args.video_backbone.upper(),
            split_name=f"Val ({args.split_strategy})",
        )

    # 7. Save history and summary JSON + CSV
    csv_path = output_dir / f"history_{args.video_backbone}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "train_acc", "val_acc"])
        for i in range(len(history["train_loss"])):
            writer.writerow([
                i + 1,
                round(history["train_loss"][i], 4),
                round(history["val_loss"][i], 4),
                round(history["train_acc"][i], 2),
                round(history["val_acc"][i], 2),
            ])
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
    with open(output_dir / f"training_summary_{args.video_backbone}.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(output_dir / "training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(output_dir / f"history_{args.video_backbone}.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"  - Saved training summaries to: {output_dir / f'training_summary_{args.video_backbone}.json'}")


if __name__ == "__main__":
    main()
