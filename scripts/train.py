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


def get_git_commit_hash() -> str:
    try:
        import subprocess
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode("utf-8").strip()
    except Exception:
        return "unknown"


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
    parser.add_argument("--min_epochs", type=int, default=50, help="Minimum number of epochs to train before early stopping can trigger (default: 50)")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume training from")
    parser.add_argument("--auto_resume", action="store_true", help="Automatically resume from latest checkpoint in output_dir if present")
    parser.add_argument("--skip_if_completed", action="store_true", help="Skip training if experiment has already completed successfully")
    parser.add_argument("--checkpoint_name", type=str, default=None, help="Custom filename for saved checkpoint")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Directory for artifacts")
    parser.add_argument("--cache_dir", type=str, default="outputs/cache", help="Cache directory for preprocessed tensors")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader workers")
    parser.add_argument("--prefetch_factor", type=int, default=2, help="DataLoader prefetch factor (when num_workers > 0)")
    parser.add_argument("--pin_memory", action="store_true", default=True, help="Enable DataLoader pin_memory (default: True)")
    parser.add_argument("--no_amp", action="store_true", help="Disable Automatic Mixed Precision (AMP)")
    parser.add_argument("--cudnn_benchmark", action="store_true", default=True, help="Enable cuDNN benchmark autotuner")
    parser.add_argument("--benchmark_warmup", action="store_true", help="Run 1-batch calibration forward/backward check before full training")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--smoke_test", action="store_true", help="Force run on synthetic data for end-to-end pipeline verification")
    return parser.parse_args()


def train_one_epoch(model, loader, criterion, optimizer, device, mode, scaler=None, use_amp=False):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (video, sensor, label, _) in enumerate(loader):
        video = video.to(device, non_blocking=True)
        sensor = sensor.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        try:
            if use_amp and device.type == "cuda" and scaler is not None:
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    logits = model(video=video, sensor=sensor, override_mode=mode)
                    loss = criterion(logits, label)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(video=video, sensor=sensor, override_mode=mode)
                loss = criterion(logits, label)
                loss.backward()
                optimizer.step()
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                raise RuntimeError(
                    f"\n[CUDA OOM] Out of memory encountered during training batch {batch_idx} (batch size {len(label)}).\n"
                    f"Model: {mode}, Allocated: {torch.cuda.memory_allocated() / (1024**2):.1f} MB, "
                    f"Reserved: {torch.cuda.memory_reserved() / (1024**2):.1f} MB.\n"
                    f"Remedy: Reduce --batch_size (e.g. from {len(label)} to {max(1, len(label)//2)}) and re-run."
                ) from e
            raise e

        batch_size = len(label)
        total_loss += loss.detach().item() * batch_size
        preds = logits.detach().argmax(dim=-1)
        correct += (preds == label).sum().item()
        total += batch_size

        if len(loader) >= 8 and (batch_idx + 1) % max(1, len(loader) // 4) == 0:
            pct = int((batch_idx + 1) / len(loader) * 100)
            print(f"    [Batch {batch_idx + 1:02d}/{len(loader):02d} ({pct}%)] Loss: {loss.detach().item():.4f}")

    avg_loss = total_loss / total if total > 0 else 0.0
    accuracy = (correct / total * 100.0) if total > 0 else 0.0
    return avg_loss, accuracy


@torch.no_grad()
def evaluate(model, loader, criterion, device, mode, class_names, use_amp=False):
    model.eval()
    total_loss = 0.0
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
                loss = criterion(logits, label)
        else:
            logits = model(video=video, sensor=sensor, override_mode=mode)
            loss = criterion(logits, label)

        probs = torch.softmax(logits, dim=-1)

        total_loss += loss.detach().item() * len(label)
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

    # 0. Check if experiment already completed (avoid redundant work)
    target_ckpt_name = args.checkpoint_name or f"best_model_{args.video_backbone}.pth"
    best_ckpt_file = ckpt_dir / target_ckpt_name
    summary_file = output_dir / f"training_summary_{args.video_backbone}.json"

    if args.skip_if_completed and summary_file.exists() and best_ckpt_file.exists():
        try:
            with open(summary_file) as f:
                prev_summary = json.load(f)
            prev_eps = len(prev_summary.get("history", {}).get("train_loss", []))
            if prev_eps >= min(args.epochs, args.min_epochs):
                test_ckpt = torch.load(best_ckpt_file, map_location="cpu")
                if "model_state_dict" in test_ckpt:
                    best_f1 = prev_summary.get("best_metrics", {}).get("macro_f1", 0.0)
                    print("\n" + "=" * 80)
                    print(f"[SKIP] Experiment in {output_dir} has already completed successfully ({prev_eps}/{args.epochs} epochs).")
                    print(f"       Found valid checkpoint: {best_ckpt_file} (Best Val F1: {best_f1:.4f})")
                    print("       Skipping training run to prevent redundant execution.")
                    print("=" * 80)
                    return
        except Exception as e:
            print(f"[Warning] Could not verify existing summary/checkpoint: {e}")

    # Auto-resume from latest checkpoint if requested and available
    if args.auto_resume and not args.resume:
        latest_file = ckpt_dir / f"latest_model_{args.video_backbone}.pth"
        if not latest_file.exists():
            latest_file = ckpt_dir / "latest_model.pth"
        if latest_file.exists():
            args.resume = str(latest_file)
            print(f"\n[Auto-Resume] Found existing interrupted checkpoint at: {args.resume}")

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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_mem = (device.type == "cuda") and args.pin_memory
    persistent = (args.num_workers > 0)
    prefetch = args.prefetch_factor if args.num_workers > 0 else None

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_mem,
        persistent_workers=persistent,
        prefetch_factor=prefetch,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_mem,
        persistent_workers=persistent,
        prefetch_factor=prefetch,
    )

    # 4. Model Setup & Hardware Acceleration
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        gpu_cap = torch.cuda.get_device_capability(0)
        print(f"\n[Step 4] Initializing MultimodalHARModel on device: {device} ({gpu_name}, Compute Capability {gpu_cap[0]}.{gpu_cap[1]})")
        print(f"  - CUDA Device Count    : {torch.cuda.device_count()}")
        print(f"  - Initial Allocated Mem: {torch.cuda.memory_allocated(0) / (1024**2):.2f} MB")
        if args.cudnn_benchmark:
            torch.backends.cudnn.benchmark = True
            print("  - cuDNN Benchmark Tuner: ENABLED")
    else:
        print(f"\n[Step 4] Initializing MultimodalHARModel on device: {device} (CUDA not available, running on CPU)")

    # Mixed precision scaler setup
    use_amp = (device.type == "cuda") and (not args.no_amp)
    scaler = None
    if use_amp:
        try:
            scaler = torch.amp.GradScaler("cuda")
        except Exception:
            scaler = torch.cuda.amp.GradScaler()
        print("  - Automatic Mixed Precision (AMP): ENABLED (float16)")
    else:
        print("  - Automatic Mixed Precision (AMP): DISABLED (FP32)")

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

    # Optional calibration warm-up check to guarantee memory safety before full run
    if args.benchmark_warmup and len(train_loader) > 0:
        print("\n[Calibration] Running 1-batch memory-safety warm-up check...")
        try:
            model.train()
            for v_w, s_w, l_w, _ in train_loader:
                v_w = v_w.to(device, non_blocking=True)
                s_w = s_w.to(device, non_blocking=True)
                l_w = l_w.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                if use_amp and scaler:
                    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                        log_w = model(video=v_w, sensor=s_w, override_mode=args.mode)
                        loss_w = criterion(log_w, l_w)
                    scaler.scale(loss_w).backward()
                else:
                    log_w = model(video=v_w, sensor=s_w, override_mode=args.mode)
                    loss_w = criterion(log_w, l_w)
                    loss_w.backward()
                optimizer.zero_grad(set_to_none=True)
                break
            print("  --> Memory-safety calibration passed cleanly.")
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                raise RuntimeError(
                    f"CUDA OutOfMemory during calibration warm-up with batch_size={args.batch_size}.\n"
                    f"Please reduce --batch_size (e.g. --batch_size {max(1, args.batch_size // 2)})."
                ) from e
            raise e

    # 5. Training Loop
    start_epoch = 1
    best_val_f1 = -1.0
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "train_time_sec": [],
        "val_time_sec": [],
        "peak_gpu_mem_mb": [],
    }
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

        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, args.mode, scaler=scaler, use_amp=use_amp)
        train_dur = time.time() - t0

        t1 = time.time()
        val_metrics, val_labels, val_probs = evaluate(model, val_loader, criterion, device, args.mode, VERIFIED_TARGET_CLASSES, use_amp=use_amp)
        val_dur = time.time() - t1
        scheduler.step()

        val_loss = val_metrics["loss"]
        val_acc = val_metrics["accuracy"] * 100.0
        val_f1 = val_metrics["macro_f1"]
        epoch_dur = train_dur + val_dur

        peak_mem = round(torch.cuda.max_memory_allocated(0) / (1024**2), 1) if device.type == "cuda" else 0.0

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["train_time_sec"].append(round(train_dur, 2))
        history["val_time_sec"].append(round(val_dur, 2))
        history["peak_gpu_mem_mb"].append(peak_mem)

        mem_info = f", Peak GPU: {peak_mem}MB" if device.type == "cuda" else ""
        elapsed_total = time.time() - start_time
        completed_eps = epoch - start_epoch + 1
        rem_eps = args.epochs - epoch
        avg_ep_dur = elapsed_total / max(1, completed_eps)
        eta_sec = rem_eps * avg_ep_dur
        eta_str = time.strftime("%H:%M:%S", time.gmtime(max(0, eta_sec)))

        print(
            f"Epoch {epoch:03d}/{args.epochs:03d} [{epoch_dur:.1f}s, ETA: {eta_str}{mem_info}] - "
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
                if epoch < args.min_epochs:
                    print(f"  [Early Stopping Deferred] Patience exceeded ({patience_counter} eps without improvement), but continuing to satisfy required --min_epochs {args.min_epochs} (Current: {epoch}).")
                else:
                    print(f"\n[Early Stopping] No improvement in validation F1 for {patience_counter} consecutive epochs and minimum {args.min_epochs} epochs completed. Stopping early at epoch {epoch}.")
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

    # 7. Save history, experiment manifest, and runtime report
    csv_path = output_dir / f"history_{args.video_backbone}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "train_acc", "val_acc", "train_time_sec", "val_time_sec", "peak_gpu_mem_mb"])
        for i in range(len(history["train_loss"])):
            writer.writerow([
                i + 1,
                round(history["train_loss"][i], 4),
                round(history["val_loss"][i], 4),
                round(history["train_acc"][i], 2),
                round(history["val_acc"][i], 2),
                history["train_time_sec"][i] if i < len(history["train_time_sec"]) else 0.0,
                history["val_time_sec"][i] if i < len(history["val_time_sec"]) else 0.0,
                history["peak_gpu_mem_mb"][i] if i < len(history["peak_gpu_mem_mb"]) else 0.0,
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

    # Experiment Manifest
    manifest = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "commit_hash": get_git_commit_hash(),
        "model_name": args.video_backbone,
        "mode": args.mode,
        "video_backbone": args.video_backbone,
        "sensor_backbone": args.sensor_backbone,
        "split_strategy": args.split_strategy,
        "epochs": args.epochs,
        "min_epochs": args.min_epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "dropout": args.dropout,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "N/A",
        "amp_enabled": use_amp,
        "num_workers": args.num_workers,
        "pin_memory": pin_mem,
        "cudnn_benchmark": args.cudnn_benchmark,
        "target_classes": VERIFIED_TARGET_CLASSES,
        "train_samples_count": len(train_samples),
        "val_samples_count": len(val_samples),
        "test_samples_count": len(test_samples),
        "package_versions": {
            "torch": torch.__version__,
            "numpy": np.__version__,
            "scipy": getattr(sys.modules.get("scipy"), "__version__", "unknown"),
        },
    }
    with open(output_dir / f"experiment_manifest_{args.video_backbone}.json", "w") as f:
        json.dump(manifest, f, indent=2)
    with open(output_dir / "experiment_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    # Runtime report
    peak_mem_val = max(history["peak_gpu_mem_mb"], default=0.0)
    runtime_report = {
        "model": args.video_backbone,
        "mode": args.mode,
        "epochs_requested": args.epochs,
        "epochs_completed": len(history["train_loss"]),
        "total_runtime_seconds": round(total_training_time, 2),
        "avg_epoch_duration_seconds": round(total_training_time / max(1, len(history["train_loss"])), 2),
        "peak_gpu_memory_mb": peak_mem_val,
        "best_val_f1": round(best_val_f1, 4),
    }
    with open(output_dir / f"runtime_report_{args.video_backbone}.json", "w") as f:
        json.dump(runtime_report, f, indent=2)
    with open(output_dir / "runtime_report.json", "w") as f:
        json.dump(runtime_report, f, indent=2)

    with open(output_dir / f"runtime_report_{args.video_backbone}.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for k, v in runtime_report.items():
            writer.writerow([k, v])
    with open(output_dir / "runtime_report.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for k, v in runtime_report.items():
            writer.writerow([k, v])

    print(f"  - Saved training summaries and manifests to: {output_dir}")


if __name__ == "__main__":
    main()
