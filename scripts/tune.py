"""
tune.py - Hyperparameter Search on Train/Validation Splits

Strict Isolation: The held-out test split is NEVER touched or evaluated during tuning.
Search Space: Explores learning rate, weight decay, and dropout regularization.
Artifacts Generated:
- outputs/tuning/tuning_history.json
- outputs/tuning/tuning_history.csv
- outputs/tuning/best_hyperparameters.json
- outputs/plots/hyperparameter_trials.png
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

from src.config import SplitConfig, set_seed, VERIFIED_TARGET_CLASSES
from src.data.indexer import MMActIndexer
from src.data.splits import DatasetSplitter, verify_split_isolation
from src.data.dataset import MMActMultimodalDataset
from src.models.fusion_model import MultimodalHARModel
from src.utils.metrics import compute_classification_metrics
from src.utils.plots import plot_hyperparameter_trials


SEARCH_SPACE = [
    {"trial_id": 1, "lr": 1e-4, "weight_decay": 1e-3, "dropout": 0.35, "name": "Baseline Regularized"},
    {"trial_id": 2, "lr": 3e-4, "weight_decay": 1e-3, "dropout": 0.35, "name": "Higher Learning Rate"},
    {"trial_id": 3, "lr": 1e-4, "weight_decay": 5e-3, "dropout": 0.45, "name": "Heavy Regularization"},
    {"trial_id": 4, "lr": 5e-5, "weight_decay": 5e-4, "dropout": 0.25, "name": "Fine-Grained Low LR"},
]


def parse_args():
    parser = argparse.ArgumentParser(description="Hyperparameter Tuning for Custom CNN Baseline")
    parser.add_argument("--data_root", type=str, default="data/genuine_mmact", help="Dataset root directory")
    parser.add_argument("--split_strategy", type=str, default="subject", choices=["subject", "cross_view", "cross_scene", "session"])
    parser.add_argument("--epochs_per_trial", type=int, default=10, help="Epochs per tuning trial")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--output_dir", type=str, default="outputs/tuning", help="Tuning output directory")
    parser.add_argument("--plots_dir", type=str, default="outputs/plots", help="Plots directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--smoke_test", action="store_true", help="Run with synthetic smoke test data")
    return parser.parse_args()


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for video, sensor, label, _ in loader:
        video, sensor, label = video.to(device), sensor.to(device), label.to(device)
        optimizer.zero_grad()
        logits = model(video=video, sensor=sensor, override_mode="video_only")
        loss = criterion(logits, label)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(label)
        preds = logits.argmax(dim=-1)
        correct += (preds == label).sum().item()
        total += len(label)
    return total_loss / total if total > 0 else 0.0, (correct / total * 100.0) if total > 0 else 0.0


@torch.no_grad()
def evaluate_epoch(model, loader, criterion, device, class_names):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []
    for video, sensor, label, _ in loader:
        video, sensor, label = video.to(device), sensor.to(device), label.to(device)
        logits = model(video=video, sensor=sensor, override_mode="video_only")
        loss = criterion(logits, label)
        total_loss += loss.item() * len(label)
        all_preds.extend(logits.argmax(dim=-1).cpu().numpy())
        all_labels.extend(label.cpu().numpy())

    avg_loss = total_loss / len(all_labels) if len(all_labels) > 0 else 0.0
    metrics = compute_classification_metrics(y_true=all_labels, y_pred=all_preds, classes=class_names)
    metrics["loss"] = avg_loss
    return metrics


def main():
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    plots_dir = Path(args.plots_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    data_path = Path("data/synthetic_smoke_test") if args.smoke_test else Path(args.data_root)
    if not data_path.exists():
        print(f"[ERROR] Data root not found: {data_path}")
        sys.exit(1)

    print("=" * 80)
    print(" HAR_MMAct: Hyperparameter Search Pipeline")
    print(f" Split Strategy    : {args.split_strategy} (Held-out Test Split ISOLATED)")
    print(f" Trials Count      : {len(SEARCH_SPACE)}")
    print(f" Epochs per Trial  : {args.epochs_per_trial}")
    print(f" Optimization Goal : Maximizing Validation Accuracy / Macro F1")
    print("=" * 80)

    # 1. Index dataset and create train/val splits strictly
    indexer = MMActIndexer(dataset_root=data_path)
    report = indexer.index()
    if report.paired_samples_count == 0:
        print("[ERROR] No paired samples found.")
        sys.exit(1)

    splitter = DatasetSplitter(SplitConfig(split_strategy=args.split_strategy))
    train_samples, val_samples, test_samples = splitter.split(report.paired_samples)

    print(f"\nDataset Partitions:")
    print(f"  - Train Partition (Used for fitting)     : {len(train_samples)} samples")
    print(f"  - Val Partition   (Used for model selection): {len(val_samples)} samples")
    print(f"  - Test Partition  (STRICTLY UNTOUCHED)   : {len(test_samples)} samples")

    isolation = verify_split_isolation(train_samples, val_samples, test_samples, strategy=args.split_strategy)
    for check_name, passed in isolation.items():
        assert passed, f"CRITICAL: Leakage detected across splits ({check_name})!"

    train_ds = MMActMultimodalDataset(train_samples, enable_cache=True)
    val_ds = MMActMultimodalDataset(val_samples, enable_cache=True)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()

    trials_history = []
    best_trial = None
    best_score = -1.0

    # 2. Iterate through search space
    for trial in SEARCH_SPACE:
        t_id = trial["trial_id"]
        lr = trial["lr"]
        wd = trial["weight_decay"]
        dp = trial["dropout"]
        name = trial["name"]

        print(f"\n--- Running Trial {t_id}/{len(SEARCH_SPACE)}: {name} ---")
        print(f"    Params: lr={lr}, weight_decay={wd}, dropout={dp}")

        set_seed(args.seed + t_id)
        model = MultimodalHARModel(
            num_classes=len(VERIFIED_TARGET_CLASSES),
            video_channels=3,
            sensor_channels=9,
            video_embed_dim=128,
            sensor_embed_dim=128,
            fusion_hidden_dim=128,
            dropout=dp,
            mode="video_only",
            video_backbone="cnn",
        ).to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs_per_trial, eta_min=1e-6)

        trial_start = time.time()
        best_trial_val_acc = 0.0
        best_trial_val_f1 = 0.0

        for ep in range(1, args.epochs_per_trial + 1):
            t_loss, t_acc = train_epoch(model, train_loader, criterion, optimizer, device)
            v_metrics = evaluate_epoch(model, val_loader, criterion, device, VERIFIED_TARGET_CLASSES)
            scheduler.step()

            v_acc = v_metrics["accuracy"] * 100.0
            v_f1 = v_metrics["macro_f1"]
            if v_acc > best_trial_val_acc:
                best_trial_val_acc = v_acc
                best_trial_val_f1 = v_f1

        trial_dur = time.time() - trial_start
        record = {
            "trial_id": t_id,
            "name": name,
            "lr": lr,
            "weight_decay": wd,
            "dropout": dp,
            "val_accuracy": round(best_trial_val_acc, 2),
            "val_macro_f1": round(best_trial_val_f1, 4),
            "duration_sec": round(trial_dur, 1),
        }
        trials_history.append(record)
        print(f"    Result: Best Val Acc={best_trial_val_acc:.2f}%, Val Macro F1={best_trial_val_f1:.4f} ({trial_dur:.1f}s)")

        if best_trial_val_acc > best_score:
            best_score = best_trial_val_acc
            best_trial = record

    # 3. Save trial results
    print("\n" + "=" * 80)
    print(" HYPERPARAMETER TUNING SUMMARY")
    print("=" * 80)
    print(f"{'Trial':<8} | {'LR':<10} | {'Weight Decay':<14} | {'Dropout':<10} | {'Val Acc (%)':<12} | {'Val Macro F1':<12}")
    print("-" * 80)
    for r in trials_history:
        print(f"{r['trial_id']:<8} | {r['lr']:<10} | {r['weight_decay']:<14} | {r['dropout']:<10} | {r['val_accuracy']:<12} | {r['val_macro_f1']:<12}")
    print("=" * 80)
    print(f"Best Configuration: Trial {best_trial['trial_id']} ({best_trial['name']}) with {best_trial['val_accuracy']:.2f}% Val Acc")

    # JSON history
    with open(output_dir / "tuning_history.json", "w") as f:
        json.dump(trials_history, f, indent=2)

    # Best config JSON
    with open(output_dir / "best_hyperparameters.json", "w") as f:
        json.dump(best_trial, f, indent=2)

    # CSV history
    csv_file = output_dir / "tuning_history.csv"
    with open(csv_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["trial_id", "name", "lr", "weight_decay", "dropout", "val_accuracy", "val_macro_f1", "duration_sec"])
        writer.writeheader()
        writer.writerows(trials_history)

    # Generate trial comparison plot
    plot_hyperparameter_trials(trials_history, plots_dir / "hyperparameter_trials.png")
    print(f"\nArtifacts Saved:")
    print(f"  - {output_dir / 'tuning_history.json'}")
    print(f"  - {output_dir / 'tuning_history.csv'}")
    print(f"  - {output_dir / 'best_hyperparameters.json'}")
    print(f"  - {plots_dir / 'hyperparameter_trials.png'}")


if __name__ == "__main__":
    main()
