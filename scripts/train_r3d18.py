#!/usr/bin/env python3
"""Train the isolated Kinetics-400 R3D-18 MMAct video-only experiment."""

import argparse
import csv
import json
import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.config import SplitConfig, VERIFIED_TARGET_CLASSES, set_seed
from src.data.dataset import MMActMultimodalDataset
from src.data.indexer import MMActIndexer
from src.data.splits import DatasetSplitter, verify_split_isolation
from src.models.fusion_model import MultimodalHARModel
from src.models.r3d18_branch import R3D18_WEIGHT_ID
from src.utils.metrics import compute_classification_metrics

DEFAULT_OUTPUT = "outputs/video_model_improvements/r3d18"
EXPECTED_TEST_SAMPLES = 852


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", default="data/genuine_mmact")
    parser.add_argument("--video_backbone", choices=["r3d18"], default="r3d18")
    parser.add_argument("--mode", choices=["video_only"], default="video_only")
    parser.add_argument("--experiment_name", default="r3d18_kinetics400_video_only")
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--cache_dir", default="outputs/cache")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.35)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--freeze_video", action="store_true")
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--no_resume", dest="auto_resume", action="store_false")
    parser.set_defaults(auto_resume=True)
    return parser.parse_args()


def atomic_json(path, payload):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    os.replace(temp, path)


def atomic_torch_save(path, payload):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temp)
    os.replace(temp, path)


def ids(samples):
    return sorted(sample.sample_id for sample in samples)


def evaluate(model, loader, device, use_amp):
    model.eval()
    labels, predictions, probabilities, sample_ids = [], [], [], []
    with torch.inference_mode():
        for video, _sensor, label, batch_ids in loader:
            video, label = video.to(device, non_blocking=True), label.to(device, non_blocking=True)
            if use_amp and device.type == "cuda":
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    logits = model(video=video, override_mode="video_only")
            else:
                logits = model(video=video, override_mode="video_only")
            probs = torch.softmax(logits.float(), dim=-1)
            labels.extend(label.cpu().tolist())
            predictions.extend(probs.argmax(dim=-1).cpu().tolist())
            probabilities.extend(probs.cpu().tolist())
            sample_ids.extend(list(batch_ids))
    metrics = compute_classification_metrics(labels, predictions, np.asarray(probabilities), VERIFIED_TARGET_CLASSES)
    return metrics, labels, predictions, probabilities, sample_ids


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output = Path(args.output_dir)
    checkpoints = output / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    cache = Path(args.cache_dir)

    if args.video_backbone != "r3d18" or args.mode != "video_only":
        raise ValueError("This dedicated experiment supports only r3d18 video_only mode.")
    data_root = Path(args.data_root)
    if not data_root.exists():
        raise FileNotFoundError(f"Dataset directory not found: {data_root.resolve()}")
    report = MMActIndexer(dataset_root=data_root).index()
    if not report.paired_samples:
        raise RuntimeError(f"No paired MMAct samples found under {data_root}.")

    split_config = SplitConfig(split_strategy="subject", dataset_root=data_root, seed=args.seed)
    splitter = DatasetSplitter(split_config, dataset_root=data_root)
    train_samples, val_samples, test_samples = splitter.split(report.paired_samples)
    if {s.subject for s in train_samples} - {f"subject{i}" for i in range(1, 15)}:
        raise RuntimeError("Unexpected training subject membership; expected subjects 1-14.")
    if {s.subject for s in val_samples} - {f"subject{i}" for i in range(15, 18)}:
        raise RuntimeError("Unexpected validation subject membership; expected subjects 15-17.")
    if {s.subject for s in test_samples} - {f"subject{i}" for i in range(18, 21)}:
        raise RuntimeError("Unexpected test subject membership; expected subjects 18-20.")
    isolation = verify_split_isolation(train_samples, val_samples, test_samples, "subject")
    if not isolation.get("all_isolation_checks_passed", False):
        raise RuntimeError(f"Subject split isolation check failed: {isolation}")
    if len(test_samples) != EXPECTED_TEST_SAMPLES:
        raise RuntimeError(
            f"Expected the existing held-out test membership of {EXPECTED_TEST_SAMPLES} samples, "
            f"but indexing/splitting produced {len(test_samples)}. Refusing to train/evaluate on a changed split."
        )

    train_ds = MMActMultimodalDataset(train_samples, cache_dir=cache / "r3d18_train", enable_cache=True, mode="video_only")
    val_ds = MMActMultimodalDataset(val_samples, cache_dir=cache / "r3d18_val", enable_cache=True, mode="video_only")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                              pin_memory=device.type == "cuda", persistent_workers=args.num_workers > 0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                            pin_memory=device.type == "cuda", persistent_workers=args.num_workers > 0)

    model = MultimodalHARModel(
        num_classes=len(VERIFIED_TARGET_CLASSES), video_channels=3, mode="video_only",
        video_backbone="r3d18", pretrained_video=True, dropout=args.dropout,
        freeze_video_backbone=args.freeze_video,
    ).to(device)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    criterion = nn.CrossEntropyLoss()
    use_amp = device.type == "cuda" and not args.no_amp
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_path = checkpoints / "best_model_r3d18.pth"
    latest_path = checkpoints / "latest_model_r3d18.pth"
    history_path = output / "epoch_history.json"
    history = []
    completed_epochs = 0
    best_val_f1 = -1.0
    if args.auto_resume and latest_path.exists():
        state = torch.load(latest_path, map_location=device)
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        scheduler.load_state_dict(state["scheduler_state_dict"])
        if state.get("scaler_state_dict"):
            scaler.load_state_dict(state["scaler_state_dict"])
        completed_epochs = int(state["completed_epochs"])
        best_val_f1 = float(state["best_val_f1"])
        history = state.get("history", [])
        print(f"Resuming from completed epoch {completed_epochs}; best validation macro-F1={best_val_f1:.4f}")

    gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "N/A (CPU)"
    split_ids = {"train": ids(train_samples), "validation": ids(val_samples), "test": ids(test_samples)}
    manifest = {
        "experiment_name": args.experiment_name,
        "video_backbone": args.video_backbone,
        "mode": args.mode,
        "pretrained_weight_identifier": R3D18_WEIGHT_ID,
        "seed": args.seed,
        "device": str(device),
        "gpu": gpu_name,
        "hyperparameters": {"epochs": args.epochs, "batch_size": args.batch_size, "learning_rate": args.lr,
                             "weight_decay": args.weight_decay, "dropout": args.dropout,
                             "optimizer": "AdamW", "amp_enabled": use_amp,
                             "freeze_video_backbone": args.freeze_video},
        "split_strategy": "subject",
        "split_subjects": {"train": [f"subject{i}" for i in range(1, 15)],
                           "validation": [f"subject{i}" for i in range(15, 18)],
                           "test": [f"subject{i}" for i in range(18, 21)]},
        "split_counts": {key: len(value) for key, value in split_ids.items()},
        "split_sample_ids": split_ids,
        "completed_epochs": completed_epochs,
        "best_validation_macro_f1": best_val_f1,
        "test_evaluation_policy": "Evaluate once after validation-based best checkpoint selection; never during training/tuning.",
    }
    atomic_json(output / "experiment_manifest.json", manifest)

    for epoch in range(completed_epochs + 1, args.epochs + 1):
        model.train()
        train_loss, train_correct, seen = 0.0, 0, 0
        for video, _sensor, label, _sample_ids in train_loader:
            video = video.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            if use_amp:
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    logits = model(video=video, override_mode="video_only")
                    loss = criterion(logits, label)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(video=video, override_mode="video_only")
                loss = criterion(logits, label)
                loss.backward()
                optimizer.step()
            batch_n = label.size(0)
            train_loss += loss.detach().item() * batch_n
            train_correct += (logits.detach().argmax(-1) == label).sum().item()
            seen += batch_n
        val_metrics, _vl, _vp, _vprobs, _vids = evaluate(model, val_loader, device, use_amp)
        scheduler.step()
        row = {"epoch": epoch, "train_loss": train_loss / max(seen, 1),
               "train_accuracy": train_correct / max(seen, 1),
               "validation_loss": val_metrics["loss"], "validation_accuracy": val_metrics["accuracy"],
               "validation_macro_f1": val_metrics["macro_f1"], "learning_rate": optimizer.param_groups[0]["lr"]}
        history.append(row)
        completed_epochs = epoch
        improved = row["validation_macro_f1"] > best_val_f1
        if improved:
            best_val_f1 = row["validation_macro_f1"]
            atomic_torch_save(best_path, {"epoch": epoch, "model_state_dict": model.state_dict(),
                "validation_metrics": val_metrics, "best_val_f1": best_val_f1,
                "classes": VERIFIED_TARGET_CLASSES, "experiment_name": args.experiment_name})
        state = {"completed_epochs": completed_epochs, "model_state_dict": model.state_dict(),
                 "optimizer_state_dict": optimizer.state_dict(), "scheduler_state_dict": scheduler.state_dict(),
                 "scaler_state_dict": scaler.state_dict() if use_amp else None,
                 "best_val_f1": best_val_f1, "history": history}
        atomic_torch_save(latest_path, state)
        atomic_json(history_path, history)
        manifest["completed_epochs"] = completed_epochs
        manifest["best_validation_macro_f1"] = best_val_f1
        manifest["last_validation_metrics"] = val_metrics
        atomic_json(output / "experiment_manifest.json", manifest)
        print(f"Epoch {epoch:03d}/{args.epochs}: train_loss={row['train_loss']:.4f}, "
              f"val_macro_f1={row['validation_macro_f1']:.4f}" + (" [best]" if improved else ""))

    if not best_path.exists():
        raise RuntimeError("No validation-selected best checkpoint exists; refusing test evaluation.")
    best_state = torch.load(best_path, map_location=device)
    model.load_state_dict(best_state["model_state_dict"])
    test_ds = MMActMultimodalDataset(test_samples, cache_dir=cache / "r3d18_test", enable_cache=True, mode="video_only")
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                             pin_memory=device.type == "cuda", persistent_workers=args.num_workers > 0)
    test_metrics, labels, predictions, probabilities, sample_ids = evaluate(model, test_loader, device, use_amp)
    if len(sample_ids) != EXPECTED_TEST_SAMPLES or set(sample_ids) != set(split_ids["test"]):
        raise RuntimeError("Test loader membership differs from the recorded 852 held-out sample IDs.")
    atomic_json(output / "metrics.json", {"best_validation_metrics": best_state["validation_metrics"],
                                           "test_metrics": test_metrics,
                                           "test_evaluated_after_best_checkpoint_selection": True})
    atomic_json(output / "predictions.json", {"sample_ids": sample_ids, "labels": labels,
        "predictions": predictions, "probabilities": probabilities, "classes": VERIFIED_TARGET_CLASSES})
    with (output / "epoch_history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]) if history else ["epoch"])
        writer.writeheader()
        writer.writerows(history)
    manifest["completed_epochs"] = completed_epochs
    manifest["best_checkpoint_epoch"] = best_state["epoch"]
    manifest["test_evaluation_completed"] = True
    manifest["test_metrics"] = test_metrics
    atomic_json(output / "experiment_manifest.json", manifest)
    print(f"Finished. Artifacts saved in {output.resolve()}; best validation macro-F1={best_val_f1:.4f}.")


if __name__ == "__main__":
    main()
