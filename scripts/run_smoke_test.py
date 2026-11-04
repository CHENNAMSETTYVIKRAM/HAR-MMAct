"""
run_smoke_test.py - Synthetic Smoke-Test Runner for HAR_MMAct

Validates the complete end-to-end pipeline:
1. Data indexing and anomaly detection
2. Video frame extraction and sensor resampling/windowing
3. Leakage-free subject partitioning
4. Preprocessing caching
5. MultimodalHARModel forward & backward pass
6. Metric computation (accuracy, macro F1, confusion matrix, OvR ROC AUC)
7. Checkpoint saving

CRITICAL: ALL ARTIFACTS AND LOGS ARE EXPLICITLY LABELED [SYNTHETIC SMOKE TEST].
NO GENUINE MMACT TRAINING IS PERFORMED IN THIS SMOKE TEST.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.config import ProjectConfig, set_seed, VERIFIED_TARGET_CLASSES
from src.data.indexer import MMActIndexer
from src.data.synthetic import generate_synthetic_fixture
from src.data.splits import DatasetSplitter, verify_split_isolation
from src.data.dataset import MMActMultimodalDataset
from src.models.fusion_model import MultimodalHARModel
from src.utils.metrics import compute_classification_metrics


def run_smoke_test():
    set_seed(42)
    cfg = ProjectConfig()

    print("=" * 75)
    print("HAR_MMAct: [SYNTHETIC SMOKE TEST RUNNER]")
    print("NOTE: This runner uses synthetic data to verify pipeline functionality.")
    print("      NO GENUINE MMACT DATA IS TRAINED HERE.")
    print("=" * 75)

    # 1. Ensure synthetic fixture exists
    if not cfg.synthetic_root.exists() or not (cfg.synthetic_root / "SYNTHETIC_METADATA.json").exists():
        print("\n[Step 1] Generating synthetic fixture...")
        generate_synthetic_fixture(cfg.synthetic_root, num_subjects=3, num_cams=2)
    else:
        print("\n[Step 1] Synthetic fixture detected at:", cfg.synthetic_root)

    # 2. Index dataset
    print("\n[Step 2] Running MMActIndexer on synthetic dataset...")
    indexer = MMActIndexer(dataset_root=cfg.synthetic_root)
    report = indexer.index()
    print(f"  - Total Videos Scanned   : {report.total_video_files_found}")
    print(f"  - Total Sensors Scanned  : {report.total_sensor_files_found}")
    print(f"  - Paired Samples Created : {report.paired_samples_count}")
    print(f"  - Unknown Labels Filtered: {len(report.unknown_label_files)}")
    assert report.paired_samples_count > 0, "No paired samples found in synthetic fixture!"

    # 3. Partition into Train / Val / Test
    print("\n[Step 3] Partitioning dataset into leakage-free splits...")
    splitter = DatasetSplitter(cfg.split)
    train_samples, val_samples, test_samples = splitter.split(report.paired_samples)
    print(f"  - Train Samples : {len(train_samples)}")
    print(f"  - Val Samples   : {len(val_samples)}")
    print(f"  - Test Samples  : {len(test_samples)}")

    isolation = verify_split_isolation(train_samples, val_samples, test_samples, strategy=cfg.split.split_strategy)
    for check_name, passed in isolation.items():
        status = "PASSED" if passed else "FAILED"
        print(f"      * {check_name:<30}: {status}")
        assert passed, f"Split isolation check failed: {check_name}"

    # 4. Create PyTorch Datasets & DataLoaders
    print("\n[Step 4] Creating PyTorch Datasets with caching enabled...")
    smoke_cache = cfg.output_root / "cache" / "smoke_test"
    train_ds = MMActMultimodalDataset(train_samples, cache_dir=smoke_cache, enable_cache=True)
    val_ds = MMActMultimodalDataset(val_samples, cache_dir=smoke_cache, enable_cache=True)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0)

    # 5. Instantiate Model
    print("\n[Step 5] Initializing MultimodalHARModel...")
    model = MultimodalHARModel(
        num_classes=len(VERIFIED_TARGET_CLASSES),
        video_channels=3,
        sensor_channels=9,
        video_embed_dim=128,
        sensor_embed_dim=128,
        fusion_hidden_dim=128,
        dropout=0.2,
        mode="multimodal",
    ).to(cfg.device)
    print(f"  - Device: {cfg.device}")
    print(f"  - Model Total Parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    # 6. Run 2-Epoch Miniature Training Loop
    print("\n[Step 6] Running miniature 2-epoch training loop [SYNTHETIC DATA]...")
    for epoch in range(1, 3):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for vids, sens, labels, _ in train_loader:
            vids = vids.to(cfg.device)
            sens = sens.to(cfg.device)
            labels = labels.to(cfg.device)

            optimizer.zero_grad()
            logits = model(vids, sens)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(labels)
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += len(labels)

        epoch_loss = train_loss / max(1, total)
        epoch_acc = correct / max(1, total)
        print(f"  - Epoch {epoch}/2 [Train] Loss: {epoch_loss:.4f} | Acc: {epoch_acc * 100:.1f}%")

    # 7. Validation & Metric Evaluation
    print("\n[Step 7] Evaluating on validation set and computing metrics...")
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for vids, sens, labels, _ in val_loader:
            vids = vids.to(cfg.device)
            sens = sens.to(cfg.device)
            logits = model(vids, sens)
            probs = torch.softmax(logits, dim=-1)

            all_probs.append(probs.cpu().numpy())
            all_preds.extend(logits.argmax(dim=-1).cpu().numpy())
            all_labels.extend(labels.numpy())

    import numpy as np
    val_probs = np.concatenate(all_probs, axis=0) if all_probs else None
    metrics = compute_classification_metrics(all_labels, all_preds, val_probs, classes=VERIFIED_TARGET_CLASSES)

    print("  [Validation Metrics on Synthetic Fixture]:")
    print(f"    * Accuracy         : {metrics['accuracy']:.4f}")
    print(f"    * Macro F1         : {metrics['macro_f1']:.4f}")
    print(f"    * Macro Precision  : {metrics['macro_precision']:.4f}")
    print(f"    * Macro Recall     : {metrics['macro_recall']:.4f}")
    if "roc_auc_ovr" in metrics:
        print(f"    * OvR ROC AUC      : {metrics['roc_auc_ovr']:.4f}")

    # 8. Unimodal Forward Passes (Verification)
    print("\n[Step 8] Verifying unimodal forward passes...")
    sample_vid, sample_sen, _, _ = train_ds[0]
    sample_vid = sample_vid.unsqueeze(0).to(cfg.device)
    sample_sen = sample_sen.unsqueeze(0).to(cfg.device)

    v_out = model(video=sample_vid, override_mode="video_only")
    s_out = model(sensor=sample_sen, override_mode="sensor_only")
    print(f"  - Video-only output shape  : {v_out.shape} -> SUCCESS")
    print(f"  - Sensor-only output shape : {s_out.shape} -> SUCCESS")

    # 9. Save Checkpoint
    checkpoint_dir = cfg.output_root / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = checkpoint_dir / "synthetic_smoke_test_model.pt"
    torch.save(
        {
            "is_synthetic": True,
            "warning": "SMOKE TEST CHECKPOINT ONLY",
            "model_state_dict": model.state_dict(),
            "target_classes": VERIFIED_TARGET_CLASSES,
        },
        ckpt_path,
    )
    print(f"\n[Step 9] Checkpoint saved at: {ckpt_path}")

    print("=" * 75)
    print("[ALL SMOKE TEST CHECKS PASSED SUCCESSFULLY!]")
    print("=" * 75)


if __name__ == "__main__":
    run_smoke_test()
