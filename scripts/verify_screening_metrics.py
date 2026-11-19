import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
from src.config import VERIFIED_TARGET_CLASSES
from src.data.indexer import MMActIndexer
from src.data.splits import DatasetSplitter, SplitConfig
from src.data.dataset import MMActMultimodalDataset
from torch.utils.data import DataLoader
from src.models.fusion_model import MultimodalHARModel
from src.utils.metrics import compute_classification_metrics

def main():
    indexer = MMActIndexer(dataset_root=Path("data/synthetic_smoke_test"))
    report = indexer.index()
    split_cfg = SplitConfig(split_strategy="subject", dataset_root=Path("data/synthetic_smoke_test"))
    splitter = DatasetSplitter(split_cfg, dataset_root=Path("data/synthetic_smoke_test"))
    train_s, val_s, test_s = splitter.split(report.paired_samples)

    print(f"Train samples: {len(train_s)}, Val samples: {len(val_s)}, Test samples: {len(test_s)}")
    train_ids = {s.sample_id for s in train_s}
    val_ids = {s.sample_id for s in val_s}
    test_ids = {s.sample_id for s in test_s}
    print(f"Train/Val overlap: {len(train_ids.intersection(val_ids))}")
    print(f"Train/Test overlap: {len(train_ids.intersection(test_ids))}")
    print(f"Val/Test overlap: {len(val_ids.intersection(test_ids))}")

    # Check class representation
    val_classes = {}
    for s in val_s:
        val_classes[s.action] = val_classes.get(s.action, 0) + 1
    print(f"Validation class distribution (10 classes present: {len(val_classes) == 10}):")
    print(val_classes)

    val_dataset = MMActMultimodalDataset(val_s, mode="video_only", enable_cache=False)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    criterion = nn.CrossEntropyLoss()

    checkpoints = [
        ("cnn_dropout035", "outputs/diagnostics/cnn_dropout035/checkpoints/best_model_custom_cnn.pth", "custom_cnn", 0.35, False),
        ("cnn_dropout020", "outputs/diagnostics/cnn_dropout020/checkpoints/best_model_custom_cnn.pth", "custom_cnn", 0.20, False),
        ("cnn_dropout025_cw", "outputs/diagnostics/cnn_dropout025_cw/checkpoints/best_model_custom_cnn.pth", "custom_cnn", 0.25, False),
        ("mobilenet_frozen10", "outputs/diagnostics/mobilenet_frozen10/checkpoints/best_model_mobilenet_v2.pth", "mobilenet_v2", 0.20, True),
        ("mobilenet_staged8", "outputs/diagnostics/mobilenet_staged8/checkpoints/best_model_mobilenet_v2.pth", "mobilenet_v2", 0.20, False),
    ]

    for name, ckpt_path, backbone, dropout, freeze in checkpoints:
        print(f"\n=================== {name} ===================")
        if not Path(ckpt_path).exists():
            print(f"File not found: {ckpt_path}")
            continue
        ckpt = torch.load(ckpt_path, map_location="cpu")
        saved_loss = ckpt.get("val_loss")
        saved_acc = ckpt.get("val_acc")
        saved_f1 = ckpt.get("val_f1")
        saved_epoch = ckpt.get("epoch")
        print(f"Saved Checkpoint Epoch: {saved_epoch}")
        print(f"Saved in Checkpoint: Loss={saved_loss}, Acc={saved_acc}%, Macro-F1={saved_f1}")

        model = MultimodalHARModel(
            num_classes=10,
            mode="video_only",
            video_backbone=backbone,
            dropout=dropout,
            freeze_video_backbone=freeze,
            pretrained_video=False,
        )
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        all_preds, all_labels, all_probs = [], [], []
        total_loss, total_count = 0.0, 0
        with torch.no_grad():
            for v, s, l, _ in val_loader:
                out = model(video=v, override_mode="video_only")
                loss = criterion(out, l)
                total_loss += loss.item() * len(l)
                total_count += len(l)
                all_preds.extend(out.argmax(-1).tolist())
                all_labels.extend(l.tolist())
                all_probs.extend(torch.softmax(out, dim=-1).tolist())

        recalculated_loss = total_loss / total_count
        metrics = compute_classification_metrics(all_labels, all_preds, all_probs, VERIFIED_TARGET_CLASSES)
        print(f"Recalculated from Checkpoint Weights on Val Split:")
        print(f"  Loss:      {recalculated_loss:.4f} (Diff: {abs(recalculated_loss - (saved_loss or 0)):.6f})")
        print(f"  Accuracy:  {metrics['accuracy']*100:.2f}% (Diff: {abs(metrics['accuracy']*100 - (saved_acc or 0)):.6f})")
        print(f"  Macro-F1:  {metrics['macro_f1']:.4f} (Diff: {abs(metrics['macro_f1'] - (saved_f1 or 0)):.6f})")
        print(f"  Per-Class F1: {[round(metrics['per_class'][c]['f1'], 3) for c in VERIFIED_TARGET_CLASSES]}")

if __name__ == "__main__":
    main()
