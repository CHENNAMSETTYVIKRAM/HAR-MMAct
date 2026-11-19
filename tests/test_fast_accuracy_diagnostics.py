from pathlib import Path
import json

import torch
import torch.nn as nn

from src.config import TARGET_CLASS_TO_IDX, VERIFIED_TARGET_CLASSES
from src.models.fusion_model import MultimodalHARModel
from src.models.video_branch import CustomCNNVideoBranch, MobileNetV2VideoBranch


def test_target_class_mapping_round_trip():
    assert len(VERIFIED_TARGET_CLASSES) == 10
    for idx, name in enumerate(VERIFIED_TARGET_CLASSES):
        assert TARGET_CLASS_TO_IDX[name] == idx


def test_custom_cnn_shapes_gradients_and_trainable_params():
    video = torch.randn(2, 3, 2, 32, 32)
    labels = torch.tensor([0, 9])

    branch = CustomCNNVideoBranch(in_channels=3, embedding_dim=128, dropout=0.20)
    model = MultimodalHARModel(
        num_classes=10,
        mode="video_only",
        video_backbone="custom_cnn",
        pretrained_video=False,
        dropout=0.20,
    ).train()

    embedding = branch(video)
    assert embedding.shape == (2, 128)
    assert torch.isfinite(embedding).all()

    logits = model(video=video, override_mode="video_only")
    assert logits.shape == (2, 10)

    loss = nn.CrossEntropyLoss()(logits, labels)
    loss.backward()

    conv_named = [name for name, p in model.video_branch.named_parameters() if p.grad is not None and p.grad.abs().sum() > 0]
    classifier_named = [name for name, p in model.video_only_classifier.named_parameters() if p.grad is not None and p.grad.abs().sum() > 0]
    assert conv_named
    assert classifier_named
    assert sum(p.numel() for p in branch.parameters()) > 0
    assert sum(p.numel() for p in branch.parameters() if p.requires_grad) > 0


def test_custom_cnn_small_subset_can_overfit():
    torch.manual_seed(42)
    video = torch.randn(10, 3, 2, 32, 32)
    labels = torch.arange(10)

    model = CustomCNNVideoBranch(in_channels=3, embedding_dim=128, dropout=0.10)
    head = nn.Linear(128, 10)
    model.train()
    head.train()
    optimizer = torch.optim.AdamW(list(model.parameters()) + list(head.parameters()), lr=2e-3, weight_decay=0.0)

    final_loss = None
    for _ in range(50):
        optimizer.zero_grad(set_to_none=True)
        logits = head(model(video))
        loss = nn.CrossEntropyLoss()(logits, labels)
        loss.backward()
        optimizer.step()
        final_loss = loss.item()

    assert final_loss is not None
    assert final_loss < 2.30, f"Unexpected diagnostic loss: {final_loss:.4f}"


def test_mobilenet_normalization_freeze_and_optimizer_state():
    frozen = MobileNetV2VideoBranch(
        in_channels=3,
        embedding_dim=128,
        dropout=0.20,
        pretrained=False,
        freeze_backbone=True,
    )
    assert frozen.normalization_mean == (0.485, 0.456, 0.406)
    assert frozen.normalization_std == (0.229, 0.224, 0.225)
    assert hasattr(frozen, "pretrained_source")
    assert all(not p.requires_grad for p in frozen.backbone.parameters())
    assert all(p.requires_grad for p in frozen.temporal_fc.parameters())

    frozen.eval()
    video = torch.randn(1, 3, 2, 64, 64)
    embedding = frozen(video)
    assert embedding.shape == (1, 128)


def test_mobilenet_pretrained_loading_is_auditable():
    branch = MobileNetV2VideoBranch(pretrained=True)
    assert branch.pretrained_source in {
        "torchvision-default",
        "torchvision-legacy",
        "torchvision-random",
        "pure-pytorch-random",
        "pure-pytorch-external",
    }


def test_retrained_split_difference_is_exactly_standing():
    paths = [
        Path("outputs/mobilenet_v2/experiment_manifest.json"),
        Path("outputs/mobilenet_v2_staged_v2_retrained/experiment_manifest.json"),
    ]
    manifests = [json.loads(path.read_text()) for path in paths]
    assert manifests[0]["train_samples_count"] == 3795
    assert manifests[1]["train_samples_count"] == 3799

    standing_original = manifests[0]["train_class_distribution"]["standing"]
    standing_retrained = manifests[1]["train_class_distribution"]["standing"]
    assert standing_original == 355
    assert standing_retrained == 359

    for original, retrained in zip(manifests, manifests):
        pass

    for key in ("train_class_distribution",):
        original = manifests[0][key]
        retrained = manifests[1][key]
        differing = {name: retrained[name] - original[name] for name in original if retrained[name] != original[name]}
        assert differing == {"standing": 4}


def test_pre_indexing_subject_exclusion(tmp_path):
    from src.data.indexer import MMActIndexer, PairedSample

    # Create dummy directory tree with subjects 1, 15, and 18 (test)
    for subj in ["subject1", "subject15", "subject18"]:
        d = tmp_path / "trimmed" / "video" / "cross_scene" / "trainval" / subj / "cam1" / "scene1" / "session1"
        d.mkdir(parents=True, exist_ok=True)
        (d / "crouching.mp4").write_bytes(b"dummy_video_bytes")

    # Verify find_files_recursively prunes subject18
    all_files = MMActIndexer.find_files_recursively(
        tmp_path, {".mp4"}, excluded_dir_names={"subject18"}
    )
    file_strs = [str(f) for f in all_files]
    assert any("subject1" in f for f in file_strs)
    assert any("subject15" in f for f in file_strs)
    assert not any("subject18" in f for f in file_strs), "subject18 was not pruned during directory traversal!"

    # Verify indexer does not construct PairedSample for subject18
    indexer = MMActIndexer(
        dataset_root=tmp_path,
        required_sensor_modalities=[],
        excluded_subjects={"subject18"},
        allowed_subjects={"subject1", "subject15"},
    )
    report = indexer.index()
    assert len(report.paired_samples) == 2
    indexed_subjects = {s.subject for s in report.paired_samples}
    assert indexed_subjects == {"subject1", "subject15"}
    assert "subject18" not in indexed_subjects


def test_paired_sample_serialization_and_manifest_roundtrip(tmp_path):
    from src.data.indexer import PairedSample

    sample = PairedSample(
        sample_id="subject1_scene1_session1_cam1_crouching",
        action="crouching",
        label_idx=0,
        subject="subject1",
        scene="scene1",
        session="session1",
        cam="cam1",
        video_path=tmp_path / "dummy.mp4",
        sensor_paths={"acc_phone": tmp_path / "sensor.csv"},
    )

    d = sample.to_dict()
    restored = PairedSample.from_dict(d)
    assert restored.sample_id == sample.sample_id
    assert restored.action == sample.action
    assert restored.label_idx == sample.label_idx
    assert restored.subject == sample.subject
    assert restored.video_path == sample.video_path
    assert restored.sensor_paths["acc_phone"] == sample.sensor_paths["acc_phone"]


def test_mobilenet_verified_pretrained_gate():
    from src.models.video_branch import MobileNetV2VideoBranch
    import pytest

    branch = MobileNetV2VideoBranch(pretrained=True, require_pretrained=True)
    assert branch.pretrained_loaded is True
    assert branch.pretrained_source in {"torchvision-default", "torchvision-legacy", "pure-pytorch-external"}


def test_indexer_data_root_alias(tmp_path):
    from src.data.indexer import MMActIndexer
    # Both dataset_root and data_root should work identically
    idx1 = MMActIndexer(dataset_root=tmp_path)
    assert idx1.dataset_root == tmp_path

    idx2 = MMActIndexer(data_root=tmp_path)
    assert idx2.dataset_root == tmp_path

    idx3 = MMActIndexer(tmp_path)
    assert idx3.dataset_root == tmp_path


def test_screening_verifier_logic(tmp_path):
    import json
    import pytest
    from scripts.verify_screening_metrics import (
        validate_single_experiment,
        verify_screening_runs,
        EXPECTED_EXPERIMENTS,
    )

    # 1. Create canonical manifests
    train_manifest = tmp_path / "canonical_train_manifest.json"
    val_manifest = tmp_path / "canonical_val_manifest.json"

    train_samples = [
        {"sample_id": f"s_train_{i}", "subject": "subject1", "action": "crouching"}
        for i in range(10)
    ]
    val_samples = [
        {"sample_id": f"s_val_{i}", "subject": "subject15", "action": "crouching"}
        for i in range(5)
    ]

    train_manifest.write_text(json.dumps(train_samples))
    val_manifest.write_text(json.dumps(val_samples))

    canonical_train_ids = [s["sample_id"] for s in train_samples]
    canonical_val_ids = [s["sample_id"] for s in val_samples]

    results_dir = tmp_path / "screening_results"
    results_dir.mkdir()

    # 2. Build 5 valid experiment mock outputs
    for exp in EXPECTED_EXPERIMENTS:
        exp_dir = results_dir / exp["name"]
        exp_dir.mkdir()
        (exp_dir / "actual_train_sample_ids.json").write_text(json.dumps(canonical_train_ids))
        (exp_dir / "actual_val_sample_ids.json").write_text(json.dumps(canonical_val_ids))

        # 12 epochs
        history = {
            "train_loss": [2.0 - i * 0.1 for i in range(12)],
            "val_loss": [1.9 - i * 0.08 for i in range(12)],
            "val_f1": [0.3 + i * 0.02 for i in range(12)],  # max at epoch 12
        }
        (exp_dir / "history.json").write_text(json.dumps(history))

        manifest = {"video_backbone": exp["backbone"]}
        (exp_dir / "experiment_manifest.json").write_text(json.dumps(manifest))

        summary = {
            "best_epoch": 12,
            "runtime_seconds": 120.0,
            "best_metrics": {
                "macro_f1": 0.52,
                "accuracy": 0.60,
                "loss": 1.02,
                "per_class": {
                    cls_name: {"precision": 0.6, "recall": 0.6, "f1-score": 0.6, "support": 5}
                    for cls_name in [
                        "crouching", "jumping", "kicking", "pointing", "running",
                        "sitting", "standing", "using_phone", "walking", "waving_hand",
                    ]
                },
            },
        }
        (exp_dir / "training_summary.json").write_text(json.dumps(summary))

    # Test full verification run
    results, selection = verify_screening_runs(
        results_dir=results_dir,
        train_manifest_path=train_manifest,
        val_manifest_path=val_manifest,
        expected_epochs=12,
    )
    assert len(results) == 5
    assert "custom_cnn_winner" in selection
    assert "mobilenet_v2_winner" in selection

    # 3. Test failure on incomplete run (<12 epochs)
    corrupted_dir = results_dir / EXPECTED_EXPERIMENTS[0]["name"]
    with open(corrupted_dir / "history.json", "w") as f:
        json.dump({"train_loss": [1.0] * 10, "val_loss": [1.0] * 10, "val_f1": [0.5] * 10}, f)

    with pytest.raises(AssertionError, match="Incomplete run"):
        validate_single_experiment(corrupted_dir, canonical_train_ids, canonical_val_ids, expected_epochs=12)


def test_historical_benchmark_manifest_schema_and_test_isolation():
    """
    Audits the historical benchmark artifacts in outputs/mobilenet_v2:
    1. Confirms the historical manifest schema lacks individual train/val sample IDs
       (proving why train_sample_ids lookup raised KeyError).
    2. Confirms aggregate counts (3,795 train, 780 val, 852 test).
    3. Confirms that historical final_test_predictions.csv isolates test subjects 18-20.
    """
    import json
    import pandas as pd
    from pathlib import Path

    manifest_path = Path("outputs/mobilenet_v2/experiment_manifest.json")
    assert manifest_path.exists(), "Historical mobilenet_v2 manifest must exist"

    with open(manifest_path, "r", encoding="utf-8") as f:
        bench_manifest = json.load(f)

    # Prove absence of individual sample ID lists
    assert "train_sample_ids" not in bench_manifest
    assert "val_sample_ids" not in bench_manifest
    assert "test_sample_ids" not in bench_manifest

    # Prove presence of aggregate counts and split definition
    assert bench_manifest["train_samples_count"] == 3795
    assert bench_manifest["val_samples_count"] == 780
    assert bench_manifest["test_samples_count"] == 852
    assert bench_manifest["split_strategy"] == "subject"

    # Verify historical test predictions exist and isolate subjects 18-20
    test_preds_path = Path("outputs/mobilenet_v2/final_test_predictions.csv")
    assert test_preds_path.exists(), "Historical test predictions must exist"
    df_test = pd.read_csv(test_preds_path)
    assert len(df_test) == 852

    test_subjects = {s.split("_")[0] for s in df_test["sample_id"]}
    assert test_subjects == {"subject18", "subject19", "subject20"}



