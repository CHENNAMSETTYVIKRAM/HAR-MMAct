"""
test_evaluation.py - Tests for checkpoint compatibility, evaluation correctness, and unimodal isolation
"""

import json
from pathlib import Path
import torch
import pytest

from src.config import VERIFIED_TARGET_CLASSES
from src.models.fusion_model import MultimodalHARModel
from scripts.evaluate import load_model_from_checkpoint, run_evaluation_for_mode
from torch.utils.data import DataLoader, TensorDataset


@pytest.fixture
def dummy_checkpoints(tmp_path):
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()

    # 1. Video-only checkpoint (Custom CNN)
    model_cnn = MultimodalHARModel(num_classes=10, mode="video_only", video_backbone="custom_cnn")
    cnn_path = ckpt_dir / "best_model_cnn.pth"
    torch.save({
        "model_state_dict": model_cnn.state_dict(),
        "config": {"mode": "video_only", "video_backbone": "custom_cnn", "sensor_backbone": "inception_time"},
        "classes": VERIFIED_TARGET_CLASSES,
    }, cnn_path)

    # 2. Sensor-only checkpoint (InceptionTime)
    model_sen = MultimodalHARModel(num_classes=10, mode="sensor_only", video_backbone="custom_cnn", sensor_backbone="inception_time")
    sen_path = ckpt_dir / "best_model_sensor.pth"
    torch.save({
        "model_state_dict": model_sen.state_dict(),
        "config": {"mode": "sensor_only", "video_backbone": "custom_cnn", "sensor_backbone": "inception_time"},
        "classes": VERIFIED_TARGET_CLASSES,
    }, sen_path)

    # 3. Multimodal checkpoint
    model_mm = MultimodalHARModel(num_classes=10, mode="multimodal", video_backbone="custom_cnn", sensor_backbone="inception_time")
    mm_path = ckpt_dir / "best_model_multimodal.pth"
    torch.save({
        "model_state_dict": model_mm.state_dict(),
        "config": {"mode": "multimodal", "video_backbone": "custom_cnn", "sensor_backbone": "inception_time"},
        "classes": VERIFIED_TARGET_CLASSES,
    }, mm_path)

    return cnn_path, sen_path, mm_path


def test_checkpoint_compatibility_loading(dummy_checkpoints):
    cnn_path, sen_path, mm_path = dummy_checkpoints
    device = torch.device("cpu")

    model_cnn, _, cfg_cnn = load_model_from_checkpoint(cnn_path, device, VERIFIED_TARGET_CLASSES)
    assert cfg_cnn["mode"] == "video_only"
    assert model_cnn is not None

    model_sen, _, cfg_sen = load_model_from_checkpoint(sen_path, device, VERIFIED_TARGET_CLASSES)
    assert cfg_sen["mode"] == "sensor_only"
    assert model_sen is not None

    model_mm, _, cfg_mm = load_model_from_checkpoint(mm_path, device, VERIFIED_TARGET_CLASSES)
    assert cfg_mm["mode"] == "multimodal"
    assert model_mm is not None


def test_unimodal_evaluation_correctness(dummy_checkpoints):
    cnn_path, sen_path, _ = dummy_checkpoints
    device = torch.device("cpu")

    # Create dummy DataLoader: video (B, 3, 16, 112, 112), sensor (B, 9, 100), label (B,)
    batch_size = 2
    dummy_video = torch.randn(4, 3, 16, 112, 112)
    dummy_sensor = torch.randn(4, 9, 100)
    dummy_label = torch.tensor([0, 1, 2, 3])
    dummy_ids = torch.tensor([0, 1, 2, 3])

    loader = DataLoader(
        TensorDataset(dummy_video, dummy_sensor, dummy_label, dummy_ids),
        batch_size=batch_size,
    )

    # Evaluate Video-only
    model_cnn, _, _ = load_model_from_checkpoint(cnn_path, device, VERIFIED_TARGET_CLASSES)
    vo_metrics, y_true, probs = run_evaluation_for_mode(model_cnn, loader, device, "video_only", VERIFIED_TARGET_CLASSES)
    assert "accuracy" in vo_metrics
    assert "macro_f1" in vo_metrics
    assert len(y_true) == 4
    assert probs.shape == (4, 10)

    # Evaluate Sensor-only
    model_sen, _, _ = load_model_from_checkpoint(sen_path, device, VERIFIED_TARGET_CLASSES)
    so_metrics, y_true_s, probs_s = run_evaluation_for_mode(model_sen, loader, device, "sensor_only", VERIFIED_TARGET_CLASSES)
    assert "accuracy" in so_metrics
    assert "macro_f1" in so_metrics
    assert len(y_true_s) == 4
    assert probs_s.shape == (4, 10)


def test_checkpoint_class_mismatch_rejection(dummy_checkpoints, tmp_path):
    _, _, mm_path = dummy_checkpoints
    bad_ckpt_path = tmp_path / "bad_classes.pth"
    ckpt = torch.load(mm_path)
    ckpt["classes"] = ["crouching", "jumping"]  # only 2 classes
    torch.save(ckpt, bad_ckpt_path)

    with pytest.raises(ValueError, match="Checkpoint classes.*do not match"):
        load_model_from_checkpoint(bad_ckpt_path, torch.device("cpu"), VERIFIED_TARGET_CLASSES)


def test_checkpoint_invalid_mode_rejection(dummy_checkpoints, tmp_path):
    _, _, mm_path = dummy_checkpoints
    bad_mode_path = tmp_path / "bad_mode.pth"
    ckpt = torch.load(mm_path)
    ckpt["config"]["mode"] = "invalid_mode_xyz"
    torch.save(ckpt, bad_mode_path)

    with pytest.raises(ValueError, match="Checkpoint mode 'invalid_mode_xyz' is invalid"):
        load_model_from_checkpoint(bad_mode_path, torch.device("cpu"), VERIFIED_TARGET_CLASSES)


def test_evaluate_cli_missing_checkpoint():
    import subprocess
    import sys
    res = subprocess.run(
        [sys.executable, "scripts/evaluate.py", "--checkpoint", "non_existent_file.pth"],
        capture_output=True,
        text=True,
    )
    assert res.returncode != 0
    assert "Checkpoint not found" in res.stdout or "Checkpoint not found" in res.stderr


def test_infer_cli_missing_inputs():
    import subprocess
    import sys
    # Both video and sensors missing
    res = subprocess.run(
        [sys.executable, "scripts/infer.py", "--checkpoint", "outputs/checkpoints/best_model.pth"],
        capture_output=True,
        text=True,
    )
    # Either checkpoint not found or input not provided
    assert res.returncode != 0

    # Non-existent video path provided
    res2 = subprocess.run(
        [
            sys.executable,
            "scripts/infer.py",
            "--checkpoint",
            "outputs/checkpoints/best_model.pth",
            "--video",
            "non_existent_video.mp4",
        ],
        capture_output=True,
        text=True,
    )
    assert res2.returncode != 0
    assert "Specified video file does not exist" in res2.stdout or "Checkpoint not found" in res2.stdout
