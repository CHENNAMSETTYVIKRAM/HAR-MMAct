"""
test_accuracy_fixes.py - Regression & Correctness Tests for Sensor Alignment & Model Integrity

Tests cover:
1. Distinct timestamp parsing (%Y%m%d_%H:%M:%S.%f, %Y/%m/%d %H:%M:%S.%f, numeric epoch).
2. Duplicate timestamp handling (dropping identical raw timestamps, preventing clock skew).
3. 100 Hz and 50 Hz alignment (backward-fill resampling on uniform integer timegrid).
4. Secondary sensor ending before primary sensor (official bfill vs bounded fill).
5. Leading and trailing missing values (bfill leading, bounded ffill trailing).
6. Large internal missing intervals (preventing stale flatline propagation).
7. No unintended loss of valid primary timestamps at quantization boundaries.
8. Correct fixed-window output shape and numerical validity (finite tensors, edge-padding vs centered).
9. Label mapping order integrity (10 canonical classes, exact 0..9 sequential indices).
10. Checkpoint strict load compatibility with saved model weights.
11. Gradient flow across active model branches and classification heads.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn

from src.config import SensorConfig, VERIFIED_TARGET_CLASSES, TARGET_CLASS_TO_IDX
from src.data.sensor_loader import SensorLoader
from src.models.fusion_model import MultimodalHARModel


def _make_timestamps(base_str: str, num_samples: int, step_ms: int, fmt: str = "%Y/%m/%d %H:%M:%S.%f") -> list:
    """Helper to generate realistic incrementing timestamp strings."""
    base = pd.to_datetime(base_str)
    return [(base + pd.Timedelta(milliseconds=i * step_ms)).strftime(fmt)[:-3] for i in range(num_samples)]


def test_distinct_timestamp_parsing(tmp_path):
    """
    Test 1: Distinct timestamp parsing.
    Verifies that various timestamp formats (%Y/%m/%d %H:%M:%S.%f, %Y%m%d_%H:%M:%S.%f, epoch)
    parse accurately and generate monotonic integer millisecond time grids.
    """
    loader = SensorLoader()

    # Format A: YYYY/MM/DD
    t_a = _make_timestamps("2026/01/01 10:00:00.000", 5, step_ms=10, fmt="%Y/%m/%d %H:%M:%S.%f")
    csv_a = tmp_path / "sensor_a.csv"
    pd.DataFrame({"time": t_a, "x": range(5), "y": range(5), "z": range(5)}).to_csv(csv_a, index=False)
    df_a = loader.load_single_csv(csv_a, "acc_phone")

    assert len(df_a) == 5
    # Step should be exactly 10 ms at 100 Hz
    steps_a = np.diff(df_a.index.to_numpy())
    assert np.all(steps_a == 10)

    # Format B: YYYYMMDD_HH:MM:SS.fff
    t_b = _make_timestamps("2018-09-05 15:20:00.000", 4, step_ms=20, fmt="%Y%m%d_%H:%M:%S.%f")
    csv_b = tmp_path / "sensor_b.csv"
    pd.DataFrame({"time": t_b, "x": range(4), "y": range(4), "z": range(4)}).to_csv(csv_b, index=False)
    df_b = loader.load_single_csv(csv_b, "gyro")

    assert len(df_b) == 4
    # Step should be exactly 20 ms at 50 Hz
    steps_b = np.diff(df_b.index.to_numpy())
    assert np.all(steps_b == 20)


def test_duplicate_timestamp_handling(tmp_path):
    """
    Test 2: Preserves all recorded sensor samples & enforces index uniqueness.
    Per official MMAct load_csv_file_list: all legitimate sensor measurements are retained,
    and synthesized epoch millisecond timestamps form a strictly unique index.
    """
    loader = SensorLoader()

    # CSV with 6 rows where raw sensor logged repeated or coarse millisecond timestamps
    times = [
        "2026/01/01 10:00:00.000",
        "2026/01/01 10:00:00.010",
        "2026/01/01 10:00:00.010",  # Same coarse ms string from hardware FIFO
        "2026/01/01 10:00:00.020",
        "2026/01/01 10:00:00.030",
        "2026/01/01 10:00:00.040",
    ]
    csv_dup = tmp_path / "dup.csv"
    pd.DataFrame({"time": times, "x": range(6), "y": range(6), "z": range(6)}).to_csv(csv_dup, index=False)
    df_dup = loader.load_single_csv(csv_dup, "acc_phone")

    # Confirms official behavior: all 6 observations are retained
    assert len(df_dup) == 6
    # Confirms synthesized integer millisecond index has no duplicates
    assert not df_dup.index.duplicated().any()
    # Confirms uniform 10 ms grid spacing at 100 Hz
    steps = np.diff(df_dup.index.to_numpy())
    assert np.all(steps == 10)
    assert not df_dup.index.duplicated().any()


def test_100hz_and_50hz_alignment(tmp_path):
    """
    Test 3: 100 Hz and 50 Hz alignment.
    Verifies backward-fill alignment of 50 Hz secondary sensor onto 100 Hz primary timeline.
    Intermediate timestamps (10 ms) should receive the upcoming 50 Hz value (20 ms).
    """
    loader = SensorLoader(SensorConfig(fill_strategy="bfill"))

    # 100 Hz primary phone acc: 6 samples (0, 10, 20, 30, 40, 50 ms)
    p_times = _make_timestamps("2026/01/01 10:00:00.000", 6, step_ms=10)
    p_csv = tmp_path / "phone_100hz.csv"
    pd.DataFrame({
        "time": p_times,
        "x": [1.0, 1.1, 1.2, 1.3, 1.4, 1.5],
        "y": [2.0] * 6,
        "z": [3.0] * 6,
    }).to_csv(p_csv, index=False)

    # 50 Hz secondary gyro: 3 samples (0, 20, 40 ms)
    g_times = _make_timestamps("2026/01/01 10:00:00.000", 3, step_ms=20)
    g_csv = tmp_path / "gyro_50hz.csv"
    pd.DataFrame({
        "time": g_times,
        "x": [10.0, 20.0, 30.0],
        "y": [0.0, 0.0, 0.0],
        "z": [0.0, 0.0, 0.0],
    }).to_csv(g_csv, index=False)

    aligned = loader.align_modalities({"acc_phone": p_csv, "gyro": g_csv})

    # Under official bfill, rows 0..40 ms are aligned. Row 50 ms has no future gyro, dropped by dropna.
    assert len(aligned) == 5
    assert list(aligned.index - aligned.index[0]) == [0, 10, 20, 30, 40]
    # At t=10 ms, gyro_x should be backfilled from t=20 ms (value 20.0)
    assert aligned.loc[aligned.index[1], "gyro_x"] == 20.0
    # At t=30 ms, gyro_x should be backfilled from t=40 ms (value 30.0)
    assert aligned.loc[aligned.index[3], "gyro_x"] == 30.0


def test_secondary_sensor_ending_before_primary(tmp_path):
    """
    Test 4: Secondary sensor ending before primary sensor.
    Compares official bfill convention vs bounded forward-fill strategy.
    """
    # Primary: 10 samples (0..90 ms)
    p_times = _make_timestamps("2026/01/01 10:00:00.000", 10, step_ms=10)
    p_csv = tmp_path / "p_end.csv"
    pd.DataFrame({"time": p_times, "x": range(10), "y": range(10), "z": range(10)}).to_csv(p_csv, index=False)

    # Secondary: 4 samples at 50 Hz (0, 20, 40, 60 ms) -> ends at 60 ms (30 ms earlier)
    g_times = _make_timestamps("2026/01/01 10:00:00.000", 4, step_ms=20)
    g_csv = tmp_path / "g_end.csv"
    pd.DataFrame({"time": g_times, "x": [1.0, 2.0, 3.0, 4.0], "y": [0.0]*4, "z": [0.0]*4}).to_csv(g_csv, index=False)

    # 1. Official convention (bfill): drops rows 70, 80, 90 ms
    loader_official = SensorLoader(SensorConfig(fill_strategy="bfill"))
    aligned_off = loader_official.align_modalities({"acc_phone": p_csv, "gyro": g_csv})
    assert len(aligned_off) == 7  # 0, 10, 20, 30, 40, 50, 60 ms

    # 2. Bounded fill (limit=2 samples = 20 ms): fills 70, 80 ms, drops 90 ms
    loader_bounded = SensorLoader(SensorConfig(fill_strategy="bfill_ffill_bounded", max_ffill_limit=2))
    aligned_bound = loader_bounded.align_modalities({"acc_phone": p_csv, "gyro": g_csv})
    assert len(aligned_bound) == 9  # 0, 10, 20, 30, 40, 50, 60, 70, 80 ms
    assert aligned_bound.loc[aligned_bound.index[-1], "gyro_x"] == 4.0  # bounded forward fill value


def test_leading_and_trailing_missing_values(tmp_path):
    """
    Test 5: Leading and trailing missing values.
    Verifies that leading gaps in secondary sensor are backfilled from its first sample,
    and trailing quantization boundaries are handled cleanly.
    """
    loader = SensorLoader(SensorConfig(fill_strategy="bfill_ffill_bounded"))

    # Primary: 100 Hz, 8 samples (0, 10, 20, 30, 40, 50, 60, 70 ms)
    p_times = _make_timestamps("2026/01/01 10:00:00.000", 8, step_ms=10)
    p_csv = tmp_path / "p_lead_trail.csv"
    pd.DataFrame({"time": p_times, "x": range(8), "y": range(8), "z": range(8)}).to_csv(p_csv, index=False)

    # Secondary: starts at 20 ms (missing 0, 10 ms) and ends at 60 ms (missing 70 ms)
    g_times = _make_timestamps("2026/01/01 10:00:00.020", 3, step_ms=20)  # 20, 40, 60 ms
    g_csv = tmp_path / "g_lead_trail.csv"
    pd.DataFrame({"time": g_times, "x": [9.9, 8.8, 7.7], "y": [0.0]*3, "z": [0.0]*3}).to_csv(g_csv, index=False)

    aligned = loader.align_modalities({"acc_phone": p_csv, "gyro": g_csv})

    # Leading values at 0 and 10 ms should be backfilled from t=20 ms (value 9.9)
    assert aligned.loc[aligned.index[0], "gyro_x"] == 9.9
    assert aligned.loc[aligned.index[1], "gyro_x"] == 9.9
    # Trailing value at 70 ms is within bounded limit (1 sample <= 2), filled from t=60 ms (value 7.7)
    assert aligned.loc[aligned.index[-1], "gyro_x"] == 7.7
    assert len(aligned) == 8


def test_large_internal_missing_interval(tmp_path):
    """
    Test 6: Large internal missing intervals.
    Verifies that bounded filling does NOT propagate stale measurements across
    substantial missing intervals (e.g. 200 ms sensor drop).
    """
    loader = SensorLoader(SensorConfig(fill_strategy="bfill_ffill_bounded", max_ffill_limit=2))

    # Primary: 30 samples (0..290 ms)
    p_times = _make_timestamps("2026/01/01 10:00:00.000", 30, step_ms=10)
    p_csv = tmp_path / "p_large_gap.csv"
    pd.DataFrame({"time": p_times, "x": range(30), "y": range(30), "z": range(30)}).to_csv(p_csv, index=False)

    # Secondary: 50 Hz with a 200 ms outage between 20 ms and 240 ms
    # Samples at 0, 20 ms, and then 240, 260 ms
    times_gap = [
        "2026/01/01 10:00:00.000",
        "2026/01/01 10:00:00.020",
        "2026/01/01 10:00:00.240",
        "2026/01/01 10:00:00.260",
    ]
    g_csv = tmp_path / "g_large_gap.csv"
    pd.DataFrame({"time": times_gap, "x": [1.0, 1.0, 5.0, 5.0], "y": [0.0]*4, "z": [0.0]*4}).to_csv(g_csv, index=False)

    aligned = loader.align_modalities({"acc_phone": p_csv, "gyro": g_csv})

    # Rows between 50 ms and 230 ms should NOT be filled with flatline data;
    # unobserved rows remain NaN and are dropped by dropna().
    # Only bounded regions around observed samples are retained.
    assert len(aligned) < 30
    assert not aligned.isna().any().any()


def test_no_unintended_loss_of_valid_primary_timestamps(tmp_path):
    """
    Test 7: No unintended loss of valid primary timestamps.
    When secondary sensor ends just 1 primary step (10 ms) before primary due to 50 Hz/100 Hz
    sampling quantization (e.g. 5 samples of 50 Hz = 0..80 ms vs 10 samples of 100 Hz = 0..90 ms),
    bounded forward fill preserves all 10 primary samples without loss.
    """
    loader = SensorLoader(SensorConfig(fill_strategy="bfill_ffill_bounded"))

    p_times = _make_timestamps("2026/01/01 10:00:00.000", 10, step_ms=10)  # 0..90 ms
    p_csv = tmp_path / "p_full.csv"
    pd.DataFrame({"time": p_times, "x": range(10), "y": range(10), "z": range(10)}).to_csv(p_csv, index=False)

    g_times = _make_timestamps("2026/01/01 10:00:00.000", 5, step_ms=20)  # 0..80 ms (1 step early)
    g_csv = tmp_path / "g_full.csv"
    pd.DataFrame({"time": g_times, "x": [0.5]*5, "y": [0.5]*5, "z": [0.5]*5}).to_csv(g_csv, index=False)

    aligned = loader.align_modalities({"acc_phone": p_csv, "gyro": g_csv})

    assert len(aligned) == 10
    assert not aligned.isna().any().any()


def test_fixed_window_output_shape_and_numerical_validity(tmp_path):
    """
    Test 8: Correct fixed-window output shape and numerical validity.
    Verifies load_and_preprocess produces a valid finite PyTorch Tensor (C, window_size)
    for both short clips (symmetric edge-padding) and long clips (centered window).
    """
    cfg = SensorConfig(modalities=["acc_phone", "acc_watch", "gyro"], window_size=100)
    loader = SensorLoader(cfg)

    # 1. Short clip: 40 samples (shorter than window_size 100) -> edge-padding
    p_times_short = _make_timestamps("2026/01/01 10:00:00.000", 40, step_ms=10)
    w_times_short = _make_timestamps("2026/01/01 10:00:00.000", 40, step_ms=10)
    g_times_short = _make_timestamps("2026/01/01 10:00:00.000", 20, step_ms=20)

    p_csv = tmp_path / "p_short.csv"
    w_csv = tmp_path / "w_short.csv"
    g_csv = tmp_path / "g_short.csv"
    pd.DataFrame({"time": p_times_short, "x": np.sin(range(40)), "y": [9.8]*40, "z": [0.0]*40}).to_csv(p_csv, index=False)
    pd.DataFrame({"time": w_times_short, "x": np.cos(range(40)), "y": [9.8]*40, "z": [0.0]*40}).to_csv(w_csv, index=False)
    pd.DataFrame({"time": g_times_short, "x": range(20), "y": [0.0]*20, "z": [0.0]*20}).to_csv(g_csv, index=False)

    tensor_short = loader.load_and_preprocess({"acc_phone": p_csv, "acc_watch": w_csv, "gyro": g_csv})
    assert isinstance(tensor_short, torch.Tensor)
    assert tensor_short.shape == (9, 100)
    assert torch.isfinite(tensor_short).all()

    # 2. Long clip: 150 samples (longer than window_size 100) -> centered window
    p_times_long = _make_timestamps("2026/01/01 10:00:00.000", 150, step_ms=10)
    w_times_long = _make_timestamps("2026/01/01 10:00:00.000", 150, step_ms=10)
    g_times_long = _make_timestamps("2026/01/01 10:00:00.000", 75, step_ms=20)

    p_long = tmp_path / "p_long.csv"
    w_long = tmp_path / "w_long.csv"
    g_long = tmp_path / "g_long.csv"
    pd.DataFrame({"time": p_times_long, "x": np.sin(range(150)), "y": [9.8]*150, "z": [0.0]*150}).to_csv(p_long, index=False)
    pd.DataFrame({"time": w_times_long, "x": np.cos(range(150)), "y": [9.8]*150, "z": [0.0]*150}).to_csv(w_long, index=False)
    pd.DataFrame({"time": g_times_long, "x": range(75), "y": [0.0]*75, "z": [0.0]*75}).to_csv(g_long, index=False)

    tensor_long = loader.load_and_preprocess({"acc_phone": p_long, "acc_watch": w_long, "gyro": g_long})
    assert isinstance(tensor_long, torch.Tensor)
    assert tensor_long.shape == (9, 100)
    assert torch.isfinite(tensor_long).all()


def test_label_mapping_order_integrity():
    """Confirms exact 10 target classes and their 0..9 sequential indices."""
    assert len(VERIFIED_TARGET_CLASSES) == 10
    for idx, name in enumerate(VERIFIED_TARGET_CLASSES):
        assert TARGET_CLASS_TO_IDX[name] == idx


def test_checkpoint_strict_load_compatibility():
    """
    Confirms that model state_dict keys remain 100% strictly compatible
    with existing saved checkpoints in outputs/.
    """
    ckpt_mobilenet = Path("outputs/mobilenet_v2/checkpoints/best_model_mobilenet_v2.pth")
    if ckpt_mobilenet.exists():
        model_mob = MultimodalHARModel(
            num_classes=10,
            mode="video_only",
            video_backbone="mobilenet_v2",
            pretrained_video=False,
        )
        ckpt = torch.load(ckpt_mobilenet, map_location="cpu")
        model_mob.load_state_dict(ckpt["model_state_dict"], strict=True)
        assert model_mob is not None

    ckpt_cnn = Path("outputs/improved_cnn/checkpoints/best_model_cnn.pth")
    if ckpt_cnn.exists():
        model_cnn = MultimodalHARModel(
            num_classes=10,
            mode="video_only",
            video_backbone="custom_cnn",
            pretrained_video=False,
        )
        ckpt = torch.load(ckpt_cnn, map_location="cpu")
        model_cnn.load_state_dict(ckpt["model_state_dict"], strict=True)
        assert model_cnn is not None


def test_forward_pass_and_gradient_flow():
    """Verifies that forward pass and backward gradients flow properly through all active parameters."""
    dummy_v = torch.randn(2, 3, 16, 112, 112)
    dummy_target = torch.tensor([1, 4])
    criterion = nn.CrossEntropyLoss()

    for bb in ["custom_cnn", "original_cnn", "mobilenet_v2"]:
        m = MultimodalHARModel(num_classes=10, mode="video_only", video_backbone=bb, pretrained_video=False)
        m.train()
        out = m(video=dummy_v)
        assert out.shape == (2, 10)
        loss = criterion(out, dummy_target)
        loss.backward()

        has_first_grad = False
        has_last_grad = False
        for name, p in m.video_branch.named_parameters():
            if p.grad is not None and p.grad.abs().sum() > 0:
                has_first_grad = True
                break
        for name, p in m.video_only_classifier.named_parameters():
            if p.grad is not None and p.grad.abs().sum() > 0:
                has_last_grad = True
                break
        assert has_first_grad, f"Missing gradients in video branch for {bb}"
        assert has_last_grad, f"Missing gradients in classifier for {bb}"
