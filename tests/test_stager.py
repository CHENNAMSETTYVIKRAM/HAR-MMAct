"""
test_stager.py - Tests for fast, idempotent, bounded MMAct dataset discovery and staging.
"""

import os
import shutil
import tempfile
from pathlib import Path
import pytest

from src.data.stager import FastMMActStager, StagingReport


@pytest.fixture
def mock_dataset_env(tmp_path):
    """Creates a mock Kaggle input layout with video, sensor, and split files."""
    input_root = tmp_path / "mock_kaggle_input" / "mmact_dataset"
    canonical_dir = tmp_path / "canonical_mmact"

    # Create mock video: subject16/cam1/scene1/session1/crouching.mp4
    video_dir = input_root / "trimmed_x-scene_subject16" / "subject16" / "cam1" / "scene1" / "session1"
    video_dir.mkdir(parents=True, exist_ok=True)
    (video_dir / "crouching.mp4").write_bytes(b"dummy video data")

    # Create mock sensor: trimmed_sensor/acc_phone_clip/subject16/scene1/session1/crouching.csv
    sensor_dir = input_root / "trimmed_sensor" / "acc_phone_clip" / "subject16" / "scene1" / "session1"
    sensor_dir.mkdir(parents=True, exist_ok=True)
    (sensor_dir / "crouching.csv").write_text("timestamp,acc_x,acc_y,acc_z\n0,0.1,0.2,0.3\n")

    watch_dir = input_root / "trimmed_sensor" / "acc_watch_clip" / "subject16" / "scene1" / "session1"
    watch_dir.mkdir(parents=True, exist_ok=True)
    (watch_dir / "crouching.csv").write_text("timestamp,acc_x,acc_y,acc_z\n0,0.1,0.2,0.3\n")

    gyro_dir = input_root / "trimmed_sensor" / "gyro_clip" / "subject16" / "scene1" / "session1"
    gyro_dir.mkdir(parents=True, exist_ok=True)
    (gyro_dir / "crouching.csv").write_text("timestamp,gyro_x,gyro_y,gyro_z\n0,0.01,0.02,0.03\n")

    # Create mock split: splits/trimmed_x-scene_train.txt
    split_dir = input_root / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    (split_dir / "trimmed_x-scene_train.txt").write_text("dummy split content")

    return input_root, canonical_dir


def test_first_staging_and_idempotency(mock_dataset_env):
    input_root, canonical_dir = mock_dataset_env
    stager = FastMMActStager(canonical_dir=canonical_dir, input_roots=[input_root])

    # 1. First run: should discover and stage video, sensors, and splits
    report1 = stager.discover_and_stage()
    assert report1.video_count >= 1
    assert report1.sensor_count >= 3
    assert report1.split_count >= 1
    assert "subject16" in report1.video_subjects
    assert report1.reused is False
    assert report1.staging_duration_sec < 5.0

    # 2. Second run: must detect existing canonical data and reuse immediately without scanning
    report2 = stager.discover_and_stage()
    assert report2.reused is True
    assert report2.video_count == report1.video_count
    assert report2.sensor_count == report1.sensor_count
    assert report2.split_count == report1.split_count
    assert report2.staging_duration_sec < 0.5  # Instantaneous check


def test_nested_links_prevention(mock_dataset_env):
    input_root, canonical_dir = mock_dataset_env
    stager = FastMMActStager(canonical_dir=canonical_dir, input_roots=[input_root])

    # Stage first time
    _ = stager.discover_and_stage()

    # Re-stage multiple times
    for _ in range(3):
        FastMMActStager.safe_link_or_copy(
            input_root / "trimmed_x-scene_subject16" / "subject16",
            canonical_dir / "trimmed" / "video" / "cross_scene" / "trainval" / "subject16",
        )

    # Ensure no nested subject16/subject16 directory was created
    subject_dir = canonical_dir / "trimmed" / "video" / "cross_scene" / "trainval" / "subject16"
    assert not (subject_dir / "subject16").exists(), "Nested subject16/subject16 directory detected!"


def test_broken_symlink_repair(mock_dataset_env, tmp_path):
    input_root, canonical_dir = mock_dataset_env
    stager = FastMMActStager(canonical_dir=canonical_dir, input_roots=[input_root])

    # Run initial staging
    _ = stager.discover_and_stage()

    # Intentionally corrupt a symlink by pointing it to a deleted file
    test_split = canonical_dir / "trimmed" / "splits" / "trimmed_x-scene_train.txt"
    test_split.unlink(missing_ok=True)
    dead_target = tmp_path / "non_existent_file.txt"
    try:
        os.symlink(dead_target, test_split)
    except OSError:
        pass  # If symlink creation requires admin on Windows, skip manual corrupt symlink

    # Staging should repair the broken link gracefully without throwing FileNotFoundError
    res = FastMMActStager.safe_link_or_copy(
        input_root / "splits" / "trimmed_x-scene_train.txt",
        test_split,
    )
    assert test_split.exists()


def test_missing_modality_reporting(tmp_path):
    # Test with empty input root
    empty_root = tmp_path / "empty_input"
    empty_root.mkdir()
    canonical_dir = tmp_path / "canonical_empty"

    stager = FastMMActStager(canonical_dir=canonical_dir, input_roots=[empty_root])
    report = stager.discover_and_stage()

    assert not report.is_valid
    assert len(report.missing_required_paths) > 0
    assert "No video files found" in report.missing_required_paths[0]
