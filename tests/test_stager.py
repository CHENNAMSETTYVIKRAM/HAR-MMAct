"""
test_stager.py - Tests for fast, idempotent, bounded MMAct dataset discovery,
merged recording inventory staging, full-file SHA-256 duplicate resolution, and integrity checks.
"""

import json
import os
import shutil
import tempfile
from pathlib import Path
import pytest

from src.data.stager import (
    FastMMActStager,
    StagingReport,
    discover_kaggle_input_roots,
    full_file_hash,
    fast_file_hash,
    EXPECTED_MODALITIES,
    EXPECTED_OFFICIAL_SPLITS,
)


@pytest.fixture
def mock_dataset_env(tmp_path):
    """Creates a mock Kaggle input layout with video, all 4 sensor modalities, and all 4 official splits."""
    input_root = tmp_path / "mock_kaggle_input" / "mmact_dataset"
    canonical_dir = tmp_path / "canonical_mmact"

    # Create mock video: subject16/cam1/scene1/session1/crouching.mp4
    video_dir = input_root / "trimmed_x-scene_subject16" / "subject16" / "cam1" / "scene1" / "session1"
    video_dir.mkdir(parents=True, exist_ok=True)
    (video_dir / "crouching.mp4").write_bytes(b"dummy video data for subject 16 crouching")

    # Create all 4 mock sensor modalities
    for mod in EXPECTED_MODALITIES:
        s_dir = input_root / "trimmed_sensor" / mod / "subject16" / "scene1" / "session1"
        s_dir.mkdir(parents=True, exist_ok=True)
        (s_dir / "crouching.csv").write_text("timestamp,acc_x,acc_y,acc_z\n0,0.1,0.2,0.3\n1,0.2,0.3,0.4\n")

    # Create all 4 mock official splits
    split_dir = input_root / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    for sp_name in EXPECTED_OFFICIAL_SPLITS:
        (split_dir / sp_name).write_text("subject16/scene1/session1/crouching 0\n")

    return input_root, canonical_dir


@pytest.fixture
def mock_kaggle_multi_dataset(tmp_path):
    """
    Creates realistic mock Kaggle layout with Dataset A (sensors, splits, subjects 8 & 16)
    and Dataset B (expanded subject video folders subject1, subject8, subject16).
    """
    kaggle_input = tmp_path / "kaggle" / "input"
    canonical_dir = tmp_path / "canonical_staged"

    # Dataset A: /kaggle/input/datasets/vikramche/mmact-trimmed/
    dataset_a = kaggle_input / "datasets" / "vikramche" / "mmact-trimmed"

    # Nested sensor path: trimmed_sensor/sensor/
    for mod in EXPECTED_MODALITIES:
        for subj in ["subject8", "subject16"]:
            s_dir = dataset_a / "trimmed_sensor" / "sensor" / mod / subj / "scene1" / "session1"
            s_dir.mkdir(parents=True, exist_ok=True)
            (s_dir / "jumping.csv").write_text("timestamp,x,y,z\n0,0.1,0.2,0.3\n1,0.2,0.3,0.4\n")

    # Nested splits: trimmed_splits/splits/
    splits_dir = dataset_a / "trimmed_splits" / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    for sp_name in EXPECTED_OFFICIAL_SPLITS:
        (splits_dir / sp_name).write_text("subject8/scene1/session1/jumping 0\n")

    # Dataset A video: subject8 and subject16
    vid_a8 = dataset_a / "trimmed_x-scene_subject8" / "subject8" / "cam1" / "scene1" / "session1"
    vid_a8.mkdir(parents=True, exist_ok=True)
    (vid_a8 / "jumping.mp4").write_bytes(b"SUBJECT8_COMMON_VIDEO_PAYLOAD_EXACT_BYTES")

    vid_a16 = dataset_a / "trimmed_x-scene_subject16" / "subject16" / "cam1" / "scene1" / "session1"
    vid_a16.mkdir(parents=True, exist_ok=True)
    (vid_a16 / "jumping.mp4").write_bytes(b"SUBJECT16_COMMON_VIDEO_PAYLOAD_EXACT_BYTES")

    # Dataset B: /kaggle/input/datasets/vikramche/trimmed-scene-video/
    dataset_b = kaggle_input / "datasets" / "vikramche" / "trimmed-scene-video"

    # Dataset B expanded videos: subject1, subject8 (identical dup), subject16 (identical dup)
    vid_b1 = dataset_b / "trimmed" / "cross_scene_video" / "trimmed_x-scene_subject1" / "video" / "cross_scene" / "trainval" / "subject1" / "cam1" / "scene1" / "session1"
    vid_b1.mkdir(parents=True, exist_ok=True)
    (vid_b1 / "jumping.mp4").write_bytes(b"SUBJECT1_UNIQUE_VIDEO_PAYLOAD")

    vid_b8 = dataset_b / "trimmed" / "cross_scene_video" / "trimmed_x-scene_subject8" / "video" / "cross_scene" / "trainval" / "subject8" / "cam1" / "scene1" / "session1"
    vid_b8.mkdir(parents=True, exist_ok=True)
    (vid_b8 / "jumping.mp4").write_bytes(b"SUBJECT8_COMMON_VIDEO_PAYLOAD_EXACT_BYTES")

    vid_b16 = dataset_b / "trimmed" / "cross_scene_video" / "trimmed_x-scene_subject16" / "video" / "cross_scene" / "trainval" / "subject16" / "cam1" / "scene1" / "session1"
    vid_b16.mkdir(parents=True, exist_ok=True)
    (vid_b16 / "jumping.mp4").write_bytes(b"SUBJECT16_COMMON_VIDEO_PAYLOAD_EXACT_BYTES")

    return kaggle_input, canonical_dir, dataset_a, dataset_b


def test_first_staging_and_idempotency(mock_dataset_env):
    input_root, canonical_dir = mock_dataset_env
    stager = FastMMActStager(canonical_dir=canonical_dir, input_roots=[input_root])

    # 1. First run: should discover and stage video, sensors, and splits
    report1 = stager.discover_and_stage()
    assert report1.is_valid is True
    assert report1.video_count >= 1
    assert report1.sensor_count >= 4
    assert report1.split_count == 4
    assert "subject16" in report1.video_subjects
    assert report1.reused is False
    assert report1.staging_duration_sec < 5.0

    # 2. Second run: must detect existing canonical data and reuse idempotently
    report2 = stager.discover_and_stage()
    assert report2.is_valid is True
    assert report2.reused is True
    assert report2.video_count == report1.video_count
    assert report2.sensor_count == report1.sensor_count
    assert report2.split_count == report1.split_count


def test_nested_links_prevention(mock_dataset_env):
    input_root, canonical_dir = mock_dataset_env
    stager = FastMMActStager(canonical_dir=canonical_dir, input_roots=[input_root])

    _ = stager.discover_and_stage()

    for _ in range(3):
        FastMMActStager.safe_link_or_copy(
            input_root / "trimmed_x-scene_subject16" / "subject16" / "cam1" / "scene1" / "session1" / "crouching.mp4",
            canonical_dir / "trimmed" / "video" / "cross_scene" / "trainval" / "subject16" / "cam1" / "scene1" / "session1" / "crouching.mp4",
        )

    subject_dir = canonical_dir / "trimmed" / "video" / "cross_scene" / "trainval" / "subject16"
    assert not (subject_dir / "subject16").exists(), "Nested subject16/subject16 directory detected!"


def test_broken_symlink_repair(mock_dataset_env, tmp_path):
    input_root, canonical_dir = mock_dataset_env
    stager = FastMMActStager(canonical_dir=canonical_dir, input_roots=[input_root])

    _ = stager.discover_and_stage()

    test_split = canonical_dir / "trimmed" / "splits" / "trimmed_x-scene_train.txt"
    test_split.unlink(missing_ok=True)
    dead_target = tmp_path / "non_existent_file.txt"
    try:
        os.symlink(dead_target, test_split)
    except OSError:
        pass

    res = FastMMActStager.safe_link_or_copy(
        input_root / "splits" / "trimmed_x-scene_train.txt",
        test_split,
    )
    assert test_split.exists()
    assert res in {"repaired", "linked", "copied"}


def test_missing_modality_reporting(tmp_path):
    empty_root = tmp_path / "empty_input"
    empty_root.mkdir()
    canonical_dir = tmp_path / "canonical_empty"

    stager = FastMMActStager(canonical_dir=canonical_dir, input_roots=[empty_root])
    report = stager.discover_and_stage()

    assert not report.is_valid
    assert len(report.missing_required_paths) > 0


def test_discover_kaggle_input_roots(mock_kaggle_multi_dataset):
    """Verifies discover_kaggle_input_roots discovers nested user datasets and avoids intermediate directories."""
    kaggle_input, _, dataset_a, dataset_b = mock_kaggle_multi_dataset

    direct_dataset = kaggle_input / "extra-direct-dataset"
    direct_dataset.mkdir(parents=True, exist_ok=True)

    discovered = discover_kaggle_input_roots(kaggle_input)
    assert len(discovered) == 3
    assert dataset_a in discovered
    assert dataset_b in discovered
    assert direct_dataset in discovered
    assert (kaggle_input / "datasets") not in discovered
    assert (kaggle_input / "datasets" / "vikramche") not in discovered


def test_nested_sensor_discovery_and_four_modalities(mock_kaggle_multi_dataset):
    """Verifies discovery of nested trimmed_sensor/sensor/ and all 4 modalities."""
    kaggle_input, canonical_dir, _, _ = mock_kaggle_multi_dataset
    input_roots = discover_kaggle_input_roots(kaggle_input)

    stager = FastMMActStager(canonical_dir=canonical_dir, input_roots=input_roots)
    report = stager.discover_and_stage()

    assert report.is_valid
    assert report.sensor_count > 0
    for mod in EXPECTED_MODALITIES:
        assert mod in report.discovered_modalities, f"Modality {mod} missing from discovered_modalities!"
        assert mod in report.sensor_modality_paths, f"Modality {mod} missing from sensor_modality_paths!"

    canonical_sensors = canonical_dir / "trimmed" / "sensors"
    assert canonical_sensors.exists()
    for mod in EXPECTED_MODALITIES:
        assert (canonical_sensors / mod).exists(), f"Canonical directory missing {mod}!"


def test_nested_splits_discovery(mock_kaggle_multi_dataset):
    """Verifies discovery of nested trimmed_splits/splits/ and split line details."""
    kaggle_input, canonical_dir, _, _ = mock_kaggle_multi_dataset
    input_roots = discover_kaggle_input_roots(kaggle_input)

    stager = FastMMActStager(canonical_dir=canonical_dir, input_roots=input_roots)
    report = stager.discover_and_stage()

    assert report.split_count == 4
    assert len(report.split_file_line_counts) == 4
    for sp_name in EXPECTED_OFFICIAL_SPLITS:
        assert sp_name in report.split_file_line_counts
        assert report.split_file_line_counts[sp_name] == 1

    canonical_splits = canonical_dir / "trimmed" / "splits"
    assert canonical_splits.exists()
    for sp_name in EXPECTED_OFFICIAL_SPLITS:
        assert (canonical_splits / sp_name).exists()


def test_check_symlink_support(tmp_path):
    """Verifies check_symlink_support probe runs without crashing and cleans up probe file."""
    probe_dir = tmp_path / "probe_dir"
    probe_dir.mkdir()
    res = FastMMActStager.check_symlink_support(probe_dir)
    assert isinstance(res, bool)
    assert not (probe_dir / ".test_symlink_probe").exists()


# =====================================================================
# Specific Required Regression Tests (13 Targeted Areas)
# =====================================================================

def test_full_hash_identical_large_files(tmp_path):
    """1. Two large files with identical content produce identical full hashes."""
    f1 = tmp_path / "large1.bin"
    f2 = tmp_path / "large2.bin"
    # Create 3MB files
    content = b"0123456789ABCDEF" * (3 * 1024 * 1024 // 16)
    f1.write_bytes(content)
    f2.write_bytes(content)

    h1 = full_file_hash(f1)
    h2 = full_file_hash(f2)
    assert h1 == h2
    assert len(h1) == 64


def test_full_hash_different_middle_content(tmp_path):
    """
    2. Two large files with the same size but different middle content produce different hashes.
    Sampled hashing would fail here; full-file SHA-256 must distinguish them.
    """
    f1 = tmp_path / "file1.bin"
    f2 = tmp_path / "file2.bin"

    # Both files have identical first 512KB and identical last 512KB
    head = b"H" * (512 * 1024)
    tail = b"T" * (512 * 1024)

    # Different middle 2MB
    mid1 = b"A" * (2 * 1024 * 1024)
    mid2 = b"B" * (2 * 1024 * 1024)

    f1.write_bytes(head + mid1 + tail)
    f2.write_bytes(head + mid2 + tail)

    assert f1.stat().st_size == f2.stat().st_size
    # A sampled hash inspecting only head and tail would match:
    assert fast_file_hash(f1) == fast_file_hash(f2)
    # But full_file_hash MUST produce different hashes:
    assert full_file_hash(f1) != full_file_hash(f2)


def test_three_sources_duplicate_comparison(tmp_path):
    """3. Three sources containing the same recording are all compared."""
    canonical_dir = tmp_path / "canonical"
    src_a = tmp_path / "source_a"
    src_b = tmp_path / "source_b"
    src_c = tmp_path / "source_c"

    for s in [src_a, src_b, src_c]:
        p = s / "trimmed_x-scene_subject8" / "subject8" / "cam1" / "scene1" / "session1"
        p.mkdir(parents=True, exist_ok=True)
        (p / "jumping.mp4").write_bytes(b"IDENTICAL_PAYLOAD_ABC")

    # Add sensors and splits to src_a to allow validation
    for mod in EXPECTED_MODALITIES:
        sp = src_a / "trimmed_sensor" / mod / "subject8" / "scene1" / "session1"
        sp.mkdir(parents=True, exist_ok=True)
        (sp / "jumping.csv").write_text("t,x,y,z\n0,1,2,3\n")
    for s_name in EXPECTED_OFFICIAL_SPLITS:
        s_dir = src_a / "splits"
        s_dir.mkdir(parents=True, exist_ok=True)
        (s_dir / s_name).write_text("subject8/scene1/session1/jumping 0\n")

    stager = FastMMActStager(canonical_dir=canonical_dir, input_roots=[src_a, src_b, src_c])
    report = stager.discover_and_stage()

    # All three sources must be compared and deduplicated into one canonical recording
    assert report.duplicate_recordings_count == 2
    assert len(report.duplicate_conflicts) == 0

    # Now corrupt source C to verify conflict detection across all 3
    shutil.rmtree(canonical_dir)
    (src_c / "trimmed_x-scene_subject8" / "subject8" / "cam1" / "scene1" / "session1" / "jumping.mp4").write_bytes(b"DIFFERENT_PAYLOAD_C")

    stager2 = FastMMActStager(canonical_dir=canonical_dir, input_roots=[src_a, src_b, src_c])
    report2 = stager2.discover_and_stage()

    assert len(report2.duplicate_conflicts) == 1
    conflict = report2.duplicate_conflicts[0]
    assert len(conflict["sources"]) == 3
    assert not report2.is_valid


def test_recording_only_in_secondary_source_preserved(tmp_path):
    """4. A recording found only in a secondary source is preserved in the merged canonical inventory."""
    canonical_dir = tmp_path / "canonical"
    src_a = tmp_path / "source_a"
    src_b = tmp_path / "source_b"

    # Source A has crouching
    p_a = src_a / "trimmed_x-scene_subject8" / "subject8" / "cam1" / "scene1" / "session1"
    p_a.mkdir(parents=True, exist_ok=True)
    (p_a / "crouching.mp4").write_bytes(b"CROUCHING_DATA")

    # Source B has crouching AND walking (only in Source B!)
    p_b1 = src_b / "trimmed" / "cross_scene_video" / "trimmed_x-scene_subject8" / "video" / "cross_scene" / "trainval" / "subject8" / "cam1" / "scene1" / "session1"
    p_b1.mkdir(parents=True, exist_ok=True)
    (p_b1 / "crouching.mp4").write_bytes(b"CROUCHING_DATA")

    p_b2 = src_b / "trimmed" / "cross_scene_video" / "trimmed_x-scene_subject8" / "video" / "cross_scene" / "trainval" / "subject8" / "cam2" / "scene1" / "session1"
    p_b2.mkdir(parents=True, exist_ok=True)
    (p_b2 / "walking.mp4").write_bytes(b"WALKING_DATA_UNIQUE_TO_SOURCE_B")

    # Add sensors and splits
    for mod in EXPECTED_MODALITIES:
        sp = src_a / "sensor" / mod / "subject8" / "scene1" / "session1"
        sp.mkdir(parents=True, exist_ok=True)
        (sp / "crouching.csv").write_text("t,x,y,z\n0,1,2,3\n")
    for s_name in EXPECTED_OFFICIAL_SPLITS:
        s_dir = src_a / "splits"
        s_dir.mkdir(parents=True, exist_ok=True)
        (s_dir / s_name).write_text("subject8/scene1/session1/crouching 0\n")

    stager = FastMMActStager(canonical_dir=canonical_dir, input_roots=[src_a, src_b])
    report = stager.discover_and_stage()

    assert report.is_valid is True
    # Merged inventory must contain BOTH crouching and walking
    assert report.video_count == 2
    c_vid = canonical_dir / "trimmed" / "video" / "cross_scene" / "trainval" / "subject8"
    assert (c_vid / "cam1" / "scene1" / "session1" / "crouching.mp4").exists()
    assert (c_vid / "cam2" / "scene1" / "session1" / "walking.mp4").exists()


def test_identical_duplicates_staged_only_once(mock_kaggle_multi_dataset):
    """5. Identical duplicates are staged only once."""
    kaggle_input, canonical_dir, _, _ = mock_kaggle_multi_dataset
    input_roots = discover_kaggle_input_roots(kaggle_input)

    stager = FastMMActStager(canonical_dir=canonical_dir, input_roots=input_roots)
    report = stager.discover_and_stage()

    assert report.duplicate_recordings_count >= 2
    assert len(report.duplicate_conflicts) == 0

    s8_videos = list((canonical_dir / "trimmed" / "video" / "cross_scene" / "trainval" / "subject8").rglob("*.mp4"))
    assert len(s8_videos) == 1
    assert s8_videos[0].name == "jumping.mp4"


def test_conflicting_duplicates_block_training(mock_kaggle_multi_dataset):
    """6. Conflicting duplicates block training."""
    kaggle_input, canonical_dir, _, dataset_b = mock_kaggle_multi_dataset

    # Corrupt subject8 video in dataset B
    s8_b = dataset_b / "trimmed" / "cross_scene_video" / "trimmed_x-scene_subject8" / "video" / "cross_scene" / "trainval" / "subject8" / "cam1" / "scene1" / "session1" / "jumping.mp4"
    s8_b.write_bytes(b"CONFLICTING_SUBJECT8_DIFFERENT_CONTENT_HERE")

    input_roots = discover_kaggle_input_roots(kaggle_input)
    stager = FastMMActStager(canonical_dir=canonical_dir, input_roots=input_roots)
    report = stager.discover_and_stage()

    assert len(report.duplicate_conflicts) == 1
    assert report.is_valid is False
    assert any("duplicate recording conflicts" in p for p in report.missing_required_paths)


def test_same_size_different_content_not_reused(tmp_path):
    """7. Existing same-size but different-content destination files are not reused."""
    src = tmp_path / "src.mp4"
    dst = tmp_path / "dst.mp4"

    src.write_bytes(b"A" * 2048)
    dst.write_bytes(b"B" * 2048)
    assert src.stat().st_size == dst.stat().st_size

    action = FastMMActStager.safe_link_or_copy(src, dst)
    # Must repair/replace, NEVER return 'reused'
    assert action == "repaired"
    assert full_file_hash(dst) == full_file_hash(src)


def test_incomplete_destination_directory_repaired(tmp_path):
    """8. Non-empty but incomplete destination directories are repaired or rejected."""
    src_dir = tmp_path / "src_dir"
    dst_dir = tmp_path / "dst_dir"

    (src_dir / "sub").mkdir(parents=True)
    (src_dir / "file1.txt").write_text("file 1 content")
    (src_dir / "sub" / "file2.txt").write_text("file 2 content")

    # dst_dir only has file1.txt, missing file2.txt
    (dst_dir / "file1.txt").parent.mkdir(parents=True, exist_ok=True)
    (dst_dir / "file1.txt").write_text("file 1 content")

    action = FastMMActStager.safe_link_or_copy(src_dir, dst_dir)
    assert action == "repaired"
    assert (dst_dir / "sub" / "file2.txt").exists()
    assert (dst_dir / "sub" / "file2.txt").read_text() == "file 2 content"


def test_missing_orientation_clip_makes_report_invalid(mock_dataset_env):
    """9. Missing orientation_clip makes the report invalid."""
    input_root, canonical_dir = mock_dataset_env
    # Delete orientation_clip from input
    shutil.rmtree(input_root / "trimmed_sensor" / "orientation_clip", ignore_errors=True)

    stager = FastMMActStager(canonical_dir=canonical_dir, input_roots=[input_root])
    report = stager.discover_and_stage()

    assert report.is_valid is False
    assert "orientation_clip" not in report.discovered_modalities
    assert any("orientation_clip" in msg for msg in report.missing_required_paths)


def test_missing_official_split_makes_report_invalid(mock_dataset_env):
    """10. Missing any required official split file makes the report invalid."""
    input_root, canonical_dir = mock_dataset_env
    # Delete trimmed_x-view_val.txt
    (input_root / "splits" / "trimmed_x-view_val.txt").unlink(missing_ok=True)

    stager = FastMMActStager(canonical_dir=canonical_dir, input_roots=[input_root])
    report = stager.discover_and_stage()

    assert report.is_valid is False
    assert report.split_count < 4
    assert any("trimmed_x-view_val.txt" in msg for msg in report.missing_required_paths)


def test_canonical_missing_modality_cannot_trigger_early_return(tmp_path):
    """11. Existing canonical data missing a modality cannot trigger early-return reuse."""
    canonical_dir = tmp_path / "canonical_partial"

    # Create canonical video
    v_dir = canonical_dir / "trimmed" / "video" / "cross_scene" / "trainval" / "subject16" / "cam1" / "scene1" / "session1"
    v_dir.mkdir(parents=True, exist_ok=True)
    (v_dir / "jumping.mp4").write_bytes(b"VIDEO")

    # Create only 3 sensor modalities in canonical (missing orientation_clip)
    for mod in ["acc_phone_clip", "acc_watch_clip", "gyro_clip"]:
        s_dir = canonical_dir / "trimmed" / "sensors" / mod / "subject16" / "scene1" / "session1"
        s_dir.mkdir(parents=True, exist_ok=True)
        (s_dir / "jumping.csv").write_text("t,x,y,z\n0,1,2,3\n")

    # Create all 4 splits
    sp_dir = canonical_dir / "trimmed" / "splits"
    sp_dir.mkdir(parents=True, exist_ok=True)
    for sp_name in EXPECTED_OFFICIAL_SPLITS:
        (sp_dir / sp_name).write_text("subject16/scene1/session1/jumping 0\n")

    # Call stager with empty input roots
    stager = FastMMActStager(canonical_dir=canonical_dir, input_roots=[])
    report = stager.discover_and_stage()

    # Must NOT trigger early-return reuse as valid
    assert report.reused is False
    assert report.is_valid is False


def test_rerunning_staging_produces_same_complete_inventory(mock_kaggle_multi_dataset):
    """12. Re-running staging produces the same complete recording inventory."""
    kaggle_input, canonical_dir, _, _ = mock_kaggle_multi_dataset
    input_roots = discover_kaggle_input_roots(kaggle_input)

    stager1 = FastMMActStager(canonical_dir=canonical_dir, input_roots=input_roots)
    report1 = stager1.discover_and_stage()
    assert report1.is_valid is True

    # List all staged videos
    videos1 = sorted([str(p.relative_to(canonical_dir)) for p in canonical_dir.rglob("*.mp4")])

    stager2 = FastMMActStager(canonical_dir=canonical_dir, input_roots=input_roots)
    report2 = stager2.discover_and_stage()

    videos2 = sorted([str(p.relative_to(canonical_dir)) for p in canonical_dir.rglob("*.mp4")])

    assert report2.is_valid is True
    assert report2.reused is True
    assert videos1 == videos2
    assert report1.video_count == report2.video_count
    assert report1.sensor_count == report2.sensor_count
    assert report1.split_count == report2.split_count


def test_original_source_files_remain_unchanged(mock_kaggle_multi_dataset):
    """13. Original source files remain unchanged."""
    kaggle_input, canonical_dir, _, _ = mock_kaggle_multi_dataset

    initial_stats = {}
    for f in kaggle_input.rglob("*"):
        if f.is_file():
            stat = f.stat()
            h = full_file_hash(f)
            initial_stats[str(f)] = (stat.st_size, stat.st_mtime, h)

    input_roots = discover_kaggle_input_roots(kaggle_input)
    stager = FastMMActStager(canonical_dir=canonical_dir, input_roots=input_roots)
    _ = stager.discover_and_stage()

    for path_str, (init_size, init_mtime, init_hash) in initial_stats.items():
        p = Path(path_str)
        assert p.exists(), f"Source file was deleted or moved: {p}"
        stat = p.stat()
        assert stat.st_size == init_size, f"Source file size changed: {p}"
        assert stat.st_mtime == init_mtime, f"Source file mtime changed: {p}"
        assert full_file_hash(p) == init_hash, f"Source file content changed: {p}"


def test_summary_and_manifest(mock_kaggle_multi_dataset):
    """Verifies summary string formatting and manifest file contents."""
    kaggle_input, canonical_dir, _, _ = mock_kaggle_multi_dataset
    input_roots = discover_kaggle_input_roots(kaggle_input)

    stager = FastMMActStager(canonical_dir=canonical_dir, input_roots=input_roots)
    report = stager.discover_and_stage()

    summary_str = report.summary()
    assert "DATASET PREFLIGHT DISCOVERY & STAGING REPORT" in summary_str
    assert "Video Files Count" in summary_str
    assert "Sensor CSV Files Count" in summary_str
    assert "Official Split Files" in summary_str
    assert report.is_valid is True

    manifest_path = canonical_dir / "dataset_manifest.json"
    assert manifest_path.exists()
    data = json.loads(manifest_path.read_text())
    assert data["is_valid"] is True
    assert data["total_video_count"] == report.video_count
    assert data["total_sensor_count"] == report.sensor_count
    assert data["total_split_count"] == report.split_count


def test_missing_split_line_counts_makes_report_invalid(tmp_path):
    """
    Regression Test (Fix 1):
    All 4 official split filenames are present in official_split_files,
    but split_file_line_counts is empty. Expected result: is_valid=False.
    """
    report = StagingReport(
        dataset_root_selected=tmp_path,
        canonical_dir=tmp_path,
        expected_modalities=EXPECTED_MODALITIES,
        discovered_modalities=EXPECTED_MODALITIES,
        video_count=10,
        sensor_count=10,
        split_count=4,
        official_split_files={sp: f"/path/to/{sp}" for sp in EXPECTED_OFFICIAL_SPLITS},
        split_file_line_counts={},  # Empty line counts
    )
    assert report.is_valid is False

    # Also verify that a split with line_count == 0 makes it invalid
    report_zero = StagingReport(
        dataset_root_selected=tmp_path,
        canonical_dir=tmp_path,
        expected_modalities=EXPECTED_MODALITIES,
        discovered_modalities=EXPECTED_MODALITIES,
        video_count=10,
        sensor_count=10,
        split_count=4,
        official_split_files={sp: f"/path/to/{sp}" for sp in EXPECTED_OFFICIAL_SPLITS},
        split_file_line_counts={
            "trimmed_x-scene_train.txt": 100,
            "trimmed_x-scene_val.txt": 50,
            "trimmed_x-view_train.txt": 100,
            "trimmed_x-view_val.txt": 0,  # Zero lines
        },
    )
    assert report_zero.is_valid is False


def test_directory_inventory_with_extra_stale_file_repaired(tmp_path):
    """
    Regression Test (Fix 2):
    Destination directory contains every expected source file plus an extra stale file.
    Must detect extra file, remove it, and return 'repaired'.
    """
    src_dir = tmp_path / "src"
    dst_dir = tmp_path / "dst"

    src_dir.mkdir()
    dst_dir.mkdir()

    (src_dir / "valid1.txt").write_text("valid content 1")
    (src_dir / "valid2.txt").write_text("valid content 2")

    (dst_dir / "valid1.txt").write_text("valid content 1")
    (dst_dir / "valid2.txt").write_text("valid content 2")
    (dst_dir / "stale_extra.txt").write_text("stale unneeded file")

    action = FastMMActStager.safe_link_or_copy(src_dir, dst_dir)
    assert action == "repaired"

    # Stale file must be removed
    assert not (dst_dir / "stale_extra.txt").exists()

    # Exact inventories must match
    src_files = sorted([p.name for p in src_dir.iterdir()])
    dst_files = sorted([p.name for p in dst_dir.iterdir()])
    assert src_files == dst_files


def test_directory_inventory_with_modified_file_repaired(tmp_path):
    """
    Regression Test (Fix 2):
    Destination directory contains a modified file (different content).
    Must repair destination file to match source and return 'repaired'.
    """
    src_dir = tmp_path / "src"
    dst_dir = tmp_path / "dst"

    src_dir.mkdir()
    dst_dir.mkdir()

    (src_dir / "file1.txt").write_text("SOURCE_CANONICAL_CONTENT")
    (dst_dir / "file1.txt").write_text("CORRUPTED_MODIFIED_CONTENT")

    action = FastMMActStager.safe_link_or_copy(src_dir, dst_dir)
    assert action == "repaired"
    assert (dst_dir / "file1.txt").read_text() == "SOURCE_CANONICAL_CONTENT"


def test_directory_inventory_missing_source_file_repaired(tmp_path):
    """
    Regression Test (Fix 2):
    Destination is missing a source file. Must repair and return 'repaired'.
    """
    src_dir = tmp_path / "src"
    dst_dir = tmp_path / "dst"

    src_dir.mkdir()
    dst_dir.mkdir()

    (src_dir / "file1.txt").write_text("file 1")
    (src_dir / "file2.txt").write_text("file 2")

    (dst_dir / "file1.txt").write_text("file 1")

    action = FastMMActStager.safe_link_or_copy(src_dir, dst_dir)
    assert action == "repaired"
    assert (dst_dir / "file2.txt").exists()
    assert (dst_dir / "file2.txt").read_text() == "file 2"


def test_directory_inventory_second_run_reused(tmp_path):
    """
    Regression Test (Fix 2):
    A second staging run over an identical destination preserves exact expected inventory
    and returns 'reused'.
    """
    src_dir = tmp_path / "src"
    dst_dir = tmp_path / "dst"

    src_dir.mkdir()
    (src_dir / "sub").mkdir()
    (src_dir / "f1.txt").write_text("content 1")
    (src_dir / "sub" / "f2.txt").write_text("content 2")

    # Initial copy/link
    action1 = FastMMActStager.safe_link_or_copy(src_dir, dst_dir)
    assert action1 in {"copied", "linked"}

    # Second run over matching directory
    action2 = FastMMActStager.safe_link_or_copy(src_dir, dst_dir)
    assert action2 == "reused"

    # Verify identical file sets
    src_rel = sorted([str(p.relative_to(src_dir)) for p in src_dir.rglob("*") if p.is_file()])
    dst_rel = sorted([str(p.relative_to(dst_dir)) for p in dst_dir.rglob("*") if p.is_file()])
    assert src_rel == dst_rel
