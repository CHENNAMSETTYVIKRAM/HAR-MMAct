"""
test_indexer.py - Tests for multimodal dataset indexing and diagnostic anomaly reporting
"""

import os
from pathlib import Path
import pytest

from src.config import ProjectConfig, VERIFIED_TARGET_CLASSES
from src.data.indexer import MMActIndexer
from src.data.synthetic import generate_synthetic_fixture


@pytest.fixture(scope="module")
def synthetic_dataset_dir(tmp_path_factory):
    tmp_dir = tmp_path_factory.mktemp("synthetic_mmact")
    generate_synthetic_fixture(
        tmp_dir,
        num_subjects=3,
        num_cams=2,
        include_test_anomalies=True,
    )
    return tmp_dir


def test_indexer_finds_paired_samples(synthetic_dataset_dir):
    indexer = MMActIndexer(dataset_root=synthetic_dataset_dir)
    report = indexer.index()

    # 3 subjects * 2 scenes * 1 session * 2 cams * 10 classes = 120 paired samples
    assert report.paired_samples_count == 120
    assert len(report.paired_samples) == 120

    # Every verified target class must have paired samples
    for cls_name in VERIFIED_TARGET_CLASSES:
        assert report.class_distribution[cls_name] == 12


def test_indexer_filters_unknown_labels(synthetic_dataset_dir):
    indexer = MMActIndexer(dataset_root=synthetic_dataset_dir)
    report = indexer.index()

    # Anomaly file 'drinking.mp4' must be filtered as an unknown label
    assert any("drinking" in path for path in report.unknown_label_files)


def test_indexer_handles_nonexistent_dir(tmp_path):
    empty_dir = tmp_path / "does_not_exist"
    indexer = MMActIndexer(dataset_root=empty_dir)
    report = indexer.index()
    assert report.paired_samples_count == 0
    assert report.total_video_files_found == 0


def test_indexer_traverses_symlinks(synthetic_dataset_dir, tmp_path):
    symlink_root = tmp_path / "symlinked_root"
    symlink_root.mkdir()
    target_link = symlink_root / "trimmed"
    try:
        os.symlink(synthetic_dataset_dir / "trimmed", target_link, target_is_directory=True)
    except OSError:
        pytest.skip("Symlink creation not permitted in this environment")

    indexer = MMActIndexer(dataset_root=symlink_root)
    report = indexer.index()
    assert report.total_video_files_found > 0
    assert report.total_sensor_files_found > 0
    assert report.paired_samples_count == 120


def test_indexer_duplicate_recordings_prevention(tmp_path):
    # Create dataset with duplicate recording files
    dup_dir = tmp_path / "dup_dataset"
    generate_synthetic_fixture(dup_dir, num_subjects=1, num_cams=1, include_test_anomalies=False)

    # Intentionally duplicate a video file under an alias directory structure that maps to same key
    vid_file = list(dup_dir.glob("**/*.mp4"))[0]
    dup_vid = vid_file.parent / f"copy_{vid_file.name}"
    import shutil
    shutil.copyfile(vid_file, dup_vid)

    indexer = MMActIndexer(dataset_root=dup_dir)
    report = indexer.index()
    # Duplicate ID should be flagged, and paired samples must not contain duplicates
    sample_ids = [s.sample_id for s in report.paired_samples]
    assert len(sample_ids) == len(set(sample_ids)), "Duplicate sample IDs found in paired samples!"
