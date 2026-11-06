"""
test_splits.py - Comprehensive Unit Tests for MMAct Dataset Splitting & Verification Protocols
"""

from pathlib import Path
import pytest

from src.config import SplitConfig, VERIFIED_TARGET_CLASSES
from src.data.indexer import MMActIndexer, PairedSample
from src.data.splits import (
    DatasetSplitter,
    find_official_split_files,
    verify_split_isolation,
    verify_split_representation,
)
from src.data.synthetic import generate_synthetic_fixture


@pytest.fixture(scope="module")
def multi_subject_samples(tmp_path_factory):
    tmp_dir = tmp_path_factory.mktemp("splits_fixture")
    generate_synthetic_fixture(tmp_dir, num_subjects=4, num_cams=2, include_test_anomalies=False)
    indexer = MMActIndexer(dataset_root=tmp_dir)
    report = indexer.index()
    return report.paired_samples


def test_subject_split_isolation(multi_subject_samples):
    config = SplitConfig(split_strategy="subject", partition_val_for_test=True)
    splitter = DatasetSplitter(config)
    train, val, test = splitter.split(multi_subject_samples)

    assert len(train) > 0
    assert len(val) > 0
    assert len(test) > 0
    assert len(train) + len(val) + len(test) == len(multi_subject_samples)

    isolation = verify_split_isolation(train, val, test, strategy="subject")
    assert isolation["all_isolation_checks_passed"]
    assert isolation["video_disjoint_train_val"]
    assert isolation["video_disjoint_train_test"]
    assert isolation["session_disjoint_train_val"]
    assert isolation["session_disjoint_train_test"]
    assert isolation["subject_disjoint_train_val"]
    assert isolation["subject_disjoint_train_test"]


def test_session_split_isolation(multi_subject_samples):
    config = SplitConfig(split_strategy="session")
    splitter = DatasetSplitter(config)
    train, val, test = splitter.split(multi_subject_samples)

    assert len(train) > 0
    assert len(val) > 0
    assert len(test) > 0

    isolation = verify_split_isolation(train, val, test, strategy="session")
    assert isolation["all_isolation_checks_passed"]
    assert isolation["video_disjoint_train_val"]
    assert isolation["video_disjoint_train_test"]
    assert isolation["video_disjoint_val_test"]
    assert isolation["session_disjoint_train_val"]
    assert isolation["session_disjoint_train_test"]
    assert isolation["session_disjoint_val_test"]


def test_cross_scene_split_isolation(multi_subject_samples):
    config = SplitConfig(split_strategy="cross_scene", partition_val_for_test=True)
    splitter = DatasetSplitter(config)
    train, val, test = splitter.split(multi_subject_samples)

    assert len(train) > 0
    assert len(val) > 0

    isolation = verify_split_isolation(train, val, test, strategy="cross_scene")
    assert isolation["all_isolation_checks_passed"]
    assert isolation["video_disjoint_train_val"]
    assert isolation["session_disjoint_train_val"]
    assert isolation["scene_disjoint_train_val"]


def test_cross_view_split_isolation(multi_subject_samples):
    config = SplitConfig(split_strategy="cross_view", partition_val_for_test=True)
    splitter = DatasetSplitter(config)
    train, val, test = splitter.split(multi_subject_samples)

    assert len(train) > 0
    assert len(val) > 0

    isolation = verify_split_isolation(train, val, test, strategy="cross_view")
    assert isolation["all_isolation_checks_passed"]
    assert isolation["video_disjoint_train_val"]
    assert isolation["view_disjoint_train_val"]


def test_missing_official_split_raises_error(multi_subject_samples, tmp_path):
    # Empty directory with no split files
    empty_dir = tmp_path / "empty_splits"
    empty_dir.mkdir(parents=True, exist_ok=True)

    config = SplitConfig(split_strategy="official_x_scene", split_dir=empty_dir)
    splitter = DatasetSplitter(config, dataset_root=empty_dir)

    # Must raise FileNotFoundError identifying missing files, NEVER silently fall back
    with pytest.raises(FileNotFoundError) as exc_info:
        splitter.split(multi_subject_samples)

    assert "Official MMAct split files for mode 'scene'" in str(exc_info.value)
    assert "trimmed_x-scene_train.txt" in str(exc_info.value)


def test_official_cross_scene_loading_synthetic(tmp_path):
    split_dir = tmp_path / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)

    # Write synthetic split files
    train_lines = [
        "cross_scene/trainval/subject1/cam1/scene1/session1",
        "cross_scene/trainval/subject1/cam2/scene1/session1",
        "cross_scene/trainval/subject2/cam1/scene1/session1",
    ]
    val_lines = [
        "cross_scene/trainval/subject3/cam1/scene3/session1",
        "cross_scene/trainval/subject3/cam2/scene3/session1",
        "cross_scene/trainval/subject4/cam1/scene3/session2",
    ]
    with open(split_dir / "trimmed_x-scene_train.txt", "w") as f:
        f.write("\n".join(train_lines) + "\n")
    with open(split_dir / "trimmed_x-scene_val.txt", "w") as f:
        f.write("\n".join(val_lines) + "\n")

    # Create dummy samples matching these recordings
    samples = [
        PairedSample("s1_c1_sc1_sess1_walking", "walking", 0, "subject1", "scene1", "session1", "cam1", Path("v1.mp4"), {}),
        PairedSample("s1_c2_sc1_sess1_walking", "walking", 0, "subject1", "scene1", "session1", "cam2", Path("v2.mp4"), {}),
        PairedSample("s2_c1_sc1_sess1_running", "running", 1, "subject2", "scene1", "session1", "cam1", Path("v3.mp4"), {}),
        PairedSample("s3_c1_sc3_sess1_sitting", "sitting", 2, "subject3", "scene3", "session1", "cam1", Path("v4.mp4"), {}),
        PairedSample("s3_c2_sc3_sess1_sitting", "sitting", 2, "subject3", "scene3", "session1", "cam2", Path("v5.mp4"), {}),
        PairedSample("s4_c1_sc3_sess2_standing", "standing", 3, "subject4", "scene3", "session2", "cam1", Path("v6.mp4"), {}),
        # Unmatched sample
        PairedSample("s9_c1_sc9_sess9_jumping", "jumping", 4, "subject9", "scene9", "session9", "cam1", Path("v7.mp4"), {}),
    ]

    config = SplitConfig(split_strategy="official_x_scene", split_dir=split_dir, partition_val_for_test=False)
    splitter = DatasetSplitter(config)
    train, val, test = splitter.split(samples)

    assert len(train) == 3
    assert len(val) == 3
    assert len(test) == 0  # Default: official validation remains intact

    audit = splitter.last_audit
    assert audit is not None
    assert audit.is_official
    assert not audit.is_custom_test
    assert len(audit.unmatched_indexed_samples) == 1
    assert audit.unmatched_indexed_samples[0] == "s9_c1_sc9_sess9_jumping"


def test_official_cross_view_loading_synthetic(tmp_path):
    split_dir = tmp_path / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)

    train_lines = [
        "cross_view/trainval/subject1/cam1/scene1/session1",
        "cross_view/trainval/subject1/cam2/scene1/session1",
    ]
    val_lines = [
        "cross_view/trainval/subject1/cam3/scene1/session1",
    ]
    with open(split_dir / "trimmed_x-view_train.txt", "w") as f:
        f.write("\n".join(train_lines) + "\n")
    with open(split_dir / "trimmed_x-view_val.txt", "w") as f:
        f.write("\n".join(val_lines) + "\n")

    samples = [
        PairedSample("s1_c1_sc1_sess1_walking", "walking", 0, "subject1", "scene1", "session1", "cam1", Path("v1.mp4"), {}),
        PairedSample("s1_c2_sc1_sess1_walking", "walking", 0, "subject1", "scene1", "session1", "cam2", Path("v2.mp4"), {}),
        PairedSample("s1_c3_sc1_sess1_walking", "walking", 0, "subject1", "scene1", "session1", "cam3", Path("v3.mp4"), {}),
    ]

    config = SplitConfig(split_strategy="official_x_view", split_dir=split_dir)
    splitter = DatasetSplitter(config)
    train, val, test = splitter.split(samples)

    assert len(train) == 2
    assert len(val) == 1
    assert len(test) == 0
    assert {s.cam for s in train} == {"cam1", "cam2"}
    assert {s.cam for s in val} == {"cam3"}


def test_train_val_split_overlap_detected(tmp_path):
    split_dir = tmp_path / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)

    # Overlapping recording: subject1/cam1/scene1/session1
    train_lines = ["cross_scene/trainval/subject1/cam1/scene1/session1"]
    val_lines = ["cross_scene/trainval/subject1/cam1/scene1/session1"]

    with open(split_dir / "trimmed_x-scene_train.txt", "w") as f:
        f.write("\n".join(train_lines) + "\n")
    with open(split_dir / "trimmed_x-scene_val.txt", "w") as f:
        f.write("\n".join(val_lines) + "\n")

    samples = [PairedSample("s1", "walking", 0, "subject1", "scene1", "session1", "cam1", Path("v.mp4"), {})]
    config = SplitConfig(split_strategy="official_x_scene", split_dir=split_dir)
    splitter = DatasetSplitter(config)

    with pytest.raises(ValueError) as exc_info:
        splitter.split(samples)
    assert "overlapping recording IDs" in str(exc_info.value)


def test_duplicate_split_entries_detected(tmp_path):
    split_dir = tmp_path / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)

    train_lines = [
        "cross_scene/trainval/subject1/cam1/scene1/session1",
        "cross_scene/trainval/subject1/cam1/scene1/session1",  # duplicate
    ]
    val_lines = ["cross_scene/trainval/subject2/cam1/scene3/session1"]

    with open(split_dir / "trimmed_x-scene_train.txt", "w") as f:
        f.write("\n".join(train_lines) + "\n")
    with open(split_dir / "trimmed_x-scene_val.txt", "w") as f:
        f.write("\n".join(val_lines) + "\n")

    samples = [
        PairedSample("s1", "walking", 0, "subject1", "scene1", "session1", "cam1", Path("v1.mp4"), {}),
        PairedSample("s2", "walking", 0, "subject2", "scene3", "session1", "cam1", Path("v2.mp4"), {}),
    ]
    config = SplitConfig(split_strategy="official_x_scene", split_dir=split_dir)
    splitter = DatasetSplitter(config)
    splitter.split(samples)

    audit = splitter.last_audit
    assert audit is not None
    assert len(audit.duplicate_train_entries) == 1
    assert "subject1/cam1/scene1/session1" in audit.duplicate_train_entries[0]


def test_no_recording_group_divided_across_val_and_test(tmp_path):
    split_dir = tmp_path / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)

    train_lines = ["cross_scene/trainval/subject1/cam1/scene1/session1"]
    val_lines = [
        "cross_scene/trainval/subject2/cam1/scene3/session1",
        "cross_scene/trainval/subject2/cam2/scene3/session1",  # same session, diff cam
        "cross_scene/trainval/subject3/cam1/scene3/session2",
        "cross_scene/trainval/subject3/cam2/scene3/session2",  # same session, diff cam
    ]
    with open(split_dir / "trimmed_x-scene_train.txt", "w") as f:
        f.write("\n".join(train_lines) + "\n")
    with open(split_dir / "trimmed_x-scene_val.txt", "w") as f:
        f.write("\n".join(val_lines) + "\n")

    samples = [
        PairedSample("s1_c1", "walking", 0, "subject1", "scene1", "session1", "cam1", Path("v1.mp4"), {}),
        PairedSample("s2_c1", "walking", 0, "subject2", "scene3", "session1", "cam1", Path("v2.mp4"), {}),
        PairedSample("s2_c2", "walking", 0, "subject2", "scene3", "session1", "cam2", Path("v3.mp4"), {}),
        PairedSample("s3_c1", "running", 1, "subject3", "scene3", "session2", "cam1", Path("v4.mp4"), {}),
        PairedSample("s3_c2", "running", 1, "subject3", "scene3", "session2", "cam2", Path("v5.mp4"), {}),
    ]

    config = SplitConfig(split_strategy="official_x_scene", split_dir=split_dir, partition_val_for_test=True, seed=42)
    splitter = DatasetSplitter(config)
    train, val, test = splitter.split(samples)

    assert len(train) == 1
    assert len(val) > 0
    assert len(test) > 0

    val_groups = {(s.subject, s.scene, s.session) for s in val}
    test_groups = {(s.subject, s.scene, s.session) for s in test}

    # Zero group overlap between val and test
    assert len(val_groups & test_groups) == 0

    # Ensure all cameras of subject2/scene3/session1 are together
    s2_partitions = set()
    if any(s.subject == "subject2" for s in val):
        s2_partitions.add("val")
    if any(s.subject == "subject2" for s in test):
        s2_partitions.add("test")
    assert len(s2_partitions) == 1, "subject2 session1 was split across val and test!"


def test_custom_test_partitioning_determinism(tmp_path):
    split_dir = tmp_path / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)

    train_lines = ["cross_scene/trainval/subject1/cam1/scene1/session1"]
    val_lines = [
        f"cross_scene/trainval/subject{i}/cam1/scene3/session{i}" for i in range(2, 6)
    ]
    with open(split_dir / "trimmed_x-scene_train.txt", "w") as f:
        f.write("\n".join(train_lines) + "\n")
    with open(split_dir / "trimmed_x-scene_val.txt", "w") as f:
        f.write("\n".join(val_lines) + "\n")

    samples = [
        PairedSample("s1", "walking", 0, "subject1", "scene1", "session1", "cam1", Path("v.mp4"), {})
    ] + [
        PairedSample(f"s{i}", "walking" if i % 2 == 0 else "running", 0, f"subject{i}", "scene3", f"session{i}", "cam1", Path(f"v{i}.mp4"), {})
        for i in range(2, 6)
    ]

    cfg1 = SplitConfig(split_strategy="official_x_scene", split_dir=split_dir, partition_val_for_test=True, seed=123)
    t1, v1, te1 = DatasetSplitter(cfg1).split(samples)

    cfg2 = SplitConfig(split_strategy="official_x_scene", split_dir=split_dir, partition_val_for_test=True, seed=123)
    t2, v2, te2 = DatasetSplitter(cfg2).split(samples)

    assert [s.sample_id for s in v1] == [s.sample_id for s in v2]
    assert [s.sample_id for s in te1] == [s.sample_id for s in te2]


def test_empty_test_isolation_representation(multi_subject_samples):
    config = SplitConfig(split_strategy="subject", partition_val_for_test=False)
    # Configure so test_subjects is empty
    config.train_subjects = ["subject1", "subject2"]
    config.val_subjects = ["subject3", "subject4"]
    config.test_subjects = []

    splitter = DatasetSplitter(config)
    train, val, test = splitter.split(multi_subject_samples)

    assert len(test) == 0
    isolation = verify_split_isolation(train, val, test, strategy="subject")

    # Empty test must NOT be reported as proven test isolation
    assert isolation["test_split_present"] is False
    assert isolation["video_disjoint_train_test"] is None
    assert isolation["session_disjoint_val_test"] is None


def test_missing_sensor_modality_detected_individually():
    sample_bad = PairedSample(
        sample_id="test_bad",
        action="walking",
        label_idx=0,
        subject="subject1",
        scene="scene1",
        session="session1",
        cam="cam1",
        video_path=Path("fake_video.mp4"),
        sensor_paths={
            "acc_phone": Path("phone.csv"),
            # missing acc_watch and gyro
        },
    )

    audit = verify_split_representation(
        train=[sample_bad],
        val=[],
        test=[],
        required_sensor_modalities=["acc_phone", "acc_watch", "gyro"],
        check_disk_files=False,
        enforce=False,
    )

    assert not audit["all_modalities_verified"]
    missing_mods = [item[2] for item in audit["missing_modality_incidents"]]
    assert "acc_watch" in missing_mods
    assert "gyro" in missing_mods
    assert "acc_phone" not in missing_mods


def test_unknown_and_missing_labels_reported():
    sample_unknown = PairedSample("s_unk", "flying", 99, "subject1", "scene1", "session1", "cam1", Path("v.mp4"), {"acc_phone": Path("p.csv"), "acc_watch": Path("w.csv"), "gyro": Path("g.csv")})
    sample_missing = PairedSample("s_mis", "", -1, "subject1", "scene1", "session1", "cam1", Path("v.mp4"), {"acc_phone": Path("p.csv"), "acc_watch": Path("w.csv"), "gyro": Path("g.csv")})

    audit = verify_split_representation(
        train=[sample_unknown],
        val=[sample_missing],
        test=[],
        expected_classes=VERIFIED_TARGET_CLASSES,
        check_disk_files=False,
        enforce=False,
    )

    assert len(audit["unknown_labels"]) == 1
    assert audit["unknown_labels"][0][2] == "flying"
    assert len(audit["missing_labels"]) == 1


def test_missing_target_classes_reported():
    # Only walking present
    sample_walking = PairedSample("s_w", "walking", 8, "subject1", "scene1", "session1", "cam1", Path("v.mp4"), {"acc_phone": Path("p.csv"), "acc_watch": Path("w.csv"), "gyro": Path("g.csv")})

    audit = verify_split_representation(
        train=[sample_walking],
        val=[sample_walking],
        test=[],
        expected_classes=VERIFIED_TARGET_CLASSES,
        check_disk_files=False,
        enforce=False,
    )

    assert not audit["all_classes_present_train"]
    assert "jumping" in audit["missing_classes_train"]
    assert "running" in audit["missing_classes_train"]


def test_representation_enforce_raises_error():
    sample_bad = PairedSample("s_bad", "jumping", 1, "subject1", "scene1", "session1", "cam1", Path("v.mp4"), {})

    with pytest.raises(ValueError) as exc_info:
        verify_split_representation(
            train=[sample_bad],
            val=[sample_bad],
            test=[],
            required_sensor_modalities=["acc_phone", "acc_watch", "gyro"],
            check_disk_files=False,
            enforce=True,
        )
    assert "Split representation verification failed" in str(exc_info.value)
    assert "Missing required sensor modalities" in str(exc_info.value)


def test_official_split_membership_and_isolation():
    genuine_root = Path("data/genuine_mmact")
    split_dir = Path("data/genuine_mmact/trimmed/splits")
    if not genuine_root.exists() or not (split_dir / "trimmed_x-scene_train.txt").exists():
        pytest.skip("Genuine data or official split files not present locally.")

    indexer = MMActIndexer(dataset_root=genuine_root)
    rep = indexer.index()
    assert rep.paired_samples_count > 0

    config = SplitConfig(split_strategy="official_x_scene", partition_val_for_test=True, split_dir=split_dir)
    splitter = DatasetSplitter(config, dataset_root=genuine_root)
    train, val, test = splitter.split(rep.paired_samples)

    assert len(train) > 0
    assert len(val) > 0
    assert len(test) > 0

    isolation = verify_split_isolation(train, val, test, strategy="official_x_scene")
    assert isolation["all_isolation_checks_passed"]
    assert isolation["video_disjoint_train_val"]
    assert isolation["video_disjoint_train_test"]
    assert isolation["video_disjoint_val_test"]
    assert isolation["session_disjoint_train_val"]
    assert isolation["session_disjoint_train_test"]
    assert isolation["session_disjoint_val_test"]

    with open(split_dir / "trimmed_x-scene_train.txt") as f:
        train_records = set(line.strip() for line in f if line.strip())
    with open(split_dir / "trimmed_x-scene_val.txt") as f:
        val_records = set(line.strip() for line in f if line.strip())

    for s in train:
        rec = f"cross_scene/trainval/{s.subject}/{s.cam}/{s.scene}/{s.session}"
        assert rec in train_records, f"Sample {rec} not in official train records!"

    for s in val + test:
        rec = f"cross_scene/trainval/{s.subject}/{s.cam}/{s.scene}/{s.session}"
        assert rec in val_records, f"Sample {rec} not in official val records!"

    # Unpartitioned official split
    config_off = SplitConfig(split_strategy="official_x_scene", partition_val_for_test=False, split_dir=split_dir)
    splitter_off = DatasetSplitter(config_off, dataset_root=genuine_root)
    train_off, val_off, test_off = splitter_off.split(rep.paired_samples)
    assert len(train_off) == len(train)
    assert len(val_off) == len(val) + len(test)
    assert len(test_off) == 0
