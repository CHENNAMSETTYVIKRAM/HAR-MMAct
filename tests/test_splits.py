"""
test_splits.py - Tests for subject-independent and session-independent dataset splitting
"""

from pathlib import Path
import pytest

from src.config import SplitConfig
from src.data.indexer import MMActIndexer
from src.data.splits import DatasetSplitter, verify_split_isolation
from src.data.synthetic import generate_synthetic_fixture


@pytest.fixture(scope="module")
def multi_subject_samples(tmp_path_factory):
    tmp_dir = tmp_path_factory.mktemp("splits_fixture")
    generate_synthetic_fixture(tmp_dir, num_subjects=4, num_cams=2, include_test_anomalies=False)
    indexer = MMActIndexer(dataset_root=tmp_dir)
    report = indexer.index()
    return report.paired_samples


def test_subject_split_isolation(multi_subject_samples):
    config = SplitConfig(split_strategy="subject")
    splitter = DatasetSplitter(config)
    train, val, test = splitter.split(multi_subject_samples)

    assert len(train) > 0
    assert len(val) > 0
    assert len(test) > 0
    assert len(train) + len(val) + len(test) == len(multi_subject_samples)

    isolation = verify_split_isolation(train, val, test, strategy="subject")
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
    assert isolation["video_disjoint_train_val"]
    assert isolation["video_disjoint_train_test"]
    assert isolation["session_disjoint_train_val"]
    assert isolation["session_disjoint_train_test"]
