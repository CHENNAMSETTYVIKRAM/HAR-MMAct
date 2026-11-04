"""
test_preprocessing.py - Tests for video frame extraction and sensor resampling/windowing
"""

from pathlib import Path
import numpy as np
import torch
import pytest

from src.config import VideoConfig, SensorConfig
from src.data.video_loader import VideoLoader
from src.data.sensor_loader import SensorLoader
from src.data.synthetic import generate_synthetic_fixture
from src.data.indexer import MMActIndexer


@pytest.fixture(scope="module")
def sample_pair(tmp_path_factory):
    tmp_dir = tmp_path_factory.mktemp("prep_fixture")
    generate_synthetic_fixture(tmp_dir, num_subjects=1, num_cams=1, include_test_anomalies=False)
    indexer = MMActIndexer(dataset_root=tmp_dir)
    report = indexer.index()
    assert report.paired_samples_count > 0
    return report.paired_samples[0]


def test_video_loader_shape_and_range(sample_pair):
    cfg = VideoConfig(num_frames=16, frame_height=112, frame_width=112)
    loader = VideoLoader(cfg)

    tensor = loader.load_and_preprocess(sample_pair.video_path)
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (3, 16, 112, 112)
    assert tensor.dtype == torch.float32
    assert not torch.isnan(tensor).any()


def test_sensor_loader_shape_and_channels(sample_pair):
    cfg = SensorConfig(
        modalities=["acc_phone", "acc_watch", "gyro"],
        window_size=100,
        convert_acc_to_g=True,
    )
    loader = SensorLoader(cfg)

    tensor = loader.load_and_preprocess(sample_pair.sensor_paths)
    assert isinstance(tensor, torch.Tensor)
    # 3 channels (phone acc) + 3 channels (watch acc) + 3 channels (gyro) = 9
    assert tensor.shape == (9, 100)
    assert tensor.dtype == torch.float32
    assert not torch.isnan(tensor).any()
