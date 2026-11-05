"""
Tests for MMActMultimodalDataset deterministic caching, atomic writes,
corruption recovery, and cache warm-up.
"""

import shutil
import tempfile
from pathlib import Path
import pytest
import torch
import numpy as np

from src.data.dataset import MMActMultimodalDataset
from src.data.indexer import PairedSample
from src.config import VideoConfig, SensorConfig


@pytest.fixture
def temp_cache_dir():
    d = tempfile.mkdtemp(prefix="test_cache_")
    yield Path(d)
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def dummy_sample(tmp_path):
    # Create fake video file
    video_path = tmp_path / "dummy_video.mp4"
    video_path.touch()

    # Create fake sensor files
    phone_path = tmp_path / "dummy_phone.csv"
    phone_path.write_text("timestamp,acc_x,acc_y,acc_z\n0,0.1,0.2,0.3\n1,0.1,0.2,0.3\n")

    watch_path = tmp_path / "dummy_watch.csv"
    watch_path.write_text("timestamp,acc_x,acc_y,acc_z\n0,0.1,0.2,0.3\n1,0.1,0.2,0.3\n")

    gyro_path = tmp_path / "dummy_gyro.csv"
    gyro_path.write_text("timestamp,gyro_x,gyro_y,gyro_z\n0,0.01,0.02,0.03\n1,0.01,0.02,0.03\n")

    return PairedSample(
        sample_id="test_sub01_scene1_s01_walking",
        action="walking",
        label_idx=0,
        subject="subject1",
        scene="scene1",
        session="session1",
        cam="cam1",
        video_path=video_path,
        sensor_paths={
            "acc_phone": phone_path,
            "acc_watch": watch_path,
            "gyro": gyro_path,
        },
    )


class MockVideoLoader:
    def __init__(self, shape=(3, 16, 112, 112)):
        self.shape = shape
        self.calls = 0
        self.config = VideoConfig()

    def load_and_preprocess(self, path):
        self.calls += 1
        return torch.zeros(self.shape, dtype=torch.float32)


class MockSensorLoader:
    def __init__(self, shape=(9, 100)):
        self.shape = shape
        self.calls = 0
        self.config = SensorConfig()

    def load_and_preprocess(self, paths):
        self.calls += 1
        return torch.zeros(self.shape, dtype=torch.float32)


def test_cache_miss_then_hit(temp_cache_dir, dummy_sample):
    mock_v = MockVideoLoader()
    mock_s = MockSensorLoader()
    ds = MMActMultimodalDataset(
        samples=[dummy_sample],
        video_loader=mock_v,
        sensor_loader=mock_s,
        cache_dir=temp_cache_dir,
        enable_cache=True,
    )

    # First access: cache miss -> loads via loaders -> writes to cache
    v1, s1, l1, id1 = ds[0]
    assert mock_v.calls == 1
    assert mock_s.calls == 1
    assert ds.cache_misses == 1
    assert ds.cache_writes == 1
    assert ds.cache_hits == 0

    # Second access: cache hit -> directly loaded from cache without calling loaders
    v2, s2, l2, id2 = ds[0]
    assert mock_v.calls == 1  # Loader not called again
    assert mock_s.calls == 1
    assert ds.cache_hits == 1
    assert torch.equal(v1, v2)
    assert torch.equal(s1, s2)
    assert l1 == l2
    assert id1 == id2


def test_cache_config_invalidation(temp_cache_dir, dummy_sample):
    # Test that different video or sensor configs yield different cache paths
    from src.data.video_loader import VideoLoader
    cfg_a = VideoConfig(num_frames=16, frame_height=112, frame_width=112)
    cfg_b = VideoConfig(num_frames=8, frame_height=112, frame_width=112)

    ds_a = MMActMultimodalDataset([dummy_sample], video_loader=VideoLoader(config=cfg_a), cache_dir=temp_cache_dir)
    ds_b = MMActMultimodalDataset([dummy_sample], video_loader=VideoLoader(config=cfg_b), cache_dir=temp_cache_dir)

    path_a = ds_a._get_cache_path(dummy_sample)
    path_b = ds_b._get_cache_path(dummy_sample)

    assert path_a != path_b
    assert path_a.name != path_b.name


def test_corruption_recovery(temp_cache_dir, dummy_sample):
    mock_v = MockVideoLoader()
    mock_s = MockSensorLoader()
    ds = MMActMultimodalDataset(
        samples=[dummy_sample],
        video_loader=mock_v,
        sensor_loader=mock_s,
        cache_dir=temp_cache_dir,
        enable_cache=True,
    )

    # Generate initial valid cache
    _ = ds[0]
    cache_path = ds._get_cache_path(dummy_sample)
    assert cache_path.exists()

    # Corrupt the cache file (write invalid non-torch bytes)
    cache_path.write_bytes(b"corrupted binary data here")

    # Access sample: should detect corruption, remove bad file, increment corruptions, and recompute
    v, s, _, _ = ds[0]
    assert ds.cache_corruptions >= 1
    assert mock_v.calls == 2  # Recomputed
    assert v.shape == (3, 16, 112, 112)


def test_nan_inf_corruption_validation(temp_cache_dir, dummy_sample):
    mock_v = MockVideoLoader()
    mock_s = MockSensorLoader()
    ds = MMActMultimodalDataset(
        samples=[dummy_sample],
        video_loader=mock_v,
        sensor_loader=mock_s,
        cache_dir=temp_cache_dir,
        enable_cache=True,
    )
    # Cache entry containing NaN
    cache_path = ds._get_cache_path(dummy_sample)
    nan_tensor = torch.full((3, 16, 112, 112), float("nan"))
    sensor_tensor = torch.zeros((9, 100))
    torch.save({"video": nan_tensor, "sensor": sensor_tensor}, cache_path)

    # Accessing should detect NaN corruption and recompute with valid data
    v, s, _, _ = ds[0]
    assert not torch.isnan(v).any()
    assert ds.cache_corruptions >= 1
    assert mock_v.calls == 1


def test_warm_up_cache(temp_cache_dir, dummy_sample):
    mock_v = MockVideoLoader()
    mock_s = MockSensorLoader()
    ds = MMActMultimodalDataset(
        samples=[dummy_sample],
        video_loader=mock_v,
        sensor_loader=mock_s,
        cache_dir=temp_cache_dir,
        enable_cache=True,
    )

    stats = ds.warm_up_cache(verbose=False)
    assert stats["writes"] == 1
    assert stats["hits"] == 0

    # Second warm-up call should register hits
    stats2 = ds.warm_up_cache(verbose=False)
    assert stats2["hits"] >= 1
