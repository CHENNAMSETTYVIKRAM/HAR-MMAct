"""
test_models.py - Tests for VideoBranch, SensorBranch, and MultimodalHARModel
"""

import os
import torch
import pytest

from src.models.video_branch import LightweightVideoBranch, ResNet18VideoBranch
from src.models.sensor_branch import Sensor1DCNNBranch, InceptionTimeSensorBranch
from src.models.fusion_model import MultimodalHARModel


def test_lightweight_video_branch_forward():
    branch = LightweightVideoBranch(in_channels=3, embedding_dim=128)
    dummy_video = torch.randn(2, 3, 8, 112, 112)
    out = branch(dummy_video)
    assert out.shape == (2, 128)
    assert not torch.isnan(out).any()


def test_resnet18_video_branch_forward():
    pretrained_exists = os.path.exists("outputs/pretrained/resnet18.pth")
    branch = ResNet18VideoBranch(
        in_channels=3,
        embedding_dim=128,
        pretrained=pretrained_exists,
        pretrained_path="outputs/pretrained/resnet18.pth",
    )
    dummy_video = torch.randn(2, 3, 4, 112, 112)
    out = branch(dummy_video)
    assert out.shape == (2, 128)
    assert not torch.isnan(out).any()


def test_sensor1d_cnn_branch_forward():
    branch = Sensor1DCNNBranch(in_channels=9, embedding_dim=128)
    dummy_sensor = torch.randn(2, 9, 100)
    out = branch(dummy_sensor)
    assert out.shape == (2, 128)
    assert not torch.isnan(out).any()


def test_inception_time_sensor_branch_forward():
    branch = InceptionTimeSensorBranch(in_channels=9, embedding_dim=128)
    dummy_sensor = torch.randn(2, 9, 100)
    out = branch(dummy_sensor)
    assert out.shape == (2, 128)
    assert not torch.isnan(out).any()


def test_multimodal_model_forward():
    model = MultimodalHARModel(
        num_classes=10,
        video_channels=3,
        sensor_channels=9,
        video_backbone="lightweight",
        sensor_backbone="cnn1d",
    )
    dummy_video = torch.randn(2, 3, 8, 112, 112)
    dummy_sensor = torch.randn(2, 9, 100)

    # 1. Multimodal mode
    logits_mm = model(video=dummy_video, sensor=dummy_sensor)
    assert logits_mm.shape == (2, 10)

    # 2. Video-only mode
    logits_vo = model(video=dummy_video, override_mode="video_only")
    assert logits_vo.shape == (2, 10)

    # 3. Sensor-only mode
    logits_so = model(sensor=dummy_sensor, override_mode="sensor_only")
    assert logits_so.shape == (2, 10)


def test_multimodal_model_resnet_inception():
    model = MultimodalHARModel(
        num_classes=10,
        video_channels=3,
        sensor_channels=9,
        video_backbone="resnet18",
        sensor_backbone="inception_time",
    )
    dummy_video = torch.randn(2, 3, 4, 112, 112)
    dummy_sensor = torch.randn(2, 9, 100)

    # Test forward and predict methods
    logits_mm = model(video=dummy_video, sensor=dummy_sensor)
    assert logits_mm.shape == (2, 10)

    p_fused = model.predict_fused(dummy_video, dummy_sensor)
    assert p_fused.shape == (2, 10)
    assert torch.allclose(p_fused.sum(dim=-1), torch.ones(2), atol=1e-5)

    p_vo = model.predict_video_only(dummy_video)
    assert p_vo.shape == (2, 10)
    assert torch.allclose(p_vo.sum(dim=-1), torch.ones(2), atol=1e-5)

    p_so = model.predict_sensor_only(dummy_sensor)
    assert p_so.shape == (2, 10)
    assert torch.allclose(p_so.sum(dim=-1), torch.ones(2), atol=1e-5)
