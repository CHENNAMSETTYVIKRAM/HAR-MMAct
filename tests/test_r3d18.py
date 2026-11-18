import torch
import pytest

from src.models.r3d18_branch import R3D18VideoBranch, R3D18_WEIGHT_ID
from src.models.fusion_model import MultimodalHARModel


def test_r3d18_constructs_with_official_kinetics_weights(monkeypatch):
    video_models = pytest.importorskip("torchvision.models.video")
    calls = {}

    class FakeWeights:
        KINETICS400_V1 = object()

    class FakeBackbone(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = torch.nn.Linear(512, 400)
            self.marker = torch.nn.Parameter(torch.tensor([7.0]))

        def forward(self, x):
            return x.new_zeros((x.shape[0], 512)) + self.marker

    def fake_r3d_18(*, weights):
        calls["weights"] = weights
        return FakeBackbone()

    monkeypatch.setattr(video_models, "R3D_18_Weights", FakeWeights)
    monkeypatch.setattr(video_models, "r3d_18", fake_r3d_18)
    branch = R3D18VideoBranch(pretrained=True)
    assert calls["weights"] is FakeWeights.KINETICS400_V1
    assert branch.backbone.fc.__class__ is torch.nn.Identity
    assert branch.backbone.marker.item() == 7.0
    assert R3D18_WEIGHT_ID == "torchvision:R3D_18_Weights.KINETICS400_V1"


def test_r3d18_fusion_model_video_only_logits_shape(monkeypatch):
    video_models = pytest.importorskip("torchvision.models.video")

    def fake_r3d_18(*, weights):
        class TinyBackbone(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = torch.nn.Linear(512, 400)

            def forward(self, x):
                return x.mean(dim=(2, 3, 4)).mean(dim=1, keepdim=True).expand(-1, 512)
        return TinyBackbone()

    monkeypatch.setattr(video_models, "r3d_18", fake_r3d_18)
    model = MultimodalHARModel(mode="video_only", video_backbone="r3d18", pretrained_video=True)
    model.eval()
    with torch.no_grad():
        logits = model(video=torch.randn(2, 3, 4, 32, 32))
    assert logits.shape == (2, 10)


def test_r3d18_real_forward_16_frames():
    pytest.importorskip("torchvision.models.video")
    model = MultimodalHARModel(mode="video_only", video_backbone="r3d18", pretrained_video=False)
    model.eval()
    with torch.no_grad():
        logits = model(video=torch.randn(1, 3, 16, 112, 112), override_mode="video_only")
    assert logits.shape == (1, 10)


def test_r3d18_frozen_backbone_keeps_trainable_projection():
    video_models = pytest.importorskip("torchvision.models.video")
    monkeypatch = pytest.MonkeyPatch()

    class FakeWeights:
        KINETICS400_V1 = object()

    monkeypatch.setattr(video_models, "R3D_18_Weights", FakeWeights)
    branch = R3D18VideoBranch(pretrained=False, freeze_backbone=True)
    assert all(not parameter.requires_grad for parameter in branch.backbone.parameters())
    assert any(parameter.requires_grad for parameter in branch.projection.parameters())
    branch.train()
    assert not branch.backbone.training
    monkeypatch.undo()
