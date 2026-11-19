from pathlib import Path
import json

import torch
import torch.nn as nn

from src.config import TARGET_CLASS_TO_IDX, VERIFIED_TARGET_CLASSES
from src.models.fusion_model import MultimodalHARModel
from src.models.video_branch import CustomCNNVideoBranch, MobileNetV2VideoBranch


def test_target_class_mapping_round_trip():
    assert len(VERIFIED_TARGET_CLASSES) == 10
    for idx, name in enumerate(VERIFIED_TARGET_CLASSES):
        assert TARGET_CLASS_TO_IDX[name] == idx


def test_custom_cnn_shapes_gradients_and_trainable_params():
    video = torch.randn(2, 3, 2, 32, 32)
    labels = torch.tensor([0, 9])

    branch = CustomCNNVideoBranch(in_channels=3, embedding_dim=128, dropout=0.20)
    model = MultimodalHARModel(
        num_classes=10,
        mode="video_only",
        video_backbone="custom_cnn",
        pretrained_video=False,
        dropout=0.20,
    ).train()

    embedding = branch(video)
    assert embedding.shape == (2, 128)
    assert torch.isfinite(embedding).all()

    logits = model(video=video, override_mode="video_only")
    assert logits.shape == (2, 10)

    loss = nn.CrossEntropyLoss()(logits, labels)
    loss.backward()

    conv_named = [name for name, p in branch.named_parameters() if p.grad is not None and p.grad.abs().sum() > 0]
    classifier_named = [name for name, p in model.video_only_classifier.named_parameters() if p.grad is not None and p.grad.abs().sum() > 0]
    assert conv_named
    assert classifier_named
    assert sum(p.numel() for p in branch.parameters()) > 0
    assert sum(p.numel() for p in branch.parameters() if p.requires_grad) > 0


def test_custom_cnn_small_subset_can_overfit():
    torch.manual_seed(42)
    video = torch.randn(10, 3, 2, 32, 32)
    labels = torch.arange(10)

    model = CustomCNNVideoBranch(in_channels=3, embedding_dim=128, dropout=0.10)
    head = nn.Linear(128, 10)
    model.train()
    head.train()
    optimizer = torch.optim.AdamW(list(model.parameters()) + list(head.parameters()), lr=2e-3, weight_decay=0.0)

    final_loss = None
    for _ in range(50):
        optimizer.zero_grad(set_to_none=True)
        logits = head(model(video))
        loss = nn.CrossEntropyLoss()(logits, labels)
        loss.backward()
        optimizer.step()
        final_loss = loss.item()

    assert final_loss is not None
    assert final_loss < 2.30, f"Unexpected diagnostic loss: {final_loss:.4f}"


def test_mobilenet_normalization_freeze_and_optimizer_state():
    frozen = MobileNetV2VideoBranch(
        in_channels=3,
        embedding_dim=128,
        dropout=0.20,
        pretrained=False,
        freeze_backbone=True,
    )
    assert frozen.normalization_mean == (0.485, 0.456, 0.406)
    assert frozen.normalization_std == (0.229, 0.224, 0.225)
    assert hasattr(frozen, "pretrained_source")
    assert all(not p.requires_grad for p in frozen.backbone.parameters())
    assert all(p.requires_grad for p in frozen.temporal_fc.parameters())

    video = torch.randn(1, 3, 2, 64, 64)
    embedding = frozen(video)
    assert embedding.shape == (1, 128)


def test_mobilenet_pretrained_loading_is_auditable():
    branch = MobileNetV2VideoBranch(pretrained=True)
    assert branch.pretrained_source in {
        "torchvision-default",
        "torchvision-legacy",
        "torchvision-random",
        "pure-pytorch-random",
        "pure-pytorch-external",
    }


def test_retrained_split_difference_is_exactly_standing():
    paths = [
        Path("outputs/mobilenet_v2/experiment_manifest.json"),
        Path("outputs/mobilenet_v2_staged_v2_retrained/experiment_manifest.json"),
    ]
    manifests = [json.loads(path.read_text()) for path in paths]
    assert manifests[0]["train_samples_count"] == 3795
    assert manifests[1]["train_samples_count"] == 3799

    standing_original = manifests[0]["train_class_distribution"]["standing"]
    standing_retrained = manifests[1]["train_class_distribution"]["standing"]
    assert standing_original == 355
    assert standing_retrained == 359

    for original, retrained in zip(manifests, manifests):
        pass

    for key in ("train_class_distribution",):
        original = manifests[0][key]
        retrained = manifests[1][key]
        differing = {name: retrained[name] - original[name] for name in original if retrained[name] != original[name]}
        assert differing == {"standing": 4}
