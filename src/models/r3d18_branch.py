"""Kinetics-400 pretrained R3D-18 video feature branch for MMAct."""

from typing import Optional

import torch
import torch.nn as nn


# torchvision R3D_18_Weights.KINETICS400_V1 preprocessing statistics.
KINETICS400_MEAN = (0.43216, 0.394666, 0.37645)
KINETICS400_STD = (0.22803, 0.22145, 0.216989)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
R3D18_WEIGHT_ID = "torchvision:R3D_18_Weights.KINETICS400_V1"


class R3D18VideoBranch(nn.Module):
    """3D ResNet-18 clip encoder. Input is (B, 3, T, H, W), ImageNet-normalized.

    VideoLoader normalizes frames using ImageNet statistics. The pretrained
    torchvision video weights expect Kinetics-400 statistics, so this module
    first recovers RGB values in [0, 1] and then applies those statistics.
    """

    def __init__(
        self,
        in_channels: int = 3,
        embedding_dim: int = 128,
        dropout: float = 0.35,
        pretrained: bool = True,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()
        if in_channels != 3:
            raise ValueError("R3D-18 Kinetics weights require exactly 3 RGB channels.")
        self.embedding_dim = embedding_dim
        self.freeze_backbone = freeze_backbone

        try:
            from torchvision.models.video import R3D_18_Weights, r3d_18
        except Exception as exc:
            raise RuntimeError(
                "R3D-18 requires a working torchvision video module; install a "
                "torchvision build compatible with the installed PyTorch."
            ) from exc

        weights = R3D_18_Weights.KINETICS400_V1 if pretrained else None
        try:
            self.backbone = r3d_18(weights=weights)
        except Exception as exc:
            if pretrained:
                raise RuntimeError(
                    "Could not load torchvision R3D-18 Kinetics-400 weights "
                    f"({R3D18_WEIGHT_ID}). Check internet/cache and torchvision compatibility; "
                    "no random-weight fallback was used."
                ) from exc
            raise RuntimeError("Could not construct torchvision R3D-18.") from exc

        # Keep the pretrained 3D feature extractor and discard its 400-class head.
        self.backbone.fc = nn.Identity()
        if freeze_backbone:
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False

        self.register_buffer(
            "imagenet_mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1, 1), persistent=False
        )
        self.register_buffer(
            "imagenet_std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1, 1), persistent=False
        )
        self.register_buffer(
            "kinetics_mean", torch.tensor(KINETICS400_MEAN).view(1, 3, 1, 1, 1), persistent=False
        )
        self.register_buffer(
            "kinetics_std", torch.tensor(KINETICS400_STD).view(1, 3, 1, 1, 1), persistent=False
        )
        self.projection = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(512, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
        )

    def train(self, mode: bool = True):
        super().train(mode)
        # Frozen parameters stay frozen and their batch-normalization statistics do too.
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5 or x.shape[1] != 3:
            raise ValueError(f"Expected video shape (B, 3, T, H, W), received {tuple(x.shape)}")
        x = x.float()
        rgb = (x * self.imagenet_std + self.imagenet_mean).clamp(0.0, 1.0)
        x = (rgb - self.kinetics_mean) / self.kinetics_std
        features = self.backbone(x)
        return self.projection(features)
