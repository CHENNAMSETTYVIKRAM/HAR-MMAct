"""
video_branch.py - Video Feature Extractors (Pretrained ResNet-18 & Lightweight CNN) for HAR_MMAct
"""

import os
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


def conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class ResNet18Backbone(nn.Module):
    """
    Standard torchvision-compatible ResNet-18 architecture implemented in pure PyTorch.
    Can directly load official ImageNet weights from resnet18.pth without torchvision dependencies.
    """

    def __init__(self, in_channels: int = 3):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, 1000)

    def _make_layer(self, planes: int, blocks: int, stride: int = 1) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes, stride),
                nn.BatchNorm2d(planes),
            )

        layers = []
        layers.append(BasicBlock(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(BasicBlock(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract 512-dim feature vector per image"""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        return torch.flatten(x, 1)


class ResNet18VideoBranch(nn.Module):
    """
    ResNet-18 video feature extractor with temporal pooling.
    Input clip shape: (B, C, T, H, W).
    """

    def __init__(
        self,
        in_channels: int = 3,
        embedding_dim: int = 128,
        dropout: float = 0.2,
        pretrained: bool = True,
        pretrained_path: str = "outputs/pretrained/resnet18.pth",
        freeze_backbone: bool = False,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.backbone = ResNet18Backbone(in_channels=in_channels)

        if pretrained and os.path.exists(pretrained_path):
            state_dict = torch.load(pretrained_path, map_location="cpu")
            self.backbone.load_state_dict(state_dict, strict=True)
            print(f"[ResNet18VideoBranch] Successfully loaded pretrained weights from {pretrained_path}")
        elif pretrained:
            print(f"[ResNet18VideoBranch] Pretrained checkpoint not found at {pretrained_path}. Using random init.")

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Temporal aggregation: Mean + Max pooling of (B, T, 512) -> 1024 dims
        self.temporal_fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(512 * 2, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input:  x of shape (B, C, T, H, W)
        Output: embedding of shape (B, embedding_dim)
        """
        b, c, t, h, w = x.shape

        # Reshape to (B * T, C, H, W)
        x_reshuffled = x.permute(0, 2, 1, 3, 4).contiguous().view(b * t, c, h, w)

        # Extract frame features: (B * T, 512)
        frame_feats = self.backbone.forward_features(x_reshuffled)

        # Reshape to (B, T, 512)
        feats_time = frame_feats.view(b, t, 512)

        # Temporal pooling (Mean + Max)
        mean_pooled = torch.mean(feats_time, dim=1)  # (B, 512)
        max_pooled, _ = torch.max(feats_time, dim=1)  # (B, 512)
        concat_pooled = torch.cat([mean_pooled, max_pooled], dim=1)  # (B, 1024)

        return self.temporal_fc(concat_pooled)


class LightweightVideoBranch(nn.Module):
    """
    Lightweight video feature extractor for clip tensors of shape (B, C, T, H, W).
    Processes each frame with shared 2D convolutional blocks, then applies temporal
    pooling (mean + max) across time steps to produce a fixed-size embedding.
    """

    def __init__(
        self,
        in_channels: int = 3,
        embedding_dim: int = 128,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim

        # Frame-level 2D feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.temporal_fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(128 * 2, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, t, h, w = x.shape
        x_reshuffled = x.permute(0, 2, 1, 3, 4).contiguous().view(b * t, c, h, w)
        frame_feats = self.features(x_reshuffled).view(b * t, 128)
        feats_time = frame_feats.view(b, t, 128)

        mean_pooled = torch.mean(feats_time, dim=1)
        max_pooled, _ = torch.max(feats_time, dim=1)
        concat_pooled = torch.cat([mean_pooled, max_pooled], dim=1)
        return self.temporal_fc(concat_pooled)
