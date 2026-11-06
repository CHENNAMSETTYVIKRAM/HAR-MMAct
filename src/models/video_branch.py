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


class InvertedResidual(nn.Module):
    def __init__(self, inp: int, oup: int, stride: int, expand_ratio: int):
        super().__init__()
        self.stride = stride
        assert stride in [1, 2]
        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
            ])
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_res_connect:
            return x + self.conv(x)
        return self.conv(x)


class MobileNetV2Backbone(nn.Module):
    """
    MobileNetV2 feature extractor implemented in pure PyTorch.
    Compatible with official torchvision checkpoints without external dependency.
    Extracts 1280-dim feature representation per frame.
    """

    def __init__(self, in_channels: int = 3, width_mult: float = 1.0):
        super().__init__()
        input_channel = int(32 * width_mult)
        last_channel = int(1280 * max(1.0, width_mult))

        interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        features = [
            nn.Conv2d(in_channels, input_channel, 3, 2, 1, bias=False),
            nn.BatchNorm2d(input_channel),
            nn.ReLU6(inplace=True),
        ]

        for t, c, n, s in interverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(InvertedResidual(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel

        features.extend([
            nn.Conv2d(input_channel, last_channel, 1, 1, 0, bias=False),
            nn.BatchNorm2d(last_channel),
            nn.ReLU6(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        ])
        self.features = nn.Sequential(*features)
        self.last_channel = last_channel

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return x.view(x.size(0), -1)


class MobileNetV2VideoBranch(nn.Module):
    """
    MobileNetV2 Video Feature Extractor for Human Activity Recognition clips (B, C, T, H, W).
    Applies transfer learning with documented pre-trained weights strategy:
    Loads pre-trained ImageNet weights when available (via torchvision or pre-downloaded checkpoint),
    pools 1280-dim frame representations temporally (mean + max pooling -> 2560 dims),
    and projects into embedding_dim with dropout regularization.
    """

    def __init__(
        self,
        in_channels: int = 3,
        embedding_dim: int = 128,
        dropout: float = 0.3,
        pretrained: bool = True,
        pretrained_path: Optional[str] = None,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.last_channel = 1280

        loaded_torchvision = False
        if pretrained_path is None:
            try:
                import torchvision.models as models
                if pretrained:
                    try:
                        from torchvision.models import MobileNet_V2_Weights
                        tv_mobilenet = models.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
                    except Exception:
                        tv_mobilenet = models.mobilenet_v2(pretrained=True)
                    print("[MobileNetV2VideoBranch] Initialized with official pretrained ImageNet weights via torchvision.")
                else:
                    try:
                        tv_mobilenet = models.mobilenet_v2(weights=None)
                    except Exception:
                        tv_mobilenet = models.mobilenet_v2(pretrained=False)
                    print("[MobileNetV2VideoBranch] Initialized unweighted torchvision MobileNetV2 architecture.")

                self.backbone = tv_mobilenet.features
                self.backbone_pool = nn.AdaptiveAvgPool2d((1, 1))
                loaded_torchvision = True
            except Exception as e:
                print(f"[MobileNetV2VideoBranch] torchvision load fallback ({e}). Using pure PyTorch MobileNetV2.")

        if not loaded_torchvision:
            self.backbone = MobileNetV2Backbone(in_channels=in_channels)
            self.last_channel = self.backbone.last_channel
            if pretrained_path and os.path.exists(pretrained_path):
                state_dict = torch.load(pretrained_path, map_location="cpu")
                self.backbone.load_state_dict(state_dict, strict=False)
                print(f"[MobileNetV2VideoBranch] Loaded weights from {pretrained_path}")

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
            print("[MobileNetV2VideoBranch] Feature extractor frozen for fine-tuning top layers.")

        # Temporal pooling (Mean + Max): 1280 * 2 = 2560 -> embedding_dim
        self.temporal_fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.last_channel * 2, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, t, h, w = x.shape
        x_reshuffled = x.permute(0, 2, 1, 3, 4).contiguous().view(b * t, c, h, w)
        if hasattr(self, "backbone_pool"):
            feats = self.backbone(x_reshuffled)
            frame_feats = self.backbone_pool(feats).view(b * t, self.last_channel)
        else:
            frame_feats = self.backbone.forward_features(x_reshuffled)

        feats_time = frame_feats.view(b, t, self.last_channel)
        mean_pooled = torch.mean(feats_time, dim=1)
        max_pooled, _ = torch.max(feats_time, dim=1)
        concat_pooled = torch.cat([mean_pooled, max_pooled], dim=1)
        return self.temporal_fc(concat_pooled)


class OriginalCNNVideoBranch(nn.Module):
    """
    Original unregularized Custom CNN baseline for MMAct video clips (B, C, T, H, W).
    Represents the initial student model before overfitting mitigations:
    - Standard 2D Conv layers without SpatialDropout
    - Standard BatchNorm2d
    - Lower dropout (0.2) on temporal projection
    """

    def __init__(
        self,
        in_channels: int = 3,
        embedding_dim: int = 128,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim

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


class CustomCNNVideoBranch(nn.Module):
    """
    Improved Custom CNN Video Branch for Human Activity Recognition clips (B, C, T, H, W).
    Engineered with explicit anti-overfitting regularization:
    - Batch Normalization after every convolution (bias=False)
    - Spatial Dropout2d between conv stages (0.15, 0.20)
    - Global average pooling per frame (bottlenecked to 128 dims)
    - Temporal Mean + Max pooling across frames
    - Higher dropout (0.35-0.5) on temporal projection to address overfitting
    """

    def __init__(
        self,
        in_channels: int = 3,
        embedding_dim: int = 128,
        dropout: float = 0.4,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim

        # Frame-level 2D feature extractor with spatial dropout
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.15),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.20),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
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


# Alias for backward compatibility
LightweightVideoBranch = OriginalCNNVideoBranch
