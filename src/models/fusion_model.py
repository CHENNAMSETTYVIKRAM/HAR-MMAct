"""
fusion_model.py - Multimodal Feature Fusion and Unimodal Baselines for HAR_MMAct
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.video_branch import (
    ResNet18VideoBranch,
    LightweightVideoBranch,
    CustomCNNVideoBranch,
    MobileNetV2VideoBranch,
)
from src.models.sensor_branch import InceptionTimeSensorBranch, Sensor1DCNNBranch


class MultimodalHARModel(nn.Module):
    """
    Multimodal Human Activity Recognition model fusing video and sensor streams.
    Supports pretrained ResNet-18, MobileNetV2, or Custom CNN for video, and InceptionTime
    or 1D-CNN for sensor time series. Also natively supports unimodal evaluation modes.
    """

    def __init__(
        self,
        num_classes: int = 10,
        video_channels: int = 3,
        sensor_channels: int = 9,
        video_embed_dim: int = 128,
        sensor_embed_dim: int = 128,
        fusion_hidden_dim: int = 128,
        dropout: float = 0.3,
        mode: str = "multimodal",  # 'multimodal', 'video_only', 'sensor_only'
        video_backbone: str = "resnet18",  # 'resnet18', 'mobilenet_v2', 'cnn', 'custom_cnn'
        sensor_backbone: str = "inception_time",  # 'inception_time' or 'cnn1d'
        pretrained_video: bool = True,
        pretrained_path: str = "outputs/pretrained/resnet18.pth",
        freeze_video_backbone: bool = False,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.mode = mode
        self.video_backbone_name = video_backbone
        self.sensor_backbone_name = sensor_backbone

        # Video branch initialization
        if video_backbone in ["mobilenet_v2", "mobilenet"]:
            self.video_branch = MobileNetV2VideoBranch(
                in_channels=video_channels,
                embedding_dim=video_embed_dim,
                dropout=dropout,
                pretrained=pretrained_video,
                pretrained_path=pretrained_path if ("mobilenet" in pretrained_path.lower()) else None,
                freeze_backbone=freeze_video_backbone,
            )
        elif video_backbone == "resnet18":
            self.video_branch = ResNet18VideoBranch(
                in_channels=video_channels,
                embedding_dim=video_embed_dim,
                dropout=dropout,
                pretrained=pretrained_video,
                pretrained_path=pretrained_path,
                freeze_backbone=freeze_video_backbone,
            )
        else:  # 'cnn', 'custom_cnn', 'lightweight'
            self.video_branch = CustomCNNVideoBranch(
                in_channels=video_channels,
                embedding_dim=video_embed_dim,
                dropout=max(0.35, dropout),
            )

        # Sensor branch initialization
        if sensor_backbone == "inception_time":
            self.sensor_branch = InceptionTimeSensorBranch(
                in_channels=sensor_channels,
                embedding_dim=sensor_embed_dim,
                dropout=dropout,
            )
        else:
            self.sensor_branch = Sensor1DCNNBranch(
                in_channels=sensor_channels,
                embedding_dim=sensor_embed_dim,
                dropout=dropout,
            )

        # Multimodal fusion classifier
        total_embed_dim = video_embed_dim + sensor_embed_dim
        self.fusion_classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(total_embed_dim, fusion_hidden_dim),
            nn.BatchNorm1d(fusion_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_dim, num_classes),
        )

        # Unimodal classifiers (for isolated baseline evaluation)
        self.video_only_classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(video_embed_dim, fusion_hidden_dim),
            nn.BatchNorm1d(fusion_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_dim, num_classes),
        )

        self.sensor_only_classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(sensor_embed_dim, fusion_hidden_dim),
            nn.BatchNorm1d(fusion_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_dim, num_classes),
        )

    def forward(
        self,
        video: Optional[torch.Tensor] = None,
        sensor: Optional[torch.Tensor] = None,
        override_mode: Optional[str] = None,
    ) -> torch.Tensor:
        """
        Forward pass.
        Args:
            video: shape (B, 3, T, H, W)
            sensor: shape (B, num_channels, W)
            override_mode: Optional runtime override ('multimodal', 'video_only', 'sensor_only')
        Returns:
            logits: shape (B, num_classes)
        """
        active_mode = override_mode or self.mode

        if active_mode == "video_only":
            if video is None:
                raise ValueError("Video tensor required for video_only mode.")
            v_embed = self.video_branch(video)
            return self.video_only_classifier(v_embed)

        elif active_mode == "sensor_only":
            if sensor is None:
                raise ValueError("Sensor tensor required for sensor_only mode.")
            s_embed = self.sensor_branch(sensor)
            return self.sensor_only_classifier(s_embed)

        else:  # Multimodal mode
            if video is None or sensor is None:
                raise ValueError("Both video and sensor tensors required for multimodal fusion.")
            v_embed = self.video_branch(video)
            s_embed = self.sensor_branch(sensor)
            fused = torch.cat([v_embed, s_embed], dim=1)
            return self.fusion_classifier(fused)

    @torch.no_grad()
    def predict_video_only(self, video: torch.Tensor) -> torch.Tensor:
        """Returns softmax probabilities for video stream alone."""
        self.eval()
        logits = self.forward(video=video, override_mode="video_only")
        return F.softmax(logits, dim=-1)

    @torch.no_grad()
    def predict_sensor_only(self, sensor: torch.Tensor) -> torch.Tensor:
        """Returns softmax probabilities for sensor stream alone."""
        self.eval()
        logits = self.forward(sensor=sensor, override_mode="sensor_only")
        return F.softmax(logits, dim=-1)

    @torch.no_grad()
    def predict_fused(self, video: torch.Tensor, sensor: torch.Tensor) -> torch.Tensor:
        """Returns softmax probabilities for multimodal fused stream."""
        self.eval()
        logits = self.forward(video=video, sensor=sensor, override_mode="multimodal")
        return F.softmax(logits, dim=-1)
