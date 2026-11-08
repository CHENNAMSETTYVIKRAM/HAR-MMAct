"""
sensor_branch.py - Multi-Channel 1D CNN and InceptionTime Sensor Feature Extractors for HAR_MMAct
"""

from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F


class InceptionModule(nn.Module):
    """
    Core Inception module for 1D time-series data (Fawaz et al., 2020),
    matching the InceptionTime architecture used in MMAct challenge sensor classification.
    """

    def __init__(
        self,
        in_channels: int,
        nb_filters: int = 32,
        kernel_sizes: List[int] = [9, 19, 39],
        bottleneck_size: int = 32,
    ):
        super().__init__()
        self.use_bottleneck = bottleneck_size > 0 and in_channels > 1
        if self.use_bottleneck:
            self.bottleneck = nn.Conv1d(in_channels, bottleneck_size, kernel_size=1, bias=False)
            conv_in = bottleneck_size
        else:
            conv_in = in_channels

        self.convs = nn.ModuleList([
            nn.Conv1d(conv_in, nb_filters, kernel_size=k, padding=k // 2, bias=False)
            for k in kernel_sizes
        ])
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        self.conv_pool = nn.Conv1d(in_channels, nb_filters, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm1d(nb_filters * (len(kernel_sizes) + 1))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_btn = self.bottleneck(x) if self.use_bottleneck else x
        out = [conv(x_btn) for conv in self.convs]
        out.append(self.conv_pool(self.maxpool(x)))
        out = torch.cat(out, dim=1)
        out = self.bn(out)
        return self.relu(out)


class InceptionBlock(nn.Module):
    """
    Residual Inception block stacking multiple Inception modules with shortcut connection.
    """

    def __init__(
        self,
        in_channels: int,
        nb_filters: int = 32,
        depth: int = 3,
    ):
        super().__init__()
        self.modules_list = nn.ModuleList()
        curr_in = in_channels
        for _ in range(depth):
            self.modules_list.append(InceptionModule(curr_in, nb_filters=nb_filters))
            curr_in = nb_filters * 4

        self.shortcut = nn.Sequential(
            nn.Conv1d(in_channels, curr_in, kernel_size=1, bias=False),
            nn.BatchNorm1d(curr_in),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.shortcut(x)
        out = x
        for mod in self.modules_list:
            out = mod(out)
        out = self.relu(out + res)
        return out


class InceptionTimeSensorBranch(nn.Module):
    """
    InceptionTime sensor feature extractor for input shape (B, channels, window_size).
    Outputs a fixed-dimension embedding (B, embedding_dim).
    """

    def __init__(
        self,
        in_channels: int = 9,
        embedding_dim: int = 128,
        nb_filters: int = 32,
        depth: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.block = InceptionBlock(in_channels, nb_filters=nb_filters, depth=depth)
        out_channels = nb_filters * 4
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(out_channels, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.block(x)
        out = self.pool(out).squeeze(-1)
        return self.fc(out)


class Sensor1DCNNBranch(nn.Module):
    """
    Standard lightweight 1D Convolutional feature extractor for multi-channel sensor windows.
    Input shape: (B, C, W).
    """

    def __init__(
        self,
        in_channels: int = 9,
        embedding_dim: int = 128,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim

        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )

        self.conv3 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )

        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(128, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = out.squeeze(-1)
        return self.fc(out)


class TemporalResidualBlock(nn.Module):
    """Dilated residual temporal block; preserves sequence order and receptive field."""
    def __init__(self, channels: int, dilation: int, dropout: float = 0.15):
        super().__init__()
        padding = dilation
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, padding=padding,
                      dilation=dilation, bias=False),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size=3, padding=padding,
                      dilation=dilation, bias=False),
            nn.BatchNorm1d(channels),
        )
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.net(x))


class TemporalConvSensorBranch(nn.Module):
    """TCN sensor encoder with dilated residual blocks and attentive temporal pooling."""
    def __init__(self, in_channels: int = 9, embedding_dim: int = 128,
                 hidden_channels: int = 96, dropout: float = 0.2):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, hidden_channels, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(hidden_channels),
            nn.GELU(),
        )
        self.temporal = nn.Sequential(
            TemporalResidualBlock(hidden_channels, dilation=1, dropout=dropout),
            TemporalResidualBlock(hidden_channels, dilation=2, dropout=dropout),
            TemporalResidualBlock(hidden_channels, dilation=4, dropout=dropout),
        )
        self.attention = nn.Conv1d(hidden_channels, 1, kernel_size=1)
        self.projection = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_channels * 3, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.temporal(self.stem(x))
        weights = torch.softmax(self.attention(h), dim=-1)
        attended = torch.sum(h * weights, dim=-1)
        mean = h.mean(dim=-1)
        maximum = h.amax(dim=-1)
        return self.projection(torch.cat([attended, mean, maximum], dim=1))
