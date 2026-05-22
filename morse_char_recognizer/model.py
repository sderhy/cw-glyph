"""1D CNN models for Morse-character glyph classification."""

from __future__ import annotations

import torch
from torch import nn


class ConvBlock1d(nn.Module):
    """Conv-BN-ReLU-Dropout block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int,
        dilation: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        padding = dilation * (kernel_size // 2)
        self.net = nn.Sequential(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResidualBlock1d(nn.Module):
    """Residual block for local Morse glyph strokes."""

    def __init__(self, channels: int, *, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            ConvBlock1d(
                channels,
                channels,
                kernel_size=kernel_size,
                dilation=dilation,
                dropout=dropout,
            ),
            nn.Conv1d(
                channels,
                channels,
                kernel_size=kernel_size,
                padding=dilation * (kernel_size // 2),
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm1d(channels),
        )
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.net(x))


class SmallCnn1d(nn.Module):
    """Compact CNN for fixed-length envelope vectors."""

    def __init__(self, num_classes: int, input_channels: int = 1) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(input_channels, 24, kernel_size=7, padding=3),
            nn.BatchNorm1d(24),
            nn.ReLU(),
            nn.Conv1d(24, 48, kernel_size=5, padding=2),
            nn.BatchNorm1d(48),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(48, 96, kernel_size=5, padding=2),
            nn.BatchNorm1d(96),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(96, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x).squeeze(-1)
        return self.classifier(x)


class MorseGlyphCnn1d(nn.Module):
    """Residual 1D CNN for Morse glyph recognition.

    The model treats a normalized envelope like a handwritten-character stroke:
    early layers catch short dits and edges, dilated residual blocks capture
    intra-character spacing, and global pooling makes the head independent of
    the fixed input length chosen for the experiment.
    """

    def __init__(self, num_classes: int, input_channels: int = 1, dropout: float = 0.08) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            ConvBlock1d(input_channels, 48, kernel_size=9, dropout=dropout),
            ConvBlock1d(48, 64, kernel_size=7, dropout=dropout),
        )
        self.stage1 = nn.Sequential(
            ResidualBlock1d(64, kernel_size=5, dilation=1, dropout=dropout),
            ResidualBlock1d(64, kernel_size=5, dilation=2, dropout=dropout),
            nn.MaxPool1d(2),
        )
        self.proj = ConvBlock1d(64, 128, kernel_size=5, dropout=dropout)
        self.stage2 = nn.Sequential(
            ResidualBlock1d(128, kernel_size=5, dilation=1, dropout=dropout),
            ResidualBlock1d(128, kernel_size=5, dilation=2, dropout=dropout),
            ResidualBlock1d(128, kernel_size=3, dilation=4, dropout=dropout),
            nn.MaxPool1d(2),
        )
        self.stage3 = nn.Sequential(
            ConvBlock1d(128, 192, kernel_size=3, dropout=dropout),
            ResidualBlock1d(192, kernel_size=3, dilation=1, dropout=dropout),
            ResidualBlock1d(192, kernel_size=3, dilation=2, dropout=dropout),
        )
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(384, 192),
            nn.ReLU(),
            nn.Dropout(dropout * 2),
            nn.Linear(192, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.proj(x)
        x = self.stage2(x)
        x = self.stage3(x)
        pooled = torch.cat([self.avg_pool(x).squeeze(-1), self.max_pool(x).squeeze(-1)], dim=1)
        return self.classifier(pooled)
