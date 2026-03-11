"""Temporal Convolutional Network model for univariate time-series forecasting."""

from __future__ import annotations

import torch
from torch import nn


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Chomp1d(nn.Module):
    """Remove extra padded timesteps to preserve causal length."""

    def __init__(self, chomp_size: int) -> None:
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.chomp_size == 0:
            return x
        return x[:, :, : -self.chomp_size]


class TemporalBlock(nn.Module):
    """Two-layer dilated causal convolution block with residual connection."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()
        padding = (kernel_size - 1) * dilation

        self.net = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation),
            Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, dilation=dilation),
            Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.downsample = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else None
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.activation(out + res)


class TCNForecaster(nn.Module):
    """Standard TCN forecaster for one-step-ahead prediction."""

    def __init__(
        self,
        input_channels: int = 1,
        channels: list[int] | None = None,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        channels = channels or [32, 32, 32]

        layers: list[nn.Module] = []
        in_ch = input_channels
        for i, out_ch in enumerate(channels):
            dilation = 2**i
            layers.append(TemporalBlock(in_ch, out_ch, kernel_size, dilation, dropout))
            in_ch = out_ch

        self.network = nn.Sequential(*layers)
        self.regressor = nn.Linear(channels[-1], 1)
        self.to(DEVICE)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Input shape: [batch, seq_len, features], output: [batch]."""
        x = x.transpose(1, 2)  # [B, C, L]
        y = self.network(x)
        last_state = y[:, :, -1]
        pred = self.regressor(last_state)
        return pred.squeeze(-1)
