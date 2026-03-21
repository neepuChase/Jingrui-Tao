from __future__ import annotations

import torch
from torch import nn

from src.lstm_model import LSTMForecaster
from src.tcn_model import TCNForecaster


class AutoFormerLite(nn.Module):
    def __init__(self, input_size: int, lookback: int, horizon: int, hidden_size: int = 64, dropout: float = 0.1) -> None:
        super().__init__()
        self.embedding = nn.Linear(input_size, hidden_size)
        self.position = nn.Parameter(torch.zeros(1, lookback, hidden_size))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=4,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, horizon),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.embedding(x) + self.position[:, : x.size(1), :]
        encoded = self.encoder(encoded)
        pooled = encoded.mean(dim=1)
        return self.head(pooled)


class SCINetLite(nn.Module):
    def __init__(self, input_size: int, horizon: int, hidden_size: int = 64, dropout: float = 0.1) -> None:
        super().__init__()
        self.branch_even = nn.Sequential(
            nn.Conv1d(input_size, hidden_size, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.branch_odd = nn.Sequential(
            nn.Conv1d(input_size, hidden_size, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, horizon),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        signal = x.transpose(1, 2)
        even_repr = self.branch_even(signal[:, :, ::2]).mean(dim=-1)
        odd_signal = signal[:, :, 1::2]
        if odd_signal.size(-1) == 0:
            odd_signal = signal[:, :, -1:]
        odd_repr = self.branch_odd(odd_signal).mean(dim=-1)
        return self.head(torch.cat([even_repr, odd_repr], dim=1))


class ModelFactory:
    SUPPORTED_MODELS = ("tcn", "lstm", "autoformer", "scinet")

    @staticmethod
    def create(name: str, *, input_size: int, lookback: int, horizon: int, dropout: float) -> nn.Module:
        model_name = name.lower()
        if model_name == "tcn":
            return TCNForecaster(input_channels=input_size, channels=[32, 64, 64], dropout=dropout, output_size=horizon)
        if model_name == "lstm":
            return LSTMForecaster(input_size=input_size, hidden_size=64, num_layers=2, dropout=dropout, output_size=horizon)
        if model_name == "autoformer":
            return AutoFormerLite(input_size=input_size, lookback=lookback, horizon=horizon, hidden_size=64, dropout=dropout)
        if model_name == "scinet":
            return SCINetLite(input_size=input_size, horizon=horizon, hidden_size=64, dropout=dropout)
        raise ValueError(f"不支持的模型类型：{name}")
