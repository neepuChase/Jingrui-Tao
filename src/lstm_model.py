"""LSTM model definition for IMF forecasting."""

from __future__ import annotations

import torch
from torch import nn


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LSTMForecaster(nn.Module):
    """Simple single-output LSTM forecaster."""

    def __init__(self, input_size: int = 1, hidden_size: int = 32, num_layers: int = 1, dropout: float = 0.0) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_size, 1)
        self.to(DEVICE)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.lstm(x)
        last_hidden = output[:, -1, :]
        return self.head(last_hidden)
