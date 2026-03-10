"""Dataset helpers for IMF-wise LSTM modeling."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


def create_sequences(series: np.ndarray, lookback: int) -> tuple[np.ndarray, np.ndarray]:
    """Create supervised learning sequences from a 1D series."""
    x, y = [], []
    for i in range(len(series) - lookback):
        x.append(series[i : i + lookback])
        y.append(series[i + lookback])

    if not x:
        return np.empty((0, lookback), dtype=np.float32), np.empty((0,), dtype=np.float32)

    return np.asarray(x, dtype=np.float32), np.asarray(y, dtype=np.float32)


def build_dataloader(x: np.ndarray, y: np.ndarray, batch_size: int) -> DataLoader:
    """Create a PyTorch DataLoader from sequence arrays."""
    features = torch.from_numpy(x).unsqueeze(-1)  # [N, lookback, 1]
    targets = torch.from_numpy(y).unsqueeze(-1)   # [N, 1]
    dataset = TensorDataset(features, targets)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
