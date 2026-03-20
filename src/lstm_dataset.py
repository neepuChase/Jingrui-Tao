"""Dataset helpers for sequence modeling."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


def create_sequences(
    features: np.ndarray,
    targets: np.ndarray | None = None,
    lookback: int = 1,
    horizon: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Create supervised learning sequences from feature and target arrays."""
    feature_array = np.asarray(features, dtype=np.float32)
    if feature_array.ndim == 1:
        feature_array = feature_array.reshape(-1, 1)

    target_array = feature_array[:, 0] if targets is None else np.asarray(targets, dtype=np.float32).reshape(-1)

    x, y = [], []
    for i in range(len(feature_array) - lookback - horizon + 1):
        x.append(feature_array[i : i + lookback])
        y.append(target_array[i + lookback : i + lookback + horizon])

    if not x:
        return (
            np.empty((0, lookback, feature_array.shape[1]), dtype=np.float32),
            np.empty((0, horizon), dtype=np.float32),
        )

    return np.asarray(x, dtype=np.float32), np.asarray(y, dtype=np.float32)


def build_dataloader(x: np.ndarray, y: np.ndarray, batch_size: int) -> DataLoader:
    """Create a PyTorch DataLoader from sequence arrays."""
    features = torch.from_numpy(np.asarray(x, dtype=np.float32))
    targets = torch.from_numpy(np.asarray(y, dtype=np.float32))
    dataset = TensorDataset(features, targets)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
