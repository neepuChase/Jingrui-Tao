"""End-to-end EMD + IMF-wise LSTM forecasting pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn

from src.evaluation import calculate_metrics, plot_forecast_results, save_metrics
from src.lstm_dataset import build_dataloader, create_sequences
from src.lstm_model import LSTMForecaster
from src.visualization import save_figure


@dataclass
class ForecastConfig:
    lookback: int = 96
    train_ratio: float = 0.8
    hidden_size: int = 32
    num_layers: int = 1
    dropout: float = 0.0
    learning_rate: float = 1e-3
    epochs: int = 20
    batch_size: int = 64
    random_seed: int = 42


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def _train_single_imf(
    series: np.ndarray,
    config: ForecastConfig,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    n = len(series)
    split_idx = int(n * config.train_ratio)
    split_idx = max(split_idx, config.lookback + 2)

    train_series = series[:split_idx]
    test_series = series[split_idx - config.lookback :]

    mean = float(np.mean(train_series))
    std = float(np.std(train_series)) + 1e-8

    train_norm = (train_series - mean) / std
    test_norm = (test_series - mean) / std

    x_train, y_train = create_sequences(train_norm, config.lookback)
    if len(x_train) == 0:
        raise ValueError("Not enough points to create LSTM training sequences. Reduce lookback or use more data.")

    train_loader = build_dataloader(x_train, y_train, config.batch_size)

    model = LSTMForecaster(
        input_size=1,
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        dropout=config.dropout,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    loss_history: list[float] = []
    model.train()
    for _ in range(config.epochs):
        epoch_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())

        loss_history.append(epoch_loss / max(1, len(train_loader)))

    # Recursive forecasting on test horizon
    model.eval()
    horizon = n - split_idx
    input_window = list(test_norm[: config.lookback])
    preds_norm: list[float] = []

    with torch.no_grad():
        for _ in range(horizon):
            x = torch.tensor(input_window[-config.lookback :], dtype=torch.float32, device=device).view(1, config.lookback, 1)
            yhat = float(model(x).item())
            preds_norm.append(yhat)
            input_window.append(yhat)

    y_pred = np.asarray(preds_norm) * std + mean
    y_true = series[split_idx:]
    return y_true, y_pred, loss_history


def run_imf_lstm_forecast(
    cleaned_df: pd.DataFrame,
    imf_df: pd.DataFrame,
    outputs_dir: Path,
    figures_dir: Path,
    config: ForecastConfig | None = None,
) -> dict[str, float]:
    """Train one LSTM per IMF, reconstruct final prediction, evaluate, and save outputs."""
    cfg = config or ForecastConfig()
    _set_seed(cfg.random_seed)
    torch.set_num_threads(max(1, torch.get_num_threads()))
    device = torch.device("cpu")

    imf_columns = [c for c in imf_df.columns if c.startswith("imf_")]
    imf_values = imf_df[imf_columns].values

    all_true = []
    all_pred = []
    all_losses: dict[str, list[float]] = {}

    for i, col in enumerate(imf_columns):
        y_true_i, y_pred_i, loss_history = _train_single_imf(imf_values[:, i], cfg, device)
        all_true.append(y_true_i)
        all_pred.append(y_pred_i)
        all_losses[col] = loss_history

    true_reconstructed = np.sum(np.vstack(all_true), axis=0)
    pred_reconstructed = np.sum(np.vstack(all_pred), axis=0)

    split_idx = int(len(cleaned_df) * cfg.train_ratio)
    split_idx = max(split_idx, cfg.lookback + 2)
    timestamps_test = cleaned_df["timestamp"].reset_index(drop=True).iloc[split_idx: split_idx + len(true_reconstructed)]

    forecast_df = pd.DataFrame(
        {
            "timestamp": timestamps_test.values,
            "actual_load": true_reconstructed,
            "predicted_load": pred_reconstructed,
            "error": true_reconstructed - pred_reconstructed,
        }
    )
    forecast_df.to_csv(outputs_dir / "forecast_results.csv", index=False, encoding="utf-8-sig")

    metrics = calculate_metrics(true_reconstructed, pred_reconstructed)
    save_metrics(metrics, outputs_dir)

    # Training loss curves
    fig, ax = plt.subplots(figsize=(14, 7))
    for name, losses in all_losses.items():
        ax.plot(range(1, len(losses) + 1), losses, label=name)
    ax.set_title("IMF-wise LSTM Training Loss Curves")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.legend(ncol=3, fontsize=8)
    ax.grid(True, alpha=0.3)
    save_figure(fig, figures_dir, "23_training_loss_curves.png")

    plot_forecast_results(timestamps_test, true_reconstructed, pred_reconstructed, figures_dir)

    return metrics
