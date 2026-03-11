"""End-to-end EMD + IMF-wise forecasting pipeline with LSTM/TCN/auto selection."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.cuda.amp import GradScaler, autocast

from src.evaluation import calculate_metrics, plot_forecast_results, save_metrics
from src.lstm_dataset import build_dataloader, create_sequences
from src.lstm_model import LSTMForecaster
from src.model_comparison import compare_and_select_model
from src.tcn_model import TCNForecaster
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
    model_type: str = "auto"  # auto | lstm | tcn
    tcn_channels: tuple[int, ...] = (32, 32, 32)
    tcn_kernel_size: int = 3


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)




def _get_device() -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}")
    return device


def _build_model(model_type: str, config: ForecastConfig, device: torch.device) -> nn.Module:
    if model_type == "lstm":
        return LSTMForecaster(
            input_size=1,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            dropout=config.dropout,
        ).to(device)
    if model_type == "tcn":
        return TCNForecaster(
            input_channels=1,
            channels=list(config.tcn_channels),
            kernel_size=config.tcn_kernel_size,
            dropout=config.dropout,
        ).to(device)
    raise ValueError(f"Unsupported model_type: {model_type}")


def _train_single_imf(series: np.ndarray, config: ForecastConfig, device: torch.device, model_type: str) -> tuple[np.ndarray, np.ndarray, list[float]]:
    n = len(series)
    split_idx = max(int(n * config.train_ratio), config.lookback + 2)

    train_series = series[:split_idx]
    test_series = series[split_idx - config.lookback :]

    mean = float(np.mean(train_series))
    std = float(np.std(train_series)) + 1e-8

    train_norm = (train_series - mean) / std
    test_norm = (test_series - mean) / std

    x_train, y_train = create_sequences(train_norm, config.lookback)
    if len(x_train) == 0:
        raise ValueError("Not enough points to create training sequences. Reduce lookback or use more data.")

    train_loader = build_dataloader(x_train, y_train, config.batch_size)
    model = _build_model(model_type, config, device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    use_amp = device.type == "cuda"
    scaler = GradScaler(enabled=use_amp)

    loss_history: list[float] = []
    model.train()
    for _ in range(config.epochs):
        epoch_loss = 0.0
        for xb, yb in train_loader:
            X = xb.to(device)
            y = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=use_amp):
                pred = model(X)
                loss = criterion(pred, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += float(loss.item())
        loss_history.append(epoch_loss / max(1, len(train_loader)))

    model.eval()
    horizon = n - split_idx
    input_window = list(test_norm[: config.lookback])
    preds_norm: list[float] = []

    with torch.no_grad():
        for _ in range(horizon):
            x = torch.tensor(input_window[-config.lookback :], dtype=torch.float32, device=device).view(1, config.lookback, 1)
            with autocast(enabled=device.type == "cuda"):
                yhat = float(model(x).item())
            preds_norm.append(yhat)
            input_window.append(yhat)

    y_pred = np.asarray(preds_norm) * std + mean
    y_true = series[split_idx: split_idx + len(y_pred)]
    return y_true, y_pred, loss_history


def _run_single_model(cleaned_df: pd.DataFrame, imf_df: pd.DataFrame, config: ForecastConfig, outputs_dir: Path, model_type: str, device: torch.device) -> tuple[pd.DataFrame, dict[str, list[float]], np.ndarray, np.ndarray, pd.Series]:
    imf_columns = [c for c in imf_df.columns if c.startswith("imf_")]
    imf_values = imf_df[imf_columns].values

    all_true, all_pred = [], []
    all_losses: dict[str, list[float]] = {}

    for i, col in enumerate(imf_columns):
        y_true_i, y_pred_i, loss_history = _train_single_imf(imf_values[:, i], config, device, model_type=model_type)
        all_true.append(y_true_i)
        all_pred.append(y_pred_i)
        all_losses[col] = loss_history

    true_reconstructed = np.sum(np.vstack(all_true), axis=0)
    pred_reconstructed = np.sum(np.vstack(all_pred), axis=0)

    split_idx = max(int(len(cleaned_df) * config.train_ratio), config.lookback + 2)
    timestamps_test = cleaned_df["timestamp"].reset_index(drop=True).iloc[split_idx: split_idx + len(true_reconstructed)]

    forecast_df = pd.DataFrame(
        {
            "timestamp": timestamps_test.values,
            "actual_load": true_reconstructed,
            "predicted_load": pred_reconstructed,
            "error": true_reconstructed - pred_reconstructed,
        }
    )
    forecast_df.to_csv(outputs_dir / f"{model_type}_forecast.csv", index=False, encoding="utf-8-sig")
    return forecast_df, all_losses, true_reconstructed, pred_reconstructed, timestamps_test


def _plot_loss_curves(loss_by_model: dict[str, dict[str, list[float]]], figures_dir: Path) -> None:
    fig, axes = plt.subplots(len(loss_by_model), 1, figsize=(14, 5 * len(loss_by_model)), squeeze=False)
    for idx, (model_name, losses_dict) in enumerate(loss_by_model.items()):
        ax = axes[idx, 0]
        for name, losses in losses_dict.items():
            ax.plot(range(1, len(losses) + 1), losses, label=name)
        ax.set_title(f"IMF-wise {model_name.upper()} Training Loss Curves")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE Loss")
        ax.legend(ncol=3, fontsize=8)
        ax.grid(True, alpha=0.3)
    save_figure(fig, figures_dir, "23_training_loss_curves.png")


def run_imf_forecast(
    cleaned_df: pd.DataFrame,
    imf_df: pd.DataFrame,
    outputs_dir: Path,
    figures_dir: Path,
    config: ForecastConfig | None = None,
) -> dict[str, float]:
    cfg = config or ForecastConfig()
    model_type = cfg.model_type.lower().strip()
    if model_type not in {"auto", "lstm", "tcn"}:
        raise ValueError("MODEL_TYPE must be one of: auto | lstm | tcn")

    _set_seed(cfg.random_seed)
    torch.backends.cudnn.benchmark = True
    torch.set_num_threads(max(1, torch.get_num_threads()))
    device = _get_device()

    selected_models = ["lstm", "tcn"] if model_type == "auto" else [model_type]

    model_results: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    loss_by_model: dict[str, dict[str, list[float]]] = {}
    forecast_tables: dict[str, pd.DataFrame] = {}
    timestamps_test: pd.Series | None = None

    for m in selected_models:
        forecast_df, losses, y_true, y_pred, ts = _run_single_model(cleaned_df, imf_df, cfg, outputs_dir, model_type=m, device=device)
        model_results[m] = (y_true, y_pred)
        forecast_tables[m] = forecast_df
        loss_by_model[m] = losses
        timestamps_test = ts

    _plot_loss_curves(loss_by_model, figures_dir)

    if len(selected_models) == 2:
        _, best_model_upper = compare_and_select_model(model_results, outputs_dir, figures_dir)
        best_key = best_model_upper.lower()
    else:
        best_key = selected_models[0]
        single_metrics = calculate_metrics(*model_results[best_key])
        comparison_df = pd.DataFrame(
            [{"model": best_key.upper(), "RMSE": single_metrics["RMSE"], "MAE": single_metrics["MAE"], "MAPE": single_metrics["MAPE"]}]
        )
        comparison_df.to_csv(outputs_dir / "model_comparison.csv", index=False, encoding="utf-8-sig")
        (outputs_dir / "best_model.txt").write_text(best_key.upper(), encoding="utf-8")

    final_forecast = forecast_tables[best_key]
    final_forecast.to_csv(outputs_dir / "final_forecast.csv", index=False, encoding="utf-8-sig")
    # Backward compatibility
    final_forecast.to_csv(outputs_dir / "forecast_results.csv", index=False, encoding="utf-8-sig")

    y_true_final, y_pred_final = model_results[best_key]
    metrics = calculate_metrics(y_true_final, y_pred_final)
    save_metrics(metrics, outputs_dir)

    if timestamps_test is None:
        raise RuntimeError("No forecast timestamps were generated.")
    plot_forecast_results(timestamps_test, y_true_final, y_pred_final, figures_dir)
    return metrics


# Backward-compatible name
run_imf_lstm_forecast = run_imf_forecast
