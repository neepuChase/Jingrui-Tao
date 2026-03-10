"""Evaluation utilities for forecast quality assessment."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.visualization import save_figure


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute MAE, RMSE, MAPE, and R2."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    denominator = np.where(np.abs(y_true) < 1e-8, np.nan, y_true)
    mape = float(np.nanmean(np.abs((y_true - y_pred) / denominator)) * 100)

    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

    return {"MAE": mae, "RMSE": rmse, "MAPE": mape, "R2": r2}


def save_metrics(metrics: dict[str, float], outputs_dir: Path) -> pd.DataFrame:
    """Save evaluation metrics to CSV."""
    metrics_df = pd.DataFrame({"metric": list(metrics.keys()), "value": list(metrics.values())})
    metrics_df.to_csv(outputs_dir / "forecast_metrics.csv", index=False, encoding="utf-8-sig")
    return metrics_df


def plot_forecast_results(
    timestamps: pd.Series,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    figures_dir: Path,
) -> None:
    """Create required forecast/evaluation figures."""
    error = y_true - y_pred

    # Forecast vs actual
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(timestamps, y_true, label="Actual", linewidth=1.0)
    ax.plot(timestamps, y_pred, label="Predicted", linewidth=1.0)
    ax.set_title("Forecast vs Actual (Reconstructed Load)")
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Load")
    ax.legend()
    ax.grid(True, alpha=0.3)
    save_figure(fig, figures_dir, "24_forecast_vs_actual.png")

    # Error distribution
    fig, ax = plt.subplots()
    sns.histplot(error, bins=50, kde=True, ax=ax, color="tab:red")
    ax.set_title("Forecast Error Distribution")
    ax.set_xlabel("Error (Actual - Predicted)")
    ax.set_ylabel("Frequency")
    save_figure(fig, figures_dir, "25_forecast_error_distribution.png")

    # Zoomed prediction
    zoom_n = min(500, len(y_true))
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(timestamps.iloc[:zoom_n], y_true[:zoom_n], label="Actual", linewidth=1.2)
    ax.plot(timestamps.iloc[:zoom_n], y_pred[:zoom_n], label="Predicted", linewidth=1.2)
    ax.set_title(f"Zoomed Forecast vs Actual (First {zoom_n} Points)")
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Load")
    ax.legend()
    ax.grid(True, alpha=0.3)
    save_figure(fig, figures_dir, "26_zoomed_prediction_plot.png")
