"""Utilities for forecasting model comparison and selection."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.visualization import save_figure


def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def compute_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denominator = np.where(np.abs(y_true) < 1e-8, np.nan, y_true)
    return float(np.nanmean(np.abs((y_true - y_pred) / denominator)) * 100)


def compare_and_select_model(
    results: dict[str, tuple[np.ndarray, np.ndarray]],
    outputs_dir: Path,
    figures_dir: Path,
) -> tuple[pd.DataFrame, str]:
    """Compare models by RMSE/MAE/MAPE and select best by lowest RMSE."""
    rows = []
    for model_name, (y_true, y_pred) in results.items():
        rows.append(
            {
                "model": model_name.upper(),
                "RMSE": compute_rmse(y_true, y_pred),
                "MAE": compute_mae(y_true, y_pred),
                "MAPE": compute_mape(y_true, y_pred),
            }
        )

    comparison_df = pd.DataFrame(rows).sort_values("RMSE", ascending=True).reset_index(drop=True)
    comparison_df.to_csv(outputs_dir / "model_comparison.csv", index=False, encoding="utf-8-sig")

    best_model = str(comparison_df.loc[0, "model"])
    (outputs_dir / "best_model.txt").write_text(best_model, encoding="utf-8")

    # Bar chart for RMSE/MAE/MAPE
    melted = comparison_df.melt(id_vars="model", value_vars=["RMSE", "MAE", "MAPE"], var_name="metric", value_name="value")
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, metric in enumerate(["RMSE", "MAE", "MAPE"]):
        subset = melted[melted["metric"] == metric]
        x = np.arange(len(subset)) + (i - 1) * 0.25
        ax.bar(x, subset["value"], width=0.25, label=metric)

    ax.set_xticks(np.arange(len(comparison_df)))
    ax.set_xticklabels(comparison_df["model"].tolist())
    ax.set_title("Model Comparison (Lower is Better)")
    ax.set_xlabel("Model")
    ax.set_ylabel("Metric Value")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    save_figure(fig, figures_dir, "27_model_comparison_bar.png")

    return comparison_df, best_model
