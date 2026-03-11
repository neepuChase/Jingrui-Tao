"""预测评估工具。"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.visualization import save_figure


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
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


def save_metrics(metrics: dict[str, float], outputs_dir: Path, filename: str = "预测评估指标.csv") -> pd.DataFrame:
    metrics_df = pd.DataFrame({"指标": list(metrics.keys()), "数值": list(metrics.values())})
    metrics_df.to_csv(outputs_dir / filename, index=False, encoding="utf-8-sig")
    return metrics_df


def plot_forecast_results(
    timestamps: pd.Series,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    figures_dir: Path,
    title_suffix: str = "",
) -> None:
    error = y_true - y_pred

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(timestamps, y_true, label="真实值", linewidth=1.0)
    ax.plot(timestamps, y_pred, label="预测值", linewidth=1.0)
    ax.set_title(f"预测值与真实值对比图{title_suffix}")
    ax.set_xlabel("时间")
    ax.set_ylabel("负荷")
    ax.legend()
    ax.grid(True, alpha=0.3)
    save_figure(fig, figures_dir, "预测值与真实值对比图.png")

    fig, ax = plt.subplots()
    sns.histplot(error, bins=50, kde=True, ax=ax, color="tab:red")
    ax.set_title(f"预测误差分布图{title_suffix}")
    ax.set_xlabel("误差（真实值-预测值）")
    ax.set_ylabel("频数")
    save_figure(fig, figures_dir, "预测误差分布图.png")
