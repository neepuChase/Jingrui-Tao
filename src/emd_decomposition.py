"""EMD decomposition utilities."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PyEMD import EMD

from src.visualization import save_figure

MAX_IMF = 10


def perform_emd(load_series: pd.Series, max_imf: int = MAX_IMF) -> np.ndarray:
    """Run EMD on the load series and cap IMF count at max_imf."""
    emd = EMD()
    values = load_series.astype(float).values
    imfs = emd.emd(values)
    if imfs.shape[0] > max_imf:
        imfs = imfs[:max_imf, :]
    return imfs


def save_imfs(imfs: np.ndarray, timestamps: pd.Series, outputs_dir: Path) -> pd.DataFrame:
    """Save IMF components and return a DataFrame."""
    imf_columns = [f"IMF{i + 1}" for i in range(imfs.shape[0])]
    imf_df = pd.DataFrame(imfs.T, columns=imf_columns)
    imf_df.insert(0, "timestamp", pd.to_datetime(timestamps).values)

    outputs_dir.mkdir(parents=True, exist_ok=True)
    imf_df.to_csv(outputs_dir / "emd_decomposition_results.csv", index=False, encoding="utf-8-sig")
    return imf_df


def plot_emd_overview(load_series: pd.Series, imfs: np.ndarray, figures_dir: Path) -> None:
    n_imfs = imfs.shape[0]
    fig, axes = plt.subplots(n_imfs + 1, 1, figsize=(14, 2.2 * (n_imfs + 1)), sharex=True)

    axes[0].plot(load_series.values, color="black", linewidth=1.0)
    axes[0].set_title("Raw Load Series")
    axes[0].grid(True, alpha=0.2)

    for i in range(n_imfs):
        axes[i + 1].plot(imfs[i], linewidth=0.8)
        axes[i + 1].set_title(f"IMF Component {i + 1}")
        axes[i + 1].grid(True, alpha=0.2)

    axes[-1].set_xlabel("Time Index")
    save_figure(fig, figures_dir, "emd_decomposition_overview.png")


def plot_imf_components(imf_df: pd.DataFrame, figures_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(14, 7))
    imf_columns = [c for c in imf_df.columns if c.startswith("IMF")]
    for col in imf_columns:
        ax.plot(imf_df["timestamp"], imf_df[col], linewidth=0.8, label=col)

    ax.set_title("IMF Components")
    ax.set_xlabel("Time")
    ax.set_ylabel("Component Value")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=3, fontsize=8)
    save_figure(fig, figures_dir, "imf_components.png")
