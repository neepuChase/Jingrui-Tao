"""Seasonal analysis plotting."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.statistics_analysis import peak_valley_metrics
from src.visualization import save_figure

SEASON_MAP = {"Spring": "Spring", "Summer": "Summer", "Autumn": "Autumn", "Winter": "Winter"}


def create_season_figures(df: pd.DataFrame, figures_dir: Path) -> None:
    season_order = ["Spring", "Summer", "Autumn", "Winter"]

    seasonal_hour = df.groupby(["season", "hour"])["load"].mean().reset_index()
    fig, ax = plt.subplots()
    for season in season_order:
        grp = seasonal_hour[seasonal_hour["season"] == season]
        if not grp.empty:
            ax.plot(grp["hour"], grp["load"], marker="o", label=SEASON_MAP.get(season, season))
    ax.set_title("Seasonal Load Analysis")
    ax.set_xlabel("Hour")
    ax.set_ylabel("Average Load (MW)")
    ax.legend()
    save_figure(fig, figures_dir, "seasonal_load_analysis.png")


def create_statistical_character_figures(df: pd.DataFrame, figures_dir: Path, outputs_dir: Path) -> None:
    fig, ax = plt.subplots()
    sns.histplot(df["load"], bins=50, kde=True, ax=ax, color="tab:blue")
    ax.set_title("Load Distribution Histogram")
    ax.set_xlabel("Load (MW)")
    ax.set_ylabel("Frequency")
    save_figure(fig, figures_dir, "load_distribution_histogram.png")

    sorted_load = np.sort(df["load"].dropna().values)
    ecdf = np.arange(1, len(sorted_load) + 1) / len(sorted_load)
    fig, ax = plt.subplots()
    ax.plot(sorted_load, ecdf, color="tab:purple")
    ax.set_title("Load Empirical Distribution Function")
    ax.set_xlabel("Load (MW)")
    ax.set_ylabel("Cumulative Distribution")
    ax.grid(True, alpha=0.3)
    save_figure(fig, figures_dir, "load_empirical_distribution_function.png")

    df_sorted = df.sort_values("timestamp").copy()
    df_sorted["ramp"] = df_sorted["load"].diff()
    fig, ax = plt.subplots()
    sns.histplot(df_sorted["ramp"].dropna(), bins=60, kde=True, ax=ax, color="tab:red")
    ax.set_title("Load Ramp Distribution")
    ax.set_xlabel("Load Change (MW)")
    ax.set_ylabel("Frequency")
    save_figure(fig, figures_dir, "load_ramp_distribution.png")

    daily_peak_valley = peak_valley_metrics(df_sorted)
    daily_peak_valley.rename(
        columns={
            "date": "date",
            "peak_load": "peak_load",
            "valley_load": "valley_load",
            "avg_load": "avg_load",
            "std_load": "std_load",
            "peak_valley_diff": "peak_valley_diff",
            "peak_valley_ratio": "peak_valley_ratio",
        }
    ).to_csv(outputs_dir / "daily_peak_valley_metrics.csv", index=False, encoding="utf-8-sig")
