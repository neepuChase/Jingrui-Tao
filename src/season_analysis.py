"""Seasonal and distribution-focused analysis plots."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.statistics_analysis import peak_valley_metrics
from src.visualization import save_figure


def create_season_figures(df: pd.DataFrame, figures_dir: Path) -> None:
    season_order = ["Spring", "Summer", "Autumn", "Winter"]

    # 5. Seasonal boxplot
    fig, ax = plt.subplots()
    sns.boxplot(data=df, x="season", y="load", order=season_order, ax=ax)
    ax.set_title("Seasonal Load Distribution (Boxplot)")
    ax.set_xlabel("Season")
    ax.set_ylabel("Load")
    save_figure(fig, figures_dir, "05_seasonal_boxplot.png")

    # 6. Seasonal average 24-hour curves
    seasonal_hour = df.groupby(["season", "hour"])["load"].mean().reset_index()
    fig, ax = plt.subplots()
    for season in season_order:
        grp = seasonal_hour[seasonal_hour["season"] == season]
        if not grp.empty:
            ax.plot(grp["hour"], grp["load"], marker="o", label=season)
    ax.set_title("Seasonal Average 24-hour Curves")
    ax.set_xlabel("Hour")
    ax.set_ylabel("Average Load")
    ax.legend()
    save_figure(fig, figures_dir, "06_seasonal_average_24h_curves.png")

    # Seasonal distribution KDE
    fig, ax = plt.subplots()
    for season in season_order:
        values = df[df["season"] == season]["load"].dropna()
        if len(values) > 1:
            sns.kdeplot(values, ax=ax, label=season, fill=False)
    ax.set_title("Seasonal Load Density Distribution")
    ax.set_xlabel("Load")
    ax.set_ylabel("Density")
    ax.legend()
    save_figure(fig, figures_dir, "20_seasonal_distribution_kde.png")


def create_statistical_character_figures(df: pd.DataFrame, figures_dir: Path, outputs_dir: Path) -> None:
    # 10. Histogram + KDE
    fig, ax = plt.subplots()
    sns.histplot(df["load"], bins=50, kde=True, ax=ax, color="tab:blue")
    ax.set_title("Load Distribution: Histogram with KDE")
    ax.set_xlabel("Load")
    ax.set_ylabel("Frequency")
    save_figure(fig, figures_dir, "10_histogram_with_kde.png")

    # 11. ECDF plot
    sorted_load = np.sort(df["load"].dropna().values)
    ecdf = np.arange(1, len(sorted_load) + 1) / len(sorted_load)
    fig, ax = plt.subplots()
    ax.plot(sorted_load, ecdf, color="tab:purple")
    ax.set_title("Empirical Cumulative Distribution Function (ECDF)")
    ax.set_xlabel("Load")
    ax.set_ylabel("ECDF")
    ax.grid(True, alpha=0.3)
    save_figure(fig, figures_dir, "11_ecdf_plot.png")

    # Ramp / change series
    df_sorted = df.sort_values("timestamp").copy()
    df_sorted["ramp"] = df_sorted["load"].diff()
    df_sorted["abs_ramp"] = df_sorted["ramp"].abs()

    # 17. Load ramp distribution
    fig, ax = plt.subplots()
    sns.histplot(df_sorted["ramp"].dropna(), bins=60, kde=True, ax=ax, color="tab:red")
    ax.set_title("Load Ramp Distribution")
    ax.set_xlabel("Load Change (Ramp)")
    ax.set_ylabel("Frequency")
    save_figure(fig, figures_dir, "17_load_ramp_distribution.png")

    daily_peak_valley = peak_valley_metrics(df_sorted)
    daily_peak_valley.to_csv(outputs_dir / "daily_peak_valley_metrics.csv", index=False, encoding="utf-8-sig")

    # 14. Daily peak load trend
    fig, ax = plt.subplots()
    ax.plot(pd.to_datetime(daily_peak_valley["date"]), daily_peak_valley["peak_load"], color="tab:orange")
    ax.set_title("Daily Peak Load Trend")
    ax.set_xlabel("Date")
    ax.set_ylabel("Peak Load")
    save_figure(fig, figures_dir, "14_daily_peak_load_trend.png")

    # 15. Daily valley load trend
    fig, ax = plt.subplots()
    ax.plot(pd.to_datetime(daily_peak_valley["date"]), daily_peak_valley["valley_load"], color="tab:green")
    ax.set_title("Daily Valley Load Trend")
    ax.set_xlabel("Date")
    ax.set_ylabel("Valley Load")
    save_figure(fig, figures_dir, "15_daily_valley_load_trend.png")

    # 16. Daily peak-valley difference trend
    fig, ax = plt.subplots()
    ax.plot(pd.to_datetime(daily_peak_valley["date"]), daily_peak_valley["peak_valley_diff"], color="tab:blue")
    ax.set_title("Daily Peak-Valley Difference Trend")
    ax.set_xlabel("Date")
    ax.set_ylabel("Peak - Valley")
    save_figure(fig, figures_dir, "16_daily_peak_valley_difference_trend.png")
