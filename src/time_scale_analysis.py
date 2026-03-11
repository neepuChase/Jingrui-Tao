"""Multi-timescale analysis and plotting."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.visualization import save_figure


def compute_time_scale_tables(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    tables: dict[str, pd.DataFrame] = {}
    tables["yearly"] = df.groupby("year")["load"].describe().reset_index()
    tables["monthly"] = df.groupby(["year", "month"])["load"].describe().reset_index()
    tables["weekly"] = df.assign(year_week=df["timestamp"].dt.strftime("%Y-%W")).groupby("year_week")["load"].describe().reset_index()
    tables["daily"] = df.groupby("date")["load"].describe().reset_index()
    tables["hourly"] = df.groupby("hour")["load"].describe().reset_index()
    return tables


def create_required_time_scale_figures(df: pd.DataFrame, figures_dir: Path) -> None:
    fig, ax = plt.subplots()
    ax.plot(df["timestamp"], df["load"], linewidth=0.8)
    ax.set_title("Raw Load Time Series")
    ax.set_xlabel("Time")
    ax.set_ylabel("Load (MW)")
    ax.grid(True, alpha=0.3)
    save_figure(fig, figures_dir, "raw_load_timeseries.png")

    monthly_avg = df.groupby("month")["load"].mean().reindex(range(1, 13))
    fig, ax = plt.subplots()
    monthly_avg.plot(kind="bar", ax=ax, color="tab:blue")
    ax.set_title("Monthly Average Load")
    ax.set_xlabel("Month")
    ax.set_ylabel("Average Load (MW)")
    save_figure(fig, figures_dir, "monthly_average_load.png")

    fig, ax = plt.subplots()
    sns.boxplot(data=df, x="month", y="load", ax=ax)
    ax.set_title("Monthly Load Boxplot")
    ax.set_xlabel("Month")
    ax.set_ylabel("Load (MW)")
    save_figure(fig, figures_dir, "monthly_load_boxplot.png")

    profile_weekpart = df.groupby(["is_weekend", "hour"])["load"].mean().reset_index()
    fig, ax = plt.subplots()
    for is_weekend, grp in profile_weekpart.groupby("is_weekend"):
        label = "Weekend" if is_weekend else "Weekday"
        ax.plot(grp["hour"], grp["load"], marker="o", label=label)
    ax.set_title("Weekday vs Weekend Load")
    ax.set_xlabel("Hour")
    ax.set_ylabel("Average Load (MW)")
    ax.legend()
    save_figure(fig, figures_dir, "weekday_vs_weekend_load.png")
