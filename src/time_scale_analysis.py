"""Multi-time-scale analysis and plotting."""

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
    tables["weekly"] = (
        df.assign(year_week=df["timestamp"].dt.strftime("%Y-%W"))
        .groupby("year_week")["load"]
        .describe()
        .reset_index()
    )
    tables["daily"] = df.groupby("date")["load"].describe().reset_index()
    tables["hourly"] = df.groupby("hour")["load"].describe().reset_index()

    return tables


def create_required_time_scale_figures(df: pd.DataFrame, figures_dir: Path) -> None:
    # 1. Raw load time series
    fig, ax = plt.subplots()
    ax.plot(df["timestamp"], df["load"], linewidth=0.8)
    ax.set_title("Raw Load Time Series")
    ax.set_xlabel("Time")
    ax.set_ylabel("Load")
    ax.grid(True, alpha=0.3)
    save_figure(fig, figures_dir, "01_raw_load_time_series.png")

    # 2. Rolling mean trend
    rolling = df.set_index("timestamp")["load"].rolling(window=24, min_periods=1).mean()
    fig, ax = plt.subplots()
    ax.plot(rolling.index, rolling.values, color="tab:orange", linewidth=1.0)
    ax.set_title("Rolling Mean Trend (24 samples)")
    ax.set_xlabel("Time")
    ax.set_ylabel("Rolling Mean Load")
    ax.grid(True, alpha=0.3)
    save_figure(fig, figures_dir, "02_rolling_mean_trend.png")

    # 3. Monthly average load
    monthly_avg = df.groupby("month")["load"].mean().reindex(range(1, 13))
    fig, ax = plt.subplots()
    monthly_avg.plot(kind="bar", ax=ax, color="tab:blue")
    ax.set_title("Monthly Average Load")
    ax.set_xlabel("Month")
    ax.set_ylabel("Average Load")
    save_figure(fig, figures_dir, "03_monthly_average_load.png")

    # 4. Monthly boxplot
    fig, ax = plt.subplots()
    sns.boxplot(data=df, x="month", y="load", ax=ax)
    ax.set_title("Monthly Load Distribution (Boxplot)")
    ax.set_xlabel("Month")
    ax.set_ylabel("Load")
    save_figure(fig, figures_dir, "04_monthly_boxplot.png")

    # 7. Weekday vs weekend average 24-hour curves
    profile_weekpart = df.groupby(["is_weekend", "hour"])["load"].mean().reset_index()
    fig, ax = plt.subplots()
    for is_weekend, grp in profile_weekpart.groupby("is_weekend"):
        label = "Weekend" if is_weekend else "Weekday"
        ax.plot(grp["hour"], grp["load"], marker="o", label=label)
    ax.set_title("Weekday vs Weekend Average 24-hour Curves")
    ax.set_xlabel("Hour")
    ax.set_ylabel("Average Load")
    ax.legend()
    save_figure(fig, figures_dir, "07_weekday_vs_weekend_24h_curves.png")

    # 8. Day-of-week average curves
    dow_profile = df.groupby(["weekday_name", "hour"], sort=False)["load"].mean().reset_index()
    ordered_days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    fig, ax = plt.subplots()
    for day in ordered_days:
        grp = dow_profile[dow_profile["weekday_name"] == day]
        if not grp.empty:
            ax.plot(grp["hour"], grp["load"], label=day)
    ax.set_title("Day-of-Week Average Curves")
    ax.set_xlabel("Hour")
    ax.set_ylabel("Average Load")
    ax.legend(ncol=2, fontsize=9)
    save_figure(fig, figures_dir, "08_day_of_week_average_curves.png")

    # 9. Hourly average load curve
    hourly_avg = df.groupby("hour")["load"].mean()
    fig, ax = plt.subplots()
    ax.plot(hourly_avg.index, hourly_avg.values, marker="o", color="tab:green")
    ax.set_title("Hourly Average Load Curve")
    ax.set_xlabel("Hour")
    ax.set_ylabel("Average Load")
    save_figure(fig, figures_dir, "09_hourly_average_load_curve.png")

    # 12. Month-hour heatmap
    mh = df.pivot_table(values="load", index="month", columns="hour", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(14, 6))
    sns.heatmap(mh, cmap="YlOrRd", ax=ax)
    ax.set_title("Month-Hour Average Load Heatmap")
    ax.set_xlabel("Hour")
    ax.set_ylabel("Month")
    save_figure(fig, figures_dir, "12_month_hour_heatmap.png")

    # 13. Weekday-hour heatmap
    wh = df.pivot_table(values="load", index="weekday_name", columns="hour", aggfunc="mean")
    wh = wh.reindex(ordered_days)
    fig, ax = plt.subplots(figsize=(14, 6))
    sns.heatmap(wh, cmap="Blues", ax=ax)
    ax.set_title("Weekday-Hour Average Load Heatmap")
    ax.set_xlabel("Hour")
    ax.set_ylabel("Weekday")
    save_figure(fig, figures_dir, "13_weekday_hour_heatmap.png")

    # Extra: daily load curve samples
    sample_days = sorted(df["date"].astype(str).unique())[:7]
    fig, ax = plt.subplots()
    for d in sample_days:
        grp = df[df["date"].astype(str) == d]
        ax.plot(grp["hour"], grp["load"], marker=".", label=d)
    ax.set_title("Daily Load Curve Samples (First 7 Days)")
    ax.set_xlabel("Hour")
    ax.set_ylabel("Load")
    ax.legend(fontsize=8, ncol=2)
    save_figure(fig, figures_dir, "19_daily_load_curve_samples.png")
