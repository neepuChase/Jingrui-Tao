"""Heavy overload indicator analysis for daily and monthly levels."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.visualization import save_figure


def analyze_heavy_overload(cleaned_df: pd.DataFrame, outputs_dir: Path, figures_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    daily_df = cleaned_df.groupby("date", as_index=False)["load"].mean().rename(columns={"load": "daily_load", "date": "day"})
    daily_df["month"] = pd.to_datetime(daily_df["day"]).dt.to_period("M").astype(str)

    monthly_df = cleaned_df.groupby(["year", "month"], as_index=False)["load"].mean().rename(columns={"load": "monthly_average"})
    monthly_df["month_period"] = pd.to_datetime(
        monthly_df["year"].astype(str) + "-" + monthly_df["month"].astype(str).str.zfill(2)
    ).dt.to_period("M").astype(str)

    monthly_lookup = monthly_df.set_index("month_period")["monthly_average"]
    daily_df["monthly_average"] = daily_df["month"].map(monthly_lookup)
    daily_df["threshold"] = daily_df["monthly_average"] * 1.3
    daily_df["is_heavy_overload"] = daily_df["daily_load"] > daily_df["threshold"]

    heavy_days = daily_df[daily_df["is_heavy_overload"]].copy()
    heavy_days.to_csv(outputs_dir / "heavy_overload_days.csv", index=False, encoding="utf-8-sig")

    yearly_avg = cleaned_df.groupby("year", as_index=False)["load"].mean().rename(columns={"load": "yearly_average"})
    monthly_vs_yearly = monthly_df.merge(yearly_avg, on="year", how="left")
    monthly_vs_yearly["threshold"] = monthly_vs_yearly["yearly_average"] * 1.3
    monthly_vs_yearly["is_heavy_overload"] = monthly_vs_yearly["monthly_average"] > monthly_vs_yearly["threshold"]

    heavy_months = monthly_vs_yearly[monthly_vs_yearly["is_heavy_overload"]].copy()
    heavy_months.to_csv(outputs_dir / "heavy_overload_months.csv", index=False, encoding="utf-8-sig")

    # Daily overload plot
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(pd.to_datetime(daily_df["day"]), daily_df["daily_load"], label="Daily Load", linewidth=1)
    ax.plot(pd.to_datetime(daily_df["day"]), daily_df["threshold"], label="Monthly Avg × 1.3", linestyle="--")
    if not heavy_days.empty:
        ax.scatter(pd.to_datetime(heavy_days["day"]), heavy_days["daily_load"], color="red", s=16, label="Heavy Overload Day")
    ax.set_title("Daily Heavy Overload Detection")
    ax.set_xlabel("Day")
    ax.set_ylabel("Load")
    ax.legend()
    ax.grid(True, alpha=0.3)
    save_figure(fig, figures_dir, "heavy_overload_days.png")

    # Monthly overload plot
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(monthly_vs_yearly["month_period"], monthly_vs_yearly["monthly_average"], marker="o", label="Monthly Average")
    ax.plot(monthly_vs_yearly["month_period"], monthly_vs_yearly["threshold"], linestyle="--", label="Yearly Avg × 1.3")
    if not heavy_months.empty:
        ax.scatter(heavy_months["month_period"], heavy_months["monthly_average"], color="red", s=40, label="Heavy Overload Month")
    ax.set_title("Monthly Heavy Overload Detection")
    ax.set_xlabel("Month")
    ax.set_ylabel("Load")
    ax.tick_params(axis="x", rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3)
    save_figure(fig, figures_dir, "heavy_overload_months.png")

    return heavy_days, heavy_months
