"""Heavy overload indicator analysis for daily and monthly levels."""

from __future__ import annotations

from pathlib import Path

import pandas as pd



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
    heavy_days.to_csv(outputs_dir / "heavy_overload_day_detection_results.csv", index=False, encoding="utf-8-sig")

    yearly_avg = cleaned_df.groupby("year", as_index=False)["load"].mean().rename(columns={"load": "yearly_average"})
    monthly_vs_yearly = monthly_df.merge(yearly_avg, on="year", how="left")
    monthly_vs_yearly["threshold"] = monthly_vs_yearly["yearly_average"] * 1.3
    monthly_vs_yearly["is_heavy_overload"] = monthly_vs_yearly["monthly_average"] > monthly_vs_yearly["threshold"]

    heavy_months = monthly_vs_yearly[monthly_vs_yearly["is_heavy_overload"]].copy()
    heavy_months.to_csv(outputs_dir / "heavy_overload_month_detection_results.csv", index=False, encoding="utf-8-sig")

    return heavy_days, heavy_months
