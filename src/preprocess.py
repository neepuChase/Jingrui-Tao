"""Preprocessing utilities: infer columns, clean time series, and quality report."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


@dataclass
class ColumnInference:
    timestamp_col: str
    load_col: str


def _normalize_colname(col: str) -> str:
    return str(col).strip().lower().replace(" ", "")


def infer_timestamp_and_load_columns(df: pd.DataFrame) -> ColumnInference:
    """Infer timestamp and load columns with English/Chinese keyword support."""
    if df.empty:
        raise ValueError("Input DataFrame is empty; cannot infer columns.")

    cols = list(df.columns)
    normalized = {_normalize_colname(c): c for c in cols}

    time_keywords = [
        "time",
        "date",
        "datetime",
        "timestamp",
        "时刻",
        "时间",
        "日期",
        "采样时间",
        "记录时间",
    ]
    load_keywords = [
        "load",
        "power",
        "demand",
        "mw",
        "kw",
        "负荷",
        "有功",
        "电力",
        "功率",
    ]

    timestamp_col = None
    load_col = None

    for norm_name, original in normalized.items():
        if any(k in norm_name for k in time_keywords):
            timestamp_col = original
            break

    for norm_name, original in normalized.items():
        if any(k in norm_name for k in load_keywords):
            load_col = original
            break

    if timestamp_col is None:
        best_col = None
        best_ratio = -1.0
        for c in cols:
            parsed = pd.to_datetime(df[c], errors="coerce")
            ratio = parsed.notna().mean()
            if ratio > best_ratio:
                best_ratio = ratio
                best_col = c
        timestamp_col = best_col

    if load_col is None:
        numeric_scores: List[Tuple[str, float]] = []
        for c in cols:
            if c == timestamp_col:
                continue
            numeric = pd.to_numeric(df[c], errors="coerce")
            score = numeric.notna().mean()
            numeric_scores.append((c, score))
        if not numeric_scores:
            raise ValueError("Unable to infer load column from dataset.")
        load_col = max(numeric_scores, key=lambda x: x[1])[0]

    if timestamp_col == load_col:
        raise ValueError("Timestamp and load columns are the same; please inspect dataset.")

    return ColumnInference(timestamp_col=timestamp_col, load_col=load_col)


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add time-derived feature columns for analysis and multivariate forecasting."""
    out = df.copy()
    out["year"] = out["timestamp"].dt.year
    out["month"] = out["timestamp"].dt.month
    out["day"] = out["timestamp"].dt.day
    out["hour"] = out["timestamp"].dt.hour
    out["minute"] = out["timestamp"].dt.minute
    out["weekday"] = out["timestamp"].dt.dayofweek
    out["weekday_name"] = out["timestamp"].dt.day_name()
    out["is_weekend"] = out["weekday"] >= 5
    out["date"] = out["timestamp"].dt.date

    month = out["month"]
    out["season"] = np.select(
        [month.isin([3, 4, 5]), month.isin([6, 7, 8]), month.isin([9, 10, 11])],
        ["Spring", "Summer", "Autumn"],
        default="Winter",
    )

    hour_of_day = out["hour"] + out["minute"] / 60.0
    out["hour_sin"] = np.sin(2 * np.pi * hour_of_day / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * hour_of_day / 24.0)
    out["weekday_sin"] = np.sin(2 * np.pi * out["weekday"] / 7.0)
    out["weekday_cos"] = np.cos(2 * np.pi * out["weekday"] / 7.0)
    out["month_sin"] = np.sin(2 * np.pi * (out["month"] - 1) / 12.0)
    out["month_cos"] = np.cos(2 * np.pi * (out["month"] - 1) / 12.0)
    return out


def clean_load_data(raw_df: pd.DataFrame, timestamp_col: str, load_col: str) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Clean load time-series data, retaining usable exogenous variables."""
    quality_report: Dict[str, int] = {}

    feature_cols = [c for c in raw_df.columns if c not in {timestamp_col, load_col}]
    selected_cols = [timestamp_col, load_col, *feature_cols]
    df = raw_df[selected_cols].copy()
    rename_map = {timestamp_col: "timestamp", load_col: "load"}
    df = df.rename(columns=rename_map)

    quality_report["raw_rows"] = len(df)

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["load"] = pd.to_numeric(df["load"], errors="coerce")

    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    quality_report["invalid_timestamp_rows"] = int(df["timestamp"].isna().sum())
    quality_report["invalid_load_rows"] = int(df["load"].isna().sum())

    df = df.dropna(subset=["timestamp"])
    df = df.sort_values("timestamp")

    duplicated_count = int(df.duplicated(subset=["timestamp"]).sum())
    quality_report["duplicated_timestamp_rows"] = duplicated_count
    if duplicated_count > 0:
        numeric_cols = [c for c in df.columns if c != "timestamp"]
        df = df.groupby("timestamp", as_index=False)[numeric_cols].mean()

    interpolation_index = pd.DatetimeIndex(df["timestamp"])

    if df["load"].isna().any():
        load_series = pd.Series(df["load"].to_numpy(), index=interpolation_index)
        load_series = load_series.interpolate(method="time", limit_direction="both")
        df["load"] = load_series.ffill().bfill().to_numpy()

    missing_feature_total = 0
    for col in feature_cols:
        if df[col].isna().any():
            feature_series = pd.Series(df[col].to_numpy(), index=interpolation_index)
            feature_series = feature_series.interpolate(method="time", limit_direction="both")
            df[col] = feature_series.ffill().bfill().to_numpy()
        missing_feature_total += int(df[col].isna().sum())

    quality_report["remaining_missing_load"] = int(df["load"].isna().sum())
    quality_report["remaining_missing_features"] = missing_feature_total
    quality_report["cleaned_rows"] = len(df)

    df = add_time_features(df)
    return df, quality_report
