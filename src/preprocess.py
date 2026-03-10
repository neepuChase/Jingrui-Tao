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

    # Candidate keywords (include Chinese and English variants).
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

    # Fallback: detect timestamp by parse success ratio.
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

    # Fallback: detect load as numeric column with highest non-null ratio.
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
    """Add time-derived feature columns for analysis."""
    out = df.copy()
    out["year"] = out["timestamp"].dt.year
    out["month"] = out["timestamp"].dt.month
    out["day"] = out["timestamp"].dt.day
    out["hour"] = out["timestamp"].dt.hour
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
    return out


def clean_load_data(raw_df: pd.DataFrame, timestamp_col: str, load_col: str) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Clean load time-series data and return quality report counts."""
    quality_report: Dict[str, int] = {}

    df = raw_df[[timestamp_col, load_col]].copy()
    df = df.rename(columns={timestamp_col: "timestamp", load_col: "load"})

    quality_report["raw_rows"] = len(df)

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["load"] = pd.to_numeric(df["load"], errors="coerce")

    quality_report["invalid_timestamp_rows"] = int(df["timestamp"].isna().sum())
    quality_report["invalid_load_rows"] = int(df["load"].isna().sum())

    df = df.dropna(subset=["timestamp"])
    df = df.sort_values("timestamp")

    # Handle duplicated timestamps by averaging load values.
    duplicated_count = int(df.duplicated(subset=["timestamp"]).sum())
    quality_report["duplicated_timestamp_rows"] = duplicated_count
    if duplicated_count > 0:
        df = df.groupby("timestamp", as_index=False)["load"].mean()

    # Missing load values: time interpolation + forward/backward fill fallback.
    if df["load"].isna().any():
        df["load"] = df["load"].interpolate(method="time", limit_direction="both")
        df["load"] = df["load"].ffill().bfill()

    quality_report["remaining_missing_load"] = int(df["load"].isna().sum())
    quality_report["cleaned_rows"] = len(df)

    df = add_time_features(df)
    return df, quality_report
