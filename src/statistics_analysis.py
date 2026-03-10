"""Basic and advanced statistical analysis for load data."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from scipy import stats


def basic_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate required basic statistics for load."""
    s = df["load"].dropna()
    q1, q2, q3 = s.quantile([0.25, 0.5, 0.75])

    result = pd.DataFrame(
        {
            "metric": [
                "count",
                "mean",
                "median",
                "min",
                "max",
                "std",
                "variance",
                "coefficient_of_variation",
                "skewness",
                "kurtosis",
                "q1",
                "q2",
                "q3",
            ],
            "value": [
                s.count(),
                s.mean(),
                s.median(),
                s.min(),
                s.max(),
                s.std(),
                s.var(),
                s.std() / s.mean() if s.mean() != 0 else float("nan"),
                stats.skew(s, bias=False),
                stats.kurtosis(s, fisher=True, bias=False),
                q1,
                q2,
                q3,
            ],
        }
    )
    return result


def peak_valley_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Daily peak-valley metrics and overall summary."""
    daily = (
        df.groupby("date")["load"]
        .agg(peak_load="max", valley_load="min", avg_load="mean", std_load="std")
        .reset_index()
    )
    daily["peak_valley_diff"] = daily["peak_load"] - daily["valley_load"]
    daily["peak_valley_ratio"] = daily["peak_load"] / daily["valley_load"].replace(0, pd.NA)
    return daily


def monthly_volatility(df: pd.DataFrame) -> pd.DataFrame:
    """Monthly variability comparison table."""
    out = (
        df.groupby(["year", "month"])["load"]
        .agg(monthly_mean="mean", monthly_std="std", monthly_var="var")
        .reset_index()
    )
    out["monthly_cv"] = out["monthly_std"] / out["monthly_mean"].replace(0, pd.NA)
    return out


def save_dataframe(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")
