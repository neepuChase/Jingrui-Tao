"""Basic and advanced statistical analysis for load data."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from src.visualization import save_figure


LAG_VALUES = (1, 24, 168)
WEATHER_FEATURE_CANDIDATES = {
    "temperature": ["平均温度", "avg_temp", "average_temp", "temperature", "temp", "气温", "温度"],
    "humidity": ["相对湿度", "humidity", "humid", "湿度"],
    "rainfall": ["降雨量", "rainfall", "precipitation", "rain", "降水"],
}
WEATHER_FEATURE_LABELS = {
    "temperature": "Temperature",
    "humidity": "Humidity",
    "rainfall": "Rainfall",
}


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


def create_difference_and_correlation_figures(df: pd.DataFrame, figures_dir: Path) -> None:
    """Create difference, correlation, lag, and FFT figures for the load series."""
    df_sorted = df.sort_values("timestamp").copy()
    df_sorted["ramp"] = df_sorted["load"].diff()
    df_sorted["diff2"] = df_sorted["ramp"].diff()

    _plot_difference_timeseries(df_sorted, figures_dir, column="ramp", filename="diff1_timeseries.png", title="First-Order Difference Time Series")
    _plot_difference_distribution(df_sorted, figures_dir, column="ramp", filename="diff1_distribution.png", title="First-Order Difference Distribution")
    _plot_difference_timeseries(df_sorted, figures_dir, column="diff2", filename="diff2_timeseries.png", title="Second-Order Difference Time Series")
    _plot_difference_distribution(df_sorted, figures_dir, column="diff2", filename="diff2_distribution.png", title="Second-Order Difference Distribution")

    _plot_acf_pacf(df_sorted["load"], figures_dir)
    _plot_lag_scatter(df_sorted, figures_dir)
    _plot_correlation_heatmap(df_sorted, figures_dir)
    _plot_weather_correlation(df_sorted, figures_dir)
    _plot_fft_spectrum(df_sorted, figures_dir)


def _plot_difference_timeseries(
    df: pd.DataFrame,
    figures_dir: Path,
    *,
    column: str,
    filename: str,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(df["timestamp"], df[column], linewidth=0.8, color="tab:blue")
    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("diff(load)")
    ax.grid(True, alpha=0.3)
    save_figure(fig, figures_dir, filename)


def _plot_difference_distribution(
    df: pd.DataFrame,
    figures_dir: Path,
    *,
    column: str,
    filename: str,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.histplot(df[column].dropna(), bins=60, kde=True, ax=ax, color="tab:orange")
    ax.set_title(title)
    ax.set_xlabel(column)
    ax.set_ylabel("Frequency")
    save_figure(fig, figures_dir, filename)


def _compute_max_lags(series: pd.Series, preferred: int = 100) -> int:
    available = series.dropna().shape[0]
    if available <= 2:
        return 1
    return max(1, min(preferred, available // 2 - 1))


def _plot_acf_pacf(series: pd.Series, figures_dir: Path) -> None:
    clean_series = series.dropna()
    max_lags = _compute_max_lags(clean_series)

    fig, ax = plt.subplots(figsize=(12, 6))
    plot_acf(clean_series, lags=max_lags, ax=ax, zero=False)
    ax.set_title(f"Autocorrelation Function (lags={max_lags})")
    ax.set_xlabel("Lag")
    ax.set_ylabel("ACF")
    save_figure(fig, figures_dir, "acf_plot.png")

    fig, ax = plt.subplots(figsize=(12, 6))
    pacf_lags = max(1, min(max_lags, clean_series.shape[0] // 2 - 1))
    plot_pacf(clean_series, lags=pacf_lags, ax=ax, method="ywm", zero=False)
    ax.set_title(f"Partial Autocorrelation Function (lags={pacf_lags})")
    ax.set_xlabel("Lag")
    ax.set_ylabel("PACF")
    save_figure(fig, figures_dir, "pacf_plot.png")


def _plot_lag_scatter(df: pd.DataFrame, figures_dir: Path) -> None:
    fig, axes = plt.subplots(1, len(LAG_VALUES), figsize=(18, 5))

    for ax, lag in zip(axes, LAG_VALUES):
        lag_df = pd.DataFrame({"current": df["load"], "lagged": df["load"].shift(lag)}).dropna()
        ax.scatter(lag_df["lagged"], lag_df["current"], s=10, alpha=0.4, color="tab:green")
        ax.set_title(f"load(t) vs load(t-{lag})")
        ax.set_xlabel(f"load(t-{lag})")
        ax.set_ylabel("load(t)")
        ax.grid(True, alpha=0.2)

    save_figure(fig, figures_dir, "lag_scatter_plots.png")


def _plot_correlation_heatmap(df: pd.DataFrame, figures_dir: Path) -> None:
    feature_df = pd.DataFrame(
        {
            "load(t)": df["load"],
            "load(t-1)": df["load"].shift(1),
            "load(t-24)": df["load"].shift(24),
            "load(t-168)": df["load"].shift(168),
            "hour": df["hour"],
            "weekday": df["weekday"],
        }
    ).dropna()

    corr = feature_df.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True, ax=ax)
    ax.set_title("Feature Correlation Heatmap")
    save_figure(fig, figures_dir, "correlation_heatmap.png")


def _match_weather_feature_columns(df: pd.DataFrame) -> dict[str, str]:
    matched: dict[str, str] = {}
    normalized = {col: str(col).strip().lower().replace(" ", "") for col in df.columns}

    for feature_name, keywords in WEATHER_FEATURE_CANDIDATES.items():
        for keyword in keywords:
            normalized_keyword = keyword.strip().lower().replace(" ", "")
            for col, norm_col in normalized.items():
                if col in {"timestamp", "load"}:
                    continue
                if normalized_keyword in norm_col:
                    matched[feature_name] = col
                    break
            if feature_name in matched:
                break

    return matched


def _plot_weather_correlation(df: pd.DataFrame, figures_dir: Path) -> None:
    matched_columns = _match_weather_feature_columns(df)
    if len(matched_columns) < 3:
        return

    plot_df = pd.DataFrame(
        {WEATHER_FEATURE_LABELS[name]: pd.to_numeric(df[column], errors="coerce") for name, column in matched_columns.items()}
    )
    plot_df.insert(0, "Load", pd.to_numeric(df["load"], errors="coerce"))
    plot_df = plot_df.dropna()
    if plot_df.empty:
        return

    corr_series = plot_df.corr(numeric_only=True).loc["Load", ["Temperature", "Humidity", "Rainfall"]]

    fig, ax = plt.subplots(figsize=(9, 6))
    colors = ["#d62728" if value >= 0 else "#1f77b4" for value in corr_series]
    bars = ax.bar(corr_series.index, corr_series.values, color=colors, width=0.55)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylim(-1.0, 1.0)
    ax.set_ylabel("Pearson Correlation")
    ax.set_title("Load Correlation with Temperature, Humidity, and Rainfall")
    ax.grid(axis="y", alpha=0.3)

    for bar, value in zip(bars, corr_series.values):
        offset = 0.03 if value >= 0 else -0.06
        va = "bottom" if value >= 0 else "top"
        ax.text(bar.get_x() + bar.get_width() / 2, value + offset, f"{value:.2f}", ha="center", va=va)

    save_figure(fig, figures_dir, "load_weather_correlation.svg")


def _infer_sampling_hours(timestamps: pd.Series) -> float:
    diffs = timestamps.sort_values().diff().dropna()
    if diffs.empty:
        return 1.0
    step_seconds = diffs.dt.total_seconds().median()
    if pd.isna(step_seconds) or step_seconds <= 0:
        return 1.0
    return step_seconds / 3600.0


def _plot_fft_spectrum(df: pd.DataFrame, figures_dir: Path) -> None:
    signal = df["load"].astype(float).to_numpy()
    signal = signal - np.mean(signal)
    n = signal.size
    if n == 0:
        return

    sampling_hours = _infer_sampling_hours(df["timestamp"])
    fft_values = np.fft.fft(signal)
    frequencies = np.fft.fftfreq(n, d=sampling_hours)
    positive = frequencies > 0

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(frequencies[positive], np.abs(fft_values[positive]), color="tab:purple", linewidth=0.9)
    ax.set_title("FFT Spectrum of Load Signal")
    ax.set_xlabel("Frequency (cycles/hour)")
    ax.set_ylabel("Amplitude")
    ax.grid(True, alpha=0.3)
    save_figure(fig, figures_dir, "fft_spectrum.png")


def save_dataframe(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")
