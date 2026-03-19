"""Forecast evaluation utilities."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from src.visualization import save_figure


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    denominator = np.where(np.abs(y_true) < 1e-8, np.nan, y_true)
    mape = float(np.nanmean(np.abs((y_true - y_pred) / denominator)) * 100)

    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape, "R2": r2}


def save_metrics(metrics: dict[str, float], outputs_dir: Path, filename: str = "forecast_evaluation_metrics.csv") -> pd.DataFrame:
    metrics_df = pd.DataFrame({"metric": list(metrics.keys()), "value": list(metrics.values())})
    metrics_df.to_csv(outputs_dir / filename, index=False, encoding="utf-8-sig")
    return metrics_df


def _prompt_plot_date(timestamps: pd.Series) -> pd.Timestamp:
    normalized_dates = pd.to_datetime(timestamps).dt.normalize()
    available_dates = pd.Index(normalized_dates.unique()).sort_values()
    start_date = available_dates.min().strftime("%Y-%m-%d")
    end_date = available_dates.max().strftime("%Y-%m-%d")

    while True:
        user_input = input(f"请输入要生成预测对比图的日期（YYYY-MM-DD，范围 {start_date} ~ {end_date}）: ").strip()
        try:
            selected_date = pd.Timestamp(user_input).normalize()
        except ValueError:
            print("日期格式无效，请按 YYYY-MM-DD 重新输入。")
            continue

        if selected_date in available_dates:
            return selected_date

        print("该日期不在预测结果范围内，请重新输入。")


def plot_forecast_results(
    timestamps: pd.Series,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    figures_dir: Path,
    title_suffix: str = "",
) -> None:
    timestamps = pd.to_datetime(pd.Series(timestamps).reset_index(drop=True))
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    error = y_true - y_pred

    selected_date = _prompt_plot_date(timestamps)
    day_mask = timestamps.dt.normalize() == selected_date
    day_timestamps = timestamps.loc[day_mask]
    day_true = y_true[day_mask.to_numpy()]
    day_pred = y_pred[day_mask.to_numpy()]
    date_label = selected_date.strftime("%Y-%m-%d")

    fig, ax = plt.subplots(figsize=(14, 6), dpi=300)
    ax.plot(day_timestamps, day_true, label="Actual Load", linewidth=1.0)
    ax.plot(day_timestamps, day_pred, label="Predicted Load", linewidth=1.0)
    ax.set_title(f"Actual vs Predicted Load ({date_label}){title_suffix}")
    ax.set_xlabel("Time")
    ax.set_ylabel("Load (MW)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    save_figure(fig, figures_dir, "actual_vs_predicted_load.png")

    fig, ax = plt.subplots(dpi=300)
    sns.histplot(error, bins=50, kde=True, ax=ax, color="tab:red")
    ax.set_title(f"Prediction Error Distribution{title_suffix}")
    ax.set_xlabel("Error (Actual - Predicted)")
    ax.set_ylabel("Frequency")
    save_figure(fig, figures_dir, "prediction_error_distribution.png")


def _save_error_stats(df: pd.DataFrame, output_path: Path) -> None:
    metrics = calculate_metrics(df["actual_load"].values, df["predicted_load"].values)
    stats_df = pd.DataFrame(
        {
            "metric": ["MAE", "RMSE", "MAPE", "sample_count"],
            "value": [metrics["MAE"], metrics["RMSE"], metrics["MAPE"], int(len(df))],
        }
    )
    stats_df.to_csv(output_path, index=False, encoding="utf-8-sig")


def _is_cn_holiday(dates: pd.Series) -> pd.Series:
    try:
        import holidays

        years = sorted(set(dates.dt.year.tolist()))
        cn_holidays = holidays.country_holidays("CN", years=years)
        return dates.dt.date.astype("object").isin(cn_holidays)
    except Exception:
        try:
            import chinese_calendar as calendar

            return dates.dt.date.apply(calendar.is_holiday)
        except Exception as exc:  # pragma: no cover - fallback path
            raise RuntimeError("Unable to load China holiday libraries. Install holidays or chinese-calendar.") from exc


def generate_error_analysis_outputs(forecast_df: pd.DataFrame, outputs_dir: Path, figures_dir: Path) -> None:
    analysis_df = forecast_df.copy()
    analysis_df["timestamp"] = pd.to_datetime(analysis_df["timestamp"])
    analysis_df["error"] = analysis_df["predicted_load"] - analysis_df["actual_load"]
    analysis_df["hour"] = analysis_df["timestamp"].dt.hour

    peak_df = analysis_df[analysis_df["hour"].between(18, 21)].copy()
    valley_df = analysis_df[analysis_df["hour"].between(2, 4)].copy()

    holiday_mask = _is_cn_holiday(analysis_df["timestamp"])
    holiday_df = analysis_df.copy()
    holiday_df["day_type"] = np.where(holiday_mask, "Holiday", "Non-Holiday")

    _save_error_stats(peak_df, outputs_dir / "peak_period_error_stats.csv")
    _save_error_stats(valley_df, outputs_dir / "valley_period_error_stats.csv")

    holiday_stats = []
    for label, group in holiday_df.groupby("day_type"):
        metrics = calculate_metrics(group["actual_load"].values, group["predicted_load"].values)
        holiday_stats.append({"category": label, "MAE": metrics["MAE"], "RMSE": metrics["RMSE"], "MAPE": metrics["MAPE"], "sample_count": len(group)})
    pd.DataFrame(holiday_stats).to_csv(outputs_dir / "holiday_error_stats.csv", index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    sns.boxplot(data=peak_df.assign(period="Peak Period"), x="period", y="error", ax=ax)
    ax.set_title("Peak Period Error")
    ax.set_xlabel("Forecast Segment")
    ax.set_ylabel("Prediction Error (kW)")
    save_figure(fig, figures_dir, "peak_period_error.png")

    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    sns.boxplot(data=valley_df.assign(period="Valley Period"), x="period", y="error", ax=ax)
    ax.set_title("Valley Period Error")
    ax.set_xlabel("Forecast Segment")
    ax.set_ylabel("Prediction Error (kW)")
    save_figure(fig, figures_dir, "valley_period_error.png")

    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    sns.boxplot(data=holiday_df, x="day_type", y="error", ax=ax)
    ax.set_title("Holiday Prediction Error")
    ax.set_xlabel("Day Type")
    ax.set_ylabel("Prediction Error (kW)")
    save_figure(fig, figures_dir, "holiday_prediction_error.png")

    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    sns.histplot(analysis_df["error"], bins=50, kde=True, color="tab:blue", ax=ax)
    ax.set_title("Error Histogram")
    ax.set_xlabel("Error (kW)")
    ax.set_ylabel("Frequency")
    save_figure(fig, figures_dir, "error_histogram.png")

    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    stats.probplot(analysis_df["error"].values, dist="norm", plot=ax)
    ax.set_title("Error QQ Plot")
    ax.set_xlabel("Theoretical Quantiles")
    ax.set_ylabel("Sample Quantiles")
    save_figure(fig, figures_dir, "error_qq_plot.png")
