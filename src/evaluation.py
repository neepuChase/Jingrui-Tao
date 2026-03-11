"""预测评估工具。"""

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


def save_metrics(metrics: dict[str, float], outputs_dir: Path, filename: str = "预测评估指标.csv") -> pd.DataFrame:
    metrics_df = pd.DataFrame({"指标": list(metrics.keys()), "数值": list(metrics.values())})
    metrics_df.to_csv(outputs_dir / filename, index=False, encoding="utf-8-sig")
    return metrics_df


def plot_forecast_results(
    timestamps: pd.Series,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    figures_dir: Path,
    title_suffix: str = "",
) -> None:
    error = y_true - y_pred

    fig, ax = plt.subplots(figsize=(14, 6), dpi=300)
    ax.plot(timestamps, y_true, label="真实值", linewidth=1.0)
    ax.plot(timestamps, y_pred, label="预测值", linewidth=1.0)
    ax.set_title(f"预测值与真实值对比图{title_suffix}")
    ax.set_xlabel("时间")
    ax.set_ylabel("负荷")
    ax.legend()
    ax.grid(True, alpha=0.3)
    save_figure(fig, figures_dir, "预测值与真实值对比图.png")

    fig, ax = plt.subplots(dpi=300)
    sns.histplot(error, bins=50, kde=True, ax=ax, color="tab:red")
    ax.set_title(f"预测误差分布图{title_suffix}")
    ax.set_xlabel("误差（真实值-预测值）")
    ax.set_ylabel("频数")
    save_figure(fig, figures_dir, "预测误差分布图.png")


def _save_error_stats(df: pd.DataFrame, output_path: Path) -> None:
    metrics = calculate_metrics(df["真实负荷"].values, df["预测负荷"].values)
    stats_df = pd.DataFrame(
        {
            "指标": ["MAE", "RMSE", "MAPE", "样本数量"],
            "数值": [metrics["MAE"], metrics["RMSE"], metrics["MAPE"], int(len(df))],
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
            raise RuntimeError("无法加载中国节假日库，请安装 holidays 或 chinese-calendar 包。") from exc


def generate_error_analysis_outputs(forecast_df: pd.DataFrame, outputs_dir: Path, figures_dir: Path) -> None:
    analysis_df = forecast_df.copy()
    analysis_df["时间戳"] = pd.to_datetime(analysis_df["时间戳"])
    analysis_df["预测误差"] = analysis_df["预测负荷"] - analysis_df["真实负荷"]
    analysis_df["小时"] = analysis_df["时间戳"].dt.hour

    peak_df = analysis_df[analysis_df["小时"].between(18, 21)].copy()
    valley_df = analysis_df[analysis_df["小时"].between(2, 4)].copy()

    holiday_mask = _is_cn_holiday(analysis_df["时间戳"])
    holiday_df = analysis_df.copy()
    holiday_df["日期类型"] = np.where(holiday_mask, "节假日", "非节假日")

    _save_error_stats(peak_df, outputs_dir / "高峰时段误差统计.csv")
    _save_error_stats(valley_df, outputs_dir / "低谷时段误差统计.csv")

    holiday_stats = []
    for label, group in holiday_df.groupby("日期类型"):
        metrics = calculate_metrics(group["真实负荷"].values, group["预测负荷"].values)
        holiday_stats.append({"类别": label, "MAE": metrics["MAE"], "RMSE": metrics["RMSE"], "MAPE": metrics["MAPE"], "样本数量": len(group)})
    pd.DataFrame(holiday_stats).to_csv(outputs_dir / "节假日误差统计.csv", index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    sns.boxplot(data=peak_df.assign(预测方案="高峰时段"), x="预测方案", y="预测误差", ax=ax)
    ax.set_title("高峰时段预测误差分布")
    ax.set_xlabel("预测方案")
    ax.set_ylabel("预测误差 (kW)")
    save_figure(fig, figures_dir, "高峰时段预测误差对比图.png")

    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    sns.boxplot(data=valley_df.assign(预测方案="低谷时段"), x="预测方案", y="预测误差", ax=ax)
    ax.set_title("低谷时段预测误差分布")
    ax.set_xlabel("预测方案")
    ax.set_ylabel("预测误差 (kW)")
    save_figure(fig, figures_dir, "低谷时段预测误差对比图.png")

    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    sns.boxplot(data=holiday_df, x="日期类型", y="预测误差", ax=ax)
    ax.set_title("节假日与非节假日预测误差对比")
    ax.set_xlabel("日期类型")
    ax.set_ylabel("预测误差 (kW)")
    save_figure(fig, figures_dir, "节假日预测误差对比图.png")

    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    sns.histplot(analysis_df["预测误差"], bins=50, kde=True, color="tab:blue", ax=ax)
    ax.set_title("预测误差直方图")
    ax.set_xlabel("预测误差 (kW)")
    ax.set_ylabel("频数")
    save_figure(fig, figures_dir, "误差直方图.png")

    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    stats.probplot(analysis_df["预测误差"].values, dist="norm", plot=ax)
    ax.set_title("预测误差QQ图")
    ax.set_xlabel("理论分位数")
    ax.set_ylabel("样本分位数")
    save_figure(fig, figures_dir, "误差QQ图.png")
