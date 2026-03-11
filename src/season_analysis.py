"""季节性分析绘图。"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.statistics_analysis import peak_valley_metrics
from src.visualization import save_figure

SEASON_MAP = {"Spring": "春季", "Summer": "夏季", "Autumn": "秋季", "Winter": "冬季"}


def create_season_figures(df: pd.DataFrame, figures_dir: Path) -> None:
    season_order = ["Spring", "Summer", "Autumn", "Winter"]

    seasonal_hour = df.groupby(["season", "hour"])["load"].mean().reset_index()
    fig, ax = plt.subplots()
    for season in season_order:
        grp = seasonal_hour[seasonal_hour["season"] == season]
        if not grp.empty:
            ax.plot(grp["hour"], grp["load"], marker="o", label=SEASON_MAP.get(season, season))
    ax.set_title("季节性负荷分析图")
    ax.set_xlabel("小时")
    ax.set_ylabel("平均负荷")
    ax.legend()
    save_figure(fig, figures_dir, "季节性负荷分析图.png")


def create_statistical_character_figures(df: pd.DataFrame, figures_dir: Path, outputs_dir: Path) -> None:
    fig, ax = plt.subplots()
    sns.histplot(df["load"], bins=50, kde=True, ax=ax, color="tab:blue")
    ax.set_title("负荷分布直方图")
    ax.set_xlabel("负荷")
    ax.set_ylabel("频数")
    save_figure(fig, figures_dir, "负荷分布直方图.png")

    sorted_load = np.sort(df["load"].dropna().values)
    ecdf = np.arange(1, len(sorted_load) + 1) / len(sorted_load)
    fig, ax = plt.subplots()
    ax.plot(sorted_load, ecdf, color="tab:purple")
    ax.set_title("负荷经验分布函数图")
    ax.set_xlabel("负荷")
    ax.set_ylabel("累积分布")
    ax.grid(True, alpha=0.3)
    save_figure(fig, figures_dir, "负荷经验分布函数图.png")

    df_sorted = df.sort_values("timestamp").copy()
    df_sorted["ramp"] = df_sorted["load"].diff()
    fig, ax = plt.subplots()
    sns.histplot(df_sorted["ramp"].dropna(), bins=60, kde=True, ax=ax, color="tab:red")
    ax.set_title("负荷爬坡分布图")
    ax.set_xlabel("负荷变化量")
    ax.set_ylabel("频数")
    save_figure(fig, figures_dir, "负荷爬坡分布图.png")

    daily_peak_valley = peak_valley_metrics(df_sorted)
    daily_peak_valley.rename(
        columns={
            "date": "日期",
            "peak_load": "峰值负荷",
            "valley_load": "谷值负荷",
            "avg_load": "平均负荷",
            "std_load": "负荷标准差",
            "peak_valley_diff": "峰谷差",
            "peak_valley_ratio": "峰谷比",
        }
    ).to_csv(outputs_dir / "日峰谷特征指标.csv", index=False, encoding="utf-8-sig")
