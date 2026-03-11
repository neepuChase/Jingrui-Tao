"""多时间尺度分析与绘图。"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.visualization import save_figure


def compute_time_scale_tables(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    tables: dict[str, pd.DataFrame] = {}
    tables["年度"] = df.groupby("year")["load"].describe().reset_index()
    tables["月度"] = df.groupby(["year", "month"])["load"].describe().reset_index()
    tables["周度"] = df.assign(year_week=df["timestamp"].dt.strftime("%Y-%W")).groupby("year_week")["load"].describe().reset_index()
    tables["日度"] = df.groupby("date")["load"].describe().reset_index()
    tables["小时"] = df.groupby("hour")["load"].describe().reset_index()
    return tables


def create_required_time_scale_figures(df: pd.DataFrame, figures_dir: Path) -> None:
    fig, ax = plt.subplots()
    ax.plot(df["timestamp"], df["load"], linewidth=0.8)
    ax.set_title("原始负荷时序图")
    ax.set_xlabel("时间")
    ax.set_ylabel("负荷")
    ax.grid(True, alpha=0.3)
    save_figure(fig, figures_dir, "原始负荷时序图.png")

    monthly_avg = df.groupby("month")["load"].mean().reindex(range(1, 13))
    fig, ax = plt.subplots()
    monthly_avg.plot(kind="bar", ax=ax, color="tab:blue")
    ax.set_title("月平均负荷图")
    ax.set_xlabel("月份")
    ax.set_ylabel("平均负荷")
    save_figure(fig, figures_dir, "月平均负荷图.png")

    fig, ax = plt.subplots()
    sns.boxplot(data=df, x="month", y="load", ax=ax)
    ax.set_title("月负荷箱线图")
    ax.set_xlabel("月份")
    ax.set_ylabel("负荷")
    save_figure(fig, figures_dir, "月负荷箱线图.png")

    profile_weekpart = df.groupby(["is_weekend", "hour"])["load"].mean().reset_index()
    fig, ax = plt.subplots()
    for is_weekend, grp in profile_weekpart.groupby("is_weekend"):
        label = "周末" if is_weekend else "工作日"
        ax.plot(grp["hour"], grp["load"], marker="o", label=label)
    ax.set_title("工作日与周末对比图")
    ax.set_xlabel("小时")
    ax.set_ylabel("平均负荷")
    ax.legend()
    save_figure(fig, figures_dir, "工作日与周末对比图.png")
