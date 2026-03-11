"""项目主入口。"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch

from src.data_loader import choose_primary_dataset, load_csv_robust
from src.emd_decomposition import MAX_IMF, perform_emd, plot_emd_overview, plot_imf_components, save_imfs
from src.forecast_pipeline import ForecastConfig, run_tcn_forecast_comparison
from src.overload_analysis import analyze_heavy_overload
from src.preprocess import clean_load_data, infer_timestamp_and_load_columns
from src.season_analysis import create_season_figures, create_statistical_character_figures
from src.statistics_analysis import basic_statistics, monthly_volatility, save_dataframe
from src.time_scale_analysis import compute_time_scale_tables, create_required_time_scale_figures
from src.visualization import configure_style


def save_quality_report(report: dict, output_path: Path) -> None:
    quality_df = pd.DataFrame({"指标": list(report.keys()), "数值": list(report.values())})
    quality_df.to_csv(output_path, index=False, encoding="utf-8-sig")


def main() -> None:
    repo_root = Path(__file__).resolve().parent
    figures_dir = repo_root / "figures"
    outputs_dir = repo_root / "outputs"
    figures_dir.mkdir(exist_ok=True)
    outputs_dir.mkdir(exist_ok=True)

    configure_style()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("使用设备:", device)
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}")

    dataset_path = choose_primary_dataset(repo_root)
    raw_df, encoding = load_csv_robust(dataset_path)

    inferred = infer_timestamp_and_load_columns(raw_df)
    cleaned_df, quality = clean_load_data(raw_df, inferred.timestamp_col, inferred.load_col)
    save_dataframe(cleaned_df, outputs_dir / "清洗后数据.csv")
    save_quality_report(quality, outputs_dir / "数据质量报告.csv")

    metadata = pd.DataFrame(
        {
            "字段": ["数据集路径", "编码", "时间列", "负荷列"],
            "取值": [str(dataset_path.relative_to(repo_root)), encoding, inferred.timestamp_col, inferred.load_col],
        }
    )
    metadata.to_csv(outputs_dir / "数据集元信息.csv", index=False, encoding="utf-8-sig")

    basic_stats_df = basic_statistics(cleaned_df).rename(columns={"metric": "指标", "value": "数值"})
    save_dataframe(basic_stats_df, outputs_dir / "基础统计结果.csv")

    scale_tables = compute_time_scale_tables(cleaned_df)
    for name, table in scale_tables.items():
        save_dataframe(table, outputs_dir / f"{name}统计结果.csv")

    monthly_vol_df = monthly_volatility(cleaned_df).rename(
        columns={
            "year": "年份",
            "month": "月份",
            "monthly_mean": "月均负荷",
            "monthly_std": "月标准差",
            "monthly_var": "月方差",
            "monthly_cv": "月变异系数",
        }
    )
    save_dataframe(monthly_vol_df, outputs_dir / "月度波动性结果.csv")

    create_required_time_scale_figures(cleaned_df, figures_dir)
    create_season_figures(cleaned_df, figures_dir)
    create_statistical_character_figures(cleaned_df, figures_dir, outputs_dir)
    analyze_heavy_overload(cleaned_df, outputs_dir, figures_dir)

    imfs = perform_emd(cleaned_df["load"], max_imf=MAX_IMF)
    imf_df = save_imfs(imfs, cleaned_df["timestamp"], outputs_dir)
    plot_emd_overview(cleaned_df["load"], imfs, figures_dir)
    plot_imf_components(imf_df, figures_dir)

    forecast_config = ForecastConfig(
        lookback=672,
        train_ratio=0.8,
        dropout=0.1,
        learning_rate=1e-3,
        epochs=20,
        batch_size=128,
        random_seed=42,
        horizon=96,
    )
    metrics = run_tcn_forecast_comparison(cleaned_df, imf_df, outputs_dir, figures_dir, forecast_config)

    print("\n=== 数据质量报告 ===")
    for k, v in quality.items():
        print(f"{k}: {v}")

    print("\n=== 预测指标 ===")
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value:.6f}")

    print("\n流程完成：读取数据 → 清洗 → 特性分析 → EMD(最多10个IMF) → IMF自动选择 → TCN预测 → 分解/未分解对比 → 自动选优")


if __name__ == "__main__":
    main()
