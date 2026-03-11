"""Project entry point."""

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


IMF_COMPONENTS = 3


def save_quality_report(report: dict, output_path: Path) -> None:
    quality_df = pd.DataFrame({"metric": list(report.keys()), "value": list(report.values())})
    quality_df.to_csv(output_path, index=False, encoding="utf-8-sig")


def main() -> None:
    repo_root = Path(__file__).resolve().parent
    figures_dir = repo_root / "figures"
    outputs_dir = repo_root / "outputs"
    figures_dir.mkdir(exist_ok=True)
    outputs_dir.mkdir(exist_ok=True)

    configure_style()

    if not 1 <= IMF_COMPONENTS <= 10:
        raise ValueError("IMF_COMPONENTS 必须在 1-10 之间")
    print(f"使用 IMF 分量数: {IMF_COMPONENTS}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}")

    dataset_path = choose_primary_dataset(repo_root)
    raw_df, encoding = load_csv_robust(dataset_path)

    inferred = infer_timestamp_and_load_columns(raw_df)
    cleaned_df, quality = clean_load_data(raw_df, inferred.timestamp_col, inferred.load_col)
    save_dataframe(cleaned_df, outputs_dir / "cleaned_data.csv")
    save_quality_report(quality, outputs_dir / "data_quality_report.csv")

    metadata = pd.DataFrame(
        {
            "field": ["dataset_path", "encoding", "timestamp_column", "load_column"],
            "value": [str(dataset_path.relative_to(repo_root)), encoding, inferred.timestamp_col, inferred.load_col],
        }
    )
    metadata.to_csv(outputs_dir / "dataset_metadata.csv", index=False, encoding="utf-8-sig")

    basic_stats_df = basic_statistics(cleaned_df)
    save_dataframe(basic_stats_df, outputs_dir / "basic_statistics.csv")

    scale_tables = compute_time_scale_tables(cleaned_df)
    for name, table in scale_tables.items():
        save_dataframe(table, outputs_dir / f"{name}_statistics.csv")

    monthly_vol_df = monthly_volatility(cleaned_df).rename(
        columns={
            "year": "year",
            "month": "month",
            "monthly_mean": "monthly_mean",
            "monthly_std": "monthly_std",
            "monthly_var": "monthly_var",
            "monthly_cv": "monthly_cv",
        }
    )
    save_dataframe(monthly_vol_df, outputs_dir / "monthly_volatility_statistics.csv")

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
        imf_components=IMF_COMPONENTS,
    )
    metrics = run_tcn_forecast_comparison(cleaned_df, imf_df, outputs_dir, figures_dir, forecast_config)

    print("\n=== Data Quality Report ===")
    for k, v in quality.items():
        print(f"{k}: {v}")

    print("\n=== Forecast Metrics ===")
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value:.6f}")

    print("\nPipeline completed: load data → clean → feature analysis → EMD (up to 10 IMF) → user-defined IMF decomposition forecast → decomposition/non-decomposition comparison.")


if __name__ == "__main__":
    main()
