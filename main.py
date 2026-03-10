"""Main entrypoint for power load characteristic analysis project."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.data_loader import choose_primary_dataset, load_csv_robust
from src.preprocess import clean_load_data, infer_timestamp_and_load_columns
from src.season_analysis import create_season_figures, create_statistical_character_figures
from src.emd_decomposition import perform_emd, plot_emd_overview, plot_imf_components, save_imfs
from src.forecast_pipeline import ForecastConfig, run_imf_lstm_forecast
from src.statistics_analysis import basic_statistics, monthly_volatility, save_dataframe
from src.time_scale_analysis import compute_time_scale_tables, create_required_time_scale_figures
from src.visualization import configure_style, save_figure


def save_quality_report(report: dict, output_path: Path) -> None:
    quality_df = pd.DataFrame({"item": list(report.keys()), "value": list(report.values())})
    quality_df.to_csv(output_path, index=False, encoding="utf-8-sig")


def main() -> None:
    repo_root = Path(__file__).resolve().parent
    data_dir = repo_root / "data"
    figures_dir = repo_root / "figures"
    outputs_dir = repo_root / "outputs"

    data_dir.mkdir(exist_ok=True)
    figures_dir.mkdir(exist_ok=True)
    outputs_dir.mkdir(exist_ok=True)

    configure_style()

    dataset_path = choose_primary_dataset(repo_root)
    raw_df, encoding = load_csv_robust(dataset_path)

    inferred = infer_timestamp_and_load_columns(raw_df)
    cleaned_df, quality = clean_load_data(raw_df, inferred.timestamp_col, inferred.load_col)

    # Save cleaned dataset and metadata
    save_dataframe(cleaned_df, outputs_dir / "cleaned_data.csv")
    save_quality_report(quality, outputs_dir / "data_quality_report.csv")

    metadata = pd.DataFrame(
        {
            "field": ["dataset_path", "encoding", "timestamp_column", "load_column"],
            "value": [str(dataset_path.relative_to(repo_root)), encoding, inferred.timestamp_col, inferred.load_col],
        }
    )
    metadata.to_csv(outputs_dir / "dataset_metadata.csv", index=False, encoding="utf-8-sig")

    # Basic statistics
    basic_stats_df = basic_statistics(cleaned_df)
    save_dataframe(basic_stats_df, outputs_dir / "basic_statistics.csv")

    # Multi-time-scale tables
    scale_tables = compute_time_scale_tables(cleaned_df)
    for name, table in scale_tables.items():
        save_dataframe(table, outputs_dir / f"{name}_statistics.csv")

    # Monthly/seasonal variability tables
    monthly_vol_df = monthly_volatility(cleaned_df)
    save_dataframe(monthly_vol_df, outputs_dir / "monthly_volatility.csv")

    season_vol = (
        cleaned_df.groupby("season")["load"].agg(season_mean="mean", season_std="std", season_var="var").reset_index()
    )
    season_vol["season_cv"] = season_vol["season_std"] / season_vol["season_mean"].replace(0, pd.NA)
    save_dataframe(season_vol, outputs_dir / "seasonal_volatility.csv")

    # Required and extra figures
    create_required_time_scale_figures(cleaned_df, figures_dir)
    create_season_figures(cleaned_df, figures_dir)
    create_statistical_character_figures(cleaned_df, figures_dir, outputs_dir)

    # 18. Monthly volatility comparison figure
    fig, ax = plt.subplots()
    sns.barplot(data=monthly_vol_df, x="month", y="monthly_std", ax=ax, color="tab:cyan")
    ax.set_title("Monthly Volatility Comparison (Std Dev)")
    ax.set_xlabel("Month")
    ax.set_ylabel("Monthly Std Dev of Load")
    save_figure(fig, figures_dir, "18_monthly_volatility_comparison.png")


    # EMD decomposition + IMF-wise LSTM forecasting + reconstruction
    imfs = perform_emd(cleaned_df["load"])
    imf_df = save_imfs(imfs, cleaned_df["timestamp"], outputs_dir)
    plot_emd_overview(cleaned_df["load"], imfs, figures_dir)
    plot_imf_components(imf_df, figures_dir)

    forecast_config = ForecastConfig(
        lookback=96,
        train_ratio=0.8,
        hidden_size=32,
        num_layers=1,
        dropout=0.0,
        learning_rate=1e-3,
        epochs=20,
        batch_size=64,
        random_seed=42,
    )
    forecast_metrics = run_imf_lstm_forecast(cleaned_df, imf_df, outputs_dir, figures_dir, forecast_config)

    # Console summary report for quick inspection
    print("=== Data Quality Report ===")
    for k, v in quality.items():
        print(f"{k}: {v}")
    print("\n=== Inferred Columns ===")
    print(f"Timestamp column: {inferred.timestamp_col}")
    print(f"Load column: {inferred.load_col}")
    print(f"Detected encoding: {encoding}")
    print(f"Dataset path: {dataset_path}")
    print("\n=== Forecast Metrics ===")
    for metric_name, metric_value in forecast_metrics.items():
        print(f"{metric_name}: {metric_value:.6f}")
    print("\nAnalysis complete. See outputs/ and figures/ for results.")


if __name__ == "__main__":
    main()
