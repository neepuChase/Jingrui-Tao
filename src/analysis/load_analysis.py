from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.season_analysis import create_season_figures, create_statistical_character_figures
from src.statistics_analysis import (
    basic_statistics,
    create_difference_and_correlation_figures,
    monthly_volatility,
    peak_valley_metrics,
    save_dataframe,
)
from src.time_scale_analysis import compute_time_scale_tables, create_required_time_scale_figures


def run_load_analysis(df: pd.DataFrame, output_dir: Path, selection_description: str) -> dict[str, Path]:
    figures_dir = output_dir / "figures"
    tables_dir = output_dir / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    summary_df = pd.DataFrame(
        {
            "field": ["start_time", "end_time", "rows", "selection"],
            "value": [str(df["timestamp"].min()), str(df["timestamp"].max()), len(df), selection_description],
        }
    )
    save_dataframe(summary_df, tables_dir / "analysis_scope.csv")
    save_dataframe(df, tables_dir / "filtered_load_data.csv")
    save_dataframe(basic_statistics(df), tables_dir / "basic_statistics.csv")
    save_dataframe(monthly_volatility(df), tables_dir / "monthly_volatility.csv")
    save_dataframe(peak_valley_metrics(df), tables_dir / "daily_peak_valley_metrics.csv")

    scale_tables = compute_time_scale_tables(df)
    for name, table in scale_tables.items():
        save_dataframe(table, tables_dir / f"{name}_statistics.csv")

    create_required_time_scale_figures(df, figures_dir)
    create_season_figures(df, figures_dir)
    create_statistical_character_figures(df, figures_dir, tables_dir)
    create_difference_and_correlation_figures(df, figures_dir)

    return {"root": output_dir, "figures": figures_dir, "tables": tables_dir}
