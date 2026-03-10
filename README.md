# Power Load Characteristic Analysis

This project provides a complete, beginner-friendly Python workflow to analyze power load time-series data from a CSV file.

It automatically:
- finds the uploaded CSV file,
- detects encoding (including Chinese-compatible encodings),
- infers timestamp/load columns (supports Chinese column names),
- cleans data,
- runs multi-time-scale + seasonal + statistical analysis,
- saves all figures and tables for reproducible reporting.

## Project Structure

```text
.
├── data/                     # Place source CSV data here (auto-detected)
├── figures/                  # Generated figures
├── outputs/                  # Generated tables and cleaned data
├── src/
│   ├── data_loader.py        # CSV discovery + robust loading
│   ├── preprocess.py         # Column inference + cleaning + time features
│   ├── statistics_analysis.py# Statistics and variability metrics
│   ├── time_scale_analysis.py# Year/month/week/day/hour analysis + figures
│   ├── season_analysis.py    # Seasonal and distribution/ramp analysis
│   └── visualization.py      # Plot style and save helper
├── main.py                   # One-click entrypoint
├── requirements.txt
└── README.md
```

## Installation

```bash
python -m pip install -r requirements.txt
```

## How to Run

```bash
python main.py
```

## What the Pipeline Does

1. **Data loading and preprocessing**
   - Automatically finds CSV in repo (prefers `data/`).
   - Detects encoding (`utf-8`, `gb18030`, etc.).
   - Infers timestamp and load columns using both English and Chinese keywords.
   - Converts timestamp to datetime, sorts by time.
   - Handles missing values and duplicated timestamps.
   - Outputs:
     - `outputs/cleaned_data.csv`
     - `outputs/data_quality_report.csv`
     - `outputs/dataset_metadata.csv`

2. **Basic statistical analysis**
   - Computes count, mean, median, min, max, std, variance, CV, skewness, kurtosis, quartiles.
   - Output:
     - `outputs/basic_statistics.csv`

3. **Multi-time-scale analysis**
   - Yearly / monthly / weekly / daily / hourly descriptive statistics.
   - Outputs:
     - `outputs/yearly_statistics.csv`
     - `outputs/monthly_statistics.csv`
     - `outputs/weekly_statistics.csv`
     - `outputs/daily_statistics.csv`
     - `outputs/hourly_statistics.csv`

4. **Seasonal and statistical characteristic analysis**
   - Seasonal comparison by Spring/Summer/Autumn/Winter.
   - Distribution analysis (Histogram+KDE, ECDF).
   - Volatility and ramp analysis.
   - Peak-valley analysis and daily trends.
   - Outputs:
     - `outputs/monthly_volatility.csv`
     - `outputs/seasonal_volatility.csv`
     - `outputs/daily_peak_valley_metrics.csv`

## Required Figures Produced

The project generates at least these required figures in `figures/`:
1. `01_raw_load_time_series.png`
2. `02_rolling_mean_trend.png`
3. `03_monthly_average_load.png`
4. `04_monthly_boxplot.png`
5. `05_seasonal_boxplot.png`
6. `06_seasonal_average_24h_curves.png`
7. `07_weekday_vs_weekend_24h_curves.png`
8. `08_day_of_week_average_curves.png`
9. `09_hourly_average_load_curve.png`
10. `10_histogram_with_kde.png`
11. `11_ecdf_plot.png`
12. `12_month_hour_heatmap.png`
13. `13_weekday_hour_heatmap.png`
14. `14_daily_peak_load_trend.png`
15. `15_daily_valley_load_trend.png`
16. `16_daily_peak_valley_difference_trend.png`
17. `17_load_ramp_distribution.png`
18. `18_monthly_volatility_comparison.png`

Additional figure:
- `19_daily_load_curve_samples.png`
- `20_seasonal_distribution_kde.png`

## Notes on Non-standard Column Names

If column names are non-standard (including Chinese names), the code auto-detects:
- timestamp columns by keywords and datetime parse success,
- load columns by keywords and numeric parse success.

The inferred columns are saved to `outputs/dataset_metadata.csv`.
