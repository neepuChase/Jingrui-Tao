# Power Load Characteristic Analysis + EMD-LSTM Forecasting

This project is a complete, beginner-friendly Python workflow for **power load time-series analysis and forecasting**.

It includes two major parts:
1. **Load characteristic analysis** (multi-time-scale, seasonal, statistical).
2. **EMD decomposition + IMF-wise LSTM forecasting + reconstruction**.

The pipeline is designed for **CPU-only environments** and does **not require CUDA/GPU**.

---

## End-to-End Workflow

```text
Load data
-> Preprocessing
-> Characteristic analysis
-> EMD decomposition
-> IMF-wise LSTM forecasting
-> Reconstruction of final prediction
-> Evaluation + visualization
```

---

## Project Structure

```text
.
├── data/                       # Input CSV data (auto-detected)
├── figures/                    # Generated figures
├── outputs/                    # Generated tables/results
├── src/
│   ├── data_loader.py          # CSV discovery + robust loading
│   ├── preprocess.py           # Column inference + cleaning + time features
│   ├── statistics_analysis.py  # Basic statistics + volatility metrics
│   ├── time_scale_analysis.py  # Multi-time-scale analysis + plots
│   ├── season_analysis.py      # Seasonal + statistical characteristic plots
│   ├── emd_decomposition.py    # EMD decomposition and IMF plotting
│   ├── lstm_dataset.py         # Sequence dataset utilities for LSTM
│   ├── lstm_model.py           # LSTM model definition
│   ├── forecast_pipeline.py    # IMF-wise training, prediction, reconstruction
│   ├── evaluation.py           # Metrics and forecast evaluation plots
│   └── visualization.py        # Unified plotting style and save helper
├── main.py
├── requirements.txt
└── README.md
```

---

## Installation

```bash
python -m pip install -r requirements.txt
```

### Notes on EMD package
- The code imports `PyEMD` via:
  ```python
  from PyEMD import EMD
  ```
- In `requirements.txt`, the corresponding package is `EMD-signal`, which is a commonly used and stable installation source for `PyEMD`.

### Notes on CPU-only PyTorch
- This project uses `torch` in CPU mode only.
- Code explicitly sets:
  - `device = torch.device("cpu")`
- No CUDA-specific code is used.

---

## Run

```bash
python main.py
```

---

## Input Data Handling

The pipeline automatically:
- Detects CSV files in the repository (prefers `data/`).
- Detects encoding (`utf-8`, `utf-8-sig`, `gb18030`, `gbk`, etc.).
- Infers timestamp and load columns using:
  - Chinese/English column keyword matching,
  - parse-success fallback rules.
- Handles non-standard column names.
- Cleans missing values and duplicate timestamps.

---

## Characteristic Analysis Outputs

### Key tables in `outputs/`
- `cleaned_data.csv`
- `data_quality_report.csv`
- `dataset_metadata.csv`
- `basic_statistics.csv`
- `yearly_statistics.csv`
- `monthly_statistics.csv`
- `weekly_statistics.csv`
- `daily_statistics.csv`
- `hourly_statistics.csv`
- `monthly_volatility.csv`
- `seasonal_volatility.csv`
- `daily_peak_valley_metrics.csv`

### Required characteristic figures in `figures/`
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

Additional analysis figures:
- `19_daily_load_curve_samples.png`
- `20_seasonal_distribution_kde.png`

---

## EMD-LSTM Forecasting Outputs

### EMD outputs
- `outputs/imf_components.csv`
- `figures/21_emd_decomposition_overview.png`
- `figures/22_imf_components_plot.png`

### LSTM training + forecast outputs
- `figures/23_training_loss_curves.png`
- `outputs/forecast_results.csv`
- `outputs/forecast_metrics.csv`
- `figures/24_forecast_vs_actual.png`
- `figures/25_forecast_error_distribution.png`
- `figures/26_zoomed_prediction_plot.png`

### Evaluation metrics
- MAE
- RMSE
- MAPE
- R2

---

## Beginner-friendly modeling notes

- The forecasting strategy is **IMF-wise modeling**:
  1. Decompose load into IMFs.
  2. Train one LSTM per IMF.
  3. Predict each IMF independently.
  4. Reconstruct final prediction by summing IMF predictions.
- Default configuration is intentionally simple and readable.
- You can tune `ForecastConfig` in `main.py` for lookback window, epochs, hidden size, etc.

