# Power Load Analysis and Forecasting (EMD + LSTM/TCN)

## Project overview
This project provides an end-to-end workflow for power load analysis and forecasting in CPU-only environments.  
It combines data cleaning, multi-time-scale statistics, heavy overload indicator analysis, EMD decomposition, and IMF-wise deep learning forecasting.

Supported forecasting models:
- **LSTM**
- **TCN (Temporal Convolutional Network)**
- **Auto model selection** (train both and select best by RMSE)

---

## Pipeline description
The full workflow in `main.py` is:

```text
load CSV data
→ preprocessing
→ statistical analysis
→ multi-time-scale visualization
→ heavy overload indicator analysis
→ EMD decomposition
→ IMF forecasting using LSTM and/or TCN
→ forecast reconstruction
→ model comparison
→ best model selection
→ evaluation metrics
→ visualization
```

---

## Heavy overload indicator definition
The project implements two heavy overload indicators:

1. **Daily heavy overload**
   - Condition: `daily_load > monthly_average * 1.3`
   - Output: `outputs/heavy_overload_days.csv`
   - Plot: `figures/heavy_overload_days.png`

2. **Monthly heavy overload**
   - Condition: `monthly_average > yearly_average * 1.3`
   - Output: `outputs/heavy_overload_months.csv`
   - Plot: `figures/heavy_overload_months.png`

---

## Model comparison logic
Forecasting is done IMF-by-IMF and then reconstructed by summing IMF predictions.

- `MODEL_TYPE = lstm`: train only LSTM
- `MODEL_TYPE = tcn`: train only TCN
- `MODEL_TYPE = auto`: train both and compare

Comparison metrics:
- RMSE
- MAE
- MAPE

Best model rule:
- Select model with **lowest RMSE**

Outputs:
- `outputs/lstm_forecast.csv`
- `outputs/tcn_forecast.csv`
- `outputs/model_comparison.csv`
- `outputs/best_model.txt`
- `outputs/final_forecast.csv`

Visual outputs include:
- Forecast vs actual plot
- Training loss curves
- Model comparison bar chart
- EMD decomposition plots
- Overload indicator plots

---

## How to run the project
Install dependencies:

```bash
pip install -r requirements.txt
```

Run the full pipeline:

```bash
python main.py
```

---

## Key files
- `main.py`: pipeline orchestration and `MODEL_TYPE` configuration
- `src/forecast_pipeline.py`: IMF-wise forecasting, reconstruction, and model selection
- `src/lstm_model.py`: LSTM model
- `src/tcn_model.py`: TCN model
- `src/model_comparison.py`: RMSE/MAE/MAPE comparison + best model selection
- `src/overload_analysis.py`: heavy overload analysis and plots
