# 电力负荷分析与预测项目（仅TCN版本）

## 1. 项目简介
本项目用于电力负荷数据的全流程分析与预测，当前版本已调整为**仅使用 TCN** 进行预测，不再运行 LSTM。运行时可在终端中输入 **1-10** 之间的 IMF 分解个数，程序会按输入的个数执行 EMD 分解与预测。

核心流程：

终端输入 IMF 分解个数（1-10）  
→ 读取数据  
→ 数据清洗  
→ 负荷特性分析  
→ EMD 分解（最多 10 个 IMF，并按输入值选取前 k 个分量）  
→ TCN 预测  
→ 误差评估  
→ 图表输出

## 2. 本版本关键变更
- 仅保留 **TCN** 预测流程；
- EMD 分解分量上限为 `MAX_IMF = 10`；
- 运行时在终端中输入 IMF 分解个数（`1~10`），程序按输入值构建“EMD分解 + TCN”方案；
- 自动对比“未分解 + TCN”与“EMD分解 + TCN（指定 IMF 个数）”，自动写出最优预测方案；
- 删除 `TCN训练损失曲线图`、`日内平均负荷曲线图`、`月-小时热力图`、`周-小时热力图`；
- 新增高峰/低谷/节假日误差分析、误差直方图与误差QQ图；
- 终端交互提示采用中文，图表与输出文件保留当前代码中的英文命名；
- 保持 PyTorch 的 CPU/GPU 自动选择逻辑，CUDA 可用时自动使用 GPU。

## 3. 目录结构
```text
.
├── data/
├── src/
│   ├── emd_decomposition.py
│   ├── tcn_model.py
│   ├── forecast_pipeline.py
│   ├── evaluation.py
│   ├── visualization.py
│   └── ...
├── outputs/
├── figures/
├── main.py
└── requirements.txt
```

## 4. 运行方式
```bash
pip install -r requirements.txt
python main.py
```

运行后，终端会提示：

```text
请输入 IMF 分解个数（1-10）:
```

输入整数后程序继续执行。

如需在非交互方式下运行，也可以直接通过命令行参数指定：

```bash
python main.py --imf-components 3
```

## 5. 主要输出
### 表格（`outputs/`）
- `cleaned_data.csv`
- `data_quality_report.csv`
- `dataset_metadata.csv`
- `basic_statistics.csv`
- `yearly_statistics.csv`
- `monthly_statistics.csv`
- `weekly_statistics.csv`
- `daily_statistics.csv`
- `hourly_statistics.csv`
- `monthly_volatility_statistics.csv`
- `daily_peak_valley_metrics.csv`
- `heavy_overload_day_detection_results.csv`
- `heavy_overload_month_detection_results.csv`
- `emd_decomposition_results.csv`
- `tcn_forecast_non_decomposed.csv`
- `tcn_forecast_emd_decomposed_imf{k}.csv`
- `TCN预测结果_IMF{k}.csv`
- `decomposition_vs_non_decomposition_comparison.csv`
- `best_forecast_strategy.txt`
- `forecast_results.csv`
- `forecast_metrics.csv`
- `forecast_metrics.json`
- `peak_period_error_stats.csv`
- `valley_period_error_stats.csv`
- `holiday_error_stats.csv`

### 图像（`figures/`）
- `raw_load_timeseries.png`
- `monthly_average_load.png`
- `monthly_load_boxplot.png`
- `weekday_vs_weekend_load.png`
- `seasonal_load_analysis.png`
- `load_distribution_histogram.png`
- `load_empirical_distribution_function.png`
- `load_ramp_distribution.png`
- `daily_heavy_overload_detection.png`
- `monthly_heavy_overload_detection.png`
- `emd_decomposition_overview.png`
- `imf_components.png`
- `actual_vs_predicted_load.png`
- `prediction_error_distribution.png`
- `peak_period_error.png`
- `valley_period_error.png`
- `holiday_prediction_error.png`
- `error_histogram.png`
- `error_qq_plot.png`

## 6. 新增误差分析说明
- **高峰时段（18:00-22:00）**：提取高峰时段样本，输出误差分布图与统计结果；
- **低谷时段（02:00-05:00）**：提取低谷时段样本，输出误差分布图与统计结果；
- **节假日分析**：使用中国节假日库自动识别节假日，输出节假日/非节假日误差对比图与统计表；
- **残差统计图**：输出误差直方图（含核密度曲线）与误差QQ图用于分布检验。

## 7. 说明
- 终端输入的 IMF 分解个数必须在 `1-10` 之间；
- 若 EMD 实际分量少于输入值，程序会报错提示可用 IMF 数量不足；
- 控制台会输出：当前使用的 IMF 分量数、未分解预测指标、分解预测指标、最优预测方案。
