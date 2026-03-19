# 电力负荷分析与预测项目（中文版说明）

本项目面向**电力负荷时序数据分析、特征挖掘与短期预测**场景，提供从数据读取、清洗、统计分析、时间尺度分析、EMD 分解，到基于 **TCN（Temporal Convolutional Network）** 的多步预测与误差评估的一体化流程。项目默认会自动选择仓库中的主数据集，并生成分析表格、预测结果和可视化图像，适合课程设计、论文实验、方法验证与工程原型搭建使用。

---

## 1. 项目能做什么

项目启动后会按以下顺序执行完整流程：

1. 自动查找并读取仓库内的 CSV 数据集；
2. 自动识别时间列与负荷列；
3. 清洗异常格式、重复时间戳和缺失值；
4. 生成基础统计分析和多时间尺度统计结果；
5. 输出季节性、分布、差分、相关性、频谱等分析图；
6. 对负荷序列执行 **EMD 分解**；
7. 按用户指定的 IMF 个数构建 **EMD + TCN** 预测方案；
8. 与“未分解直接使用 TCN”的方案进行对比；
9. 输出误差指标、最佳策略说明及多类误差分析结果。

简化理解如下：

```text
原始负荷数据
  → 数据清洗
  → 统计分析 / 时间尺度分析 / 可视化
  → EMD 分解
  → TCN 预测
  → 预测效果评估
  → 输出结果文件与图表
```

---

## 2. 项目特点

- **仅保留 TCN 主预测流程**：当前版本默认围绕 TCN 进行预测与对比分析。
- **支持交互式 IMF 个数输入**：运行时可输入 `1-10` 之间的 IMF 分量数。
- **支持命令行参数运行**：可通过 `--imf-components` 直接指定 IMF 个数，便于批处理或脚本运行。
- **自动识别数据列**：无需手动硬编码时间列和负荷列名称，支持中英文关键字推断。
- **自动选择 CPU / GPU**：若环境中可用 CUDA，程序会自动调用 GPU。
- **分析结果较完整**：覆盖基础统计、季节性、频域分析、重载检测、预测误差分解等多个模块。
- **输出适合留档**：CSV、JSON、TXT、PNG 等结果会自动写入 `outputs/` 与 `figures/`。

---

## 3. 项目目录结构

```text
.
├── data/                       # 原始数据集（示例：quanzhou.csv）
├── figures/                    # 图像输出目录
│   └── README.md               # 图像说明文档
├── outputs/                    # 结果表格与预测输出目录（运行后生成）
├── src/
│   ├── data_loader.py          # 数据集查找、编码识别、CSV 读取
│   ├── preprocess.py           # 列识别、数据清洗、时间特征构造
│   ├── statistics_analysis.py  # 基础统计、相关性、频谱分析
│   ├── time_scale_analysis.py  # 年/月/周/日/小时多尺度分析
│   ├── season_analysis.py      # 季节性与统计特征可视化
│   ├── overload_analysis.py    # 重载检测分析
│   ├── emd_decomposition.py    # EMD 分解与 IMF 分析
│   ├── tcn_model.py            # TCN 模型定义
│   ├── lstm_dataset.py         # 时序样本构造与 DataLoader 工具
│   ├── forecast_pipeline.py    # 预测主流程与方案对比
│   ├── evaluation.py           # 指标计算与误差分析输出
│   └── visualization.py        # 统一绘图样式与保存工具
├── main.py                     # 项目入口
├── requirements.txt            # Python 依赖说明
└── README.md                   # 本文档
```

---

## 4. 环境要求

建议使用以下环境：

- Python **3.10 及以上**（推荐 3.10/3.11）
- Windows / macOS / Linux 均可
- 建议使用虚拟环境（`venv` 或 `conda`）

如果需要 GPU 加速，请确认：

- 已正确安装 NVIDIA 驱动；
- CUDA 版本与 PyTorch 安装方式匹配；
- `python -c "import torch; print(torch.cuda.is_available())"` 返回 `True`。

---

## 5. 安装依赖

### 方式一：直接安装 requirements

```bash
pip install -r requirements.txt
```

该依赖文件默认包含 **CPU 可运行版本的 PyTorch** 写法，适合大多数本地开发与教学环境。

### 方式二：手动安装 GPU 版本 PyTorch（推荐有 CUDA 的机器）

如果你希望使用官方 CUDA 轮子，可以先安装除 PyTorch 之外的其他依赖，再根据自己的 CUDA 版本安装对应的 PyTorch。

示例（CUDA 12.1）：

```bash
pip install pandas numpy matplotlib seaborn scipy statsmodels PyEMD scikit-learn holidays chinese-calendar
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

> 说明：不同机器的 CUDA 版本可能不同，请以 PyTorch 官方安装指引为准。

---

## 6. 如何运行

### 6.1 交互式运行

```bash
python main.py
```

程序启动后会提示：

```text
请输入 IMF 分解个数（1-10）:
```

输入一个 `1` 到 `10` 之间的整数后，程序继续执行完整分析与预测流程。

### 6.2 非交互式运行

```bash
python main.py --imf-components 3
```

适用于：

- 服务器脚本运行；
- 批处理实验；
- 避免终端交互输入；
- 在 IDE / Notebook 外直接固定实验参数。

---

## 7. 输入数据要求

项目会自动在仓库内查找 CSV 文件，并优先选择 `data/` 目录下、体积较大的数据文件作为主数据集。为了保证识别准确，建议输入数据满足以下要求：

### 7.1 最低要求

CSV 中至少应包含：

- 一列时间戳；
- 一列负荷值。

### 7.2 推荐列名

时间列支持自动识别如下关键词：

- `time`
- `date`
- `datetime`
- `timestamp`
- `时刻`
- `时间`
- `日期`
- `采样时间`
- `记录时间`

负荷列支持自动识别如下关键词：

- `load`
- `power`
- `demand`
- `mw`
- `kw`
- `负荷`
- `有功`
- `电力`
- `功率`

### 7.3 数据建议

建议数据具有以下特征：

- 时间间隔尽量固定（如 15 分钟、30 分钟、1 小时）；
- 时间列尽量可被 `pandas.to_datetime()` 正确解析；
- 负荷列尽量为纯数字或可转为数字；
- 数据量不宜过少，否则可能不足以完成 `lookback + horizon` 的序列建模。

---

## 8. 数据清洗与预处理逻辑

程序会自动执行以下预处理步骤：

- 将时间列统一转换为 `datetime`；
- 将负荷列统一转换为数值类型；
- 删除无法解析的时间戳；
- 统计无法解析的负荷记录；
- 对重复时间戳按平均值聚合；
- 对缺失负荷值进行时间插值，并辅以前向/后向填充；
- 自动生成 `year`、`month`、`day`、`hour`、`weekday`、`is_weekend`、`season`、`date` 等派生特征。

同时，程序会输出：

- `cleaned_data.csv`
- `data_quality_report.csv`
- `dataset_metadata.csv`

便于回溯清洗过程与数据来源。

---

## 9. 预测与建模说明

### 9.1 当前主流程

当前预测主流程为：

- 原始序列直接使用 **TCN** 预测；
- 原始序列经过 **EMD 分解** 后，按用户给定的 IMF 个数进行组合预测；
- 对比分解与未分解两种方案的效果；
- 自动输出更优方案说明。

### 9.2 IMF 个数限制

- 支持输入范围：`1-10`
- 程序内部上限：`MAX_IMF = 10`
- 若实际分解得到的 IMF 分量少于用户指定值，则后续流程可能无法按预期进行，应降低输入值重试。

### 9.3 默认训练参数

预测主流程中的默认配置包括：

- `lookback = 672`
- `train_ratio = 0.8`
- `dropout = 0.1`
- `learning_rate = 1e-3`
- `epochs = 20`
- `batch_size = 128`
- `horizon = 96`
- `random_seed = 42`

这些参数适合一般实验用途；如果需要进一步优化精度，可在 `src/forecast_pipeline.py` 中自行调整。

---

## 10. 输出结果说明

项目运行完成后，主要输出位于两个目录：

- `outputs/`：表格、指标、文本说明；
- `figures/`：图像结果。

### 10.1 `outputs/` 常见输出

以下文件会根据流程自动生成（不同数据或流程分支下可能略有增减）：

- `cleaned_data.csv`：清洗后的时序数据；
- `data_quality_report.csv`：数据质量报告；
- `dataset_metadata.csv`：数据集元信息；
- `basic_statistics.csv`：基础统计指标；
- `yearly_statistics.csv`：年度统计；
- `monthly_statistics.csv`：月度统计；
- `weekly_statistics.csv`：周尺度统计；
- `daily_statistics.csv`：日尺度统计；
- `hourly_statistics.csv`：小时尺度统计；
- `monthly_volatility_statistics.csv`：月波动性统计；
- `daily_peak_valley_metrics.csv`：日峰谷指标；
- `heavy_overload_day_detection_results.csv`：日重载识别结果；
- `heavy_overload_month_detection_results.csv`：月重载识别结果；
- `emd_decomposition_results.csv`：EMD 分解结果；
- `imf_frequency_features.csv`：IMF 频率特征结果；
- `tcn_forecast_non_decomposed.csv`：未分解 TCN 预测结果；
- `tcn_forecast_emd_decomposed_imf{k}.csv`：分解后预测结果；
- `TCN预测结果_IMF{k}.csv`：中文命名版本预测结果；
- `decomposition_vs_non_decomposition_comparison.csv`：分解/未分解对比结果；
- `best_forecast_strategy.txt`：最佳预测策略说明；
- `freq_fusion_forecast.csv`：频率融合预测结果（启用相关流程时生成）；
- `model_comparison.csv`：模型对比结果（调用相关模块时生成）；
- `best_model.txt`：模型对比选优结果（调用相关模块时生成）；
- `forecast_results.csv`：最终预测结果；
- `forecast_metrics.csv`：预测指标表；
- `forecast_metrics.json`：预测指标 JSON；
- `peak_period_error_stats.csv`：高峰时段误差统计；
- `valley_period_error_stats.csv`：低谷时段误差统计；
- `holiday_error_stats.csv`：节假日误差统计。

### 10.2 `figures/` 常见输出

- `raw_load_timeseries.png`：原始负荷时序图；
- `monthly_average_load.png`：月平均负荷图；
- `monthly_load_boxplot.png`：月度箱线图；
- `weekday_vs_weekend_load.png`：工作日/周末负荷对比图；
- `seasonal_load_analysis.png`：季节性负荷分析图；
- `load_distribution_histogram.png`：负荷分布直方图；
- `load_empirical_distribution_function.png`：经验分布函数图；
- `load_ramp_distribution.png`：爬坡变化分布图；
- `diff1_timeseries.png` / `diff2_timeseries.png`：一阶/二阶差分时序图；
- `diff1_distribution.png` / `diff2_distribution.png`：一阶/二阶差分分布图；
- `acf_plot.png` / `pacf_plot.png`：自相关 / 偏自相关图；
- `lag_scatter_plots.png`：滞后散点图；
- `correlation_heatmap.png`：相关性热力图；
- `emd_decomposition_overview.png`：EMD 总览图；
- `imf_components.png`：IMF 分量图；
- `imf_spectrum_overview.png`：IMF 频谱图；
- `imf_frequency_classification.png`：IMF 频率分类图；
- `imf_energy_ratio.png`：IMF 能量占比图；
- `actual_vs_predicted_load.png`：实际值与预测值对比图；
- `prediction_error_distribution.png`：预测误差分布图；
- `peak_period_error.png`：高峰时段误差图；
- `valley_period_error.png`：低谷时段误差图；
- `holiday_prediction_error.png`：节假日误差图；
- `error_histogram.png`：残差直方图；
- `error_qq_plot.png`：QQ 图；
- `freq_fusion_prediction.png`：频率融合预测结果图（启用相关流程时生成）；
- `27_model_comparison_bar.png`：模型对比柱状图（调用相关模块时生成）。

如果需要查看图像的补充说明，可继续阅读 `figures/README.md`。

---

## 11. 误差分析说明

项目除了输出整体预测误差外，还会进行更细粒度的误差分析。

### 11.1 高峰时段误差

默认提取 **18:00-21:00** 区间样本，输出：

- 误差统计表；
- 箱线图等可视化结果。

### 11.2 低谷时段误差

默认提取 **02:00-04:00** 区间样本，输出：

- 误差统计表；
- 低谷时段误差分布图。

### 11.3 节假日误差

程序会优先尝试通过 `holidays` 或 `chinese-calendar` 识别中国节假日，并输出：

- 节假日 / 非节假日误差对比；
- 对应的误差统计结果。

### 11.4 误差分布检验

项目还会生成：

- 误差直方图；
- 带理论分位数对比的 QQ 图。

有助于判断预测残差是否接近某种统计分布。

---

## 12. 常见问题

### 12.1 运行时报“IMF 分解个数必须在 1-10 之间”

请确认输入值或命令行参数满足 `1 <= imf_components <= 10`。

### 12.2 程序提示样本不足

通常表示数据长度不足以支持当前的：

- `lookback = 672`
- `horizon = 96`

可以尝试：

- 提供更长时间范围的数据；
- 在代码中适当调小 `lookback` 或 `horizon`。

### 12.3 没有使用 GPU

请检查：

- PyTorch 是否安装了正确的 CUDA 版本；
- `torch.cuda.is_available()` 是否为 `True`；
- 当前机器是否具备 NVIDIA GPU 与可用驱动。

### 12.4 节假日识别失败

请确认 `holidays` 或 `chinese-calendar` 已正确安装。项目已在 `requirements.txt` 中列出这两个依赖。

---

## 13. 适用场景

本项目适合用于：

- 电力系统课程设计；
- 电力负荷预测论文实验；
- 时间序列分析方法验证；
- EMD + 深度学习预测流程演示；
- 配电网或地区负荷数据分析原型开发。

---

## 14. 后续可扩展方向

如果你准备继续完善该项目，可以考虑：

- 增加配置文件（如 `yaml` / `json`）统一管理参数；
- 支持多数据集批量实验；
- 增加交叉验证与超参数搜索；
- 引入更完整的模型注册机制；
- 将输出目录按时间戳自动归档；
- 增加 Notebook 示例或实验报告模板；
- 增加更细粒度的异常检测与数据质量诊断模块。

---

## 15. 快速开始示例

```bash
# 1) 安装依赖
pip install -r requirements.txt

# 2) 运行项目（交互式输入 IMF 个数）
python main.py

# 3) 或直接指定 IMF 个数为 3
python main.py --imf-components 3
```

运行结束后，查看：

- `outputs/` 中的表格和指标；
- `figures/` 中的图像结果；
- `best_forecast_strategy.txt` 中的最佳策略说明。

---

## 16. 说明

本 README 为当前项目的**中文使用说明版本**，重点是帮助你快速理解：

- 项目做什么；
- 如何安装；
- 如何运行；
- 会输出什么；
- 结果应该去哪里看。

如果你后续希望，我还可以继续帮你补一版：

- 更适合 **GitHub 展示风格** 的 README；
- 更适合 **课程论文附录** 的 README；
- 更适合 **工程部署** 的 README。
