# 电力负荷分析与预测项目

本项目是一个围绕电力负荷时间序列构建的端到端分析与预测仓库，覆盖：

- CSV 数据自动发现与编码识别；
- 时间列 / 负荷列自动推断；
- 数据清洗与质量报告；
- 多时间尺度统计分析；
- 季节性、相关性、频域与重过载识别；
- EMD（经验模态分解）分解与 IMF 频率分层；
- 多种预测策略对比；
- 误差评估、最佳策略选择与图表输出。

> **重要说明（以当前代码为准）**
>
> 这个仓库的文档与实现曾存在轻微偏差。根据当前代码，项目的预测部分应理解为：
>
> 1. **基础对比基线**：未分解原始序列的 **LSTM** 预测；
> 2. **主分解方案**：EMD 分解后的分量重构预测（README 中通常称为 TCN 分解方案）；
> 3. **扩展方案**：按 IMF 频率分层后，分别使用 **SCINet / TCN / Autoformer / TimeXer** 的频率融合预测；
> 4. 最终程序会按照误差指标自动挑选最佳预测策略，并输出统一的结果文件。

---

## 1. 项目概览

### 1.1 项目目标

这个项目解决的是典型的单变量电力负荷时间序列分析与预测问题。给定一份包含时间戳和负荷值的 CSV 文件，程序会自动完成：

1. 读取并识别数据；
2. 清洗异常时间列与负荷列；
3. 生成基础统计与图像；
4. 将负荷序列做 EMD 分解；
5. 构造多种预测策略并进行比较；
6. 输出最终最优策略的预测结果与误差分析。

### 1.2 当前仓库默认数据

仓库当前包含一个默认数据文件：

- `data/quanzhou.csv`

程序会自动扫描仓库中的 CSV 文件，并优先选择 `data/` 目录下、体积更大的 CSV 文件作为主数据集，因此在默认情况下会直接使用上面的数据文件。

### 1.3 项目主流程

整体执行链路如下：

```text
自动发现 CSV
→ 编码识别与读取
→ 时间列/负荷列自动推断
→ 数据清洗与质量报告
→ 基础统计与多时间尺度分析
→ 季节性 / 相关性 / 频域 / 重过载分析
→ EMD 分解
→ IMF 频率特征提取与分组
→ 多策略预测与比较
→ 最优策略输出
→ 误差分析与图表落盘
```

---

## 2. 当前实现中的预测策略说明

这是本项目最容易产生误解的地方，因此单独说明。

### 2.1 基础基线：未分解 LSTM 预测

程序首先直接对原始负荷序列做未分解预测。当前实现中，这条基线调用的是 `LSTMForecaster`，因此它的真实含义是：

> **不做 EMD 分解，直接用 LSTM 对原始负荷序列做多步预测。**

输出对比表中，这一方案会被记录为：

- `Non-decomposed LSTM Forecast`

### 2.2 分解方案：EMD 分解后重构预测

程序会对负荷序列先做 EMD 分解，再选取前 `k` 个 IMF 分量，并把剩余部分并入一个 `remainder_component`。之后对这些分量分别建模并重构预测结果。

命令行参数 `--imf-components` 或交互式输入中的 `k`，控制的就是这里保留多少个 IMF 分量参与主分解方案。

输出对比表中，这一方案会被记录为：

- `EMD-decomposed TCN Forecast (k=<IMF个数>)`

> 注意：从命名上看它被描述为 TCN 分解方案，但仓库里实际还保留了多种模型接口。阅读仓库时请以代码实现为准，不要仅根据历史文档命名判断。

### 2.3 扩展方案：频率融合预测

在 EMD 分解之后，程序还会计算每个 IMF 的主频与能量，并自动把 IMF 分为高频 / 中频 / 低频三组。随后使用以下模型映射进行扩展预测：

- 高频 IMF：`SCINet`
- 中频 IMF：`TCN`
- 低频 IMF：`Autoformer`
- 趋势项：`TimeXer`
- 未被分组命中的剩余 IMF：回退到 `TCN`

这一方案属于扩展型模型融合，最终在对比表中记为：

- `Frequency Fusion Forecast`

### 2.4 最终采用哪个预测结果？

程序会对以下候选策略按误差指标自动排序：

- 未分解 LSTM 预测；
- EMD 分解方案预测；
- 频率融合预测（若成功生成）。

最终会选择 **RMSE 优先、再看 MAE 与 MAPE** 的最佳策略，并输出：

- `outputs/best_forecast_strategy.txt`
- `outputs/forecast_results.csv`
- `outputs/forecast_metrics.csv`
- `outputs/forecast_metrics.json`

因此：

> **最终的统一结果文件并不一定来自 TCN，也不一定来自 LSTM，而是来自当前运行中误差最优的那条策略。**

---

## 3. 项目功能说明

### 3.1 数据自动发现与鲁棒读取

项目会自动在仓库内搜索 CSV 文件，并排除以下目录：

- `.git`
- `outputs`
- `figures`
- `__pycache__`
- `.venv`
- `venv`

读取时会尝试多种常见编码：

- `utf-8`
- `utf-8-sig`
- `gb18030`
- `gbk`
- `big5`
- `latin1`

### 3.2 自动识别时间列与负荷列

程序支持中英文关键词识别，例如：

- 时间列候选：`time`、`date`、`datetime`、`timestamp`、`时间`、`日期`、`采样时间` 等；
- 负荷列候选：`load`、`power`、`demand`、`mw`、`kw`、`负荷`、`功率`、`电力` 等。

如果没有明显列名，程序还会：

- 按可解析时间比例自动推断时间列；
- 按数值有效比例自动推断负荷列。

### 3.3 数据清洗

清洗步骤包括：

- 时间戳解析；
- 负荷字段转数值；
- 删除非法时间记录；
- 按时间排序；
- 对重复时间戳按平均值聚合；
- 对缺失负荷值进行时间插值，并辅以前向 / 后向填充。

同时还会输出数据质量报告，例如：

- 原始记录数；
- 无效时间行数；
- 无效负荷行数；
- 重复时间戳行数；
- 清洗后记录数。

### 3.4 多时间尺度统计分析

程序会对清洗后的负荷序列生成：

- 年尺度统计；
- 月尺度统计；
- 周尺度统计；
- 日尺度统计；
- 小时尺度统计；
- 月波动性统计；
- 峰谷特征统计。

### 3.5 季节性与统计特征分析

项目会输出以下分析图：

- 原始时间序列图；
- 月平均负荷图；
- 月负荷箱线图；
- 工作日 / 周末日曲线对比；
- 四季负荷日曲线；
- 负荷分布直方图；
- 经验分布函数图；
- 负荷爬坡分布图。

### 3.6 差分、相关性与频域分析

程序还会生成：

- 一阶差分时序图与分布图；
- 二阶差分时序图与分布图；
- ACF / PACF 图；
- 多滞后散点图；
- 相关性热力图；
- FFT 频谱图。

### 3.7 重过载识别

当前实现定义了两类重过载识别：

1. **日尺度重过载**：日均负荷 > 当月平均负荷 × 1.3
2. **月尺度重过载**：月均负荷 > 当年平均负荷 × 1.3

结果会以 CSV 形式保存，便于后续排查重点日期和重点月份。

### 3.8 EMD 分解与 IMF 频率分析

项目对清洗后的负荷序列执行 EMD 分解，并默认最多保留 `10` 个 IMF 分量。随后程序会：

- 保存 IMF 分解结果；
- 绘制 EMD 总览图；
- 绘制 IMF 分量叠加图；
- 计算各 IMF 主频与能量；
- 将 IMF 分为高频 / 中频 / 低频；
- 输出频率分类、能量占比、重构分量等图像。

### 3.9 预测评估与误差分析

对最终选中的最佳预测策略，程序会进一步生成：

- 真实值 vs 预测值图；
- 预测误差分布图；
- 高峰时段误差分析（18:00-21:00）；
- 低谷时段误差分析（02:00-04:00）；
- 节假日 / 非节假日误差分析；
- 误差直方图；
- QQ 图。

> 节假日分析默认按中国节假日逻辑处理，依赖 `holidays` 或 `chinese-calendar`。

---

## 4. 目录结构

```text
Jingrui-Tao/
├── data/
│   └── quanzhou.csv
├── figures/
│   └── README.md
├── outputs/                       # 运行后自动生成
├── src/
│   ├── data_loader.py             # CSV 搜索、编码识别与读取
│   ├── preprocess.py              # 列推断、清洗与时间特征构造
│   ├── statistics_analysis.py     # 基础统计、差分、相关性与频域分析
│   ├── time_scale_analysis.py     # 年/月/周/日/小时尺度分析
│   ├── season_analysis.py         # 季节分析与统计特征图
│   ├── overload_analysis.py       # 重过载识别
│   ├── emd_decomposition.py       # EMD 分解、IMF 保存与可视化
│   ├── lstm_dataset.py            # 序列样本构造与 DataLoader
│   ├── lstm_model.py              # LSTM 预测模型
│   ├── tcn_model.py               # TCN 模型定义
│   ├── forecast_pipeline.py       # 多策略预测、比较与最佳方案选择
│   ├── evaluation.py              # 指标计算、预测图与误差分析
│   ├── model_comparison.py        # 模型对比表与柱状图
│   └── visualization.py           # 统一绘图风格与图片保存
├── main.py                        # 程序入口
├── requirements.txt               # 依赖清单
└── README.md
```

---

## 5. 环境要求

### 5.1 Python 版本

推荐：

- **Python 3.10+**

### 5.2 核心依赖

主要依赖包括：

- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `scipy`
- `statsmodels`
- `PyEMD`
- `holidays`
- `chinese-calendar`
- `torch`

### 5.3 GPU 说明

项目会自动检测：

- 若本机可用 CUDA，则使用 GPU；
- 否则自动回退到 CPU。

PyTorch 的安装方式请根据本机 CUDA 版本选择相应轮子。

---

## 6. 安装方式

### 6.1 创建虚拟环境（推荐）

```bash
python -m venv .venv
source .venv/bin/activate
# Windows:
# .venv\Scripts\activate
```

### 6.2 安装依赖

```bash
pip install -r requirements.txt
```

### 6.3 如需单独安装 GPU 版 PyTorch

请参考 PyTorch 官方安装页面，根据系统与 CUDA 版本选择对应命令。安装完成后再执行：

```bash
pip install -r requirements.txt
```

---

## 7. 如何运行

### 7.1 交互式运行

```bash
python main.py
```

程序会提示：

```text
请输入 IMF 分解个数（1-10）:
```

你输入的值将决定主分解方案中保留多少个 IMF 分量参与预测。

### 7.2 命令行指定 IMF 个数

```bash
python main.py --imf-components 3
```

### 7.3 运行后控制台会显示什么

通常会输出：

- 当前选择的 IMF 个数；
- 使用设备（CPU / GPU）；
- 未分解预测指标；
- 分解方案预测指标；
- 最佳预测策略；
- 数据质量报告；
- 最终预测指标。

---

## 8. 预测配置说明

当前默认配置位于 `ForecastConfig` 中，主要参数如下：

- `lookback = 672`
- `train_ratio = 0.8`
- `dropout = 0.1`
- `learning_rate = 1e-3`
- `epochs = 20`
- `batch_size = 128`
- `random_seed = 42`
- `horizon = 96`
- `tcn_channels = (32, 32, 32)`
- `tcn_kernel_size = 3`

### 8.1 这些参数代表什么

如果当前数据是 **15 分钟粒度**，那么：

- `lookback = 672` 表示回看 `672 × 15 分钟 = 7 天` 的历史；
- `horizon = 96` 表示预测未来 `96 × 15 分钟 = 24 小时`。

也就是说，默认配置相当于：

> 用过去 7 天的负荷历史，预测未来 1 天的负荷变化。

### 8.2 IMF 个数如何理解

`imf_components` 决定的是主分解方案中保留前多少个 IMF 分量。程序会：

1. 保留前 `k` 个 IMF；
2. 把其余成分合并为一个剩余项；
3. 对这些分量分别建模并重构预测结果。

这个参数直接影响：

- 分解粒度；
- 主方案建模结构；
- 最终误差表现。

---

## 9. 输出文件说明

运行完成后，主要结果会写入 `outputs/` 与 `figures/`。

### 9.1 `outputs/` 目录常见文件

#### 数据与清洗类

- `cleaned_data.csv`：清洗后的标准化时序数据
- `data_quality_report.csv`：数据质量报告
- `dataset_metadata.csv`：自动识别出的数据文件、编码、时间列、负荷列

#### 统计分析类

- `basic_statistics.csv`
- `yearly_statistics.csv`
- `monthly_statistics.csv`
- `weekly_statistics.csv`
- `daily_statistics.csv`
- `hourly_statistics.csv`
- `monthly_volatility_statistics.csv`
- `daily_peak_valley_metrics.csv`

#### 重过载识别类

- `heavy_overload_day_detection_results.csv`
- `heavy_overload_month_detection_results.csv`

#### EMD 与 IMF 类

- `emd_decomposition_results.csv`
- `imf_frequency_features.csv`

#### 预测与评估类

- `tcn_forecast_non_decomposed.csv`
- `tcn_forecast_emd_decomposed_imf<k>.csv`
- `TCN预测结果_IMF<k>.csv`
- `freq_fusion_forecast.csv`（若频率融合成功生成）
- `decomposition_vs_non_decomposition_comparison.csv`
- `model_comparison.csv`
- `best_model.txt`
- `best_forecast_strategy.txt`
- `forecast_results.csv`
- `forecast_metrics.csv`
- `forecast_metrics.json`
- `peak_period_error_stats.csv`
- `valley_period_error_stats.csv`
- `holiday_error_stats.csv`

### 9.2 `figures/` 目录常见图片

图片明细可查看：

- `figures/README.md`

常见图片包括：

- `raw_load_timeseries.png`
- `monthly_average_load.png`
- `monthly_load_boxplot.png`
- `weekday_vs_weekend_load.png`
- `seasonal_load_analysis.png`
- `load_distribution_histogram.png`
- `load_empirical_distribution_function.png`
- `load_ramp_distribution.png`
- `diff1_timeseries.png`
- `diff1_distribution.png`
- `diff2_timeseries.png`
- `diff2_distribution.png`
- `acf_plot.png`
- `pacf_plot.png`
- `lag_scatter_plots.png`
- `correlation_heatmap.png`
- `fft_spectrum.png`
- `emd_decomposition_overview.png`
- `imf_components.png`
- `imf_spectrum_overview.png`
- `imf_frequency_classification.png`
- `imf_energy_ratio.png`
- `imf_reconstruction.png`
- `imf_volatility_decomposition.png`
- `actual_vs_predicted_load.png`
- `prediction_error_distribution.png`
- `freq_fusion_prediction.png`
- `peak_period_error.png`
- `valley_period_error.png`
- `holiday_prediction_error.png`
- `error_histogram.png`
- `error_qq_plot.png`
- `27_model_comparison_bar.png`

> 注意：并非所有图片在每次运行中都一定生成，具体取决于流程是否走到对应步骤，以及候选策略是否生成成功。

---

## 10. 输入数据要求

### 10.1 最低要求

至少需要一份 CSV，且满足：

- 至少有 1 列可解析为时间；
- 至少有 1 列可解析为数值型负荷；
- 数据量足以支持滑窗建模。

### 10.2 推荐要求

更推荐满足以下条件：

- 时间粒度稳定（如 15 分钟、30 分钟、1 小时）；
- 时间跨度较长，覆盖多个周期；
- 缺失值和异常值较少；
- 负荷单位清晰（MW / kW 等）。

### 10.3 对数据量的实际提醒

由于默认配置是：

- `lookback = 672`
- `horizon = 96`
- `train_ratio = 0.8`

如果数据过短，程序可能报错：

- 训练样本不足；
- 测试样本不足；
- IMF 分量数超出可用上限。

如果出现这种情况，可以：

- 使用更长时间的数据；
- 降低 `lookback`；
- 降低 `horizon`；
- 降低 `imf_components`。

---

## 11. 项目优点与当前局限

### 11.1 优点

- 从数据读入到评估输出的链路完整；
- 对 CSV 编码和列名具有较好的鲁棒性；
- 提供了丰富的统计分析与图像结果；
- 支持 EMD 分解与 IMF 多频率分析；
- 支持多策略预测比较并自动选择最佳方案；
- 同时兼顾分析研究与实验扩展场景。

### 11.2 当前局限

- 主体仍是**单变量负荷预测**，尚未把气象等外生变量纳入主预测流程；
- 节假日分析默认基于中国节假日；
- `SCINet`、`Autoformer`、`TimeXer` 当前实现更偏轻量占位 / 原型接口，而非完整论文复现；
- 文档历史上更强调 TCN，但当前代码实际是“未分解 LSTM + 分解方案 + 频率融合”的组合对比；
- 默认超参数更像实验配置，未针对所有数据集做系统调优。

---

## 12. 适合如何扩展

这个仓库比较适合作为电力负荷预测实验底座。后续可以考虑：

### 12.1 多变量预测

将以下外部变量并入主预测模型：

- 温度
- 湿度
- 天气类型
- 节假日标识
- 电价 / 工业负荷标签

### 12.2 模型增强

可以进一步替换或扩展：

- 更完整的 TCN / Transformer 家族实现；
- PatchTST / TimesNet / iTransformer / DLinear 等模型；
- 多模型集成；
- 多步滚动预测与递归预测比较；
- 贝叶斯调参或自动调参。

### 12.3 工程化提升

可继续补充：

- 配置文件系统（YAML / TOML）；
- 实验日志与参数追踪；
- 更规范的输出版本管理；
- 单元测试与集成测试；
- 模型保存、加载与复现机制。

---

## 13. 快速开始建议

如果你第一次接触这个仓库，建议按下面顺序使用：

1. 直接运行：
   ```bash
   python main.py --imf-components 3
   ```
2. 查看 `outputs/data_quality_report.csv`，确认清洗质量；
3. 查看 `outputs/dataset_metadata.csv`，确认列识别是否正确；
4. 查看 `figures/raw_load_timeseries.png` 和 `figures/seasonal_load_analysis.png`；
5. 查看 `outputs/decomposition_vs_non_decomposition_comparison.csv`；
6. 查看 `outputs/best_forecast_strategy.txt` 与 `outputs/forecast_metrics.csv`；
7. 最后查看 `figures/actual_vs_predicted_load.png` 与误差分析图。

---

## 14. 常见问题

### Q1：为什么程序会要求输入 IMF 个数？

因为主分解方案需要你指定保留多少个 IMF 分量参与预测。这个值会直接影响：

- 分解粒度；
- 分量重构结构；
- 预测误差表现。

### Q2：为什么换了自己的数据后效果不好？

常见原因包括：

- 数据粒度与默认参数不匹配；
- 数据长度太短；
- 缺失或异常值较多；
- 负荷模式与默认模型假设差异较大；
- 未针对新数据调参。

### Q3：可以不交互输入 IMF 吗？

可以，使用：

```bash
python main.py --imf-components 3
```

### Q4：项目只能做预测吗？

不是。即使你暂时不关心预测，这个仓库也可以单独用于：

- 数据清洗；
- 负荷统计分析；
- 多时间尺度分析；
- 频域分析；
- EMD 分解；
- 重过载识别。

### Q5：最终统一输出的 `forecast_results.csv` 来自哪个模型？

它不固定来自某一个模型，而是来自本次运行中误差最优的那条预测策略。

---

## 15. 许可证与说明

如需对外发布、论文复现或企业内部使用，建议补充：

- LICENSE 文件；
- 数据来源说明；
- 模型参数版本记录；
- 实验复现说明。

当前 README 的描述以仓库现有代码逻辑为准。如果后续修改了 `src/forecast_pipeline.py` 中的策略组合，请同步更新本文件，避免再次出现“文档和实现不一致”的问题。
