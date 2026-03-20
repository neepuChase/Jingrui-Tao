# 电力负荷分析与预测项目

一个面向**多变量电力负荷时间序列**的端到端分析仓库：自动发现 CSV 数据、识别时间列与负荷列，保留天气等数值字段并构造时间特征，完成清洗与统计分析、执行 EMD 分解，并对多种预测策略进行比较，最终输出统一的预测结果、误差指标与图表。

> 本 README 以**当前仓库代码实现**为准，而不是历史命名或旧文档。

---

## 1. 项目定位

本项目围绕一类典型问题展开：

- 输入：包含时间戳、负荷值以及可选天气/环境字段的 CSV 文件；
- 目标：自动完成数据读取、清洗、统计分析、EMD 分解和负荷预测；
- 输出：清洗后的数据表、统计结果、分析图、预测结果、误差分析结果，以及“最佳预测策略”的汇总文件。

项目适合以下场景：

- 电力负荷时间序列课程设计、毕业设计或科研原型；
- 负荷数据分析流程的快速复现；
- 比较“未分解预测”与“分解后预测”效果；
- 产出一组较完整的、可直接用于报告展示的图表与 CSV 结果。

---

## 2. 当前代码真实在做什么

主程序入口是 `main.py`，执行顺序大致如下：

```text
自动发现 CSV
→ 编码识别与读取
→ 时间列/负荷列自动推断
→ 数据清洗与质量报告
→ 多时间尺度统计分析与绘图
→ 季节性 / 差分 / 相关性 / 频域分析
→ EMD 分解
→ IMF 频率分类
→ 预测策略对比
→ 选择最优策略
→ 输出最终预测结果与误差分析
```

从代码上看，仓库包含三类预测逻辑：

1. **未分解基线预测**：对原始负荷序列联合天气/时间特征做多变量 LSTM 预测；
2. **EMD 分解后的重构预测**：将 IMF 分量重组后，结合共享外生特征逐分量预测，再求和重构；
3. **频率融合预测**：将 IMF 按主频分为高频 / 中频 / 低频，并映射到不同模型进行多变量融合预测。

最终程序会基于误差指标自动选择最佳方案，并将该方案的结果写入统一输出文件。  

---

## 3. 仓库结构

```text
.
├── main.py                      # 项目入口
├── requirements.txt            # Python 依赖
├── data/
│   └── quanzhou.csv            # 默认示例数据
├── figures/                    # 运行后生成图像
│   └── README.md               # figures 目录图像说明
└── src/
    ├── data_loader.py          # CSV 搜索、编码识别、数据读取
    ├── preprocess.py           # 列识别、数据清洗、时间特征构造
    ├── statistics_analysis.py  # 基础统计、差分/相关性/频谱分析
    ├── time_scale_analysis.py  # 年/月/周/日/小时尺度统计与图表
    ├── season_analysis.py      # 季节性与分布特征图表
    ├── emd_decomposition.py    # EMD 分解、IMF 频率分类与可视化
    ├── lstm_dataset.py         # 序列样本构造与 DataLoader
    ├── lstm_model.py           # LSTM 预测模型
    ├── tcn_model.py            # TCN 预测模型
    ├── model_comparison.py     # 模型指标比较与最佳模型记录
    ├── evaluation.py           # 指标计算、预测图与误差分析
    ├── forecast_pipeline.py    # 预测主流程与策略对比
    └── visualization.py        # 绘图风格与图片保存工具
```

---

## 4. 数据输入机制

### 4.1 自动发现 CSV

程序会在仓库根目录下递归搜索 `*.csv` 文件，并跳过以下目录：

- `.git`
- `outputs`
- `figures`
- `__pycache__`
- `.venv`
- `venv`

随后会优先选择：

1. 位于 `data/` 目录中的 CSV；
2. 在候选数据中体积更大的文件。

因此，默认情况下会选中：

- `data/quanzhou.csv`

### 4.2 编码识别

程序会按顺序尝试以下编码读取 CSV：

- `utf-8`
- `utf-8-sig`
- `gb18030`
- `gbk`
- `big5`
- `latin1`

仓库自带示例数据 `data/quanzhou.csv` 在当前环境下可被识别为 `gb18030`。

### 4.3 自动识别列名

程序会自动推断：

- 时间列；
- 负荷列。

支持的关键字包含中英文两类，例如：

- 时间列：`time`、`date`、`datetime`、`timestamp`、`时间`、`日期`、`采样时间`；
- 负荷列：`load`、`power`、`demand`、`mw`、`kw`、`负荷`、`功率`、`电力`。

如果列名不明显，程序还会进一步：

- 根据可解析为日期时间的比例推断时间列；
- 根据数值化成功比例推断负荷列。

### 4.4 当前示例数据概况

仓库自带数据文件：

- `data/quanzhou.csv`

数据特征：

- 共 **35040** 行；
- 共 **7** 列；
- 包含中文字段，如 `时间`、`负荷`、温度、湿度和降雨量等；
- 时间粒度为 **15 分钟**；
- 时间范围从 **2018/1/1 0:00** 开始。

当前代码会保留示例数据中的温度、湿度、降雨量等数值字段，并与派生的时间特征一起进入预测模型。

---

## 5. 数据清洗与特征构造

数据清洗主要由 `src/preprocess.py` 完成，步骤包括：

1. 解析时间列为 `timestamp`；
2. 将负荷列转为数值型 `load`；
3. 删除非法时间记录；
4. 按时间排序；
5. 对重复时间戳按平均值聚合；
6. 对缺失负荷值执行时间插值；
7. 对其余数值型外生变量也执行插值和前向/后向填充；
8. 保留可用的天气/环境数值列作为预测输入。

同时，程序会构造以下时间特征，供统计分析和多变量预测使用：

- `year`
- `month`
- `day`
- `hour`
- `weekday`
- `minute`
- `hour_sin` / `hour_cos`
- `weekday_sin` / `weekday_cos`
- `month_sin` / `month_cos`
- `weekday_name`
- `is_weekend`
- `date`
- `season`

程序还会输出数据质量报告，例如：

- 原始记录数；
- 无效时间行数；
- 无效负荷行数；
- 重复时间戳行数；
- 清洗后记录数；
- 剩余缺失值数量。

---

## 6. 统计分析与可视化内容

运行主流程后，仓库会产出较完整的分析图表，主要分为以下几类。

### 6.1 多时间尺度分析

- 原始负荷时间序列图；
- 月平均负荷图；
- 月负荷箱线图；
- 工作日 / 周末负荷对比图；
- 年 / 月 / 周 / 日 / 小时统计表。

### 6.2 季节性与分布特征

- 四季日内负荷曲线；
- 负荷分布直方图；
- 负荷经验分布函数图；
- 负荷爬坡分布图；
- 日峰谷统计表。

### 6.3 差分、相关性与频域分析

- 一阶差分时序图与分布图；
- 二阶差分时序图与分布图；
- ACF 图；
- PACF 图；
- 滞后散点图；
- 相关性热力图；
- FFT 频谱图。

### 6.4 EMD 分解与 IMF 分析

- EMD 分解总览图；
- IMF 分量叠加图；
- IMF 频谱总览图；
- IMF 主频分类图；
- IMF 能量占比图；
- IMF 重构图；
- IMF 波动性分解图。

关于 `figures/` 下图片的逐一解释，可以继续查看：

- `figures/README.md`

---

## 7. 预测部分说明

这是本仓库最重要、也最容易误读的部分。

### 7.1 未分解预测：LSTM 基线

在 `run_tcn_forecast_comparison()` 中，程序首先构造仅包含原始负荷的单列数据，然后调用 `_forecast_by_components()`。而 `_forecast_by_components()` 内部实际使用的是 `LSTMForecaster`。

也就是说，当前实现中的“未分解预测”本质上是：

> **对原始负荷序列直接做 LSTM 多步预测。**

在输出汇总中，该方案会显示为：

- `Non-decomposed LSTM Forecast`

### 7.2 EMD 分解后预测

程序会先执行 EMD 分解，最多保留 10 个 IMF 分量。之后根据 `--imf-components` 的取值，选取前 `k` 个 IMF 分量，并将原始序列减去这 `k` 个 IMF 之和，构造一个 `remainder_component`。

随后，程序会对这些分量分别训练预测模型，再把各分量预测值相加得到重构结果。

需要特别注意：

- 如果提供了 IMF 频率分组信息，代码会优先把 IMF 合并成 `high_group`、`mid_group`、`low_group` 三个组，再进行分组重构预测；
- 这意味着“EMD 分解后预测”不一定严格等于“前 k 个 IMF + remainder”的逐列预测，实际行为会受到分组信息影响。

在输出汇总中，该方案会显示为：

- `EMD-decomposed TCN Forecast (k=<k值>)`
- 最终选择阶段中也可能被记作 `TCN Forecast After EMD Decomposition`

虽然命名中出现了 `TCN`，但当前分量重构路径内部仍调用的是 `_train_single_series()`，该函数用的是 `LSTMForecaster`。因此应理解为：

> **当前代码的命名保留了 TCN 历史痕迹，但分量级重构预测核心实现实际仍是 LSTM。**

### 7.3 频率融合预测

程序还实现了一个额外的“频率融合预测”分支：

- 高频 IMF → `SCINet` 风格轻量模型；
- 中频 IMF → `TCN`；
- 低频 IMF → `Autoformer` 风格编码器；
- 趋势项 → `TimeXer` 名义下的 LSTM 适配结构；
- 未被分组使用的 IMF → 回退到 `TCN`。

这里要注意两点：

1. `SCINet`、`Autoformer`、`TimeXer` 在当前仓库中是**轻量占位式实现 / 近似接口**，不是这些论文模型的完整复现版本；
2. 频率融合预测依赖 IMF 分组结果，如果分组信息为空或样本不足，相关分支可能无法得到有效输出。

在输出汇总中，该方案会显示为：

- `Frequency Fusion Forecast`

### 7.4 最终如何选“最佳策略”

程序会将以下候选方案进行比较：

- 未分解 LSTM 预测；
- EMD 分解后的重构预测；
- 频率融合预测（若成功生成）。

最终按以下优先级选择最佳方案：

1. 先比较 `RMSE`；
2. 若相同，再比较 `MAE`；
3. 若仍相同，再比较 `MAPE`。

统一的最终结果会写入：

- `outputs/best_forecast_strategy.txt`
- `outputs/forecast_results.csv`
- `outputs/forecast_metrics.csv`
- `outputs/forecast_metrics.json`

因此：

> 这几个统一输出文件代表的是**本次运行误差最优的策略结果**，而不固定属于某一个模型名称。

---

## 8. 运行方式

### 8.1 环境要求

建议：

- Python 3.10+
- Linux / macOS / Windows 均可
- 有 CUDA 时会自动启用 GPU；否则使用 CPU

### 8.2 安装依赖

```bash
pip install -r requirements.txt
```

核心依赖包括：

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

> `torch` 请按你的设备环境安装匹配版本；如你已单独安装好 PyTorch，可直接安装其余依赖。

### 8.3 启动程序

推荐直接通过命令行提供 IMF 个数：

```bash
python main.py --imf-components 3
```

如果不传 `--imf-components`，程序会在终端中交互输入：

```text
请输入 IMF 分解个数（1-10）:
```

### 8.4 运行中的第二次交互

主流程接近结束时，程序在绘制“真实值 vs 预测值”图像前，还会要求你输入一个日期：

```text
请输入要生成预测对比图的日期（YYYY-MM-DD，范围 起始日期 ~ 结束日期）:
```

只有输入一个**位于预测结果范围内**的日期后，程序才会继续完成：

- `actual_vs_predicted_load.png`
- `prediction_error_distribution.png`
- 后续误差分析图

这意味着：

> 当前仓库主流程**不是完全无交互批处理模式**；即使你已经通过命令行提供了 IMF 个数，后面仍然需要再输入一次绘图日期。

---

## 9. 主要输出文件

程序会在根目录下自动创建：

- `outputs/`
- `figures/`

### 9.1 `outputs/` 中常见文件

基础数据与统计结果：

- `cleaned_data.csv`：清洗后的标准化数据；
- `data_quality_report.csv`：数据质量报告；
- `dataset_metadata.csv`：数据集路径、编码、识别列名等元信息；
- `basic_statistics.csv`：基础统计量；
- `yearly_statistics.csv` / `monthly_statistics.csv` / `weekly_statistics.csv` / `daily_statistics.csv` / `hourly_statistics.csv`；
- `monthly_volatility_statistics.csv`：月度波动性统计；
- `daily_peak_valley_metrics.csv`：日峰谷指标。

EMD 与 IMF 分析结果：

- `emd_decomposition_results.csv`：EMD 分解结果；
- `imf_frequency_features.csv`：IMF 主频与能量特征。

预测与比较结果：

- `tcn_forecast_non_decomposed.csv`：未分解预测结果；
- `tcn_forecast_emd_decomposed_imf{k}.csv`：分解重构预测结果；
- `TCN预测结果_IMF{k}.csv`：分解预测结果的中文命名副本；
- `freq_fusion_forecast.csv`：频率融合预测结果（若成功生成）；
- `decomposition_vs_non_decomposition_comparison.csv`：策略比较表；
- `model_comparison.csv`：模型比较表；
- `best_model.txt`：模型比较模块记录的最佳模型；
- `best_forecast_strategy.txt`：最终统一输出采用的最佳策略；
- `forecast_results.csv`：最终统一预测结果；
- `forecast_metrics.csv` / `forecast_metrics.json`：最终统一误差指标。

误差分析结果：

- `peak_period_error_stats.csv`
- `valley_period_error_stats.csv`
- `holiday_error_stats.csv`

### 9.2 `figures/` 中常见图片

包括但不限于：

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
- `peak_period_error.png`
- `valley_period_error.png`
- `holiday_prediction_error.png`
- `error_histogram.png`
- `error_qq_plot.png`
- `freq_fusion_prediction.png`
- `27_model_comparison_bar.png`

---

## 10. 代码模块职责说明

### `main.py`

负责整个项目调度：

- 解析参数；
- 准备输出目录；
- 选择数据集；
- 调用清洗、分析、EMD 与预测模块；
- 保存元信息和最终结果。

### `src/data_loader.py`

负责：

- 搜索 CSV 文件；
- 识别编码；
- 读取数据；
- 自动选定主数据集。

### `src/preprocess.py`

负责：

- 自动推断时间列与负荷列；
- 清洗时间序列；
- 构造时间衍生特征。

### `src/time_scale_analysis.py`

负责：

- 年 / 月 / 周 / 日 / 小时尺度统计；
- 原始时序图、月均图、箱线图、工作日/周末图等。

### `src/statistics_analysis.py`

负责：

- 基础统计量；
- 峰谷指标；
- 月度波动性；
- 差分、ACF/PACF、滞后散点、热力图、FFT 频谱。

### `src/season_analysis.py`

负责：

- 四季负荷曲线；
- 负荷分布图；
- ECDF 图；
- Ramp 分布图。

### `src/emd_decomposition.py`

负责：

- 执行 EMD 分解；
- 保存 IMF 结果；
- 基于主频与能量分析 IMF；
- 输出 IMF 相关图表。

### `src/forecast_pipeline.py`

负责：

- 预测参数组织；
- 未分解预测；
- IMF 频率融合预测；
- 多策略比较与最佳策略选择；
- 输出最终预测结果。

### `src/evaluation.py`

负责：

- 计算 `MAE` / `RMSE` / `MAPE` / `R2`；
- 生成预测对比图；
- 输出高峰、低谷、节假日等误差分析结果。

---

## 11. 已知实现特点与注意事项

1. **当前流程已切换为多变量负荷预测。**  
   模型会联合使用目标序列、可用天气数值字段以及派生时间特征进行训练。

2. **命名与实现存在历史痕迹。**  
   某些输出名或函数名包含 `TCN`，但当前部分核心预测路径实际调用的是 `LSTMForecaster`。

3. **频率融合模型是轻量化近似实现。**  
   `SCINet`、`Autoformer`、`TimeXer` 不是完整论文复现版本，而是为了形成多分支对比流程而构建的简化结构。

4. **主流程包含终端交互。**  
   至少会要求输入 IMF 个数；绘制最终预测对比图时还会额外要求输入日期。

5. **节假日误差分析依赖额外节假日库。**  
   代码优先尝试 `holidays`，失败后回退到 `chinese-calendar`。

6. **EMD 分解上限为 10 个 IMF。**  
   `--imf-components` 的合法范围也是 `1-10`。

7. **运行成本不低。**  
   由于要生成大量图表并训练多个模型分支，在 CPU 环境下运行可能较慢。

---

## 12. 最简使用建议

如果你只是想先把项目跑通，建议按下面步骤：

### 第一步：安装依赖

```bash
pip install -r requirements.txt
```

### 第二步：直接运行示例流程

```bash
python main.py --imf-components 3
```

### 第三步：在终端按提示输入绘图日期

输入一个位于预测区间内的日期，例如程序提示范围中的某一天。

### 第四步：重点查看这些结果

建议优先查看：

- `outputs/forecast_results.csv`
- `outputs/forecast_metrics.csv`
- `outputs/best_forecast_strategy.txt`
- `outputs/decomposition_vs_non_decomposition_comparison.csv`
- `figures/actual_vs_predicted_load.png`
- `figures/prediction_error_distribution.png`
- `figures/emd_decomposition_overview.png`

---

## 13. 后续可改进方向

如果你准备继续完善这个仓库，比较值得优先改进的方向包括：

- 统一“TCN / LSTM”相关命名，消除历史歧义；
- 把交互式输入改成纯命令行参数，方便批量运行；
- 增加配置文件支持；
- 增加训练日志、模型保存和断点恢复；
- 提供更标准的实验对比脚本；
- 为频率融合分支替换为更完整的模型实现。

---

## 14. 一句话总结

如果用一句话概括当前仓库：

> 这是一个以**电力负荷单变量时序**为对象，集**数据清洗、统计分析、EMD 分解、分解/未分解预测比较与结果可视化**于一体的完整实验型项目仓库。
