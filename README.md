# 电力负荷分析与预测项目

> 一个面向**电力负荷时序分析、EMD 分解、多尺度特征挖掘与短期预测**的完整实验项目。

本项目围绕一份真实电力负荷 CSV 数据，构建了从**数据自动发现 → 编码识别 → 字段推断 → 数据清洗 → 统计分析 → 多时间尺度分析 → 季节性分析 → 重过载识别 → EMD 分解 → TCN 预测 → 误差评估与模型对比**的一体化流程。

如果你希望把这份项目用于以下场景，这个仓库是比较合适的：

- 课程设计 / 毕业设计；
- 电力负荷预测方向的实验复现；
- 论文原型验证；
- 时序建模入门；
- 需要快速产出图表、统计表和预测结果的项目展示。

---

## 1. 项目概览

### 1.1 项目解决什么问题

电力负荷数据通常同时具有以下特点：

- 明显的日周期、周周期、季节周期；
- 不同时间尺度上的波动模式叠加；
- 存在高峰、低谷、异常突变与节假日效应；
- 需要同时兼顾“解释性分析”和“预测精度”。

本项目的目标，就是把这些问题放在同一个工程流程中统一处理：

1. 自动找到仓库中的主数据集；
2. 自动识别时间列与负荷列；
3. 对原始数据执行清洗与质量检查；
4. 输出基础统计结果与多尺度分析表；
5. 生成可直接用于汇报/论文的图像；
6. 使用 EMD 将原始负荷序列分解为不同频率成分；
7. 基于 TCN 执行多步预测；
8. 对比分解方案与未分解方案的效果；
9. 输出误差指标、误差分布图和分场景误差分析结果。

### 1.2 项目主流程

```text
CSV 原始数据
   ↓
自动查找与读取
   ↓
字段识别（时间列 / 负荷列）
   ↓
数据清洗与质量报告
   ↓
统计分析 / 时间尺度分析 / 季节分析 / 重过载识别
   ↓
EMD 分解与 IMF 频率分类
   ↓
TCN 预测（含 IMF 分解方案）
   ↓
误差评估 / 模型比较 / 图表输出
```

---

## 2. 当前项目的实际数据情况

仓库当前自带的数据集位于 `data/quanzhou.csv`，程序会优先选择它作为主数据源。

### 2.1 数据集已确认信息

根据当前仓库代码实际读取结果，这份示例数据具有以下特征：

- 数据文件：`data/quanzhou.csv`
- 文件编码：`gb18030`
- 时间列：`时间`
- 负荷列：`负荷`
- 总记录数：`35040`
- 时间范围：`2018-01-01 00:00:00` 至 `2018-12-31 23:45:00`
- 采样粒度：`15 分钟`
- 示例中未检测到无效时间、无效负荷和重复时间戳

### 2.2 当前示例数据包含字段

当前 CSV 样例列如下：

- `时间`
- `负荷`
- `最高温度℃`
- `最低温度℃`
- `平均温度℃`
- `相对湿度(平均)`
- `降雨量（mm）`

虽然当前预测主流程主要使用**时间列 + 负荷列**，但这些气象字段为后续扩展多变量建模提供了很好的基础。

---

## 3. 项目核心功能

### 3.1 自动数据发现与鲁棒读取

项目会自动在仓库中搜索 CSV 文件，并优先选择：

- 位于 `data/` 目录中的文件；
- 文件体积更大的候选文件；
- 非 `outputs/`、`figures/` 等生成目录中的文件。

同时，读取时会尝试多种常见编码：

- `utf-8`
- `utf-8-sig`
- `gb18030`
- `gbk`
- `big5`
- `latin1`

这使得项目对中文数据集尤其友好。

### 3.2 自动识别时间列与负荷列

程序无需手工写死字段名，会通过关键词与兜底策略自动推断字段：

**时间列支持关键词：**

- `time`
- `date`
- `datetime`
- `timestamp`
- `时间`
- `日期`
- `记录时间`
- `采样时间`

**负荷列支持关键词：**

- `load`
- `power`
- `demand`
- `mw`
- `kw`
- `负荷`
- `电力`
- `功率`
- `有功`

如果列名不规范，程序还会进一步：

- 用时间解析成功率判断哪个字段更像时间列；
- 用数值化成功率判断哪个字段更像负荷列。

### 3.3 数据清洗与质量报告

程序会自动执行以下预处理：

- 时间列转换为 `datetime`；
- 负荷列转换为数值型；
- 删除无法解析的时间记录；
- 检测并统计无效负荷记录；
- 对重复时间戳按平均值聚合；
- 对缺失负荷使用时间插值；
- 必要时再执行前向/后向填充；
- 自动补充多种时间派生特征。

新增时间特征包括：

- `year`
- `month`
- `day`
- `hour`
- `weekday`
- `weekday_name`
- `is_weekend`
- `date`
- `season`

### 3.4 多维分析能力

项目不仅做预测，还输出大量可解释分析结果，包括：

- 基础统计分析；
- 年 / 月 / 周 / 日 / 小时多时间尺度统计；
- 月度波动性分析；
- 日峰谷指标分析；
- 一阶差分 / 二阶差分分析；
- 自相关 / 偏自相关分析；
- 滞后散点关系分析；
- FFT 频谱分析；
- 季节日内负荷特征分析；
- 工作日 / 周末负荷差异分析；
- 重过载日与重过载月识别。

### 3.5 EMD 分解与频率分层

程序会对清洗后的负荷序列执行 EMD（经验模态分解），并进一步：

- 生成 IMF 分量结果表；
- 计算每个 IMF 的主导频率与能量；
- 自动把 IMF 划分为高频 / 中频 / 低频；
- 生成 IMF 频谱、能量占比、重构结果与波动分解图。

### 3.6 预测与对比

当前主预测流程以 **TCN（Temporal Convolutional Network）** 为核心，支持：

- 原始序列直接预测；
- EMD 分解后按给定 IMF 数量进行融合预测；
- 自动比较方案优劣并选择最佳结果；
- 输出标准误差指标与图像。

此外，项目内部还保留了若干序列模型接口，用于模型对比实验扩展。

---

## 4. 项目目录结构

```text
Jingrui-Tao/
├── data/
│   └── quanzhou.csv                 # 示例电力负荷数据
├── figures/
│   └── README.md                    # 图像输出说明文档
├── src/
│   ├── __init__.py
│   ├── data_loader.py               # CSV 搜索、编码检测、数据读取
│   ├── preprocess.py                # 字段推断、清洗、特征构造
│   ├── statistics_analysis.py       # 基础统计、差分、相关性、频谱分析
│   ├── time_scale_analysis.py       # 年/月/周/日/小时尺度统计与绘图
│   ├── season_analysis.py           # 季节性与统计特征图
│   ├── overload_analysis.py         # 重过载识别
│   ├── emd_decomposition.py         # EMD 分解、IMF 分析与绘图
│   ├── tcn_model.py                 # TCN 模型定义
│   ├── lstm_model.py                # LSTM 模型定义（扩展接口）
│   ├── lstm_dataset.py              # 序列样本构造与 DataLoader
│   ├── forecast_pipeline.py         # 预测训练与方案对比主流程
│   ├── evaluation.py                # 误差评估与误差分析图
│   ├── model_comparison.py          # 模型对比与最佳模型选择
│   └── visualization.py             # 全局绘图风格与图像保存
├── main.py                          # 项目入口
├── requirements.txt                 # Python 依赖列表
└── README.md                        # 项目说明文档
```

> 运行后还会自动生成：
>
> - `outputs/`：统计表、质量报告、分解结果、预测结果、指标文件；
> - `figures/`：分析图、分解图、预测图、误差图。

---

## 5. 运行环境要求

### 5.1 基础环境

建议使用：

- Python `3.10` 或 `3.11`
- `pip` / `venv` 或 `conda`
- Windows / macOS / Linux 均可

### 5.2 PyTorch 与 GPU 说明

本项目支持 CPU 运行，也支持在可用时自动切换到 GPU：

- 若 `torch.cuda.is_available()` 为 `True`，程序会自动使用 CUDA；
- 若没有 GPU，则会自动回退为 CPU。

如果你希望使用 GPU，请自行安装与你本机 CUDA 版本匹配的 PyTorch 版本。

---

## 6. 安装方式

### 6.1 推荐：创建虚拟环境

```bash
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate        # Windows
```

### 6.2 安装依赖

```bash
pip install -r requirements.txt
```

### 6.3 如果你需要 GPU 版 PyTorch

由于不同机器的 CUDA 版本不同，建议先参考 PyTorch 官方安装页，再执行对应命令。例如：

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

如果你已经单独安装了 GPU 版 `torch`，再次安装本仓库其他依赖即可。

---

## 7. 如何运行

### 7.1 交互式运行

```bash
python main.py
```

程序会提示输入：

```text
请输入 IMF 分解个数（1-10）:
```

输入一个 `1` 到 `10` 之间的整数后，程序将开始执行完整流程。

### 7.2 命令行直接指定 IMF 数量

```bash
python main.py --imf-components 3
```

这个方式更适合：

- 服务器非交互运行；
- 批量实验；
- 固定实验参数；
- 在 shell 脚本中调用。

---

## 8. 预测配置说明

当前默认预测配置位于 `main.py` 中，核心参数如下：

- `lookback = 672`
- `train_ratio = 0.8`
- `dropout = 0.1`
- `learning_rate = 1e-3`
- `epochs = 20`
- `batch_size = 128`
- `random_seed = 42`
- `horizon = 96`

### 8.1 这些参数意味着什么

在当前示例数据为 **15 分钟粒度** 的前提下：

- `lookback = 672` 表示回看 `672 × 15 分钟 = 7 天` 的历史窗口；
- `horizon = 96` 表示预测未来 `96 × 15 分钟 = 24 小时`。

也就是说，当前模型默认是在做：

> 用过去 7 天的负荷历史，预测未来 1 天的负荷曲线。

### 8.2 IMF 个数说明

- 输入范围限制：`1-10`
- 程序内部最大分解上限：`10`
- 如果实际 EMD 产生的有效 IMF 分量少于你输入的值，建议适当减小该参数重新运行。

---

## 9. 程序输出内容

程序运行完成后，通常会在 `outputs/` 中生成以下文件（实际生成情况以运行阶段是否成功完成为准）。

### 9.1 数据与统计类输出

- `cleaned_data.csv`：清洗后的主数据表
- `data_quality_report.csv`：数据质量报告
- `dataset_metadata.csv`：数据集元信息
- `basic_statistics.csv`：基础统计结果
- `yearly_statistics.csv`：年度统计表
- `monthly_statistics.csv`：月度统计表
- `weekly_statistics.csv`：周统计表
- `daily_statistics.csv`：日统计表
- `hourly_statistics.csv`：小时统计表
- `monthly_volatility_statistics.csv`：月波动性统计
- `daily_peak_valley_metrics.csv`：日峰谷指标
- `heavy_overload_day_detection_results.csv`：重过载日识别结果
- `heavy_overload_month_detection_results.csv`：重过载月识别结果

### 9.2 EMD 与 IMF 输出

- `emd_decomposition_results.csv`：所有 IMF 分量结果
- `imf_frequency_features.csv`：IMF 主导频率与能量特征

### 9.3 预测与评估输出

- `forecast_evaluation_metrics.csv`：预测指标
- `model_comparison.csv`：模型对比结果
- `best_model.txt`：最佳方案名称
- `peak_period_error_stats.csv`：高峰时段误差统计
- `valley_period_error_stats.csv`：低谷时段误差统计
- `holiday_error_stats.csv`：节假日 / 非节假日误差统计

---

## 10. 图像输出内容

项目会在 `figures/` 下输出分析图。常见图片包括：

### 10.1 时间尺度与分布分析图

- `raw_load_timeseries.png`
- `monthly_average_load.png`
- `monthly_load_boxplot.png`
- `weekday_vs_weekend_load.png`
- `seasonal_load_analysis.png`
- `load_distribution_histogram.png`
- `load_empirical_distribution_function.png`
- `load_ramp_distribution.png`

### 10.2 相关性与频域分析图

- `diff1_timeseries.png`
- `diff1_distribution.png`
- `diff2_timeseries.png`
- `diff2_distribution.png`
- `acf_plot.png`
- `pacf_plot.png`
- `lag_scatter_plots.png`
- `correlation_heatmap.png`
- `fft_spectrum.png`

### 10.3 EMD 分解图

- `emd_decomposition_overview.png`
- `imf_components.png`
- `imf_spectrum_overview.png`
- `imf_frequency_classification.png`
- `imf_energy_ratio.png`
- `imf_reconstruction.png`
- `imf_volatility_decomposition.png`

### 10.4 预测与误差图

- `actual_vs_predicted_load.png`
- `prediction_error_distribution.png`
- `peak_period_error.png`
- `valley_period_error.png`
- `holiday_prediction_error.png`
- `error_histogram.png`
- `error_qq_plot.png`
- `27_model_comparison_bar.png`

关于这些图片的详细中文解释，可以继续查看：

- `figures/README.md`

---

## 11. 输入数据要求

如果你打算替换为自己的数据，建议至少满足以下条件。

### 11.1 最低要求

CSV 至少包含：

- 一列时间字段；
- 一列负荷字段。

### 11.2 推荐要求

- 时间间隔尽量固定；
- 时间列可被 `pandas.to_datetime()` 正确解析；
- 负荷列应为数值或可转换为数值；
- 数据量足够长，至少明显大于 `lookback + horizon`；
- 若要做季节性和年/月统计，建议覆盖较长时间跨度。

### 11.3 切换数据时需要注意

当你替换数据集后，以下内容可能需要重新评估：

- `lookback` 是否仍合适；
- `horizon` 是否匹配业务目标；
- 采样间隔是否仍为 15 分钟；
- EMD 分解得到的 IMF 数量是否足够；
- 节假日分析是否仍适用于中国地区数据。

---

## 12. 项目的优点与局限

### 12.1 优点

- 文档化程度较高，适合教学和展示；
- 从数据读取到预测输出流程完整；
- 对中文数据集和编码兼容较友好；
- 自动化程度高，易于直接运行；
- 图表与表格输出较丰富，适合论文附图和结果汇报。

### 12.2 当前局限

- 主流程目前主要使用单变量负荷序列预测；
- 虽有气象字段，但尚未在主预测流程中显式使用；
- 默认训练参数比较偏实验原型，不一定适合所有数据集；
- 节假日分析默认基于中国节假日逻辑；
- 不同模型接口虽被保留，但主流程核心仍是 TCN 对比方案。

---

## 13. 适合如何扩展

如果你想继续完善这个项目，推荐的扩展方向包括：

### 13.1 多变量预测

把以下字段加入建模输入：

- 温度
- 湿度
- 降雨量
- 节假日标签
- 工作日 / 周末标签

### 13.2 模型增强

可以继续扩展：

- 更完整的 SCINet / Autoformer / PatchTST 等实现；
- 多模型集成；
- 超参数搜索；
- 交叉验证；
- 多步滚动预测与递归预测对比。

### 13.3 工程化提升

- 增加配置文件（如 YAML / TOML）；
- 增加日志系统；
- 增加实验记录；
- 将输出目录按时间戳区分；
- 补充单元测试和自动化检查。

---

## 14. 快速开始建议

如果你第一次接触这个项目，建议按下面步骤走：

1. 安装依赖；
2. 运行 `python main.py --imf-components 3`；
3. 先查看 `outputs/data_quality_report.csv`；
4. 再查看 `figures/raw_load_timeseries.png`；
5. 接着查看 `figures/emd_decomposition_overview.png`；
6. 最后查看预测结果图和误差分析图；
7. 根据效果再调整 `imf-components`、`epochs`、`lookback` 等参数。

---

## 15. 常见问题

### Q1：程序为什么会要求输入 IMF 个数？

因为当前流程支持“用户指定保留多少个 IMF 分量”来参与分解预测对比，这一参数会直接影响 EMD 方案的结构。

### Q2：为什么换了自己的数据后效果不好？

常见原因包括：

- 数据周期与当前参数不匹配；
- 数据长度不够；
- 时间间隔不稳定；
- 数据存在大量缺失或异常；
- 负荷模式和示例数据差异过大。

### Q3：可以不交互输入吗？

可以，直接使用：

```bash
python main.py --imf-components 3
```

### Q4：项目能不能只做分析、不做预测？

可以通过修改 `main.py`，在进入预测模块前提前结束流程；当前代码结构已经把分析模块与预测模块拆分得比较清晰。

---

## 16. 许可证与说明

本仓库当前更像一个**研究/课程/实验型项目模板**。如果你将它用于：

- 学术汇报；
- 课程作业；
- 论文实验；
- 工程原型；

建议你在正式使用前，进一步补充：

- 数据来源说明；
- 结果复现实验记录；
- 模型参数版本记录；
- 更严格的测试和评估方案。

---

如果你接下来还想继续优化这个项目，我建议优先做两件事：

1. 把气象特征并入主预测模型，升级为多变量预测；
2. 把训练参数抽离到配置文件中，让实验复现实验更方便。
