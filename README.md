# 轻量级多变量负荷分析与 EMD 预测项目

## 1. 项目简介
本项目面向电力负荷数据的**分析与预测一体化场景**，在现有轻量级代码框架上完成整理与重构，目标是提供一个能够**直接运行、结构清晰、终端交互明确、README 完整**的多变量负荷分析与预测项目。

项目保留了原有负荷数据分析中的主要统计与图表逻辑，并将预测流程统一为：

> **EMD 分解 → IMF 分量预测 → 融合重构输出**

项目不引入庞大的原版大模型库，而是使用轻量级可运行实现，便于快速实验与本地运行：
- TCN：轻量时序卷积实现
- LSTM：轻量多变量 LSTM
- AutoFormer：轻量 Transformer Encoder 风格占位实现
- SCINet：轻量 SCINet 风格占位实现
- 混合预测：对每个 IMF 分量分别选择验证效果最优的模型后再融合
- 最优预测：直接输出当前候选方案中表现最好的最终结果

## 2. 项目结构

```text
project_root/
├─ main.py
├─ README.md
├─ requirements.txt
├─ data/
│  └─ quanzhou.csv
├─ outputs/
│  ├─ analysis/
│  └─ forecast/
└─ src/
   ├─ analysis/
   │  └─ load_analysis.py
   ├─ forecast/
   │  └─ load_forecasting.py
   ├─ models/
   │  └─ sequence_models.py
   ├─ utils/
   │  └─ time_selection.py
   ├─ data_loader.py
   ├─ preprocess.py
   ├─ emd_decomposition.py
   ├─ statistics_analysis.py
   ├─ time_scale_analysis.py
   ├─ season_analysis.py
   ├─ lstm_dataset.py
   ├─ lstm_model.py
   ├─ tcn_model.py
   ├─ evaluation.py
   └─ visualization.py
```

## 3. 数据格式要求
项目会自动识别时间列与负荷列，并尽量兼容中文列名。推荐数据至少包含以下字段：

- 时间列：如 `时间`、`timestamp`、`date` 等
- 目标负荷列：如 `负荷`、`load`、`power`、`demand` 等
- 其他数值特征：如温度、湿度、降雨量等

示例：

| 时间 | 负荷 | 最高温度℃ | 最低温度℃ | 平均温度℃ | 相对湿度(平均) | 降雨量（mm） |
|---|---:|---:|---:|---:|---:|---:|
| 2018/1/1 0:00 | 4454.57 | 20.7 | 9.3 | 14.3 | 40.0 | 0.0 |

### 数据要求说明
1. 时间列必须可被 `pandas.to_datetime()` 解析。
2. 负荷列必须是可转为数值的列。
3. 建议时间间隔尽量规则。
4. 若存在缺失值，程序会进行插值和补全。
5. 为保证“分析时间范围最少一周”的要求，建议数据不少于 7 天。

## 4. 环境安装方法

### 4.1 创建虚拟环境（推荐）
```bash
python -m venv .venv
source .venv/bin/activate
```

Windows PowerShell：
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 4.2 安装依赖
```bash
pip install -r requirements.txt
```

## 5. 负荷数据分析功能说明
分析模块保留并整理了原有主要内容，且现在所有结果都只针对**用户选定时间段**输出。主要包括：

1. **基础统计分析**
   - 均值、中位数、方差、标准差、偏度、峰度、分位数等
2. **多时间尺度分析**
   - 年 / 月 / 周 / 日 / 小时尺度统计表
3. **月度波动分析**
   - 月均值、月标准差、月方差、变异系数
4. **季节性分析**
   - 季节负荷曲线
5. **工作日 / 周末分析**
   - 分小时平均负荷曲线
6. **差分与相关性分析**
   - 一阶差分、二阶差分、ACF、PACF、滞后散点图、相关热力图、FFT 频谱
7. **统计特性图表**
   - 负荷分布直方图、经验分布函数、爬坡分布图
8. **峰谷特征统计**
   - 日峰值、谷值、峰谷差、峰谷比

## 6. 时间选择模块使用说明
分析与预测前，都会先完成时间范围选择。

支持三种方式：

1. **全部数据**
2. **自定义开始时间 + 结束时间**
3. **最近 N 天 / 最近 N 周**

### 时间范围校验规则
- 最短分析范围为 **1 周**
- 若开始时间晚于结束时间，程序会给出错误提示
- 若时间超出数据边界，程序会自动截断到可用范围并提示
- 若筛选后数据为空，程序会终止并提示重新选择

## 7. 负荷预测流程说明
预测流程固定为：

1. 对目标负荷序列进行 **EMD 分解**
2. 选取指定数量的 IMF 分量，并额外构建残差分量
3. 对每个 IMF / 残差分量分别进行建模预测
4. 将各分量预测结果进行融合重构
5. 输出最终预测结果、指标文件、必要图表与记录文件

### 多变量预测说明
项目保持多变量输入，不会退化为单变量预测。模型输入包含：
- 当前 IMF 分量（作为目标分量信号）
- 时间特征（年、月、日、小时、周期编码等）
- 其他数值型外生特征（如温度、湿度、降雨量等）

## 8. 六种预测方式说明

### 8.1 TCN
对每个 IMF 分量使用轻量级 TCN 进行预测，再融合输出总预测结果。

### 8.2 LSTM
对每个 IMF 分量使用轻量级 LSTM 进行预测，再融合输出总预测结果。

### 8.3 AutoFormer
使用轻量级 Transformer Encoder 风格占位实现，对每个 IMF 分量预测后融合。

### 8.4 SCINet
使用轻量级 SCINet 风格占位实现，对每个 IMF 分量预测后融合。

### 8.5 混合预测
对每个 IMF 分量分别运行：
- TCN
- LSTM
- AutoFormer
- SCINet

然后基于验证表现，为**每个 IMF 分量**选出最优模型，再将这些最优分量结果融合为最终预测结果。

> 说明：当前轻量实现以候选结果比较的方式完成分量选模，适合快速横向实验。

### 8.6 最优预测
在候选方案中比较：
- TCN
- LSTM
- AutoFormer
- SCINet
- 混合预测

然后直接输出整体表现最好的最终结果，并保存最优方案摘要文件。

## 9. 终端命令示例

### 9.1 交互式运行
```bash
python main.py
```
运行后会依次提示：
- 选择模式：仅分析 / 仅预测 / 分析+预测
- 选择时间范围：全部 / 自定义 / 最近 N 天 / 最近 N 周
- 若包含预测，再选择预测方式：TCN / LSTM / AutoFormer / SCINet / 混合预测 / 最优预测

### 9.2 仅分析：全部数据
```bash
python main.py --mode analysis --time-mode all
```

### 9.3 仅分析：自定义时间范围
```bash
python main.py --mode analysis --time-mode range --start "2018-01-01 00:00" --end "2018-03-31 23:45"
```

### 9.4 仅分析：最近 30 天
```bash
python main.py --mode analysis --time-mode recent --recent-value 30 --recent-unit days
```

### 9.5 仅分析：最近 8 周
```bash
python main.py --mode analysis --time-mode recent --recent-value 8 --recent-unit weeks
```

### 9.6 TCN 预测
```bash
python main.py --mode forecast --time-mode recent --recent-value 12 --recent-unit weeks --forecast-method tcn
```

### 9.7 LSTM 预测
```bash
python main.py --mode forecast --time-mode recent --recent-value 12 --recent-unit weeks --forecast-method lstm
```

### 9.8 AutoFormer 预测
```bash
python main.py --mode forecast --time-mode recent --recent-value 12 --recent-unit weeks --forecast-method autoformer
```

### 9.9 SCINet 预测
```bash
python main.py --mode forecast --time-mode recent --recent-value 12 --recent-unit weeks --forecast-method scinet
```

### 9.10 混合预测
```bash
python main.py --mode forecast --time-mode recent --recent-value 12 --recent-unit weeks --forecast-method hybrid
```

### 9.11 最优预测
```bash
python main.py --mode forecast --time-mode recent --recent-value 12 --recent-unit weeks --forecast-method best
```

### 9.12 分析 + 预测
```bash
python main.py --mode both --time-mode recent --recent-value 12 --recent-unit weeks --forecast-method hybrid
```

### 9.13 指定训练参数
```bash
python main.py \
  --mode forecast \
  --time-mode recent \
  --recent-value 12 \
  --recent-unit weeks \
  --forecast-method lstm \
  --lookback 672 \
  --horizon 96 \
  --epochs 10 \
  --batch-size 128 \
  --train-ratio 0.8 \
  --imf-components 6
```

## 10. 输出结果说明

### 10.1 分析输出目录
分析结果位于：
```text
outputs/analysis/<时间标签>/
├─ dataset_metadata.csv
├─ data_quality_report.csv
├─ figures/
└─ tables/
```

其中：
- `tables/filtered_load_data.csv`：时间筛选后的数据
- `tables/basic_statistics.csv`：基础统计表
- `tables/yearly_statistics.csv` 等：多时间尺度统计表
- `figures/`：分析图表输出

### 10.2 预测输出目录
预测结果位于：
```text
outputs/forecast/<方法>_<时间标签>_<时间戳>/
├─ figures/
├─ tables/
├─ tcn_forecast.csv / lstm_forecast.csv / autoformer_forecast.csv / scinet_forecast.csv
├─ hybrid_forecast.csv
├─ best_forecast.csv
├─ *_metrics.csv
├─ *_metrics.txt
├─ *_forecast.png
├─ *_training_loss.png
├─ imf_model_selection.csv
└─ best_method_summary.txt
```

### 输出重点说明
- 单模型预测：
  - 当前模型预测结果 CSV
  - 当前模型指标 CSV/TXT
  - 当前模型预测图
  - IMF 分量指标表
- 混合预测：
  - `hybrid_forecast.csv`
  - `hybrid_metrics.csv`
  - `imf_model_selection.csv`
- 最优预测：
  - `best_forecast.csv`
  - `best_metrics.csv`
  - `best_method_summary.txt`

### 不再输出的内容
本次重构**不再生成多模型最终预测效果横向总对比图**，避免产生不必要的大对比图输出。

## 11. 注意事项
1. 本项目采用**轻量级实现**，优先保证流程清晰、结构统一和终端可运行。
2. `AutoFormer` 与 `SCINet` 为轻量占位实现，不是原论文完整复现版本。
3. 混合预测采用“**各 IMF 分量分别选取验证效果最佳模型后再融合**”的策略。
4. 最优预测采用“**直接输出当前候选方案中整体表现最佳的最终结果**”的策略。
5. 若筛选时间范围太短，无法完成 `lookback + horizon` 的窗口切片，程序会提示缩小窗口参数或扩大时间范围。
6. 若本地没有 GPU，程序会自动使用 CPU。
7. 若本地已安装 `PyEMD / EMD-signal`，程序会优先使用该实现；未安装时会自动退回到轻量级分解方案。
8. 如果你替换了自己的数据文件，建议先运行一次仅分析模式检查时间列和负荷列识别是否正确。
