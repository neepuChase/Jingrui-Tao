# 电力负荷分析与预测全流程项目（EMD + LSTM/TCN）

## 1. 项目简介
本项目是一个面向电力负荷场景的**时序分析与预测一体化流水线**，用于从原始负荷数据出发，完成数据清洗、统计分析、季节性规律挖掘、重过载风险识别、EMD 分解、多模型预测与效果评估，最终输出可复用的图表与报告。

项目目标：
- 为配网/用电负荷研究提供标准化分析流程；
- 支持从“**可解释分析**”到“**可量化预测**”的闭环；
- 在 **CPU-only** 环境即可运行，便于教学、研究和工程快速验证。

---

## 2. 功能特点
本项目聚焦以下核心能力：

- **多时间尺度负荷特性分析**（年 / 月 / 周 / 日 / 小时）
- **季节性分析**（季节分布、工作日与周末差异、月-小时热力）
- **重过载指标检测**（按日/按月统计重过载风险）
- **EMD 分解**（将负荷序列分解为多个 IMF 分量）
- **LSTM 预测**（基于时序窗口学习非线性动态）
- **TCN 预测**（并行卷积时序建模，适配长感受野）
- **模型比较与自动选择**（基于 RMSE / MAE / MAPE 等指标）
- **可视化分析**（时间序列、分解结果、预测对比、模型横向比较）

---

## 3. 项目流程
完整流程如下：

```text
数据加载
→ 数据预处理
→ 负荷特性分析
→ 重过载指标分析
→ EMD分解
→ IMF分量预测
→ LSTM / TCN预测
→ 模型比较
→ 最优模型选择
→ 预测结果输出
```

可将其理解为三层结构：
1. **数据与质量层**：读取、字段识别、缺失处理、时间索引规范化；
2. **分析与建模层**：负荷特征分析 + 分解建模 + 深度学习预测；
3. **评估与交付层**：误差评估、最佳模型选择、图表与 CSV 报告落地。

---

## 4. 项目结构
根据当前仓库结构，核心目录与文件如下：

```text
.
├── data/                       # 输入数据目录（示例：quanzhou.csv）
├── src/                        # 核心源码
│   ├── data_loader.py          # 数据加载与编码兼容
│   ├── preprocess.py           # 清洗、字段规范化、特征构建
│   ├── statistics_analysis.py  # 统计分析
│   ├── time_scale_analysis.py  # 多时间尺度分析
│   ├── season_analysis.py      # 季节性分析
│   ├── emd_decomposition.py    # EMD 分解
│   ├── lstm_dataset.py         # 序列样本构建
│   ├── lstm_model.py           # LSTM 模型定义
│   ├── forecast_pipeline.py    # 预测主流程（分量级训练与重构）
│   ├── evaluation.py           # 指标计算与评估
│   └── visualization.py        # 绘图风格与图像保存
├── outputs/                    # 结果表格输出目录（运行后生成）
├── figures/                    # 图像输出目录（运行后生成）
├── main.py                     # 统一主入口
└── requirements.txt            # Python 依赖
```

---

## 5. 环境依赖
建议使用 Python 3.9+。

安装依赖：

```bash
pip install -r requirements.txt
```

说明：
- 项目支持 **CPU-only** 环境；
- 默认不依赖 GPU/CUDA，也可完成全流程分析与预测。

---

## 6. 运行方法
在仓库根目录执行：

```bash
python main.py
```

程序将自动完成数据读取、分析、建模、评估，并在 `outputs/` 与 `figures/` 下生成结果。

---

## 7. 输出结果
### 表格输出（`outputs/`）

- `forecast_results.csv`：预测值与真实值对齐结果
- `model_comparison.csv`：模型评估指标对比（RMSE / MAE / MAPE 等）
- `heavy_overload_days.csv`：日尺度重过载识别结果
- `heavy_overload_months.csv`：月尺度重过载统计结果

### 图像输出（`figures/`）

- `load_timeseries.png`：负荷时间序列概览
- `seasonal_patterns.png`：季节性与周期规律图
- `emd_decomposition.png`：EMD 分解结果（原序列 + IMF）
- `forecast_vs_actual.png`：预测与真实值对比图
- `model_comparison.png`：模型性能横向比较图

---

## 8. 示例运行流程
以下是一个典型使用流程：

1. 将原始负荷数据（CSV）放入 `data/`；
2. 安装依赖：`pip install -r requirements.txt`；
3. 启动主程序：`python main.py`；
4. 在 `outputs/` 查看指标表与预测结果；
5. 在 `figures/` 查看趋势图、分解图、预测对比图与模型对比图；
6. 根据 `model_comparison.csv` 读取最优模型并用于后续部署或复训。

---

## 9. 技术栈
- **Python**
- **Pandas**
- **NumPy**
- **Matplotlib**
- **PyTorch**
- **PyEMD**
- **Scikit-learn**

---

## 10. 未来扩展
可进一步拓展以下方向：

- 引入 **VMD** 分解，与 EMD 做互补对比；
- 新增 **Transformer** 系列时序预测模型；
- 集成 **TimesNet / Autoformer** 等新型架构；
- 加入 **SHAP** 可解释性分析，提升模型可审计性。

---

## 11. 作者信息
- 作者：Jingrui Tao（项目维护者）
- 方向：电力负荷分析、时序预测、智能配用电数据建模
- 欢迎基于本项目进行二次开发、实验对比与工程落地。
