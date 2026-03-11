# 电力负荷分析与预测项目（仅TCN版本）

## 1. 项目简介
本项目用于电力负荷数据的全流程分析与预测，当前版本已调整为**仅使用 TCN** 进行预测，不再运行 LSTM。

核心流程：

读取数据  
→ 数据清洗  
→ 负荷特性分析  
→ EMD 分解（最多 10 个 IMF）  
→ 自动选择最优 IMF 分量数  
→ TCN 预测  
→ 对比“未分解预测”与“EMD 分解预测”  
→ 自动选择最优方案  
→ 输出中文图表与中文结果文件

## 2. 本版本关键变更
- 仅保留 **TCN** 预测流程；
- EMD 分解分量上限为 `MAX_IMF = 10`；
- 自动测试 IMF 分量数（1~实际分量数），按**验证集 RMSE 最小优先**，再参考 MAE、MAPE；
- 自动对比“未分解 + TCN”与“EMD分解 + TCN”，自动写出最优预测方案；
- 图表标题、坐标轴、图例与主要输出文件均采用中文命名；
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

## 5. 主要输出
### 表格（`outputs/`）
- `基础统计结果.csv`
- `清洗后数据.csv`
- `EMD分解结果.csv`
- `IMF分量数选择结果.csv`
- `最优IMF分量数.txt`
- `TCN预测结果_未分解.csv`
- `TCN预测结果_EMD分解.csv`
- `分解与未分解预测对比.csv`
- `最优预测方案.txt`

### 图像（`figures/`）
- `原始负荷时序图.png`
- `月平均负荷图.png`
- `月负荷箱线图.png`
- `工作日与周末对比图.png`
- `日内平均负荷曲线图.png`
- `月-小时热力图.png`
- `周-小时热力图.png`
- `季节性负荷分析图.png`
- `EMD分解总览图.png`
- `IMF分量图.png`
- `TCN训练损失曲线.png`
- `预测值与真实值对比图.png`
- `预测误差分布图.png`
- `分解与未分解效果对比图.png`
- `IMF分量数选择对比图.png`

## 6. 说明
- 若 EMD 实际分量少于 10，则按实际分量数进行 IMF 选择测试；
- 不保证“使用全部 IMF”最优，程序会自动搜索更优分量深度；
- 控制台会输出：当前测试 IMF 分量数、未分解预测指标、分解预测指标、最优 IMF 分量数、最优预测方案。
