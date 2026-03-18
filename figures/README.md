# figures 目录图像说明

本文件用于说明本仓库代码运行后，`figures/` 目录中**可能生成的图片**。为便于和程序输出结果一一对应，以下各节都保留了图片的**英文文件名**，并使用中文解释图片内容、用途与阅读方式。

> 说明：
> 1. 图片是否最终生成，取决于主流程是否顺利执行到对应步骤。
> 2. 预测相关图片会基于程序自动选择的“最佳预测方案”输出。
> 3. 若节假日依赖库不可用，节假日误差分析相关图片可能无法生成。

---

## 1. 基础时间尺度分析图

### `raw_load_timeseries.png`
**英文名称：Raw Load Time Series**

中文说明：原始负荷时间序列图。横轴是时间，纵轴是负荷值，用来展示全时间范围内负荷随时间变化的整体趋势，是所有分析中最基础的一张图。通过这张图通常可以直接观察到长期趋势、周期性变化、异常波动以及尖峰负荷出现的位置。

### `monthly_average_load.png`
**英文名称：Monthly Average Load**

中文说明：月平均负荷柱状图。该图按 1 到 12 月分别统计平均负荷，用来比较不同月份的整体用电水平差异。适合观察季节性强弱、夏季或冬季负荷是否偏高，以及全年负荷在月份维度上的集中情况。

### `monthly_load_boxplot.png`
**英文名称：Monthly Load Boxplot**

中文说明：月负荷箱线图。每个月对应一个箱体，用来展示该月负荷分布的中位数、四分位区间和离群点情况。相比月平均图，这张图更适合观察某个月内部的波动幅度是否更大，以及是否存在明显异常值。

### `weekday_vs_weekend_load.png`
**英文名称：Weekday vs Weekend Load**

中文说明：工作日与周末负荷对比图。该图按照小时维度分别绘制工作日和周末的平均负荷曲线，用来比较两类日期在一天 24 小时内的典型用电形态差异，例如早高峰、午后平台和夜间回落是否存在明显区别。

---

## 2. 季节与统计特征图

### `seasonal_load_analysis.png`
**英文名称：Seasonal Load Analysis**

中文说明：季节负荷分析图。该图会分别绘制春、夏、秋、冬四个季节在不同小时上的平均负荷曲线，用来观察各季节的日内负荷模式差异。它特别适合判断不同季节是否在高峰时段、低谷时段或峰谷差上存在显著区别。

### `load_distribution_histogram.png`
**英文名称：Load Distribution Histogram**

中文说明：负荷分布直方图。该图统计全部负荷值的频数分布，并叠加核密度曲线，用来判断样本主要集中在哪些负荷区间，以及负荷分布是否偏态、是否存在长尾现象。

### `load_empirical_distribution_function.png`
**英文名称：Load Empirical Distribution Function**

中文说明：负荷经验分布函数图。该图以累计概率形式展示负荷值分布情况，适合用来判断某个负荷阈值以下的数据占比，或者反过来估计高负荷区间出现的频率。

### `load_ramp_distribution.png`
**英文名称：Load Ramp Distribution**

中文说明：负荷爬坡分布图。这里的“Ramp”表示相邻时刻之间的负荷变化量。该图展示负荷增减幅度的分布情况，可用于分析系统负荷变化是否平稳，以及短时突升、突降出现得是否频繁。

---

## 3. 重过载识别图

### `daily_heavy_overload_detection.png`
**英文名称：Daily Heavy Overload Detection**

中文说明：日尺度重过载识别图。该图将每日平均负荷与“当月平均负荷 × 1.3”的阈值进行对比，并标出超过阈值的日期。它主要用于识别在日尺度上明显偏高的重负荷日。

### `monthly_heavy_overload_detection.png`
**英文名称：Monthly Heavy Overload Detection**

中文说明：月尺度重过载识别图。该图将每个月平均负荷与“当年平均负荷 × 1.3”的阈值进行对比，并高亮超过阈值的月份。它适合用来识别全年中负荷显著偏高的重点月份。

---

## 4. EMD 分解结果图

### `emd_decomposition_overview.png`
**英文名称：EMD Decomposition Overview**

中文说明：EMD 分解总览图。第一行显示原始负荷序列，后续每一行显示一个 IMF 分量。该图用来整体查看原始序列在经验模态分解后被拆解成哪些不同频率层次的成分，是理解分解结果的核心图像。

### `imf_components.png`
**英文名称：IMF Components**

中文说明：IMF 分量叠加图。该图在同一张坐标系中绘制所有 IMF 分量随时间变化的曲线，用来直观比较不同 IMF 分量的振荡幅度、频率差异及其时间变化特征。

---

## 5. 预测效果图

### `actual_vs_predicted_load.png`
**英文名称：Actual vs Predicted Load**

中文说明：真实值与预测值对比图。该图在同一时间轴上绘制实际负荷与预测负荷，是衡量预测模型整体拟合效果最直接的一张图。两条曲线越接近，说明模型对负荷变化趋势和波动细节的刻画越好。

### `prediction_error_distribution.png`
**英文名称：Prediction Error Distribution**

中文说明：预测误差分布图。这里的误差是实际值减去预测值。图中使用直方图和核密度曲线展示误差分布，用来判断模型是否存在系统性高估或低估，以及误差是否主要集中在零附近。

---

## 6. 分场景误差分析图

### `peak_period_error.png`
**英文名称：Peak Period Error**

中文说明：高峰时段误差图。代码中高峰时段定义为 **18:00-21:00**。该图通过箱线图展示高峰时段预测误差的分布情况，用来观察模型在用电高峰期是否更容易出现较大偏差。

### `valley_period_error.png`
**英文名称：Valley Period Error**

中文说明：低谷时段误差图。代码中低谷时段定义为 **02:00-04:00**。该图展示低谷时段的误差分布，用来比较模型在系统低负荷阶段的预测稳定性。

### `holiday_prediction_error.png`
**英文名称：Holiday Prediction Error**

中文说明：节假日预测误差图。该图按“Holiday / Non-Holiday”对预测误差进行箱线图对比，用来分析模型在节假日与非节假日两种场景下的表现差异。如果节假日负荷模式更复杂，这张图通常能反映出误差增大的情况。

### `error_histogram.png`
**英文名称：Error Histogram**

中文说明：误差直方图。这里的误差定义为预测值减去实际值。该图用于进一步观察残差分布是否集中、是否偏斜，以及误差的主要范围。它和预测误差分布图一起，可以帮助判断模型误差结构。

### `error_qq_plot.png`
**英文名称：Error QQ Plot**

中文说明：误差 QQ 图。该图将样本误差分位数与正态分布理论分位数进行比较，用来判断预测残差是否近似服从正态分布。如果散点明显偏离参考线，说明误差分布可能具有偏态、厚尾或异常点。

---

## 7. 可选的模型对比图

### `27_model_comparison_bar.png`
**英文名称：Model Comparison (Lower is Better)**

中文说明：模型对比柱状图。该图会同时展示不同模型在 `RMSE`、`MAE` 和 `MAPE` 三个指标上的结果，并通过柱状图对比模型优劣。数值越低代表效果越好。虽然当前主流程以 TCN 为主，但仓库中仍保留了生成该图的工具函数，因此如果调用模型对比模块，也可能输出这张图。

---

## 8. 推荐阅读顺序

如果你想快速理解程序输出的图片，建议按下面顺序查看：

1. `raw_load_timeseries.png`：先看原始数据整体趋势；
2. `monthly_average_load.png`、`seasonal_load_analysis.png`：再看季节性和月份差异；
3. `load_distribution_histogram.png`、`load_ramp_distribution.png`：了解统计分布和波动强度；
4. `emd_decomposition_overview.png`、`imf_components.png`：理解 EMD 分解后的多尺度成分；
5. `actual_vs_predicted_load.png`、`prediction_error_distribution.png`：检查预测是否准确；
6. `peak_period_error.png`、`valley_period_error.png`、`holiday_prediction_error.png`、`error_qq_plot.png`：最后分析模型在不同场景下的误差特征。

## 9. 与代码的对应关系

这些图片主要由以下模块生成：

- `src/time_scale_analysis.py`：基础时间尺度图；
- `src/season_analysis.py`：季节与统计特征图；
- `src/overload_analysis.py`：重过载识别图；
- `src/emd_decomposition.py`：EMD 分解图；
- `src/evaluation.py`：预测效果图与误差分析图；
- `src/model_comparison.py`：可选的模型对比柱状图。
