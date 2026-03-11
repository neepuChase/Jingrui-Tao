"""仅基于 TCN 的预测流水线：IMF 自动选优 + 分解/未分解对比。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.cuda.amp import GradScaler, autocast

from src.evaluation import calculate_metrics, generate_error_analysis_outputs, plot_forecast_results, save_metrics
from src.lstm_dataset import build_dataloader, create_sequences
from src.tcn_model import TCNForecaster
from src.visualization import save_figure


@dataclass
class ForecastConfig:
    lookback: int = 96
    train_ratio: float = 0.8
    dropout: float = 0.1
    learning_rate: float = 1e-3
    epochs: int = 20
    batch_size: int = 128
    random_seed: int = 42
    tcn_channels: tuple[int, ...] = (32, 32, 32)
    tcn_kernel_size: int = 3


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def _get_device() -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("使用设备:", device)
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}")
    return device


def _build_model(config: ForecastConfig, device: torch.device) -> nn.Module:
    return TCNForecaster(
        input_channels=1,
        channels=list(config.tcn_channels),
        kernel_size=config.tcn_kernel_size,
        dropout=config.dropout,
    ).to(device)


def _train_single_series(series: np.ndarray, config: ForecastConfig, device: torch.device) -> tuple[np.ndarray, np.ndarray, list[float]]:
    n = len(series)
    split_idx = max(int(n * config.train_ratio), config.lookback + 2)

    train_series = series[:split_idx]
    test_series = series[split_idx - config.lookback :]

    mean = float(np.mean(train_series))
    std = float(np.std(train_series)) + 1e-8

    train_norm = (train_series - mean) / std
    test_norm = (test_series - mean) / std

    x_train, y_train = create_sequences(train_norm, config.lookback)
    if len(x_train) == 0:
        raise ValueError("训练样本不足，请减小 lookback 或增加数据量。")

    train_loader = build_dataloader(x_train, y_train, config.batch_size)
    model = _build_model(config, device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    use_amp = device.type == "cuda"
    scaler = GradScaler(enabled=use_amp)

    loss_history: list[float] = []
    model.train()
    for _ in range(config.epochs):
        epoch_loss = 0.0
        for xb, yb in train_loader:
            X = xb.to(device)
            y = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=use_amp):
                pred = model(X)
                loss = criterion(pred, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += float(loss.item())
        loss_history.append(epoch_loss / max(1, len(train_loader)))

    model.eval()
    horizon = n - split_idx
    input_window = list(test_norm[: config.lookback])
    preds_norm: list[float] = []

    with torch.no_grad():
        for _ in range(horizon):
            x = torch.tensor(input_window[-config.lookback :], dtype=torch.float32, device=device).view(1, config.lookback, 1)
            with autocast(enabled=device.type == "cuda"):
                yhat = float(model(x).item())
            preds_norm.append(yhat)
            input_window.append(yhat)

    y_pred = np.asarray(preds_norm) * std + mean
    y_true = series[split_idx: split_idx + len(y_pred)]
    return y_true, y_pred, loss_history


def _forecast_by_components(
    cleaned_df: pd.DataFrame,
    component_df: pd.DataFrame,
    config: ForecastConfig,
    outputs_dir: Path,
    strategy_name: str,
    device: torch.device,
) -> tuple[pd.DataFrame, dict[str, list[float]], dict[str, float], pd.Series]:
    comp_cols = [c for c in component_df.columns if c != "时间戳"]
    comp_values = component_df[comp_cols].values

    all_true, all_pred = [], []
    all_losses: dict[str, list[float]] = {}

    for i, col in enumerate(comp_cols):
        y_true_i, y_pred_i, loss_history = _train_single_series(comp_values[:, i], config, device)
        all_true.append(y_true_i)
        all_pred.append(y_pred_i)
        all_losses[col] = loss_history

    true_reconstructed = np.sum(np.vstack(all_true), axis=0)
    pred_reconstructed = np.sum(np.vstack(all_pred), axis=0)

    split_idx = max(int(len(cleaned_df) * config.train_ratio), config.lookback + 2)
    timestamps_test = cleaned_df["timestamp"].reset_index(drop=True).iloc[split_idx: split_idx + len(true_reconstructed)]

    forecast_df = pd.DataFrame(
        {
            "时间戳": timestamps_test.values,
            "真实负荷": true_reconstructed,
            "预测负荷": pred_reconstructed,
            "误差": true_reconstructed - pred_reconstructed,
        }
    )
    forecast_df.to_csv(outputs_dir / f"TCN预测结果_{strategy_name}.csv", index=False, encoding="utf-8-sig")
    metrics = calculate_metrics(true_reconstructed, pred_reconstructed)
    return forecast_df, all_losses, metrics, timestamps_test



def _build_k_component_df(cleaned_df: pd.DataFrame, imf_df: pd.DataFrame, k: int) -> pd.DataFrame:
    imf_cols = [c for c in imf_df.columns if c.startswith("IMF")]
    selected = imf_df[imf_cols[:k]].copy()
    original = cleaned_df["load"].values
    remainder = original - selected.sum(axis=1).values
    selected[f"剩余分量(k={k})"] = remainder
    selected.insert(0, "时间戳", imf_df["时间戳"].values)
    return selected


def _plot_imf_k_comparison(df: pd.DataFrame, figures_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df["IMF分量数"], df["RMSE"], marker="o", label="RMSE")
    ax.plot(df["IMF分量数"], df["MAE"], marker="s", label="MAE")
    ax.plot(df["IMF分量数"], df["MAPE"], marker="^", label="MAPE")
    ax.set_title("IMF分量数选择效果对比图")
    ax.set_xlabel("IMF分量数")
    ax.set_ylabel("指标值")
    ax.legend()
    ax.grid(True, alpha=0.3)
    save_figure(fig, figures_dir, "IMF分量数选择对比图.png")


def _plot_strategy_comparison(df: pd.DataFrame, figures_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(df))
    width = 0.2
    ax.bar(x - 1.5 * width, df["MAE"], width=width, label="MAE")
    ax.bar(x - 0.5 * width, df["RMSE"], width=width, label="RMSE")
    ax.bar(x + 0.5 * width, df["MAPE"], width=width, label="MAPE")
    ax.bar(x + 1.5 * width, df["R2"], width=width, label="R2")
    ax.set_xticks(x)
    ax.set_xticklabels(df["方案"])
    ax.set_title("分解与未分解预测效果对比图")
    ax.set_xlabel("预测方案")
    ax.set_ylabel("指标值")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    save_figure(fig, figures_dir, "分解与未分解效果对比图.png")


def run_tcn_forecast_comparison(
    cleaned_df: pd.DataFrame,
    imf_df: pd.DataFrame,
    outputs_dir: Path,
    figures_dir: Path,
    config: ForecastConfig | None = None,
) -> dict[str, float]:
    cfg = config or ForecastConfig()
    _set_seed(cfg.random_seed)
    torch.backends.cudnn.benchmark = True
    device = _get_device()

    # A. 未分解预测
    undecomposed_df = pd.DataFrame({"时间戳": cleaned_df["timestamp"], "原始负荷": cleaned_df["load"]})
    un_forecast, _, un_metrics, ts = _forecast_by_components(cleaned_df, undecomposed_df, cfg, outputs_dir, "未分解", device)
    print("未分解预测指标:", un_metrics)

    # B. IMF分量数选择
    imf_cols = [c for c in imf_df.columns if c.startswith("IMF")]
    max_k = len(imf_cols)
    candidate_rows = []
    best = None

    for k in range(1, max_k + 1):
        print(f"当前测试的IMF分量数: {k}")
        comp_df = _build_k_component_df(cleaned_df, imf_df, k)
        _, _, metrics, _ = _forecast_by_components(cleaned_df, comp_df, cfg, outputs_dir, f"EMD分解_k{k}", device)
        row = {"IMF分量数": k, **metrics}
        candidate_rows.append(row)

        score = (metrics["RMSE"], metrics["MAE"], metrics["MAPE"])
        if best is None or score < best[0]:
            best = (score, k, metrics)

    select_df = pd.DataFrame(candidate_rows)
    select_df.to_csv(outputs_dir / "IMF分量数选择结果.csv", index=False, encoding="utf-8-sig")
    _plot_imf_k_comparison(select_df, figures_dir)

    best_k = int(best[1]) if best else 1
    (outputs_dir / "最优IMF分量数.txt").write_text(str(best_k), encoding="utf-8")
    print("最优IMF分量数:", best_k)

    best_comp_df = _build_k_component_df(cleaned_df, imf_df, best_k)
    de_forecast, _, de_metrics, _ = _forecast_by_components(cleaned_df, best_comp_df, cfg, outputs_dir, "EMD分解", device)
    print("分解预测指标:", de_metrics)

    compare_df = pd.DataFrame(
        [
            {"方案": "未分解TCN预测", **un_metrics},
            {"方案": f"EMD分解TCN预测(k={best_k})", **de_metrics},
        ]
    )
    compare_df.to_csv(outputs_dir / "分解与未分解预测对比.csv", index=False, encoding="utf-8-sig")
    _plot_strategy_comparison(compare_df, figures_dir)

    if (de_metrics["RMSE"], de_metrics["MAE"], de_metrics["MAPE"]) < (un_metrics["RMSE"], un_metrics["MAE"], un_metrics["MAPE"]):
        best_strategy = "EMD分解后TCN预测"
        final_forecast = de_forecast
        final_metrics = de_metrics
    else:
        best_strategy = "未分解TCN预测"
        final_forecast = un_forecast
        final_metrics = un_metrics

    print("最优预测方案:", best_strategy)
    (outputs_dir / "最优预测方案.txt").write_text(best_strategy, encoding="utf-8")
    final_forecast.to_csv(outputs_dir / "最终预测结果.csv", index=False, encoding="utf-8-sig")
    save_metrics(final_metrics, outputs_dir)

    plot_forecast_results(ts, final_forecast["真实负荷"].values, final_forecast["预测负荷"].values, figures_dir)
    generate_error_analysis_outputs(final_forecast, outputs_dir, figures_dir)
    return final_metrics
