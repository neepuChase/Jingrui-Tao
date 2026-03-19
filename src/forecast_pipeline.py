"""TCN-only forecasting pipeline with user-defined IMF components."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.cuda.amp import GradScaler, autocast

from src.evaluation import calculate_metrics, generate_error_analysis_outputs, plot_forecast_results, save_metrics
from src.lstm_dataset import build_dataloader, create_sequences
from src.lstm_model import LSTMForecaster
from src.model_comparison import add_frequency_fusion_result, compare_and_select_model
from src.tcn_model import TCNForecaster
from src.visualization import save_figure


@dataclass
class ForecastConfig:
    lookback: int = 672
    train_ratio: float = 0.8
    dropout: float = 0.1
    learning_rate: float = 1e-3
    epochs: int = 20
    batch_size: int = 128
    random_seed: int = 42
    tcn_channels: tuple[int, ...] = (32, 32, 32)
    tcn_kernel_size: int = 3
    horizon: int = 96
    imf_components: int = 3
    imf_groups: dict[str, list[int]] = field(default_factory=dict)




class _SCINetForecaster(nn.Module):
    """Lightweight SCINet-style placeholder for high-frequency forecasting."""

    def __init__(self, hidden_size: int, horizon: int, dropout: float) -> None:
        super().__init__()
        self.branch_even = nn.Sequential(
            nn.Conv1d(1, hidden_size, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.branch_odd = nn.Sequential(
            nn.Conv1d(1, hidden_size, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.projection = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, horizon),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        signal = x.transpose(1, 2)
        even_repr = self.branch_even(signal[:, :, ::2]).mean(dim=-1)
        odd_signal = signal[:, :, 1::2]
        if odd_signal.size(-1) == 0:
            odd_signal = signal[:, :, -1:]
        odd_repr = self.branch_odd(odd_signal).mean(dim=-1)
        return self.projection(torch.cat([even_repr, odd_repr], dim=1))


class _AutoformerForecaster(nn.Module):
    """Transformer-encoder placeholder used for low-frequency IMF forecasting."""

    def __init__(self, hidden_size: int, horizon: int, dropout: float, num_layers: int = 2, nhead: int = 4) -> None:
        super().__init__()
        self.input_projection = nn.Linear(1, hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=nhead,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, horizon),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(self.input_projection(x))
        return self.head(encoded[:, -1, :])


class _TorchSequenceAdapter:
    """Unified train/predict adapter for sequence forecasting models."""

    def __init__(self, model: nn.Module, config: ForecastConfig, device: torch.device) -> None:
        self.model = model.to(device)
        self.config = config
        self.device = device

    def train_model(self, x_train: np.ndarray, y_train: np.ndarray) -> list[float]:
        train_loader = build_dataloader(x_train, y_train, self.config.batch_size)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        use_amp = self.device.type == "cuda"
        scaler = GradScaler(enabled=use_amp)

        self.model.train()
        loss_history: list[float] = []
        for _ in range(self.config.epochs):
            epoch_loss = 0.0
            for xb, yb in train_loader:
                inputs = xb.to(self.device)
                targets = yb.to(self.device)
                optimizer.zero_grad(set_to_none=True)
                with autocast(enabled=use_amp):
                    predictions = self.model(inputs)
                    loss = criterion(predictions, targets)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                epoch_loss += float(loss.item())
            loss_history.append(epoch_loss / max(1, len(train_loader)))
        return loss_history

    def predict(self, x_test: np.ndarray) -> np.ndarray:
        self.model.eval()
        x_tensor = torch.from_numpy(x_test).unsqueeze(-1).to(self.device)
        with torch.no_grad():
            with autocast(enabled=self.device.type == "cuda"):
                predictions = self.model(x_tensor).detach().cpu().numpy()
        return predictions


def _resolve_imf_index(index: int, total_components: int) -> int | None:
    if 1 <= index <= total_components:
        return index - 1
    if 0 <= index < total_components:
        return index
    return None


def _build_sequence_adapter(model_name: str, config: ForecastConfig, device: torch.device) -> _TorchSequenceAdapter:
    if model_name == "SCINet":
        model = _SCINetForecaster(hidden_size=64, horizon=config.horizon, dropout=config.dropout)
    elif model_name == "TCN":
        model = _build_model(config, device)
    elif model_name == "Autoformer":
        model = _AutoformerForecaster(hidden_size=64, horizon=config.horizon, dropout=config.dropout)
    elif model_name == "TimeXer":
        model = nn.Sequential(
            LSTMForecaster(input_size=1, hidden_size=128, num_layers=2, dropout=config.dropout),
            nn.Flatten(start_dim=1),
            nn.Linear(1, config.horizon),
        )
    else:
        raise ValueError(f"Unsupported model name: {model_name}")
    return _TorchSequenceAdapter(model=model, config=config, device=device)


def _train_with_model_interface(
    series: np.ndarray,
    config: ForecastConfig,
    device: torch.device,
    model_name: str,
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    n = len(series)
    split_idx = max(int(n * config.train_ratio), config.lookback + config.horizon)

    train_series = series[:split_idx]
    test_series = series[split_idx - config.lookback :]

    mean = float(np.mean(train_series))
    std = float(np.std(train_series)) + 1e-8

    train_norm = (train_series - mean) / std
    test_norm = (test_series - mean) / std

    x_train, y_train = create_sequences(train_norm, config.lookback, config.horizon)
    x_test, y_test = create_sequences(test_norm, config.lookback, config.horizon)
    if len(x_train) == 0 or len(x_test) == 0:
        raise ValueError("Insufficient samples for frequency-based forecasting. Reduce lookback/horizon or provide more data.")

    adapter = _build_sequence_adapter(model_name, config, device)
    loss_history = adapter.train_model(x_train, y_train)
    preds_test = adapter.predict(x_test)

    preds_test = preds_test * std + mean
    y_test = y_test * std + mean

    total_len = n - split_idx
    pred_sum = np.zeros(total_len, dtype=np.float64)
    pred_count = np.zeros(total_len, dtype=np.int32)
    true_values = np.zeros(total_len, dtype=np.float64)

    for i in range(len(preds_test)):
        for h in range(config.horizon):
            target_idx = i + h
            if target_idx >= total_len:
                continue
            pred_sum[target_idx] += float(preds_test[i, h])
            pred_count[target_idx] += 1
            true_values[target_idx] = float(y_test[i, h])

    valid_mask = pred_count > 0
    y_pred = (pred_sum[valid_mask] / pred_count[valid_mask]).astype(np.float64)
    y_true = true_values[valid_mask].astype(np.float64)
    return y_true, y_pred, loss_history


def forecast_with_frequency_models(
    imfs: list[np.ndarray],
    classification: dict,
    config: ForecastConfig,
) -> tuple[np.ndarray, np.ndarray]:
    device = _get_device()
    pred_total: np.ndarray | int = 0
    true_total: np.ndarray | int = 0

    imf_array = [np.asarray(component, dtype=np.float32).reshape(-1) for component in imfs]
    if not imf_array:
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)

    total_components = len(imf_array)
    trend_index = total_components - 1
    used_indices: set[int] = set()

    group_model_map = {
        "high": "SCINet",
        "mid": "TCN",
        "low": "Autoformer",
    }

    for group_name, model_name in group_model_map.items():
        raw_indices = classification.get(group_name, []) if isinstance(classification, dict) else []
        resolved_indices = []
        for index in raw_indices:
            resolved = _resolve_imf_index(int(index), total_components)
            if resolved is not None and resolved != trend_index:
                resolved_indices.append(resolved)

        for component_index in resolved_indices:
            used_indices.add(component_index)
            y_true, y_pred, _ = _train_with_model_interface(imf_array[component_index], config, device, model_name)
            if isinstance(pred_total, int):
                pred_total = np.zeros_like(y_pred, dtype=np.float64)
                true_total = np.zeros_like(y_true, dtype=np.float64)
            common_length = min(len(pred_total), len(y_pred))
            pred_total = np.asarray(pred_total[:common_length], dtype=np.float64)
            true_total = np.asarray(true_total[:common_length], dtype=np.float64)
            pred_total += y_pred[:common_length]
            true_total += y_true[:common_length]

    trend_true, trend_pred, _ = _train_with_model_interface(imf_array[trend_index], config, device, "TimeXer")
    if isinstance(pred_total, int):
        pred_total = np.zeros_like(trend_pred, dtype=np.float64)
        true_total = np.zeros_like(trend_true, dtype=np.float64)
    common_length = min(len(pred_total), len(trend_pred))
    pred_total = np.asarray(pred_total[:common_length], dtype=np.float64) + trend_pred[:common_length]
    true_total = np.asarray(true_total[:common_length], dtype=np.float64) + trend_true[:common_length]

    unused_indices = [idx for idx in range(total_components - 1) if idx not in used_indices]
    for component_index in unused_indices:
        y_true, y_pred, _ = _train_with_model_interface(imf_array[component_index], config, device, "TCN")
        common_length = min(len(pred_total), len(y_pred))
        pred_total = np.asarray(pred_total[:common_length], dtype=np.float64) + y_pred[:common_length]
        true_total = np.asarray(true_total[:common_length], dtype=np.float64) + y_true[:common_length]

    return np.asarray(true_total, dtype=np.float64), np.asarray(pred_total, dtype=np.float64)


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def _get_device() -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}")
    return device


def _build_model(config: ForecastConfig, device: torch.device) -> nn.Module:
    return TCNForecaster(
        input_channels=1,
        channels=list(config.tcn_channels),
        kernel_size=config.tcn_kernel_size,
        dropout=config.dropout,
        output_size=config.horizon,
    ).to(device)


def _train_single_series(series: np.ndarray, config: ForecastConfig, device: torch.device) -> tuple[np.ndarray, np.ndarray, list[float]]:
    n = len(series)
    split_idx = max(int(n * config.train_ratio), config.lookback + config.horizon)

    train_series = series[:split_idx]
    test_series = series[split_idx - config.lookback :]

    mean = float(np.mean(train_series))
    std = float(np.std(train_series)) + 1e-8

    train_norm = (train_series - mean) / std
    test_norm = (test_series - mean) / std

    x_train, y_train = create_sequences(train_norm, config.lookback, config.horizon)
    if len(x_train) == 0:
        raise ValueError("Insufficient training samples. Reduce lookback or provide more data.")

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
    x_test, y_test = create_sequences(test_norm, config.lookback, config.horizon)
    if len(x_test) == 0:
        raise ValueError("Insufficient test samples. Increase data size or reduce lookback/horizon.")

    x_tensor = torch.from_numpy(x_test).unsqueeze(-1).to(device)
    with torch.no_grad():
        with autocast(enabled=device.type == "cuda"):
            preds_test = model(x_tensor).detach().cpu().numpy()

    preds_test = preds_test * std + mean
    y_test = y_test * std + mean

    horizon = config.horizon
    total_len = n - split_idx
    pred_sum = np.zeros(total_len, dtype=np.float64)
    pred_count = np.zeros(total_len, dtype=np.int32)
    true_values = np.zeros(total_len, dtype=np.float64)

    for i in range(len(preds_test)):
        for h in range(horizon):
            target_idx = i + h
            if target_idx >= total_len:
                continue
            pred_sum[target_idx] += float(preds_test[i, h])
            pred_count[target_idx] += 1
            true_values[target_idx] = float(y_test[i, h])

    valid_mask = pred_count > 0
    y_pred = (pred_sum[valid_mask] / pred_count[valid_mask]).astype(np.float64)
    y_true = true_values[valid_mask].astype(np.float64)
    return y_true, y_pred, loss_history


def _forecast_by_components(
    cleaned_df: pd.DataFrame,
    component_df: pd.DataFrame,
    config: ForecastConfig,
    outputs_dir: Path,
    strategy_name: str,
    device: torch.device,
) -> tuple[pd.DataFrame, dict[str, list[float]], dict[str, float], pd.Series]:
    comp_cols = [c for c in component_df.columns if c != "timestamp"]
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

    split_idx = max(int(len(cleaned_df) * config.train_ratio), config.lookback + config.horizon)
    timestamps_test = cleaned_df["timestamp"].reset_index(drop=True).iloc[split_idx: split_idx + len(true_reconstructed)]

    forecast_df = pd.DataFrame(
        {
            "timestamp": timestamps_test.values,
            "actual_load": true_reconstructed,
            "predicted_load": pred_reconstructed,
            "error": true_reconstructed - pred_reconstructed,
        }
    )
    forecast_df.to_csv(outputs_dir / f"tcn_forecast_{strategy_name}.csv", index=False, encoding="utf-8-sig")
    metrics = calculate_metrics(true_reconstructed, pred_reconstructed)
    return forecast_df, all_losses, metrics, timestamps_test


def _build_group_component_df(imf_df: pd.DataFrame, groups: dict[str, list[int]]) -> pd.DataFrame:
    component_df = pd.DataFrame({"timestamp": imf_df["timestamp"].values})
    imf_cols = [c for c in imf_df.columns if c.startswith("IMF")]

    for group_name in ("high", "mid", "low"):
        indices = groups.get(group_name, [])
        selected_cols = [imf_cols[idx - 1] for idx in indices if 1 <= idx <= len(imf_cols)]
        if selected_cols:
            component_df[f"{group_name}_group"] = imf_df[selected_cols].sum(axis=1)

    return component_df


def _build_k_component_df(cleaned_df: pd.DataFrame, imf_df: pd.DataFrame, k: int) -> pd.DataFrame:
    imf_cols = [c for c in imf_df.columns if c.startswith("IMF")]
    if not 1 <= k <= 10:
        raise ValueError("IMF_COMPONENTS 必须在 1-10 之间")
    if k > len(imf_cols):
        raise ValueError(f"IMF_COMPONENTS={k} 超出可用 IMF 数量({len(imf_cols)})")

    selected = imf_df[imf_cols[:k]].copy()
    original = cleaned_df["load"].values
    remainder = original - selected.sum(axis=1).values
    selected["remainder_component"] = remainder
    selected.insert(0, "timestamp", imf_df["timestamp"].values)
    return selected


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

    undecomposed_df = pd.DataFrame({"timestamp": cleaned_df["timestamp"], "raw_load": cleaned_df["load"]})
    un_forecast, _, un_metrics, ts = _forecast_by_components(cleaned_df, undecomposed_df, cfg, outputs_dir, "non_decomposed", device)
    print("Non-decomposed forecast metrics:", un_metrics)

    component_df = _build_k_component_df(cleaned_df, imf_df, cfg.imf_components)
    if cfg.imf_groups:
        grouped_component_df = _build_group_component_df(imf_df, cfg.imf_groups)
        if len(grouped_component_df.columns) > 1:
            component_df = grouped_component_df
    de_forecast, _, de_metrics, _ = _forecast_by_components(
        cleaned_df,
        component_df,
        cfg,
        outputs_dir,
        f"emd_decomposed_imf{cfg.imf_components}",
        device,
    )
    de_forecast.to_csv(outputs_dir / f"TCN预测结果_IMF{cfg.imf_components}.csv", index=False, encoding="utf-8-sig")
    print(f"Decomposed forecast metrics (IMF={cfg.imf_components}):", de_metrics)

    results = {
        "TCN": (un_forecast["actual_load"].to_numpy(), un_forecast["predicted_load"].to_numpy()),
    }

    freq_true, freq_pred = forecast_with_frequency_models(
        imfs=[imf_df[col].to_numpy() for col in imf_df.columns if col.startswith("IMF")],
        classification=cfg.imf_groups,
        config=cfg,
    )
    freq_metrics: dict[str, float] | None = None
    if len(freq_true) > 0 and len(freq_pred) > 0:
        timestamps_freq = cleaned_df["timestamp"].reset_index(drop=True).iloc[-len(freq_true):]
        freq_forecast = pd.DataFrame(
            {
                "timestamp": timestamps_freq.values,
                "actual_load": freq_true,
                "predicted_load": freq_pred,
                "error": freq_true - freq_pred,
            }
        )
        freq_forecast.to_csv(outputs_dir / "freq_fusion_forecast.csv", index=False, encoding="utf-8-sig")
        freq_metrics = calculate_metrics(freq_true, freq_pred)
        add_frequency_fusion_result(results, freq_true, freq_pred)

        fig, ax = plt.subplots(figsize=(14, 6), dpi=300)
        ax.plot(timestamps_freq, freq_true, label="Actual Load", linewidth=1.0)
        ax.plot(timestamps_freq, freq_pred, label="FREQ_FUSION", linewidth=1.0)
        ax.set_title("True vs Predicted Load (FREQ_FUSION)")
        ax.set_xlabel("Time")
        ax.set_ylabel("Load (MW)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        save_figure(fig, figures_dir, "freq_fusion_prediction.png")

    compare_rows = [
        {"strategy": "Non-decomposed TCN Forecast", **un_metrics},
        {"strategy": f"EMD-decomposed TCN Forecast (k={cfg.imf_components})", **de_metrics},
    ]
    if freq_metrics is not None:
        compare_rows.append({"strategy": "Frequency Fusion Forecast", **freq_metrics})

    compare_df = pd.DataFrame(compare_rows)
    compare_df.to_csv(outputs_dir / "decomposition_vs_non_decomposition_comparison.csv", index=False, encoding="utf-8-sig")
    compare_and_select_model(results, outputs_dir, figures_dir)

    final_candidates = [
        ("Non-decomposed TCN Forecast", un_forecast, un_metrics),
        ("TCN Forecast After EMD Decomposition", de_forecast, de_metrics),
    ]
    if freq_metrics is not None:
        final_candidates.append(("Frequency Fusion Forecast", freq_forecast, freq_metrics))

    best_strategy, final_forecast, final_metrics = min(
        final_candidates,
        key=lambda item: (item[2]["RMSE"], item[2]["MAE"], item[2]["MAPE"]),
    )

    print("Best forecast strategy:", best_strategy)
    (outputs_dir / "best_forecast_strategy.txt").write_text(best_strategy, encoding="utf-8")
    final_forecast.to_csv(outputs_dir / "forecast_results.csv", index=False, encoding="utf-8-sig")
    save_metrics(final_metrics, outputs_dir, filename="forecast_metrics.csv")

    import json

    with (outputs_dir / "forecast_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(final_metrics, f, ensure_ascii=False, indent=2)

    plot_forecast_results(ts, final_forecast["actual_load"].values, final_forecast["predicted_load"].values, figures_dir)
    generate_error_analysis_outputs(final_forecast, outputs_dir, figures_dir)
    return final_metrics
