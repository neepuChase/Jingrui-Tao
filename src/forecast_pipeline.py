"""Forecasting pipeline for EMD-based load prediction model comparison."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.amp import GradScaler, autocast

from src.evaluation import calculate_metrics, generate_error_analysis_outputs, plot_forecast_results, save_metrics
from src.lstm_dataset import build_dataloader, create_sequences
from src.lstm_model import LSTMForecaster
from src.model_comparison import compare_and_select_model


@dataclass
class ForecastConfig:
    lookback: int = 672
    train_ratio: float = 0.8
    dropout: float = 0.1
    learning_rate: float = 1e-3
    epochs: int = 20
    batch_size: int = 128
    eval_batch_size: int = 32
    random_seed: int = 42
    horizon: int = 96
    imf_components: int = 3
    imf_groups: dict[str, list[int]] = field(default_factory=dict)


class _SCINetForecaster(nn.Module):
    """Compact SCINet-style block using odd/even subsequences."""

    def __init__(self, input_size: int, hidden_size: int, horizon: int, dropout: float) -> None:
        super().__init__()
        self.branch_even = nn.Sequential(
            nn.Conv1d(input_size, hidden_size, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.branch_odd = nn.Sequential(
            nn.Conv1d(input_size, hidden_size, kernel_size=5, padding=2),
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


class _ITransformerForecaster(nn.Module):
    """Lightweight iTransformer-style encoder over inverted variate tokens."""

    def __init__(self, input_size: int, lookback: int, hidden_size: int, horizon: int, dropout: float) -> None:
        super().__init__()
        self.time_projection = nn.Linear(lookback, hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=4,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, horizon),
        )
        self.input_size = input_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        inverted_tokens = x.transpose(1, 2)
        encoded = self.encoder(self.time_projection(inverted_tokens))
        pooled = encoded.mean(dim=1)
        return self.head(pooled)


class _TimeXerForecaster(nn.Module):
    """Compact TimeXer-style mixer combining temporal convolution and gating."""

    def __init__(self, input_size: int, hidden_size: int, horizon: int, dropout: float) -> None:
        super().__init__()
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(input_size, hidden_size, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.gate = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Sigmoid(),
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, horizon),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        conv_features = self.temporal_conv(x.transpose(1, 2)).mean(dim=-1)
        gated = conv_features * self.gate(conv_features)
        return self.head(gated)


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
        scaler = GradScaler(device="cuda", enabled=use_amp)

        self.model.train()
        loss_history: list[float] = []
        for _ in range(self.config.epochs):
            epoch_loss = 0.0
            for xb, yb in train_loader:
                inputs = xb.to(self.device)
                targets = yb.to(self.device)
                optimizer.zero_grad(set_to_none=True)
                with autocast(device_type=self.device.type, enabled=use_amp):
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
        x_array = np.asarray(x_test, dtype=np.float32)
        if len(x_array) == 0:
            return np.empty((0, self.config.horizon), dtype=np.float32)

        outputs: list[np.ndarray] = []
        batch_size = max(1, self.config.eval_batch_size)
        with torch.inference_mode():
            for start in range(0, len(x_array), batch_size):
                batch = torch.from_numpy(x_array[start : start + batch_size]).to(self.device, non_blocking=self.device.type == "cuda")
                with autocast(device_type=self.device.type, enabled=self.device.type == "cuda"):
                    batch_pred = self.model(batch)
                outputs.append(batch_pred.detach().cpu().numpy())
                del batch, batch_pred
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        return np.concatenate(outputs, axis=0)


def _select_multivariate_feature_columns(cleaned_df: pd.DataFrame) -> list[str]:
    excluded = {"timestamp", "date", "weekday_name", "season"}
    feature_cols: list[str] = []
    for col in cleaned_df.columns:
        if col in excluded:
            continue
        if pd.api.types.is_numeric_dtype(cleaned_df[col]):
            feature_cols.append(col)
    return feature_cols


def _assemble_feature_matrix(
    cleaned_df: pd.DataFrame,
    target_series: np.ndarray,
    *,
    include_load: bool,
) -> np.ndarray:
    feature_cols = _select_multivariate_feature_columns(cleaned_df)
    base_df = cleaned_df[feature_cols].copy()
    target_array = np.asarray(target_series, dtype=np.float32).reshape(-1)
    base_df.insert(0, "target_signal", target_array)

    if not include_load and "load" in base_df.columns:
        base_df = base_df.drop(columns=["load"])

    values = base_df.astype(np.float32).to_numpy(copy=True)
    col_means = np.nanmean(values, axis=0)
    col_means = np.where(np.isnan(col_means), 0.0, col_means)
    nan_rows, nan_cols = np.where(np.isnan(values))
    if len(nan_rows) > 0:
        values[nan_rows, nan_cols] = col_means[nan_cols]
    return values


def _split_and_scale_features(
    features: np.ndarray,
    targets: np.ndarray,
    config: ForecastConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float, int]:
    n = len(targets)
    split_idx = max(int(n * config.train_ratio), config.lookback + config.horizon)
    if split_idx >= n:
        raise ValueError("Insufficient samples for forecasting. Reduce lookback/horizon or provide more data.")

    train_features = features[:split_idx]
    test_features = features[split_idx - config.lookback :]
    train_targets = targets[:split_idx]
    test_targets = targets[split_idx - config.lookback :]

    feature_mean = train_features.mean(axis=0)
    feature_std = train_features.std(axis=0) + 1e-8
    feature_std[feature_std < 1e-8] = 1.0

    train_norm = (train_features - feature_mean) / feature_std
    test_norm = (test_features - feature_mean) / feature_std

    target_mean = float(np.mean(train_targets))
    target_std = float(np.std(train_targets)) + 1e-8
    return train_norm, test_norm, train_targets, test_targets, target_mean, target_std, split_idx


def _aggregate_horizon_predictions(preds_test: np.ndarray, y_test: np.ndarray, total_len: int, horizon: int) -> tuple[np.ndarray, np.ndarray]:
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
    return y_true, y_pred


def _build_sequence_adapter(model_name: str, config: ForecastConfig, device: torch.device, input_size: int) -> _TorchSequenceAdapter:
    if model_name == "LSTM":
        model = LSTMForecaster(
            input_size=input_size,
            hidden_size=64,
            num_layers=2,
            dropout=config.dropout,
            output_size=config.horizon,
        )
    elif model_name == "SCINet":
        model = _SCINetForecaster(input_size=input_size, hidden_size=64, horizon=config.horizon, dropout=config.dropout)
    elif model_name == "iTransformer":
        model = _ITransformerForecaster(
            input_size=input_size,
            lookback=config.lookback,
            hidden_size=64,
            horizon=config.horizon,
            dropout=config.dropout,
        )
    elif model_name == "TimeXer":
        model = _TimeXerForecaster(input_size=input_size, hidden_size=64, horizon=config.horizon, dropout=config.dropout)
    else:
        raise ValueError(f"Unsupported model name: {model_name}")
    return _TorchSequenceAdapter(model=model, config=config, device=device)


def _train_component_series(
    target_series: np.ndarray,
    feature_matrix: np.ndarray,
    config: ForecastConfig,
    device: torch.device,
    model_name: str,
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    train_norm, test_norm, train_targets, test_targets, target_mean, target_std, split_idx = _split_and_scale_features(
        feature_matrix,
        np.asarray(target_series, dtype=np.float32).reshape(-1),
        config,
    )

    y_train_norm = (train_targets - target_mean) / target_std
    y_test_norm = (test_targets - target_mean) / target_std

    x_train, y_train = create_sequences(train_norm, y_train_norm, lookback=config.lookback, horizon=config.horizon)
    x_test, y_test = create_sequences(test_norm, y_test_norm, lookback=config.lookback, horizon=config.horizon)
    if len(x_train) == 0 or len(x_test) == 0:
        raise ValueError("Insufficient samples for EMD forecasting. Reduce lookback/horizon or provide more data.")

    adapter = _build_sequence_adapter(model_name, config, device, input_size=x_train.shape[-1])
    loss_history = adapter.train_model(x_train, y_train)
    preds_test = adapter.predict(x_test)

    preds_test = preds_test * target_std + target_mean
    y_test = y_test * target_std + target_mean

    total_len = len(target_series) - split_idx
    y_true, y_pred = _aggregate_horizon_predictions(preds_test, y_test, total_len, config.horizon)
    return y_true, y_pred, loss_history


def _forecast_by_components(
    cleaned_df: pd.DataFrame,
    component_df: pd.DataFrame,
    config: ForecastConfig,
    outputs_dir: Path,
    model_name: str,
    device: torch.device,
) -> tuple[pd.DataFrame, dict[str, list[float]], dict[str, float], pd.Series]:
    comp_cols = [c for c in component_df.columns if c != "timestamp"]
    comp_values = component_df[comp_cols].values

    all_true, all_pred = [], []
    all_losses: dict[str, list[float]] = {}

    for i, col in enumerate(comp_cols):
        include_load = col == "raw_load"
        feature_matrix = _assemble_feature_matrix(cleaned_df, comp_values[:, i], include_load=include_load)
        y_true_i, y_pred_i, loss_history = _train_component_series(comp_values[:, i], feature_matrix, config, device, model_name)
        all_true.append(y_true_i)
        all_pred.append(y_pred_i)
        all_losses[col] = loss_history

    true_reconstructed = np.sum(np.vstack(all_true), axis=0)
    pred_reconstructed = np.sum(np.vstack(all_pred), axis=0)

    split_idx = max(int(len(cleaned_df) * config.train_ratio), config.lookback + config.horizon)
    timestamps_test = cleaned_df["timestamp"].reset_index(drop=True).iloc[split_idx : split_idx + len(true_reconstructed)]

    forecast_df = pd.DataFrame(
        {
            "timestamp": timestamps_test.values,
            "actual_load": true_reconstructed,
            "predicted_load": pred_reconstructed,
            "error": true_reconstructed - pred_reconstructed,
        }
    )
    slug = model_name.lower()
    forecast_df.to_csv(outputs_dir / f"emd_{slug}_forecast.csv", index=False, encoding="utf-8-sig")
    metrics = calculate_metrics(true_reconstructed, pred_reconstructed)
    return forecast_df, all_losses, metrics, timestamps_test


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def _get_device() -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}")
    return device


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


def run_emd_model_comparison(
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

    component_df = _build_k_component_df(cleaned_df, imf_df, cfg.imf_components)
    model_suite = ("LSTM", "SCINet", "iTransformer", "TimeXer")

    results: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    candidate_forecasts: list[tuple[str, pd.DataFrame, dict[str, float]]] = []
    metric_rows: list[dict[str, float | str | int]] = []

    for model_name in model_suite:
        forecast_df, _, metrics, _ = _forecast_by_components(cleaned_df, component_df, cfg, outputs_dir, model_name, device)
        strategy_name = f"EMD+{model_name}"
        results[strategy_name] = (forecast_df["actual_load"].to_numpy(), forecast_df["predicted_load"].to_numpy())
        candidate_forecasts.append((strategy_name, forecast_df, metrics))
        metric_rows.append({"strategy": strategy_name, "imf_components": cfg.imf_components, **metrics})
        print(f"{strategy_name} metrics:", metrics)

    comparison_df = pd.DataFrame(metric_rows).sort_values(["RMSE", "MAE", "MAPE"], ascending=True).reset_index(drop=True)
    comparison_df.to_csv(outputs_dir / "emd_model_metrics.csv", index=False, encoding="utf-8-sig")

    _, best_model_name = compare_and_select_model(results, outputs_dir, figures_dir)
    best_strategy, final_forecast, final_metrics = min(
        candidate_forecasts,
        key=lambda item: (item[2]["RMSE"], item[2]["MAE"], item[2]["MAPE"]),
    )
    if best_model_name != best_strategy.upper():
        raise RuntimeError(f"Best model mismatch: {best_model_name} vs {best_strategy.upper()}")

    print("Best forecast strategy:", best_strategy)
    (outputs_dir / "best_forecast_strategy.txt").write_text(best_strategy, encoding="utf-8")
    final_forecast.to_csv(outputs_dir / "forecast_results.csv", index=False, encoding="utf-8-sig")
    save_metrics(final_metrics, outputs_dir, filename="forecast_metrics.csv")
    with (outputs_dir / "forecast_metrics.json").open("w", encoding="utf-8") as file_obj:
        json.dump(final_metrics, file_obj, ensure_ascii=False, indent=2)

    plot_forecast_results(final_forecast["timestamp"], final_forecast["actual_load"].values, final_forecast["predicted_load"].values, figures_dir)
    generate_error_analysis_outputs(final_forecast, outputs_dir, figures_dir)
    return final_metrics
