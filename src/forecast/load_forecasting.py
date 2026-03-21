from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn

from src.emd_decomposition import create_imf_analysis_figures, perform_emd, plot_emd_overview, plot_imf_components, save_imfs
from src.evaluation import calculate_metrics
from src.lstm_dataset import build_dataloader, create_sequences
from src.models.sequence_models import ModelFactory
from src.visualization import save_figure


@dataclass
class ForecastConfig:
    lookback: int = 672
    horizon: int = 96
    train_ratio: float = 0.8
    epochs: int = 10
    batch_size: int = 128
    eval_batch_size: int = 64
    learning_rate: float = 1e-3
    dropout: float = 0.1
    random_seed: int = 42
    imf_components: int = 6


class SequenceTrainer:
    def __init__(self, model: nn.Module, config: ForecastConfig, device: torch.device) -> None:
        self.model = model.to(device)
        self.config = config
        self.device = device

    def fit(self, x_train: np.ndarray, y_train: np.ndarray) -> list[float]:
        loader = build_dataloader(x_train, y_train, self.config.batch_size)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        criterion = nn.MSELoss()
        losses: list[float] = []

        self.model.train()
        for _ in range(self.config.epochs):
            epoch_loss = 0.0
            for xb, yb in loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                optimizer.zero_grad(set_to_none=True)
                pred = self.model(xb)
                loss = criterion(pred, yb)
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.item())
            losses.append(epoch_loss / max(1, len(loader)))
        return losses

    def predict(self, x_test: np.ndarray) -> np.ndarray:
        if len(x_test) == 0:
            return np.empty((0, self.config.horizon), dtype=np.float32)
        self.model.eval()
        outputs: list[np.ndarray] = []
        with torch.inference_mode():
            for start in range(0, len(x_test), self.config.eval_batch_size):
                batch = torch.from_numpy(x_test[start : start + self.config.eval_batch_size]).to(self.device)
                outputs.append(self.model(batch).cpu().numpy())
        return np.concatenate(outputs, axis=0)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def select_numeric_exogenous_columns(df: pd.DataFrame) -> list[str]:
    excluded = {"timestamp", "date", "weekday_name", "season", "load"}
    return [col for col in df.columns if col not in excluded and pd.api.types.is_numeric_dtype(df[col])]


def build_component_frame(df: pd.DataFrame, config: ForecastConfig) -> pd.DataFrame:
    imfs = perform_emd(df["load"], max_imf=max(10, config.imf_components))
    if imfs.shape[0] == 0:
        raise ValueError("EMD 分解失败，未生成 IMF 分量。")
    selected_count = min(config.imf_components, imfs.shape[0])
    selected_imfs = imfs[:selected_count]
    frame = pd.DataFrame(selected_imfs.T, columns=[f"imf_{i + 1}" for i in range(selected_count)])
    frame.insert(0, "timestamp", pd.to_datetime(df["timestamp"]).values)
    remainder = df["load"].to_numpy(dtype=float) - selected_imfs.sum(axis=0)
    frame["residual"] = remainder
    return frame


def assemble_features(df: pd.DataFrame, component_signal: np.ndarray) -> np.ndarray:
    exogenous_cols = select_numeric_exogenous_columns(df)
    feature_df = pd.DataFrame({"target_component": component_signal.astype(np.float32)})
    for col in exogenous_cols:
        feature_df[col] = df[col].astype(np.float32).to_numpy()
    values = feature_df.to_numpy(dtype=np.float32)
    col_means = np.nanmean(values, axis=0)
    col_means = np.where(np.isnan(col_means), 0.0, col_means)
    nan_rows, nan_cols = np.where(np.isnan(values))
    if len(nan_rows) > 0:
        values[nan_rows, nan_cols] = col_means[nan_cols]
    return values


def split_and_scale(
    features: np.ndarray,
    targets: np.ndarray,
    config: ForecastConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float, int]:
    n = len(targets)
    split_idx = max(int(n * config.train_ratio), config.lookback + config.horizon)
    if split_idx >= n:
        raise ValueError("样本量不足，无法完成训练/预测，请缩小 lookback 或 horizon，或提供更长时间范围数据。")

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


def aggregate_horizon_predictions(preds: np.ndarray, truth: np.ndarray, total_len: int, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    pred_sum = np.zeros(total_len, dtype=np.float64)
    pred_count = np.zeros(total_len, dtype=np.int32)
    true_values = np.zeros(total_len, dtype=np.float64)

    for i in range(len(preds)):
        for h in range(horizon):
            idx = i + h
            if idx >= total_len:
                continue
            pred_sum[idx] += float(preds[i, h])
            pred_count[idx] += 1
            true_values[idx] = float(truth[i, h])

    valid_mask = pred_count > 0
    return true_values[valid_mask], pred_sum[valid_mask] / pred_count[valid_mask]


def train_component_model(
    df: pd.DataFrame,
    component_signal: np.ndarray,
    model_name: str,
    config: ForecastConfig,
    device: torch.device,
) -> dict[str, object]:
    features = assemble_features(df, component_signal)
    targets = np.asarray(component_signal, dtype=np.float32)
    train_norm, test_norm, train_targets, test_targets, target_mean, target_std, split_idx = split_and_scale(features, targets, config)

    y_train_norm = (train_targets - target_mean) / target_std
    y_test_norm = (test_targets - target_mean) / target_std
    x_train, y_train = create_sequences(train_norm, y_train_norm, lookback=config.lookback, horizon=config.horizon)
    x_test, y_test = create_sequences(test_norm, y_test_norm, lookback=config.lookback, horizon=config.horizon)
    if len(x_train) == 0 or len(x_test) == 0:
        raise ValueError("切片后的训练集或测试集为空，请调整参数。")

    model = ModelFactory.create(model_name, input_size=x_train.shape[-1], lookback=config.lookback, horizon=config.horizon, dropout=config.dropout)
    trainer = SequenceTrainer(model, config, device)
    losses = trainer.fit(x_train, y_train)
    pred_norm = trainer.predict(x_test)

    pred = pred_norm * target_std + target_mean
    truth = y_test * target_std + target_mean

    total_len = len(component_signal) - split_idx
    y_true, y_pred = aggregate_horizon_predictions(pred, truth, total_len, config.horizon)
    metrics = calculate_metrics(y_true, y_pred)
    return {
        "y_true": y_true,
        "y_pred": y_pred,
        "metrics": metrics,
        "losses": losses,
        "split_idx": split_idx,
    }


def save_component_loss_plot(loss_history: dict[str, list[float]], output_path: Path, title: str) -> None:
    if not loss_history:
        return
    fig, ax = plt.subplots(figsize=(12, 6))
    for name, losses in loss_history.items():
        ax.plot(range(1, len(losses) + 1), losses, label=name)
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend(ncol=2, fontsize=8)
    ax.grid(True, alpha=0.3)
    save_figure(fig, output_path.parent, output_path.name)


def save_forecast_plot(forecast_df: pd.DataFrame, output_path: Path, title: str) -> None:
    plot_df = forecast_df.copy()
    if len(plot_df) > 7 * 24 * 4:
        plot_df = plot_df.tail(7 * 24 * 4)
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(plot_df["timestamp"], plot_df["actual_load"], label="Actual Load", linewidth=1.0)
    ax.plot(plot_df["timestamp"], plot_df["predicted_load"], label="Predicted Load", linewidth=1.0)
    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Load")
    ax.legend()
    ax.grid(True, alpha=0.3)
    save_figure(fig, output_path.parent, output_path.name)


def save_metrics_artifacts(metrics: dict[str, float], output_dir: Path, stem: str) -> None:
    metrics_df = pd.DataFrame({"metric": list(metrics.keys()), "value": list(metrics.values())})
    metrics_df.to_csv(output_dir / f"{stem}_metrics.csv", index=False, encoding="utf-8-sig")
    with (output_dir / f"{stem}_metrics.txt").open("w", encoding="utf-8") as file_obj:
        for key, value in metrics.items():
            file_obj.write(f"{key}: {value:.6f}\n")


def save_forecast_artifacts(
    forecast_df: pd.DataFrame,
    metrics: dict[str, float],
    output_dir: Path,
    method_name: str,
    loss_history: dict[str, list[float]],
) -> None:
    forecast_df.to_csv(output_dir / f"{method_name}_forecast.csv", index=False, encoding="utf-8-sig")
    save_metrics_artifacts(metrics, output_dir, method_name)
    save_forecast_plot(forecast_df, output_dir / f"{method_name}_forecast.png", f"{method_name.upper()} Forecast")
    save_component_loss_plot(loss_history, output_dir / f"{method_name}_training_loss.png", f"{method_name.upper()} IMF Component Training Loss")


def build_forecast_frame(df: pd.DataFrame, split_idx: int, actual: np.ndarray, predicted: np.ndarray) -> pd.DataFrame:
    timestamps = pd.to_datetime(df["timestamp"]).reset_index(drop=True).iloc[split_idx : split_idx + len(actual)]
    return pd.DataFrame(
        {
            "timestamp": timestamps.values,
            "actual_load": actual,
            "predicted_load": predicted,
            "error": actual - predicted,
        }
    )


def save_emd_outputs(df: pd.DataFrame, component_df: pd.DataFrame, output_dir: Path) -> None:
    figures_dir = output_dir / "figures"
    tables_dir = output_dir / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    imf_columns = [col for col in component_df.columns if col.startswith("imf_")]
    imf_values = component_df[imf_columns].to_numpy().T
    imf_export = component_df.rename(columns={col: f"IMF{idx + 1}" for idx, col in enumerate(imf_columns)})
    imf_export.to_csv(tables_dir / "emd_components.csv", index=False, encoding="utf-8-sig")
    save_imfs(imf_values, component_df["timestamp"], tables_dir)
    plot_emd_overview(df["load"], imf_values, figures_dir)
    plot_imf_components(imf_export.rename(columns={"timestamp": "timestamp"}), figures_dir)
    create_imf_analysis_figures(df, imf_values, figures_dir)


def run_single_method(df: pd.DataFrame, config: ForecastConfig, method_name: str, output_dir: Path) -> dict[str, object]:
    device = get_device()
    component_df = build_component_frame(df, config)
    save_emd_outputs(df, component_df, output_dir)

    component_columns = [col for col in component_df.columns if col != "timestamp"]
    truth_parts: list[np.ndarray] = []
    pred_parts: list[np.ndarray] = []
    loss_history: dict[str, list[float]] = {}
    component_metrics: list[dict[str, object]] = []
    split_idx = None

    for component_name in component_columns:
        result = train_component_model(df, component_df[component_name].to_numpy(), method_name, config, device)
        truth_parts.append(np.asarray(result["y_true"], dtype=float))
        pred_parts.append(np.asarray(result["y_pred"], dtype=float))
        loss_history[component_name] = list(result["losses"])
        split_idx = int(result["split_idx"])
        component_metrics.append({"component": component_name, "model": method_name, **dict(result["metrics"])})

    actual = np.sum(np.vstack(truth_parts), axis=0)
    predicted = np.sum(np.vstack(pred_parts), axis=0)
    forecast_df = build_forecast_frame(df, split_idx, actual, predicted)
    metrics = calculate_metrics(actual, predicted)

    pd.DataFrame(component_metrics).to_csv(output_dir / f"{method_name}_component_metrics.csv", index=False, encoding="utf-8-sig")
    save_forecast_artifacts(forecast_df, metrics, output_dir, method_name, loss_history)
    return {
        "method": method_name,
        "forecast_df": forecast_df,
        "metrics": metrics,
        "component_metrics": pd.DataFrame(component_metrics),
        "selected_models": None,
    }


def run_hybrid_method(df: pd.DataFrame, config: ForecastConfig, output_dir: Path) -> dict[str, object]:
    device = get_device()
    component_df = build_component_frame(df, config)
    save_emd_outputs(df, component_df, output_dir)

    component_columns = [col for col in component_df.columns if col != "timestamp"]
    candidate_models = list(ModelFactory.SUPPORTED_MODELS)
    selection_rows: list[dict[str, object]] = []
    truth_parts: list[np.ndarray] = []
    pred_parts: list[np.ndarray] = []
    loss_history: dict[str, list[float]] = {}
    split_idx = None

    for component_name in component_columns:
        candidates: list[dict[str, object]] = []
        for model_name in candidate_models:
            result = train_component_model(df, component_df[component_name].to_numpy(), model_name, config, device)
            candidates.append({"model": model_name, **result})
        best_candidate = min(candidates, key=lambda item: (item["metrics"]["RMSE"], item["metrics"]["MAE"], item["metrics"]["MAPE"]))
        truth_parts.append(np.asarray(best_candidate["y_true"], dtype=float))
        pred_parts.append(np.asarray(best_candidate["y_pred"], dtype=float))
        loss_history[component_name] = list(best_candidate["losses"])
        split_idx = int(best_candidate["split_idx"])
        selection_rows.append({"component": component_name, "selected_model": best_candidate["model"], **dict(best_candidate["metrics"])})

    actual = np.sum(np.vstack(truth_parts), axis=0)
    predicted = np.sum(np.vstack(pred_parts), axis=0)
    forecast_df = build_forecast_frame(df, split_idx, actual, predicted)
    metrics = calculate_metrics(actual, predicted)

    selection_df = pd.DataFrame(selection_rows)
    selection_df.to_csv(output_dir / "imf_model_selection.csv", index=False, encoding="utf-8-sig")
    save_forecast_artifacts(forecast_df, metrics, output_dir, "hybrid", loss_history)
    return {
        "method": "hybrid",
        "forecast_df": forecast_df,
        "metrics": metrics,
        "component_metrics": selection_df,
        "selected_models": selection_df,
    }


def run_best_method(df: pd.DataFrame, config: ForecastConfig, output_dir: Path) -> dict[str, object]:
    candidates_root = output_dir / "candidates"
    candidates_root.mkdir(parents=True, exist_ok=True)
    candidate_methods = [*ModelFactory.SUPPORTED_MODELS, "hybrid"]
    candidate_results: list[dict[str, object]] = []

    for method_name in candidate_methods:
        method_dir = candidates_root / method_name
        method_dir.mkdir(parents=True, exist_ok=True)
        if method_name == "hybrid":
            candidate_results.append(run_hybrid_method(df, config, method_dir))
        else:
            candidate_results.append(run_single_method(df, config, method_name, method_dir))

    best_result = min(candidate_results, key=lambda item: (item["metrics"]["RMSE"], item["metrics"]["MAE"], item["metrics"]["MAPE"]))
    forecast_df = best_result["forecast_df"]
    metrics = best_result["metrics"]
    summary_lines = [
        f"best_method: {best_result['method']}",
        *[f"{name}: {value:.6f}" for name, value in metrics.items()],
    ]
    (output_dir / "best_method_summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    save_forecast_artifacts(forecast_df, metrics, output_dir, "best", {})
    with (output_dir / "best_method_metrics.json").open("w", encoding="utf-8") as file_obj:
        json.dump({"best_method": best_result["method"], **metrics}, file_obj, ensure_ascii=False, indent=2)

    if best_result.get("selected_models") is not None:
        best_result["selected_models"].to_csv(output_dir / "imf_model_selection.csv", index=False, encoding="utf-8-sig")

    forecast_df.to_csv(output_dir / "best_forecast.csv", index=False, encoding="utf-8-sig")
    return {
        "method": "best",
        "forecast_df": forecast_df,
        "metrics": metrics,
        "best_method": best_result["method"],
    }


def run_forecast(df: pd.DataFrame, config: ForecastConfig, method_name: str, output_dir: Path) -> dict[str, object]:
    set_seed(config.random_seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    normalized = method_name.lower()
    if normalized in ModelFactory.SUPPORTED_MODELS:
        return run_single_method(df, config, normalized, output_dir)
    if normalized == "hybrid":
        return run_hybrid_method(df, config, output_dir)
    if normalized == "best":
        return run_best_method(df, config, output_dir)
    raise ValueError(f"不支持的预测方式：{method_name}")
