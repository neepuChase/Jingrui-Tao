"""EMD decomposition utilities."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

try:
    from PyEMD import EMD  # type: ignore
except ImportError:  # pragma: no cover
    EMD = None

from src.visualization import save_figure

MAX_IMF = 10


def _fallback_emd(values: np.ndarray, max_imf: int) -> np.ndarray:
    windows = [4, 16, 48, 96, 192, 336, 672]
    windows = windows[: max(1, max_imf - 1)]
    residual = pd.Series(values.astype(float))
    components: list[np.ndarray] = []

    for window in windows:
        if len(residual) <= window:
            break
        smooth = residual.rolling(window=window, center=True, min_periods=1).mean()
        component = (residual - smooth).to_numpy(dtype=float)
        components.append(component)
        residual = smooth

    components.append(residual.to_numpy(dtype=float))
    return np.asarray(components[:max_imf], dtype=float)


def perform_emd(load_series: pd.Series, max_imf: int = MAX_IMF) -> np.ndarray:
    """Run EMD on the load series and cap IMF count at max_imf."""
    values = load_series.astype(float).values
    if EMD is not None:
        imfs = EMD().emd(values)
    else:
        imfs = _fallback_emd(values, max_imf=max_imf)
    if imfs.shape[0] > max_imf:
        imfs = imfs[:max_imf, :]
    return imfs


def save_imfs(imfs: np.ndarray, timestamps: pd.Series, outputs_dir: Path) -> pd.DataFrame:
    """Save IMF components and return a DataFrame."""
    imf_columns = [f"IMF{i + 1}" for i in range(imfs.shape[0])]
    imf_df = pd.DataFrame(imfs.T, columns=imf_columns)
    imf_df.insert(0, "timestamp", pd.to_datetime(timestamps).values)

    outputs_dir.mkdir(parents=True, exist_ok=True)
    imf_df.to_csv(outputs_dir / "emd_decomposition_results.csv", index=False, encoding="utf-8-sig")
    return imf_df


def plot_emd_overview(load_series: pd.Series, imfs: np.ndarray, figures_dir: Path) -> None:
    n_imfs = imfs.shape[0]
    fig, axes = plt.subplots(n_imfs + 1, 1, figsize=(14, 2.2 * (n_imfs + 1)), sharex=True)

    axes[0].plot(load_series.values, color="black", linewidth=1.0)
    axes[0].set_title("Raw Load Series")
    axes[0].grid(True, alpha=0.2)

    for i in range(n_imfs):
        axes[i + 1].plot(imfs[i], linewidth=0.8)
        axes[i + 1].set_title(f"IMF Component {i + 1}")
        axes[i + 1].grid(True, alpha=0.2)

    axes[-1].set_xlabel("Time Index")
    save_figure(fig, figures_dir, "emd_decomposition_overview.png")


def plot_imf_components(imf_df: pd.DataFrame, figures_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(14, 7))
    imf_columns = [c for c in imf_df.columns if c.startswith("IMF")]
    for col in imf_columns:
        ax.plot(imf_df["timestamp"], imf_df[col], linewidth=0.8, label=col)

    ax.set_title("IMF Components")
    ax.set_xlabel("Time")
    ax.set_ylabel("Component Value")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=3, fontsize=8)
    save_figure(fig, figures_dir, "imf_components.png")


def classify_imfs_by_frequency(imfs: list[np.ndarray] | np.ndarray, sampling_interval: float = 1.0) -> dict[str, object]:
    """Classify IMFs by dominant frequency using FFT thresholding."""
    imf_array = np.asarray(imfs, dtype=float)
    features = []
    groups = {"high": [], "mid": [], "low": []}

    if imf_array.size == 0:
        empty_df = pd.DataFrame(columns=["index", "dominant_freq", "energy"])
        return {**groups, "features": empty_df}

    if imf_array.ndim == 1:
        imf_array = imf_array[np.newaxis, :]

    safe_sampling_interval = sampling_interval if sampling_interval and sampling_interval > 0 else 1.0

    for idx, imf in enumerate(imf_array, start=1):
        signal = np.asarray(imf, dtype=float)
        n = signal.size
        energy = float(np.sum(np.square(signal)))

        dominant_freq = 0.0
        if n > 0:
            centered = signal - np.mean(signal)
            fft_values = np.fft.fft(centered)
            frequencies = np.fft.fftfreq(n, d=safe_sampling_interval)
            positive_mask = frequencies > 0
            positive_freqs = frequencies[positive_mask]
            amplitude = np.abs(fft_values[positive_mask])
            if positive_freqs.size > 0 and amplitude.size > 0:
                dominant_freq = float(positive_freqs[int(np.argmax(amplitude))])

        feature = {"index": idx, "dominant_freq": dominant_freq, "energy": energy}
        features.append(feature)

        if dominant_freq > 0.1:
            groups["high"].append(idx)
        elif dominant_freq > 0.02:
            groups["mid"].append(idx)
        else:
            groups["low"].append(idx)

    features_df = pd.DataFrame(features, columns=["index", "dominant_freq", "energy"])
    return {**groups, "features": features_df}


def create_imf_analysis_figures(load_df: pd.DataFrame, imfs: np.ndarray, figures_dir: Path) -> None:
    """Create IMF spectrum, energy, reconstruction, and volatility figures."""
    if imfs.size == 0:
        return

    timestamps = pd.to_datetime(load_df["timestamp"])
    sampling_hours = _infer_sampling_hours(timestamps)
    classification = classify_imfs_by_frequency(imfs, sampling_interval=sampling_hours)
    _plot_imf_spectrum_overview(imfs, sampling_hours, figures_dir)
    _plot_imf_frequency_classification(classification, figures_dir)
    _plot_imf_energy_ratio(classification["features"], figures_dir)
    _plot_imf_reconstruction(load_df["load"].to_numpy(), imfs, timestamps, figures_dir)
    _plot_imf_volatility_decomposition(imfs, figures_dir)


def _infer_sampling_hours(timestamps: pd.Series) -> float:
    diffs = timestamps.sort_values().diff().dropna()
    if diffs.empty:
        return 1.0
    step_seconds = diffs.dt.total_seconds().median()
    if pd.isna(step_seconds) or step_seconds <= 0:
        return 1.0
    return step_seconds / 3600.0


def _positive_fft(signal: np.ndarray, sampling_hours: float) -> tuple[np.ndarray, np.ndarray]:
    centered = signal.astype(float) - np.mean(signal)
    n = centered.size
    if n == 0:
        return np.array([]), np.array([])

    fft_values = np.fft.fft(centered)
    frequencies = np.fft.fftfreq(n, d=sampling_hours)
    positive = frequencies > 0
    return frequencies[positive], np.abs(fft_values[positive])


def _plot_imf_spectrum_overview(imfs: np.ndarray, sampling_hours: float, figures_dir: Path) -> None:
    n_imfs = imfs.shape[0]
    fig, axes = plt.subplots(n_imfs, 1, figsize=(14, max(3 * n_imfs, 4)), sharex=True)
    if n_imfs == 1:
        axes = [axes]

    for idx, ax in enumerate(axes):
        frequencies, amplitude = _positive_fft(imfs[idx], sampling_hours)
        ax.plot(frequencies, amplitude, linewidth=0.8, color="tab:blue")
        ax.set_title(f"IMF{idx + 1} Spectrum")
        ax.set_ylabel("Amplitude")
        ax.grid(True, alpha=0.2)

    axes[-1].set_xlabel("Frequency (cycles/hour)")
    save_figure(fig, figures_dir, "imf_spectrum_overview.png")


def _plot_imf_frequency_classification(classification: dict[str, object], figures_dir: Path) -> None:
    features = classification["features"]
    if features.empty:
        return

    plot_df = features.copy()
    plot_df["category"] = "low"
    plot_df.loc[plot_df["index"].isin(classification["mid"]), "category"] = "mid"
    plot_df.loc[plot_df["index"].isin(classification["high"]), "category"] = "high"

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(
        data=plot_df,
        x="index",
        y="dominant_freq",
        hue="category",
        dodge=False,
        palette={"high": "tab:red", "mid": "tab:orange", "low": "tab:green"},
        ax=ax,
    )
    ax.set_title("IMF Frequency Classification")
    ax.set_xlabel("IMF Index")
    ax.set_ylabel("Dominant Frequency")
    ax.grid(True, axis="y", alpha=0.3)
    save_figure(fig, figures_dir, "imf_frequency_classification.png")


def _plot_imf_energy_ratio(features: pd.DataFrame, figures_dir: Path) -> None:
    if features.empty:
        return

    plot_df = features.copy()
    total_energy = float(plot_df["energy"].sum())
    plot_df["energy_ratio"] = plot_df["energy"] / total_energy if total_energy else 0.0

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(data=plot_df, x="index", y="energy_ratio", color="tab:cyan", ax=ax)
    ax.set_title("IMF Energy Contribution Ratio")
    ax.set_xlabel("IMF Index")
    ax.set_ylabel("Energy Ratio")
    ax.grid(True, axis="y", alpha=0.3)
    save_figure(fig, figures_dir, "imf_energy_ratio.png")


def _plot_imf_reconstruction(load: np.ndarray, imfs: np.ndarray, timestamps: pd.Series, figures_dir: Path) -> None:
    high = imfs[0:2].sum(axis=0) if imfs.shape[0] >= 1 else np.zeros_like(load)
    mid = imfs[2:4].sum(axis=0) if imfs.shape[0] >= 3 else np.zeros_like(load)
    low = imfs[4:-1].sum(axis=0) if imfs.shape[0] >= 6 else np.zeros_like(load)
    trend = imfs[-1] if imfs.shape[0] > 0 else np.zeros_like(load)

    fig, axes = plt.subplots(5, 1, figsize=(14, 14), sharex=True)
    series_map = [
        (load, "Original Load", "black"),
        (high, "High-Frequency Reconstruction (IMF1-2)", "tab:red"),
        (mid, "Mid-Frequency Reconstruction (IMF3-4)", "tab:orange"),
        (low, "Low-Frequency Reconstruction (IMF5+)", "tab:green"),
        (trend, "Residual Trend", "tab:blue"),
    ]

    for ax, (series, title, color) in zip(axes, series_map):
        ax.plot(timestamps, series, linewidth=0.8, color=color)
        ax.set_title(title)
        ax.grid(True, alpha=0.2)

    axes[-1].set_xlabel("Time")
    save_figure(fig, figures_dir, "imf_reconstruction.png")


def _plot_imf_volatility_decomposition(imfs: np.ndarray, figures_dir: Path) -> None:
    volatility = np.std(imfs, axis=1)
    labels = [f"IMF{i + 1}" for i in range(imfs.shape[0])]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(labels, volatility, color="tab:purple")
    ax.set_title("IMF Volatility Decomposition")
    ax.set_xlabel("IMF Component")
    ax.set_ylabel("Standard Deviation")
    ax.grid(True, axis="y", alpha=0.3)
    save_figure(fig, figures_dir, "imf_volatility_decomposition.png")
