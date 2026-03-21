from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.analysis import run_load_analysis
from src.data_loader import choose_primary_dataset, load_csv_robust
from src.preprocess import clean_load_data, infer_timestamp_and_load_columns
from src.utils import TimeRangeError, filter_dataframe_by_time, prompt_time_range, resolve_time_range
from src.visualization import configure_style

MODE_CHOICES = ("analysis", "forecast", "both")
FORECAST_METHOD_CHOICES = ("tcn", "lstm", "autoformer", "scinet", "hybrid", "best")
TIME_MODE_CHOICES = ("all", "range", "recent")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="轻量级多变量负荷分析与 EMD 预测项目")
    parser.add_argument("--data-path", type=str, default=None, help="CSV 数据文件路径，默认自动选择仓库内主数据集。")
    parser.add_argument("--mode", choices=MODE_CHOICES, default=None, help="运行模式：analysis / forecast / both。")
    parser.add_argument("--time-mode", choices=TIME_MODE_CHOICES, default=None, help="时间选择模式：all / range / recent。")
    parser.add_argument("--start", type=str, default=None, help="自定义开始时间。")
    parser.add_argument("--end", type=str, default=None, help="自定义结束时间。")
    parser.add_argument("--recent-value", type=int, default=None, help="最近 N 天或 N 周中的 N。")
    parser.add_argument("--recent-unit", choices=("days", "weeks"), default=None, help="最近时间长度单位：days / weeks。")
    parser.add_argument("--forecast-method", choices=FORECAST_METHOD_CHOICES, default=None, help="预测方式。")
    parser.add_argument("--lookback", type=int, default=672, help="输入窗口长度。")
    parser.add_argument("--horizon", type=int, default=96, help="预测步长。")
    parser.add_argument("--epochs", type=int, default=10, help="训练轮数。")
    parser.add_argument("--batch-size", type=int, default=128, help="训练批大小。")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="训练集比例。")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="学习率。")
    parser.add_argument("--dropout", type=float, default=0.1, help="dropout 比例。")
    parser.add_argument("--imf-components", type=int, default=6, help="参与预测的 IMF 分量数量。")
    parser.add_argument("--seed", type=int, default=42, help="随机种子。")
    return parser


def prompt_mode() -> str:
    mapping = {"1": "analysis", "2": "forecast", "3": "both"}
    print("\n请选择运行模式：")
    print("1. 仅分析")
    print("2. 仅预测")
    print("3. 分析 + 预测")
    while True:
        choice = input("请输入选项（1/2/3）: ").strip()
        if choice in mapping:
            return mapping[choice]
        print("输入无效，请输入 1/2/3。")


def prompt_forecast_method() -> str:
    mapping = {
        "1": "tcn",
        "2": "lstm",
        "3": "autoformer",
        "4": "scinet",
        "5": "hybrid",
        "6": "best",
    }
    print("\n请选择预测方式：")
    print("1. TCN")
    print("2. LSTM")
    print("3. AutoFormer")
    print("4. SCINet")
    print("5. 混合预测")
    print("6. 最优预测")
    while True:
        choice = input("请输入选项（1/2/3/4/5/6）: ").strip()
        if choice in mapping:
            return mapping[choice]
        print("输入无效，请重新输入。")


def choose_time_range(args: argparse.Namespace, timestamps: pd.Series):
    if args.time_mode:
        return resolve_time_range(
            timestamps,
            mode=args.time_mode,
            start=args.start,
            end=args.end,
            recent_value=args.recent_value,
            recent_unit=args.recent_unit,
        )
    return prompt_time_range(timestamps)


def prepare_output_dirs(repo_root: Path, selection_label: str, forecast_method: str | None) -> tuple[Path, Path | None]:
    outputs_root = repo_root / "outputs"
    analysis_dir = outputs_root / "analysis" / selection_label
    analysis_dir.mkdir(parents=True, exist_ok=True)

    forecast_dir = None
    if forecast_method:
        timestamp_label = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        forecast_dir = outputs_root / "forecast" / f"{forecast_method}_{selection_label}_{timestamp_label}"
        forecast_dir.mkdir(parents=True, exist_ok=True)
    return analysis_dir, forecast_dir


def print_summary(title: str, payload: dict[str, object]) -> None:
    print(f"\n=== {title} ===")
    for key, value in payload.items():
        print(f"{key}: {value}")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parent
    configure_style()

    try:
        dataset_path = Path(args.data_path) if args.data_path else choose_primary_dataset(repo_root)
        raw_df, encoding = load_csv_robust(dataset_path)
        inferred = infer_timestamp_and_load_columns(raw_df)
        cleaned_df, quality = clean_load_data(raw_df, inferred.timestamp_col, inferred.load_col)

        mode = args.mode or prompt_mode()
        selection = choose_time_range(args, cleaned_df["timestamp"])
        filtered_df = filter_dataframe_by_time(cleaned_df, selection)
        forecast_method = None
        if mode in {"forecast", "both"}:
            forecast_method = args.forecast_method or prompt_forecast_method()

        analysis_dir, forecast_dir = prepare_output_dirs(repo_root, selection.label, forecast_method)

        metadata = {
            "dataset_path": str(dataset_path.relative_to(repo_root) if dataset_path.is_relative_to(repo_root) else dataset_path),
            "encoding": encoding,
            "timestamp_column": inferred.timestamp_col,
            "load_column": inferred.load_col,
            "selection": selection.description,
            "rows_after_filter": len(filtered_df),
        }
        pd.DataFrame({"field": list(metadata.keys()), "value": list(metadata.values())}).to_csv(
            analysis_dir / "dataset_metadata.csv", index=False, encoding="utf-8-sig"
        )
        pd.DataFrame({"metric": list(quality.keys()), "value": list(quality.values())}).to_csv(
            analysis_dir / "data_quality_report.csv", index=False, encoding="utf-8-sig"
        )

        if mode in {"analysis", "both"}:
            analysis_outputs = run_load_analysis(filtered_df, analysis_dir, selection.description)
            print_summary("分析输出目录", {key: str(value) for key, value in analysis_outputs.items()})

        if mode in {"forecast", "both"} and forecast_dir is not None:
            from src.forecast import ForecastConfig, run_forecast

            config = ForecastConfig(
                lookback=args.lookback,
                horizon=args.horizon,
                train_ratio=args.train_ratio,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                dropout=args.dropout,
                random_seed=args.seed,
                imf_components=args.imf_components,
            )
            forecast_result = run_forecast(filtered_df, config, forecast_method, forecast_dir)
            summary_payload = {"method": forecast_method, **{k: v for k, v in forecast_result.items() if k not in {"forecast_df", "component_metrics", "selected_models"}}}
            summary_payload["output_dir"] = str(forecast_dir)
            print_summary("预测结果", summary_payload)

        print("\n任务执行完成。")
    except TimeRangeError as exc:
        print(f"时间范围错误：{exc}")
    except FileNotFoundError as exc:
        print(f"文件错误：{exc}")
    except ValueError as exc:
        print(f"参数或数据错误：{exc}")
    except Exception as exc:
        print(f"程序执行失败：{exc}")


if __name__ == "__main__":
    main()
