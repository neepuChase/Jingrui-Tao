from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

MIN_ANALYSIS_SPAN = pd.Timedelta(days=7)


@dataclass
class TimeRangeSelection:
    mode: str
    start: pd.Timestamp
    end: pd.Timestamp
    label: str
    description: str
    clipped: bool = False


class TimeRangeError(ValueError):
    pass


def _normalize_timestamp(value: str | pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        ts = ts.tz_convert(None)
    return ts


def _format_label(start: pd.Timestamp, end: pd.Timestamp) -> str:
    return f"{start:%Y%m%d%H%M}_{end:%Y%m%d%H%M}"


def resolve_time_range(
    timestamps: pd.Series,
    *,
    mode: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
    recent_value: Optional[int] = None,
    recent_unit: Optional[str] = None,
) -> TimeRangeSelection:
    if timestamps.empty:
        raise TimeRangeError("数据集中没有可用时间戳。")

    ts = pd.to_datetime(timestamps).sort_values().reset_index(drop=True)
    data_start = ts.iloc[0]
    data_end = ts.iloc[-1]
    clipped = False

    if mode == "all":
        selected_start, selected_end = data_start, data_end
        description = f"全部数据：{data_start} ~ {data_end}"
        return TimeRangeSelection(mode=mode, start=selected_start, end=selected_end, label="all_data", description=description)

    if mode == "range":
        if not start or not end:
            raise TimeRangeError("自定义时间范围模式需要同时提供开始时间和结束时间。")
        selected_start = _normalize_timestamp(start)
        selected_end = _normalize_timestamp(end)
        if selected_start > selected_end:
            raise TimeRangeError("开始时间晚于结束时间，请重新输入。")
    elif mode == "recent":
        if recent_value is None or recent_value <= 0:
            raise TimeRangeError("最近时间段模式需要提供大于 0 的长度。")
        if recent_unit not in {"days", "weeks"}:
            raise TimeRangeError("最近时间段模式的单位仅支持 days 或 weeks。")
        delta = pd.Timedelta(days=recent_value) if recent_unit == "days" else pd.Timedelta(weeks=recent_value)
        selected_end = data_end
        selected_start = data_end - delta
    else:
        raise TimeRangeError(f"不支持的时间选择模式：{mode}")

    if selected_end < data_start or selected_start > data_end:
        raise TimeRangeError("所选时间范围完全超出原始数据边界。")

    if selected_start < data_start:
        selected_start = data_start
        clipped = True
    if selected_end > data_end:
        selected_end = data_end
        clipped = True

    if selected_end - selected_start < MIN_ANALYSIS_SPAN:
        raise TimeRangeError("所选时间范围少于 1 周，请至少选择 7 天数据。")

    description = f"{selected_start} ~ {selected_end}"
    if clipped:
        description += "（已按数据边界自动截断）"
    return TimeRangeSelection(
        mode=mode,
        start=selected_start,
        end=selected_end,
        label=_format_label(selected_start, selected_end),
        description=description,
        clipped=clipped,
    )


def filter_dataframe_by_time(df: pd.DataFrame, selection: TimeRangeSelection) -> pd.DataFrame:
    timestamp_series = pd.to_datetime(df["timestamp"])
    mask = (timestamp_series >= selection.start) & (timestamp_series <= selection.end)
    filtered = df.loc[mask].copy().sort_values("timestamp").reset_index(drop=True)
    if filtered.empty:
        raise TimeRangeError("筛选后的数据为空，请调整时间范围。")
    return filtered


def prompt_time_range(timestamps: pd.Series) -> TimeRangeSelection:
    ts = pd.to_datetime(timestamps)
    data_start = ts.min()
    data_end = ts.max()
    print("\n可选时间范围：")
    print(f"- 数据起止：{data_start} ~ {data_end}")
    print("1. 全部数据")
    print("2. 自定义开始/结束时间")
    print("3. 最近 N 天")
    print("4. 最近 N 周")

    while True:
        choice = input("请选择时间范围方式（1/2/3/4）: ").strip()
        try:
            if choice == "1":
                return resolve_time_range(ts, mode="all")
            if choice == "2":
                start = input("请输入开始时间（如 2018-01-01 00:00）: ").strip()
                end = input("请输入结束时间（如 2018-03-31 23:45）: ").strip()
                return resolve_time_range(ts, mode="range", start=start, end=end)
            if choice == "3":
                days = int(input("请输入最近天数（至少 7）: ").strip())
                return resolve_time_range(ts, mode="recent", recent_value=days, recent_unit="days")
            if choice == "4":
                weeks = int(input("请输入最近周数（至少 1）: ").strip())
                return resolve_time_range(ts, mode="recent", recent_value=weeks, recent_unit="weeks")
            print("请输入 1/2/3/4 中的一个选项。")
        except (ValueError, TimeRangeError) as exc:
            print(f"时间范围选择失败：{exc}")
