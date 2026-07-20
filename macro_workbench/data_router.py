from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import date
from typing import Literal

import pandas as pd

from .akshare_source import FetchResult, fetch_akshare_observations
from .models import SeriesSpec
from .openbb_source import fetch_openbb_observations
from .tsanghi_source import fetch_tsanghi_observations


ProgressCallback = Callable[[int, int, str], None]
FetchMode = Literal["incremental", "full"]


class _Progress:
    def __init__(self, total: int, callback: ProgressCallback | None) -> None:
        self.total = max(total, 1)
        self.done = 0
        self.callback = callback

    def emit(self, message: str) -> None:
        if self.callback is not None:
            self.callback(min(self.done, self.total), self.total, message)


def resolve_start_dates(
    specs: list[SeriesSpec],
    *,
    end: str | date | None = None,
    years: int = 5,
    mode: FetchMode = "incremental",
    lookback_days: int = 30,
    latest_dates: Mapping[str, date] | None = None,
) -> dict[str, pd.Timestamp]:
    """Compute per-series fetch start dates.

    Incremental: max(observation_date) - lookback, or full window if missing.
    Full: always today - years.
    """
    end_date = pd.Timestamp(end or date.today()).normalize()
    full_start = end_date - pd.DateOffset(years=years)
    latest = latest_dates or {}
    starts: dict[str, pd.Timestamp] = {}
    for spec in specs:
        if mode == "full":
            starts[spec.id] = full_start
            continue
        last = latest.get(spec.id)
        if last is None:
            starts[spec.id] = full_start
        else:
            incremental = pd.Timestamp(last).normalize() - pd.Timedelta(days=lookback_days)
            starts[spec.id] = max(full_start, incremental)
    return starts


def fetch_all_observations(
    specs: list[SeriesSpec],
    end: str | date | None = None,
    years: int = 5,
    on_progress: ProgressCallback | None = None,
    *,
    mode: FetchMode = "incremental",
    lookback_days: int = 30,
    latest_dates: Mapping[str, date] | None = None,
) -> FetchResult:
    """Fetch observations. OpenBB failures never fall back to AKShare.

    AKShare is used only for specs with ``source == "akshare"`` (China macro).
    Tsanghi is used for specs with ``source == "tsanghi"`` (market quotes).
    """
    openbb_specs = [spec for spec in specs if spec.source == "openbb"]
    akshare_specs = [spec for spec in specs if spec.source == "akshare"]
    tsanghi_specs = [spec for spec in specs if spec.source == "tsanghi"]
    unknown = [spec for spec in specs if spec.source not in {"openbb", "akshare", "tsanghi"}]

    start_by_series = resolve_start_dates(
        specs,
        end=end,
        years=years,
        mode=mode,
        lookback_days=lookback_days,
        latest_dates=latest_dates,
    )

    frames: list[pd.DataFrame] = []
    errors: dict[str, str] = {
        spec.id: f"未知数据源: {spec.source}" for spec in unknown
    }
    progress = _Progress(len(openbb_specs) + len(akshare_specs) + len(tsanghi_specs) + len(unknown), on_progress)

    mode_label = "全量刷新" if mode == "full" else f"增量更新（回看 {lookback_days} 天）"
    progress.emit(f"模式：{mode_label}")

    if unknown:
        progress.done += len(unknown)
        progress.emit(f"跳过 {len(unknown)} 条未知数据源")

    if openbb_specs:
        base = progress.done

        def openbb_progress(local_done: int, _local_total: int, message: str) -> None:
            progress.done = base + local_done
            progress.emit(message)

        progress.emit(f"开始 OpenBB 拉取（{len(openbb_specs)} 条，无 AKShare 回退）…")
        result = fetch_openbb_observations(
            openbb_specs,
            end=end,
            years=years,
            on_progress=openbb_progress,
            start_by_series=start_by_series,
        )
        if not result.observations.empty:
            frames.append(result.observations)
        errors.update(result.errors)
        progress.done = base + len(openbb_specs)
        progress.emit(f"OpenBB 完成（{len(openbb_specs)} 条）")

    if tsanghi_specs:
        base = progress.done

        def tsanghi_progress(local_done: int, _local_total: int, message: str) -> None:
            progress.done = base + local_done
            progress.emit(message)

        progress.emit(f"开始 tsanghi 拉取（{len(tsanghi_specs)} 条行情序列）…")
        result = fetch_tsanghi_observations(
            tsanghi_specs,
            end=end,
            years=years,
            on_progress=tsanghi_progress,
            start_by_series=start_by_series,
        )
        if not result.observations.empty:
            frames.append(result.observations)
        for series_id, message in result.errors.items():
            errors[series_id] = message
        progress.done = base + len(tsanghi_specs)
        progress.emit(f"tsanghi 完成（{len(tsanghi_specs)} 条）")

    if akshare_specs:
        base = progress.done

        def akshare_progress(local_done: int, _local_total: int, message: str) -> None:
            progress.done = base + local_done
            progress.emit(message)

        progress.emit(f"开始 AKShare 拉取（{len(akshare_specs)} 条中国序列）…")
        result = fetch_akshare_observations(
            akshare_specs,
            end=end,
            years=years,
            on_progress=akshare_progress,
            start_by_series=start_by_series,
        )
        if not result.observations.empty:
            frames.append(result.observations)
        for series_id, message in result.errors.items():
            errors[series_id] = message
        progress.done = base + len(akshare_specs)
        progress.emit(f"AKShare 完成（{len(akshare_specs)} 条）")

    if on_progress is not None:
        on_progress(progress.total, progress.total, "数据获取结束")

    columns = [
        "series_id",
        "observation_date",
        "value",
        "release_time",
        "vintage_date",
        "source",
        "last_updated",
    ]
    observations = (
        pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=columns)
    )
    return FetchResult(observations[columns], errors)
