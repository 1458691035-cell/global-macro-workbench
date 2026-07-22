from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import date
from typing import Literal

import pandas as pd

from .akshare_source import FetchResult, fetch_akshare_observations
from .models import SeriesSpec
from .openbb_source import fetch_openbb_observations
from .tsanghi_source import fetch_tsanghi_observations
from .yfinance_latest import fetch_yfinance_latest


ProgressCallback = Callable[[int, int, str], None]
FetchMode = Literal["incremental", "full"]

# Frequency-aware incremental windows (calendar days).
DEFAULT_DAILY_LOOKBACK = 5
WEEKLY_LOOKBACK = 14
MONTHLY_LOOKBACK = 65
NO_NEW_VALUE_MSG = "本周期无新值"

_EMPTY_WINDOW_MARKERS = (
    "results not found",
    "返回为空",
    "没有有效数值",
    "empty",
)


def incremental_lookback_days(spec: SeriesSpec, daily_lookback: int = DEFAULT_DAILY_LOOKBACK) -> int:
    """Return lookback window by series frequency."""
    freq = (spec.frequency or "daily").lower()
    if freq in {"monthly", "quarterly"}:
        return max(MONTHLY_LOOKBACK, daily_lookback)
    if freq == "weekly":
        return max(WEEKLY_LOOKBACK, daily_lookback)
    return max(daily_lookback, 1)


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
    lookback_days: int = DEFAULT_DAILY_LOOKBACK,
    latest_dates: Mapping[str, date] | None = None,
) -> dict[str, pd.Timestamp]:
    """Compute per-series fetch start dates.

    Incremental with history (lookback depends on frequency: daily 5 / weekly 14 /
    monthly 65 unless ``lookback_days`` is larger):
    - gap <= lookback: resume from last obs (1-day overlap for revisions)
    - gap > lookback: only pull the recent lookback window (not a long backfill)
    Missing history / full mode: today - years.
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
            continue
        last_ts = pd.Timestamp(last).normalize()
        window = incremental_lookback_days(spec, lookback_days)
        gap_days = int((end_date - last_ts).days)
        if gap_days > window:
            starts[spec.id] = max(full_start, end_date - pd.Timedelta(days=window))
        else:
            starts[spec.id] = max(full_start, last_ts - pd.Timedelta(days=1))
    return starts


def expected_tsanghi_latest(end: str | date | None = None) -> pd.Timestamp:
    """Calendar T-1 relative to end (default today)."""
    return pd.Timestamp(end or date.today()).normalize() - pd.Timedelta(days=1)


def stale_tsanghi_specs(
    specs: list[SeriesSpec],
    observations: pd.DataFrame,
    errors: Mapping[str, str],
    *,
    end: str | date | None = None,
) -> list[SeriesSpec]:
    """Return tsanghi specs that need a lightweight yfinance latest-bar patch."""
    expected = expected_tsanghi_latest(end)
    stale: list[SeriesSpec] = []
    for spec in specs:
        if not spec.yfinance_symbol:
            continue
        # Soft "no new value" is not a fetch failure worth patching away.
        if spec.id in errors and errors[spec.id] == NO_NEW_VALUE_MSG:
            continue
        if spec.id in errors:
            stale.append(spec)
            continue
        if observations.empty:
            stale.append(spec)
            continue
        subset = observations.loc[observations["series_id"] == spec.id]
        if subset.empty:
            stale.append(spec)
            continue
        max_obs = pd.to_datetime(subset["observation_date"]).max().normalize()
        if max_obs < expected:
            stale.append(spec)
    return stale


def _looks_like_empty_window(message: str) -> bool:
    text = (message or "").lower()
    return any(marker in text for marker in _EMPTY_WINDOW_MARKERS)


def soften_empty_window_errors(
    errors: Mapping[str, str],
    *,
    latest_dates: Mapping[str, date] | None,
    mode: FetchMode,
) -> dict[str, str]:
    """For incremental updates, empty windows on known series are not hard failures."""
    if mode != "incremental" or not latest_dates:
        return dict(errors)
    softened: dict[str, str] = {}
    for series_id, message in errors.items():
        if series_id in latest_dates and _looks_like_empty_window(message):
            softened[series_id] = NO_NEW_VALUE_MSG
        else:
            softened[series_id] = message
    return softened


def fetch_all_observations(
    specs: list[SeriesSpec],
    end: str | date | None = None,
    years: int = 5,
    on_progress: ProgressCallback | None = None,
    *,
    mode: FetchMode = "incremental",
    lookback_days: int = DEFAULT_DAILY_LOOKBACK,
    latest_dates: Mapping[str, date] | None = None,
) -> FetchResult:
    """Fetch observations. OpenBB failures never fall back to AKShare.

    AKShare is used only for specs with ``source == "akshare"`` (China macro).
    Tsanghi is used for market quotes; when its latest bar is older than calendar
    T-1, mapped series get a single-bar yfinance patch.
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

    mode_label = (
        "全量刷新"
        if mode == "full"
        else f"增量更新（日{lookback_days}/周{WEEKLY_LOOKBACK}/月{MONTHLY_LOOKBACK}天）"
    )
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
        errors.update(
            soften_empty_window_errors(
                result.errors, latest_dates=latest_dates, mode=mode
            )
        )
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
        tsanghi_obs = result.observations
        if not tsanghi_obs.empty:
            frames.append(tsanghi_obs)
        soft_tsanghi_errors = soften_empty_window_errors(
            result.errors, latest_dates=latest_dates, mode=mode
        )
        for series_id, message in soft_tsanghi_errors.items():
            errors[series_id] = message

        stale = stale_tsanghi_specs(tsanghi_specs, tsanghi_obs, soft_tsanghi_errors, end=end)
        if stale:
            progress.emit(f"tsanghi 滞后 {len(stale)} 条，yfinance 轻量补最新点…")
            patch = fetch_yfinance_latest(stale)
            patched_ids: set[str] = set()
            if not patch.observations.empty:
                frames.append(patch.observations)
                patched_ids = set(patch.observations["series_id"].astype(str))
                for series_id in patched_ids:
                    errors.pop(series_id, None)
            for series_id, message in patch.errors.items():
                if series_id not in patched_ids:
                    errors[series_id] = message
            merged = (
                pd.concat([frame for frame in frames if not frame.empty], ignore_index=True)
                if frames
                else pd.DataFrame()
            )
            still_stale = stale_tsanghi_specs(stale, merged, {}, end=end)
            expected = expected_tsanghi_latest(end).date()
            for spec in still_stale:
                if spec.id not in errors:
                    errors[spec.id] = f"yfinance 最新仍滞后（期望 >= {expected}）"

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
        errors.update(
            soften_empty_window_errors(
                result.errors, latest_dates=latest_dates, mode=mode
            )
        )
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
