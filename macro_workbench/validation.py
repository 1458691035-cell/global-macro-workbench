from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import pandas as pd

from .memo import generate_memo
from .models import SeriesSpec
from .pipeline import calculate_snapshot
from .storage import MacroStore


@dataclass(frozen=True)
class ValidationReport:
    replay_days: int
    average_coverage: float
    average_healthy: float
    average_memo_seconds: float
    review_series: tuple[str, ...]

    def as_markdown(self) -> str:
        reviews = "、".join(self.review_series) if self.review_series else "无"
        return (
            "# 工作流试运行报告\n\n"
            f"- 回放交易日：{self.replay_days}\n"
            f"- 平均信号覆盖率：{self.average_coverage:.1%}\n"
            f"- 平均质量健康度：{self.average_healthy:.1%}\n"
            f"- 平均 memo 生成耗时：{self.average_memo_seconds:.3f} 秒\n"
            f"- 建议删减或替换：{reviews}\n\n"
            "规则：10 日中缺失/过期超过 40%，或衍生信号没有变化的序列进入复核；"
            "系统不会在缺少真实发布周期证据时自动删除指标。"
        )


def validate_replay(
    store: MacroStore, specs: list[SeriesSpec], end: str | None = None, days: int = 10
) -> ValidationReport:
    raw_max = store.query("SELECT MAX(observation_date) AS date FROM raw_observations").iloc[0, 0]
    end_date = pd.Timestamp(end or raw_max or pd.Timestamp.today()).normalize()
    replay_dates = pd.bdate_range(end=end_date, periods=days)
    quality_frames: list[pd.DataFrame] = []
    signal_frames: list[pd.DataFrame] = []
    memo_times: list[float] = []
    for replay_date in replay_dates:
        signals, quality, regime = calculate_snapshot(store, specs, replay_date.date())
        store.upsert_frame("derived_signals", signals)
        store.upsert_frame("quality_status", quality)
        store.upsert_frame("regime_snapshots", regime)
        quality_frames.append(quality.assign(replay_date=replay_date.date()))
        signal_frames.append(signals.assign(replay_date=replay_date.date()))
        started = perf_counter()
        generate_memo(store, replay_date.date())
        memo_times.append(perf_counter() - started)

    quality_all = pd.concat(quality_frames, ignore_index=True)
    signals_all = pd.concat(signal_frames, ignore_index=True)
    bad_rate = quality_all.assign(
        bad=quality_all.status.isin(["missing", "stale"])
    ).groupby("series_id")["bad"].mean()
    variability = signals_all.groupby("series_id")["transformed_value"].nunique(dropna=True)
    review = sorted(
        set(bad_rate[bad_rate > 0.4].index) | set(variability[variability <= 1].index)
    )
    coverage = signals_all.groupby("replay_date")["series_id"].nunique() / len(specs)
    healthy = quality_all.groupby("replay_date")["status"].apply(lambda x: (x == "ok").mean())
    return ValidationReport(
        replay_days=len(replay_dates),
        average_coverage=float(coverage.mean()),
        average_healthy=float(healthy.mean()),
        average_memo_seconds=float(pd.Series(memo_times).mean()),
        review_series=tuple(review),
    )
