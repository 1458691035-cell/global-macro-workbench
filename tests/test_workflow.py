from __future__ import annotations

from pathlib import Path

import pandas as pd

from macro_workbench.memo import generate_memo
from macro_workbench.models import load_series
from macro_workbench.pipeline import generate_demo_observations, run_pipeline
from macro_workbench.storage import MacroStore
from macro_workbench.validation import validate_replay


ROOT = Path(__file__).resolve().parents[1]


def test_mvp_catalog_has_37_explainable_series_after_replay_pruning() -> None:
    specs = load_series(ROOT / "config" / "series.yaml")
    assert len(specs) == 37
    assert {spec.module for spec in specs} == {
        "cross_asset",
        "growth",
        "inflation",
        "policy",
        "positioning",
        "events",
    }
    assert all(spec.transform and spec.direction and spec.staleness_days for spec in specs)


def test_pipeline_is_vintage_aware_and_generates_memo(tmp_path: Path) -> None:
    specs = load_series(ROOT / "config" / "series.yaml")
    as_of = "2026-07-15"
    observations = generate_demo_observations(specs, as_of, years=2)
    store = MacroStore(tmp_path / "test.duckdb")
    signals, quality, regime = run_pipeline(store, specs, as_of, observations)

    assert len(signals) == len(specs)
    assert set(quality.status) <= {"ok", "stale", "anomaly", "missing"}
    assert regime.iloc[0].regime in {"Goldilocks", "Reflation", "Stagflation", "Slowdown"}

    future = observations.iloc[[0]].copy()
    future["series_id"] = "sp500"
    future["observation_date"] = pd.Timestamp("2026-07-16").date()
    future["release_time"] = pd.Timestamp("2026-07-16")
    future["vintage_date"] = pd.Timestamp("2026-07-16").date()
    future["value"] = 999_999.0
    store.upsert_frame("raw_observations", future)
    rerun, _, _ = run_pipeline(store, specs, as_of)
    assert rerun.loc[rerun.series_id == "sp500", "value"].iloc[0] != 999_999.0

    memo = generate_memo(store, as_of)
    for section in (
        "一句话结论",
        "Overnight",
        "Regime",
        "What changed",
        "What is priced",
        "Positioning",
        "Catalysts",
        "Trade map",
        "Watch list",
    ):
        assert section in memo
    store.close()


def test_ten_day_replay_meets_workflow_targets(tmp_path: Path) -> None:
    specs = load_series(ROOT / "config" / "series.yaml")
    observations = generate_demo_observations(specs, "2026-07-15", years=2)
    store = MacroStore(tmp_path / "replay.duckdb")
    store.register_catalog(specs)
    store.upsert_frame("raw_observations", observations)
    report = validate_replay(store, specs, "2026-07-15", days=10)

    assert report.replay_days == 10
    assert report.average_coverage >= 0.95
    assert report.average_memo_seconds < 10
    assert "工作流试运行报告" in report.as_markdown()
    store.close()
