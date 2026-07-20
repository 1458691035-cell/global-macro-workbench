from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from macro_workbench.data_router import resolve_start_dates
from macro_workbench.models import SeriesSpec
from macro_workbench.storage import MacroStore


def _spec(series_id: str) -> SeriesSpec:
    return SeriesSpec(
        id=series_id,
        name=series_id,
        module="cross_asset",
        source="openbb",
        series_id=f"fred:{series_id}",
        frequency="daily",
        unit="percent",
        transform="change",
        direction="tightening",
        staleness_days=3,
    )


def _obs(series_id: str, day: str, value: float) -> dict[str, object]:
    ts = pd.Timestamp(day)
    return {
        "series_id": series_id,
        "observation_date": ts.date(),
        "value": value,
        "release_time": ts.to_pydatetime(),
        "vintage_date": ts.date(),
        "source": "openbb:fred",
        "last_updated": ts.to_pydatetime(),
    }


def test_resolve_start_dates_incremental_and_full() -> None:
    specs = [_spec("us_10y"), _spec("us_2y")]
    latest = {"us_10y": date(2026, 7, 1)}
    starts = resolve_start_dates(
        specs,
        end="2026-07-17",
        years=5,
        mode="incremental",
        lookback_days=30,
        latest_dates=latest,
    )
    assert starts["us_10y"].date() == date(2026, 6, 1)
    assert starts["us_2y"].date() == date(2021, 7, 17)

    full = resolve_start_dates(
        specs,
        end="2026-07-17",
        years=5,
        mode="full",
        lookback_days=30,
        latest_dates=latest,
    )
    assert full["us_10y"].date() == date(2021, 7, 17)
    assert full["us_2y"].date() == date(2021, 7, 17)


def test_upsert_keeps_history_replace_clears(tmp_path: Path) -> None:
    store = MacroStore(tmp_path / "macro.duckdb")
    try:
        old = pd.DataFrame([_obs("us_10y", "2026-01-01", 4.0)])
        new = pd.DataFrame(
            [
                _obs("us_10y", "2026-01-01", 4.1),
                _obs("us_10y", "2026-07-01", 4.5),
            ]
        )
        store.upsert_observations(old)
        store.upsert_observations(new)
        merged = store.query(
            "SELECT observation_date, value FROM raw_observations WHERE series_id = ? ORDER BY 1",
            ["us_10y"],
        )
        merged["observation_date"] = pd.to_datetime(merged["observation_date"]).dt.date
        assert len(merged) == 2
        assert float(merged.loc[merged.observation_date == date(2026, 1, 1), "value"].iloc[0]) == 4.1
        assert float(merged.loc[merged.observation_date == date(2026, 7, 1), "value"].iloc[0]) == 4.5

        latest = store.latest_observation_dates()
        assert latest["us_10y"] == date(2026, 7, 1)

        store.replace_series_observations(
            pd.DataFrame([_obs("us_10y", "2026-07-15", 4.6)])
        )
        replaced = store.query(
            "SELECT observation_date, value FROM raw_observations WHERE series_id = ?",
            ["us_10y"],
        )
        replaced["observation_date"] = pd.to_datetime(replaced["observation_date"]).dt.date
        assert len(replaced) == 1
        assert replaced.iloc[0].observation_date == date(2026, 7, 15)
    finally:
        store.close()


def test_purge_source_mismatches(tmp_path: Path) -> None:
    store = MacroStore(tmp_path / "macro.duckdb")
    try:
        mixed = pd.DataFrame(
            [
                _obs("sp500", "2026-01-01", 100.0),
                {
                    **_obs("sp500", "2026-01-02", 101.0),
                    "source": "akshare:index_us_stock_sina",
                },
                {
                    **_obs("china_pmi", "2026-06-01", 49.5),
                    "source": "akshare:macro_china_pmi",
                },
            ]
        )
        store.upsert_observations(mixed)
        removed = store.purge_source_mismatches(
            [
                SeriesSpec(
                    id="sp500",
                    name="sp500",
                    module="cross_asset",
                    source="openbb",
                    series_id="yfinance:^GSPC",
                    frequency="daily",
                    unit="index",
                    transform="return",
                    direction="risk_on",
                    staleness_days=2,
                ),
                SeriesSpec(
                    id="china_pmi",
                    name="china_pmi",
                    module="growth",
                    source="akshare",
                    series_id="macro_china_pmi:制造业-指数",
                    frequency="monthly",
                    unit="index",
                    transform="level",
                    direction="growth",
                    staleness_days=65,
                ),
            ]
        )
        assert removed == 1
        left = store.query(
            "SELECT series_id, source FROM raw_observations ORDER BY series_id, source"
        )
        assert set(zip(left.series_id, left.source, strict=True)) == {
            ("sp500", "openbb:fred"),
            ("china_pmi", "akshare:macro_china_pmi"),
        }
    finally:
        store.close()
