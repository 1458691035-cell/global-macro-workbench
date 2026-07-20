from __future__ import annotations

import pandas as pd

from macro_workbench import data_router
from macro_workbench.akshare_source import FetchResult
from macro_workbench.models import SeriesSpec


def _spec(series_id: str, source: str, ref: str) -> SeriesSpec:
    return SeriesSpec(
        id=series_id,
        name=series_id,
        module="cross_asset" if source != "akshare" else "growth",
        source=source,
        series_id=ref,
        frequency="daily",
        unit="index",
        transform="return",
        direction="risk_on",
        staleness_days=3,
    )


def test_data_router_merges_sources_without_openbb_akshare_fallback(monkeypatch) -> None:
    openbb_obs = pd.DataFrame(
        [
            {
                "series_id": "us_10y",
                "observation_date": pd.Timestamp("2026-07-15").date(),
                "value": 4.55,
                "release_time": pd.Timestamp("2026-07-15"),
                "vintage_date": pd.Timestamp("2026-07-15").date(),
                "source": "openbb:fred",
                "last_updated": pd.Timestamp("2026-07-17"),
            }
        ]
    )
    ak_obs = pd.DataFrame(
        [
            {
                "series_id": "china_pmi",
                "observation_date": pd.Timestamp("2026-06-01").date(),
                "value": 49.5,
                "release_time": pd.Timestamp("2026-06-01"),
                "vintage_date": pd.Timestamp("2026-06-01").date(),
                "source": "akshare:macro_china_pmi",
                "last_updated": pd.Timestamp("2026-07-17"),
            }
        ]
    )
    akshare_calls: list[list[str]] = []

    monkeypatch.setattr(
        data_router,
        "fetch_openbb_observations",
        lambda specs, end=None, years=5, on_progress=None, start_by_series=None: FetchResult(
            openbb_obs, {"vix": "rate limited", "sp500": "empty"}
        ),
    )

    def fake_akshare(specs, end=None, years=5, on_progress=None, start_by_series=None):
        akshare_calls.append([spec.id for spec in specs])
        return FetchResult(ak_obs, {})

    monkeypatch.setattr(data_router, "fetch_akshare_observations", fake_akshare)

    result = data_router.fetch_all_observations(
        [
            _spec("us_10y", "openbb", "fred:DGS10"),
            _spec("vix", "openbb", "yfinance:^VIX"),
            _spec("sp500", "openbb", "yfinance:^GSPC"),
            _spec("china_pmi", "akshare", "macro_china_pmi:制造业-指数"),
        ]
    )
    assert set(result.observations.series_id) == {"us_10y", "china_pmi"}
    assert result.errors["vix"] == "rate limited"
    assert result.errors["sp500"] == "empty"
    assert akshare_calls == [["china_pmi"]]
