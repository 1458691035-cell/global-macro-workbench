from __future__ import annotations

import pandas as pd

from macro_workbench import data_router
from macro_workbench.akshare_source import FetchResult
from macro_workbench.models import SeriesSpec


def _spec(
    series_id: str,
    source: str,
    ref: str,
    *,
    yfinance_symbol: str | None = None,
) -> SeriesSpec:
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
        yfinance_symbol=yfinance_symbol,
    )


def _obs(series_id: str, day: str, value: float, source: str) -> dict:
    stamp = pd.Timestamp(day)
    return {
        "series_id": series_id,
        "observation_date": stamp.date(),
        "value": value,
        "release_time": stamp,
        "vintage_date": stamp.date(),
        "source": source,
        "last_updated": stamp,
    }


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


def test_tsanghi_stale_triggers_single_yfinance_batch(monkeypatch) -> None:
    end = "2026-07-22"
    tsanghi_obs = pd.DataFrame(
        [
            _obs("gold", "2026-07-20", 4000.0, "tsanghi:forex/XAUUSD"),
            _obs("eurusd", "2026-07-20", 1.14, "tsanghi:forex/EURUSD"),
        ]
    )
    yfinance_calls: list[list[str]] = []

    monkeypatch.setattr(
        data_router,
        "fetch_tsanghi_observations",
        lambda specs, end=None, years=5, on_progress=None, start_by_series=None: FetchResult(
            tsanghi_obs, {}
        ),
    )

    def fake_yf(specs):
        yfinance_calls.append([spec.id for spec in specs])
        return FetchResult(
            pd.DataFrame(
                [
                    _obs("gold", "2026-07-21", 4010.0, "yfinance:GC=F"),
                    _obs("eurusd", "2026-07-21", 1.15, "yfinance:EURUSD=X"),
                ]
            ),
            {},
        )

    monkeypatch.setattr(data_router, "fetch_yfinance_latest", fake_yf)

    result = data_router.fetch_all_observations(
        [
            _spec("gold", "tsanghi", "tsanghi:forex/XAUUSD", yfinance_symbol="GC=F"),
            _spec("eurusd", "tsanghi", "tsanghi:forex/EURUSD", yfinance_symbol="EURUSD=X"),
        ],
        end=end,
    )

    assert yfinance_calls == [["gold", "eurusd"]]
    gold_days = set(
        pd.to_datetime(result.observations.loc[result.observations.series_id == "gold", "observation_date"])
        .dt.date
    )
    assert gold_days == {pd.Timestamp("2026-07-20").date(), pd.Timestamp("2026-07-21").date()}
    assert "gold" not in result.errors
    assert "eurusd" not in result.errors
    assert (
        result.observations.loc[
            (result.observations.series_id == "gold")
            & (result.observations.observation_date == pd.Timestamp("2026-07-21").date())
        ]
        .iloc[0]
        .source
        == "yfinance:GC=F"
    )


def test_tsanghi_fresh_skips_yfinance(monkeypatch) -> None:
    end = "2026-07-22"
    tsanghi_obs = pd.DataFrame([_obs("gold", "2026-07-21", 4000.0, "tsanghi:forex/XAUUSD")])
    yfinance_calls: list[list[str]] = []

    monkeypatch.setattr(
        data_router,
        "fetch_tsanghi_observations",
        lambda specs, end=None, years=5, on_progress=None, start_by_series=None: FetchResult(
            tsanghi_obs, {}
        ),
    )
    monkeypatch.setattr(
        data_router,
        "fetch_yfinance_latest",
        lambda specs: yfinance_calls.append([s.id for s in specs]) or FetchResult(pd.DataFrame(), {}),
    )

    result = data_router.fetch_all_observations(
        [_spec("gold", "tsanghi", "tsanghi:forex/XAUUSD", yfinance_symbol="GC=F")],
        end=end,
    )
    assert yfinance_calls == []
    assert set(result.observations.series_id) == {"gold"}
    assert result.errors == {}


def test_tsanghi_stale_without_yfinance_symbol_skips_fallback(monkeypatch) -> None:
    end = "2026-07-22"
    yfinance_calls: list[list[str]] = []

    monkeypatch.setattr(
        data_router,
        "fetch_tsanghi_observations",
        lambda specs, end=None, years=5, on_progress=None, start_by_series=None: FetchResult(
            pd.DataFrame(), {"china_stocks": "tsanghi 返回为空"}
        ),
    )
    monkeypatch.setattr(
        data_router,
        "fetch_yfinance_latest",
        lambda specs: yfinance_calls.append([s.id for s in specs]) or FetchResult(pd.DataFrame(), {}),
    )

    result = data_router.fetch_all_observations(
        [_spec("china_stocks", "tsanghi", "tsanghi:index/CHN/000300")],
        end=end,
    )
    assert yfinance_calls == []
    assert result.errors["china_stocks"] == "tsanghi 返回为空"
