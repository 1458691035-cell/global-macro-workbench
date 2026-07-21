from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx
import pandas as pd
import pytest

from macro_workbench.models import SeriesSpec
from macro_workbench.openbb_source import OpenBBFetcher, fetch_openbb_observations


def _spec(series_id: str, ref: str) -> SeriesSpec:
    return SeriesSpec(
        id=series_id,
        name=series_id,
        module="cross_asset",
        source="openbb",
        series_id=ref,
        frequency="daily",
        unit="index",
        transform="return",
        direction="risk_on",
        staleness_days=3,
    )


class _Result:
    def __init__(self, frame: pd.DataFrame) -> None:
        self._frame = frame

    def to_dataframe(self) -> pd.DataFrame:
        return self._frame


def test_openbb_fred_and_yfinance_normalize() -> None:
    fred_frame = pd.DataFrame(
        {"DGS10": [4.58, 4.55]},
        index=pd.to_datetime(["2026-07-14", "2026-07-15"]),
    )
    equity_frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-07-14", "2026-07-15"]),
            "close": [7500.0, 7572.4],
        }
    ).set_index("date")

    class EquityPrice:
        def historical(self, **kwargs):
            return _Result(equity_frame)

    class IndexPrice:
        def historical(self, **kwargs):
            raise RuntimeError("prefer equity path in this test")

    class CurrencyPrice:
        def historical(self, **kwargs):
            raise RuntimeError("unused")

    fake = SimpleNamespace(
        user=SimpleNamespace(credentials=SimpleNamespace(fred_api_key="test-key")),
        economy=SimpleNamespace(
            fred_series=lambda **kwargs: _Result(fred_frame),
        ),
        equity=SimpleNamespace(price=EquityPrice()),
        index=SimpleNamespace(price=IndexPrice()),
        currency=SimpleNamespace(price=CurrencyPrice()),
    )
    fetcher = OpenBBFetcher(fake, api_key="test-key")
    result = fetcher.fetch(
        [_spec("us_10y", "fred:DGS10"), _spec("sp500", "yfinance:^GSPC")],
        end="2026-07-16",
        years=1,
    )
    assert set(result.observations.series_id) == {"us_10y", "sp500"}
    assert result.observations.source.str.startswith("openbb:").all()
    assert (
        result.observations.loc[result.observations.series_id == "us_10y", "value"].iloc[-1]
        == 4.55
    )
    assert (
        result.observations.loc[result.observations.series_id == "sp500", "value"].iloc[-1]
        == 7572.4
    )


def test_direct_fred_http_without_openbb_import(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {
        "observations": [
            {"date": "2026-07-14", "value": "4.58"},
            {"date": "2026-07-15", "value": "4.55"},
            {"date": "2026-07-16", "value": "."},
        ]
    }

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return payload

    client = MagicMock(spec=httpx.Client)
    client.get.return_value = FakeResponse()

    fetcher = OpenBBFetcher(api_key="demo-key", client=client)
    result = fetcher.fetch([_spec("us_10y", "fred:DGS10")], end="2026-07-16", years=1)
    assert list(result.observations.series_id) == ["us_10y", "us_10y"]
    assert float(result.observations.value.iloc[-1]) == 4.55
    assert result.observations.source.iloc[0] == "openbb:fred"
    client.get.assert_called()
    args, kwargs = client.get.call_args
    assert "fred/series/observations" in args[0]
    assert kwargs["params"]["series_id"] == "DGS10"
    assert kwargs["params"]["api_key"] == "demo-key"


def test_fetch_openbb_observations_sets_auto_build_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENBB_AUTO_BUILD", raising=False)
    captured: dict[str, str] = {}

    def fake_fetch(self, *args, **kwargs):
        captured["OPENBB_AUTO_BUILD"] = __import__("os").environ.get("OPENBB_AUTO_BUILD")
        return __import__("macro_workbench.akshare_source", fromlist=["FetchResult"]).FetchResult(
            __import__("pandas").DataFrame(
                columns=[
                    "series_id",
                    "observation_date",
                    "value",
                    "release_time",
                    "vintage_date",
                    "source",
                    "last_updated",
                ]
            ),
            {},
        )

    monkeypatch.setattr(OpenBBFetcher, "fetch", fake_fetch)
    monkeypatch.setenv("FRED_API_KEY", "x")
    fetch_openbb_observations([_spec("us_10y", "fred:DGS10")], end="2026-07-16")
    assert captured["OPENBB_AUTO_BUILD"] == "0"


def test_openbb_requires_fred_key(monkeypatch: pytest.MonkeyPatch) -> None:
    from macro_workbench import openbb_source as module

    monkeypatch.setattr(module, "_load_env", lambda: None)
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    monkeypatch.setenv("FRED_API_KEY", "")

    class Dummy:
        user = SimpleNamespace(credentials=SimpleNamespace(fred_api_key=None))

    with pytest.raises(RuntimeError, match="FRED_API_KEY"):
        module._configure_credentials(Dummy())
