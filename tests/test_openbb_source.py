from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from macro_workbench.models import SeriesSpec
from macro_workbench.openbb_source import OpenBBFetcher


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
        user=SimpleNamespace(credentials=SimpleNamespace(fred_api_key=None)),
        economy=SimpleNamespace(
            fred_series=lambda **kwargs: _Result(fred_frame),
        ),
        equity=SimpleNamespace(price=EquityPrice()),
        index=SimpleNamespace(price=IndexPrice()),
        currency=SimpleNamespace(price=CurrencyPrice()),
    )
    fetcher = OpenBBFetcher(fake)
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


def test_openbb_requires_fred_key(monkeypatch: pytest.MonkeyPatch) -> None:
    from macro_workbench import openbb_source as module

    monkeypatch.setattr(module, "_load_env", lambda: None)
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    monkeypatch.setenv("FRED_API_KEY", "")

    class Dummy:
        user = SimpleNamespace(credentials=SimpleNamespace(fred_api_key=None))

    with pytest.raises(RuntimeError, match="FRED_API_KEY"):
        module._configure_credentials(Dummy())
