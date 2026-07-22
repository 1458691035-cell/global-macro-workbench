from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from macro_workbench.akshare_source import AkshareFetcher
from macro_workbench.models import SeriesSpec


def _spec(series_id: str) -> SeriesSpec:
    return SeriesSpec(
        id=series_id,
        name=series_id,
        module="growth",
        source="akshare",
        series_id="x",
        frequency="monthly",
        unit="index",
        transform="level",
        direction="growth",
        staleness_days=65,
    )


def test_akshare_fetcher_allowlist_and_rejects_unmapped_ids() -> None:
    fake = SimpleNamespace(
        stock_zh_index_daily=lambda symbol="sh000300": pd.DataFrame(
            {
                "date": ["2026-07-14", "2026-07-15"],
                "close": [4796.5, 4786.8],
            }
        ),
        macro_china_pmi=lambda: pd.DataFrame(
            {
                "月份": ["2026年05月份", "2026年06月份"],
                "制造业-指数": [49.5, 49.7],
            }
        ),
        macro_china_cpi=lambda: pd.DataFrame(
            {
                "月份": ["2026年05月份", "2026年06月份"],
                "全国-同比增长": [1.2, 1.0],
            }
        ),
        futures_foreign_hist=lambda symbol="OIL": pd.DataFrame(
            {
                "date": ["2026-07-14", "2026-07-15"],
                "close": [90.1, 91.2] if symbol == "OIL" else [64000.0, 65000.0],
            }
        ),
    )
    fetcher = AkshareFetcher(fake)
    result = fetcher.fetch(
        [
            _spec("china_stocks"),
            _spec("china_pmi"),
            _spec("china_cpi"),
            _spec("oil"),
            _spec("bitcoin"),
            _spec("us_10y"),
            _spec("sp500"),
        ],
        end="2026-07-16",
        years=1,
    )

    assert set(result.observations.series_id) == {
        "china_stocks",
        "china_pmi",
        "china_cpi",
        "oil",
        "bitcoin",
    }
    assert result.errors["us_10y"] == "AKShare 暂无可验证的一一对应接口"
    assert result.errors["sp500"] == "AKShare 暂无可验证的一一对应接口"
    assert result.observations.source.str.startswith("akshare:").all()
    assert (
        result.observations.loc[result.observations.series_id == "china_stocks", "value"].iloc[-1]
        == 4786.8
    )
    assert (
        result.observations.loc[result.observations.series_id == "china_cpi", "value"].iloc[-1]
        == 1.0
    )
    assert (
        result.observations.loc[result.observations.series_id == "oil", "value"].iloc[-1] == 91.2
    )
    assert (
        result.observations.loc[result.observations.series_id == "bitcoin", "value"].iloc[-1]
        == 65000.0
    )
