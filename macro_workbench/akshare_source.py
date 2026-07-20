from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, datetime
from time import sleep
from typing import Any, Iterator

import pandas as pd
import requests

from .models import SeriesSpec

ProgressCallback = Callable[[int, int, str], None]


@contextmanager
def _without_system_proxy() -> Iterator[None]:
    """Disable requests proxy auto-detection for Eastmoney endpoints."""
    original_init = requests.sessions.Session.__init__

    def patched_init(self, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, **kwargs)
        self.trust_env = False
        self.proxies = {}

    requests.sessions.Session.__init__ = patched_init  # type: ignore[method-assign]
    try:
        yield
    finally:
        requests.sessions.Session.__init__ = original_init  # type: ignore[method-assign]


@dataclass(frozen=True)
class Endpoint:
    function: str
    date_column: str
    value_column: str
    kwargs: dict[str, Any] | None = None
    scale: float = 1.0


@dataclass(frozen=True)
class FetchResult:
    observations: pd.DataFrame
    errors: dict[str, str]


# China-only endpoints. US/global series must go through OpenBB (FRED / yfinance)
# and must never silently fall back here.
ENDPOINTS: dict[str, Endpoint] = {
    "china_stocks": Endpoint(
        "stock_zh_index_daily", "date", "close", {"symbol": "sh000300"}
    ),
    "china_credit": Endpoint("macro_china_new_financial_credit", "月份", "当月"),
    "china_pmi": Endpoint("macro_china_pmi", "月份", "制造业-指数"),
    "global_trade": Endpoint("macro_china_exports_yoy", "日期", "今值"),
}


class AkshareFetcher:
    def __init__(self, akshare_module: Any | None = None) -> None:
        if akshare_module is None:
            import akshare as akshare_module

        self.akshare = akshare_module
        self._cache: dict[tuple[str, tuple[tuple[str, Any], ...]], pd.DataFrame] = {}

    def fetch(
        self,
        specs: list[SeriesSpec],
        end: str | date | None = None,
        years: int = 5,
        on_progress: ProgressCallback | None = None,
        start_by_series: Mapping[str, pd.Timestamp] | None = None,
    ) -> FetchResult:
        end_date = pd.Timestamp(end or date.today()).normalize()
        default_start = end_date - pd.DateOffset(years=years)
        fetched_at = datetime.now().replace(microsecond=0)
        rows: list[pd.DataFrame] = []
        errors: dict[str, str] = {}
        total = len(specs)
        done = 0

        with _without_system_proxy():
            for spec in specs:
                start_date = pd.Timestamp(
                    (start_by_series or {}).get(spec.id, default_start)
                ).normalize()
                endpoint = ENDPOINTS.get(spec.id)
                if on_progress is not None:
                    on_progress(
                        done,
                        total,
                        f"正在获取：{spec.name}（akshare:{endpoint.function if endpoint else 'n/a'}，自 {start_date.date()}）",
                    )
                if endpoint is None:
                    errors[spec.id] = "AKShare 暂无可验证的一一对应接口"
                    done += 1
                    continue
                try:
                    raw = self._call(endpoint, start_date, end_date)
                    normalized = self._normalize(
                        raw, spec, endpoint, start_date, end_date, fetched_at
                    )
                    if normalized.empty:
                        errors[spec.id] = "接口返回数据为空或没有有效数值"
                    else:
                        rows.append(normalized)
                except Exception as exc:  # network and upstream schemas are outside our control
                    errors[spec.id] = f"{type(exc).__name__}: {exc}"
                done += 1

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
            pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=columns)
        )
        return FetchResult(observations[columns], errors)

    def _call(
        self, endpoint: Endpoint, start_date: pd.Timestamp, end_date: pd.Timestamp
    ) -> pd.DataFrame:
        if endpoint.function == "derived_fx_eurusd":
            return self._derived_cross_rate("欧元", "美元", start_date, end_date)
        if endpoint.function == "derived_fx_usdjpy":
            return self._derived_cross_rate("美元", "日元", start_date, end_date)

        kwargs = dict(endpoint.kwargs or {})
        if endpoint.function == "bond_zh_us_rate":
            kwargs["start_date"] = start_date.strftime("%Y%m%d")
        if endpoint.function == "currency_boc_sina":
            kwargs["start_date"] = start_date.strftime("%Y%m%d")
            kwargs["end_date"] = end_date.strftime("%Y%m%d")
        key = (endpoint.function, tuple(sorted(kwargs.items())))
        if key not in self._cache:
            function = getattr(self.akshare, endpoint.function)
            self._cache[key] = self._call_with_retry(function, kwargs)
        return self._cache[key].copy()

    @staticmethod
    def _call_with_retry(function: Any, kwargs: dict[str, Any], attempts: int = 3) -> pd.DataFrame:
        last_error: Exception | None = None
        for attempt in range(attempts):
            try:
                return function(**kwargs)
            except Exception as exc:  # transient upstream/network failures
                last_error = exc
                if attempt + 1 < attempts:
                    sleep(1.5 * (attempt + 1))
        assert last_error is not None
        raise last_error

    def _boc_mid(
        self, symbol: str, start_date: pd.Timestamp, end_date: pd.Timestamp
    ) -> pd.DataFrame:
        key = (
            "currency_boc_sina",
            (
                ("end_date", end_date.strftime("%Y%m%d")),
                ("start_date", start_date.strftime("%Y%m%d")),
                ("symbol", symbol),
            ),
        )
        if key not in self._cache:
            self._cache[key] = self.akshare.currency_boc_sina(
                symbol=symbol,
                start_date=start_date.strftime("%Y%m%d"),
                end_date=end_date.strftime("%Y%m%d"),
            )
        frame = self._cache[key].copy()
        out = frame[["日期", "央行中间价"]].copy()
        out["日期"] = pd.to_datetime(out["日期"], errors="coerce")
        out["央行中间价"] = pd.to_numeric(out["央行中间价"], errors="coerce")
        return out.dropna()

    def _derived_cross_rate(
        self,
        numerator: str,
        denominator: str,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
    ) -> pd.DataFrame:
        left = self._boc_mid(numerator, start_date, end_date).rename(
            columns={"央行中间价": "num"}
        )
        right = self._boc_mid(denominator, start_date, end_date).rename(
            columns={"央行中间价": "den"}
        )
        merged = left.merge(right, on="日期", how="inner")
        merged["value"] = merged["num"] / merged["den"]
        return merged[["日期", "value"]]

    @staticmethod
    def _normalize(
        raw: pd.DataFrame,
        spec: SeriesSpec,
        endpoint: Endpoint,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        fetched_at: datetime,
    ) -> pd.DataFrame:
        missing = {endpoint.date_column, endpoint.value_column} - set(raw.columns)
        if missing:
            raise ValueError(f"返回字段缺失: {', '.join(sorted(missing))}")

        frame = raw[[endpoint.date_column, endpoint.value_column]].copy()
        date_text = (
            frame[endpoint.date_column]
            .astype(str)
            .str.replace("年", "-", regex=False)
            .str.replace("月份", "-01", regex=False)
            .str.replace("月", "-01", regex=False)
        )
        frame["observation_date"] = pd.to_datetime(date_text, errors="coerce")
        frame["value"] = pd.to_numeric(frame[endpoint.value_column], errors="coerce")
        frame["value"] *= endpoint.scale
        frame = frame.dropna(subset=["observation_date", "value"])
        frame = frame[
            frame["observation_date"].between(start_date, end_date, inclusive="both")
        ]
        frame = frame.sort_values("observation_date").drop_duplicates(
            "observation_date", keep="last"
        )
        frame["series_id"] = spec.id
        # AKShare does not expose ALFRED-style vintages; use observation date so
        # historical replay is not blocked by the fetch timestamp.
        frame["release_time"] = frame["observation_date"]
        frame["vintage_date"] = frame["observation_date"].dt.date
        frame["source"] = f"akshare:{endpoint.function}"
        frame["last_updated"] = fetched_at
        return frame


def fetch_akshare_observations(
    specs: list[SeriesSpec],
    end: str | date | None = None,
    years: int = 5,
    on_progress: ProgressCallback | None = None,
    start_by_series: Mapping[str, pd.Timestamp] | None = None,
) -> FetchResult:
    if on_progress is not None:
        on_progress(0, max(len(specs), 1), "正在初始化 AKShare…")
    return AkshareFetcher().fetch(
        specs,
        end=end,
        years=years,
        on_progress=on_progress,
        start_by_series=start_by_series,
    )
