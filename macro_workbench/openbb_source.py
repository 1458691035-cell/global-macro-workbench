from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from datetime import date, datetime
from pathlib import Path
from typing import Any

import httpx
import pandas as pd
from dotenv import load_dotenv
import os

from .akshare_source import FetchResult
from .models import SeriesSpec

ProgressCallback = Callable[[int, int, str], None]


ROOT = Path(__file__).resolve().parents[1]
COLUMNS = [
    "series_id",
    "observation_date",
    "value",
    "release_time",
    "vintage_date",
    "source",
    "last_updated",
]
FRED_OBS_URL = "https://api.stlouisfed.org/fred/series/observations"

_PROXY_ENV_KEYS = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
    "GIT_HTTP_PROXY",
    "GIT_HTTPS_PROXY",
)


@contextmanager
def _without_proxy_env() -> Iterator[None]:
    """Avoid Cursor/sandbox local proxies that return 403 for FRED/Yahoo."""
    saved = {key: os.environ.pop(key) for key in _PROXY_ENV_KEYS if key in os.environ}
    try:
        yield
    finally:
        os.environ.update(saved)


def _load_env() -> None:
    load_dotenv(ROOT / ".env", override=False)


def _fred_api_key() -> str:
    """Resolve FRED key from env or Streamlit secrets."""
    _load_env()
    key = os.getenv("FRED_API_KEY", "").strip()
    if key:
        return key
    try:
        import streamlit as st

        secrets = getattr(st, "secrets", None)
        if secrets is not None and "FRED_API_KEY" in secrets:
            return str(secrets["FRED_API_KEY"]).strip()
    except Exception:
        pass
    raise RuntimeError(
        "缺少 FRED_API_KEY：请在 .env、环境变量或 Streamlit Secrets 中配置"
    )


def _configure_credentials(obb: Any) -> None:
    """Legacy helper kept for tests; sets OpenBB credential from env/secrets."""
    key = _fred_api_key()
    obb.user.credentials.fred_api_key = key


def _parse_ref(series_id: str) -> tuple[str, str]:
    if ":" not in series_id:
        raise ValueError(f"OpenBB series_id 需为 provider:symbol，收到: {series_id}")
    provider, symbol = series_id.split(":", 1)
    return provider.lower(), symbol


class OpenBBFetcher:
    """Fetch FRED (and optional yfinance) series without importing ``openbb``.

    Streamlit Community Cloud cannot write OpenBB's ``.build.lock`` under
    site-packages; calling the FRED REST API directly avoids that import path.
    """

    def __init__(
        self,
        obb_module: Any | None = None,
        *,
        api_key: str | None = None,
        client: httpx.Client | None = None,
    ) -> None:
        self.obb = obb_module
        if obb_module is not None:
            existing = getattr(obb_module.user.credentials, "fred_api_key", None)
            if not (existing or api_key):
                _configure_credentials(obb_module)
            elif api_key and not existing:
                obb_module.user.credentials.fred_api_key = api_key
            self.api_key = (
                getattr(obb_module.user.credentials, "fred_api_key", None) or api_key
            )
        else:
            self.api_key = api_key if api_key is not None else _fred_api_key()
        self._client = client
        self._owns_client = client is None

    def _http(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(timeout=30.0)
        return self._client

    def close(self) -> None:
        if self._owns_client and self._client is not None:
            self._client.close()
            self._client = None

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
        starts = {
            spec.id: pd.Timestamp(
                (start_by_series or {}).get(spec.id, default_start)
            ).normalize()
            for spec in specs
        }
        fetched_at = datetime.now().replace(microsecond=0)
        rows: list[pd.DataFrame] = []
        errors: dict[str, str] = {}
        total = len(specs)
        done = 0

        def report(message: str) -> None:
            if on_progress is not None:
                on_progress(done, total, message)

        def start_for(spec: SeriesSpec) -> pd.Timestamp:
            return starts[spec.id]

        fred_specs = [s for s in specs if _safe_provider(s) == "fred"]
        market_specs = [s for s in specs if _safe_provider(s) == "yfinance"]
        other = [s for s in specs if _safe_provider(s) not in {"fred", "yfinance"}]

        try:
            for spec in other:
                report(f"跳过不支持的 provider：{spec.name}")
                errors[spec.id] = f"不支持的 OpenBB provider: {spec.series_id}"
                done += 1

            for spec in fred_specs:
                report(
                    f"正在获取：{spec.name}（{spec.series_id}，自 {start_for(spec).date()}）"
                )
                try:
                    frame = self._fetch_fred_one(
                        spec, start_for(spec), end_date, fetched_at
                    )
                    if frame.empty:
                        errors[spec.id] = "FRED 返回为空"
                    else:
                        rows.append(frame)
                except Exception as inner:
                    errors[spec.id] = f"{type(inner).__name__}: {inner}"
                done += 1

            for spec in market_specs:
                report(
                    f"正在获取：{spec.name}（{spec.series_id}，自 {start_for(spec).date()}）"
                )
                try:
                    frame = self._fetch_yfinance(
                        spec, start_for(spec), end_date, fetched_at
                    )
                    if frame.empty:
                        errors[spec.id] = "yfinance 返回为空"
                    else:
                        rows.append(frame)
                except Exception as exc:
                    errors[spec.id] = f"{type(exc).__name__}: {exc}"
                done += 1
        finally:
            self.close()

        observations = (
            pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=COLUMNS)
        )
        return FetchResult(observations[COLUMNS], errors)

    def _fetch_fred_batch(
        self,
        specs: list[SeriesSpec],
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        fetched_at: datetime,
    ) -> list[pd.DataFrame]:
        """Fetch each FRED symbol; kept for tests/callers that expect a batch API."""
        frames: list[pd.DataFrame] = []
        for spec in specs:
            frame = self._fetch_fred_one(spec, start_date, end_date, fetched_at)
            if not frame.empty:
                frames.append(frame)
        return frames

    def _fetch_fred_one(
        self,
        spec: SeriesSpec,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        fetched_at: datetime,
    ) -> pd.DataFrame:
        if self.obb is not None:
            return self._fetch_fred_via_obb(spec, start_date, end_date, fetched_at)
        _, symbol = _parse_ref(spec.series_id)
        series = self._fred_observations(symbol, start_date, end_date)
        return _normalize_series(
            series,
            spec,
            source="openbb:fred",
            fetched_at=fetched_at,
            start_date=start_date,
            end_date=end_date,
        )

    def _fetch_fred_via_obb(
        self,
        spec: SeriesSpec,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        fetched_at: datetime,
    ) -> pd.DataFrame:
        _, symbol = _parse_ref(spec.series_id)
        result = self.obb.economy.fred_series(
            symbol=symbol,
            start_date=start_date.date().isoformat(),
            end_date=end_date.date().isoformat(),
            provider="fred",
        )
        wide = result.to_dataframe()
        if wide.empty:
            return pd.DataFrame(columns=COLUMNS)
        column = symbol if symbol in wide.columns else wide.columns[0]
        return _normalize_series(
            wide[column],
            spec,
            source="openbb:fred",
            fetched_at=fetched_at,
            start_date=start_date,
            end_date=end_date,
        )

    def _fred_observations(
        self, symbol: str, start_date: pd.Timestamp, end_date: pd.Timestamp
    ) -> pd.Series:
        if not self.api_key:
            raise RuntimeError("缺少 FRED_API_KEY")
        params = {
            "series_id": symbol,
            "api_key": self.api_key,
            "file_type": "json",
            "observation_start": start_date.date().isoformat(),
            "observation_end": end_date.date().isoformat(),
        }
        response = self._http().get(FRED_OBS_URL, params=params)
        response.raise_for_status()
        payload = response.json()
        if "error_code" in payload:
            raise RuntimeError(payload.get("error_message") or str(payload["error_code"]))
        observations = payload.get("observations") or []
        if not observations:
            return pd.Series(dtype=float)
        frame = pd.DataFrame(observations)
        frame = frame.replace(".", pd.NA)
        frame["date"] = pd.to_datetime(frame["date"])
        frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
        frame = frame.dropna(subset=["value"]).set_index("date")["value"].astype(float)
        frame.index = pd.to_datetime(frame.index).tz_localize(None)
        return frame

    def _fetch_yfinance(
        self,
        spec: SeriesSpec,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        fetched_at: datetime,
    ) -> pd.DataFrame:
        _, symbol = _parse_ref(spec.series_id)
        close = self._yfinance_close(symbol, start_date, end_date)
        return _normalize_series(
            close,
            spec,
            source="openbb:yfinance",
            fetched_at=fetched_at,
            start_date=start_date,
            end_date=end_date,
        )

    def _yfinance_close(
        self, symbol: str, start_date: pd.Timestamp, end_date: pd.Timestamp
    ) -> pd.Series:
        start = start_date.date().isoformat()
        end = end_date.date().isoformat()
        if self.obb is not None:
            for caller in (
                lambda: self.obb.index.price.historical(
                    symbol=symbol, start_date=start, end_date=end, provider="yfinance"
                ),
                lambda: self.obb.equity.price.historical(
                    symbol=symbol, start_date=start, end_date=end, provider="yfinance"
                ),
                lambda: self.obb.currency.price.historical(
                    symbol=symbol.replace("=X", ""),
                    start_date=start,
                    end_date=end,
                    provider="yfinance",
                ),
            ):
                try:
                    frame = caller().to_dataframe()
                    if frame is not None and not frame.empty:
                        series = _extract_close(frame)
                        if not series.empty:
                            return series
                except Exception:
                    continue

        import yfinance as yf

        hist = yf.Ticker(symbol).history(
            start=start, end=(end_date + pd.Timedelta(days=1)).date().isoformat()
        )
        if hist.empty:
            return pd.Series(dtype=float)
        close = hist["Close"] if "Close" in hist.columns else hist.iloc[:, 0]
        close.index = pd.to_datetime(close.index).tz_localize(None)
        return close.astype(float)


def _safe_provider(spec: SeriesSpec) -> str | None:
    try:
        return _parse_ref(spec.series_id)[0]
    except ValueError:
        return None


def _extract_close(frame: pd.DataFrame) -> pd.Series:
    work = frame.copy()
    if not isinstance(work.index, pd.DatetimeIndex):
        for candidate in ("date", "Date", "datetime"):
            if candidate in work.columns:
                work = work.set_index(candidate)
                break
    work.index = pd.to_datetime(work.index).tz_localize(None)
    for candidate in ("close", "Close", "adj_close", "Adj Close"):
        if candidate in work.columns:
            return work[candidate].astype(float)
    numeric = work.select_dtypes("number")
    if numeric.empty:
        return pd.Series(dtype=float)
    return numeric.iloc[:, 0].astype(float)


def _normalize_series(
    values: pd.Series,
    spec: SeriesSpec,
    source: str,
    fetched_at: datetime,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.DataFrame:
    series = values.copy()
    series.index = pd.to_datetime(series.index).tz_localize(None)
    series = pd.to_numeric(series, errors="coerce").dropna()
    series = series[(series.index >= start_date) & (series.index <= end_date)]
    if series.empty:
        return pd.DataFrame(columns=COLUMNS)
    frame = pd.DataFrame(
        {
            "series_id": spec.id,
            "observation_date": series.index.date,
            "value": series.to_numpy(dtype=float),
            "release_time": series.index.to_pydatetime(),
            "vintage_date": series.index.date,
            "source": source,
            "last_updated": fetched_at,
        }
    )
    return frame.drop_duplicates("observation_date", keep="last")


def fetch_openbb_observations(
    specs: list[SeriesSpec],
    end: str | date | None = None,
    years: int = 5,
    on_progress: ProgressCallback | None = None,
    start_by_series: Mapping[str, pd.Timestamp] | None = None,
) -> FetchResult:
    if on_progress is not None:
        on_progress(0, max(len(specs), 1), "正在初始化 FRED / 读取 API 密钥…")
    # Avoid OpenBB package builder writing .build.lock into read-only site-packages.
    os.environ.setdefault("OPENBB_AUTO_BUILD", "0")
    with _without_proxy_env():
        return OpenBBFetcher().fetch(
            specs,
            end=end,
            years=years,
            on_progress=on_progress,
            start_by_series=start_by_series,
        )
