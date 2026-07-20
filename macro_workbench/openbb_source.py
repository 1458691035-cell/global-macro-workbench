from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from datetime import date, datetime
from pathlib import Path
from typing import Any

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


def _configure_credentials(obb: Any) -> None:
    _load_env()
    key = os.getenv("FRED_API_KEY", "").strip()
    if not key:
        raise RuntimeError("缺少 FRED_API_KEY：请在项目根目录 .env 中配置")
    obb.user.credentials.fred_api_key = key


def _parse_ref(series_id: str) -> tuple[str, str]:
    if ":" not in series_id:
        raise ValueError(f"OpenBB series_id 需为 provider:symbol，收到: {series_id}")
    provider, symbol = series_id.split(":", 1)
    return provider.lower(), symbol


class OpenBBFetcher:
    def __init__(self, obb_module: Any | None = None) -> None:
        if obb_module is None:
            from openbb import obb as obb_module

            _configure_credentials(obb_module)
        self.obb = obb_module

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

        for spec in other:
            report(f"跳过不支持的 provider：{spec.name}")
            errors[spec.id] = f"不支持的 OpenBB provider: {spec.series_id}"
            done += 1

        if fred_specs:
            shared_starts = {start_for(spec) for spec in fred_specs}
            # 批量仅在无进度回调且所有序列同一 start 时使用
            use_batch = on_progress is None and len(shared_starts) == 1
            if use_batch:
                batch_start = next(iter(shared_starts))
                preview = "、".join(spec.name for spec in fred_specs[:3])
                suffix = f" 等 {len(fred_specs)} 条" if len(fred_specs) > 3 else ""
                report(f"批量获取 FRED：{preview}{suffix}")
                try:
                    batch_frames = self._fetch_fred_batch(
                        fred_specs, batch_start, end_date, fetched_at
                    )
                    rows.extend(batch_frames)
                    done += len(fred_specs)
                    fetched_ids = {
                        frame.series_id.iloc[0] for frame in rows if not frame.empty
                    }
                    for spec in fred_specs:
                        if spec.id not in fetched_ids and spec.id not in errors:
                            errors[spec.id] = "FRED 返回为空"
                    report(f"FRED 批量完成（{len(fred_specs)} 条）")
                except Exception:
                    use_batch = False

            if not use_batch:
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
        symbols = [_parse_ref(spec.series_id)[1] for spec in specs]
        symbol_to_spec = {_parse_ref(spec.series_id)[1]: spec for spec in specs}
        result = self.obb.economy.fred_series(
            symbol=",".join(symbols),
            start_date=start_date.date().isoformat(),
            end_date=end_date.date().isoformat(),
            provider="fred",
        )
        wide = result.to_dataframe()
        if wide.empty:
            return []
        frames: list[pd.DataFrame] = []
        for column in wide.columns:
            if column not in symbol_to_spec:
                continue
            frames.append(
                _normalize_series(
                    wide[column],
                    symbol_to_spec[column],
                    source="openbb:fred",
                    fetched_at=fetched_at,
                    start_date=start_date,
                    end_date=end_date,
                )
            )
        return [frame for frame in frames if not frame.empty]

    def _fetch_fred_one(
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
        # Prefer OpenBB router; fall back to yfinance package on empty/rate-limit.
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
        on_progress(0, max(len(specs), 1), "正在初始化 OpenBB / 读取 FRED 密钥…")
    with _without_proxy_env():
        return OpenBBFetcher().fetch(
            specs,
            end=end,
            years=years,
            on_progress=on_progress,
            start_by_series=start_by_series,
        )
