from __future__ import annotations

from datetime import date, datetime
from time import sleep
from typing import Iterable

import pandas as pd

from .akshare_source import FetchResult
from .models import SeriesSpec

COLUMNS = [
    "series_id",
    "observation_date",
    "value",
    "release_time",
    "vintage_date",
    "source",
    "last_updated",
]


def fetch_yfinance_latest(specs: list[SeriesSpec]) -> FetchResult:
    """Fetch only the latest daily close for each mapped symbol.

    Uses one batched ``yf.download(..., period="5d")`` call, then falls back to
    per-ticker ``history(period="5d")`` for any symbols still missing.
    """
    targets = [(spec, spec.yfinance_symbol) for spec in specs if spec.yfinance_symbol]
    if not targets:
        return FetchResult(pd.DataFrame(columns=COLUMNS), {})

    fetched_at = datetime.now().replace(microsecond=0)
    symbol_to_specs: dict[str, list[SeriesSpec]] = {}
    for spec, symbol in targets:
        symbol_to_specs.setdefault(symbol, []).append(spec)

    closes = _download_latest_closes(list(symbol_to_specs))
    missing = [symbol for symbol in symbol_to_specs if symbol not in closes]
    if missing:
        sleep(1)
        closes.update(_history_latest_closes(missing))

    rows: list[dict] = []
    errors: dict[str, str] = {}
    for symbol, mapped in symbol_to_specs.items():
        point = closes.get(symbol)
        if point is None:
            for spec in mapped:
                errors[spec.id] = f"yfinance 最新点为空（{symbol}）"
            continue
        obs_date, value = point
        for spec in mapped:
            rows.append(
                {
                    "series_id": spec.id,
                    "observation_date": obs_date,
                    "value": value,
                    "release_time": pd.Timestamp(obs_date),
                    "vintage_date": obs_date,
                    "source": f"yfinance:{symbol}",
                    "last_updated": fetched_at,
                }
            )

    frame = pd.DataFrame(rows, columns=COLUMNS) if rows else pd.DataFrame(columns=COLUMNS)
    return FetchResult(frame, errors)


def _download_latest_closes(symbols: list[str]) -> dict[str, tuple[date, float]]:
    if not symbols:
        return {}
    import yfinance as yf

    try:
        raw = yf.download(
            tickers=symbols,
            period="5d",
            group_by="ticker",
            auto_adjust=True,
            threads=False,
            progress=False,
        )
    except Exception:
        return {}
    if raw is None or raw.empty:
        return {}

    closes: dict[str, tuple[date, float]] = {}
    for symbol in symbols:
        series = _close_series_from_download(raw, symbol, single=(len(symbols) == 1))
        point = _latest_close_point(series)
        if point is not None:
            closes[symbol] = point
    return closes


def _history_latest_closes(symbols: Iterable[str]) -> dict[str, tuple[date, float]]:
    import yfinance as yf

    closes: dict[str, tuple[date, float]] = {}
    for symbol in symbols:
        try:
            hist = yf.Ticker(symbol).history(period="5d", auto_adjust=True)
        except Exception:
            continue
        if hist is None or hist.empty:
            continue
        close = hist["Close"] if "Close" in hist.columns else hist.iloc[:, 0]
        close.index = pd.to_datetime(close.index).tz_localize(None)
        point = _latest_close_point(close.astype(float))
        if point is not None:
            closes[symbol] = point
    return closes


def _close_series_from_download(
    raw: pd.DataFrame, symbol: str, *, single: bool
) -> pd.Series:
    if isinstance(raw.columns, pd.MultiIndex):
        if symbol in raw.columns.get_level_values(0):
            frame = raw[symbol]
        elif symbol in raw.columns.get_level_values(1):
            frame = raw.xs(symbol, axis=1, level=1)
        else:
            return pd.Series(dtype=float)
    elif single:
        frame = raw
    else:
        return pd.Series(dtype=float)

    for candidate in ("Close", "close", "Adj Close", "adj_close"):
        if candidate in frame.columns:
            series = frame[candidate]
            break
    else:
        numeric = frame.select_dtypes("number")
        if numeric.empty:
            return pd.Series(dtype=float)
        series = numeric.iloc[:, 0]

    series = pd.to_numeric(series, errors="coerce")
    series.index = pd.to_datetime(series.index).tz_localize(None)
    return series.dropna()


def _latest_close_point(series: pd.Series) -> tuple[date, float] | None:
    if series is None or series.empty:
        return None
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return None
    stamp = pd.Timestamp(clean.index.max()).normalize()
    value = float(clean.loc[clean.index.max()])
    return stamp.date(), value
