from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import date, datetime
from pathlib import Path
from typing import Any

import os
import pandas as pd
import requests
from dotenv import load_dotenv

from .akshare_source import FetchResult
from .models import SeriesSpec

ProgressCallback = Callable[[int, int, str], None]

ROOT = Path(__file__).resolve().parents[1]
BASE_URL = "https://tsanghi.com/api/fin"

COLUMNS = [
    "series_id",
    "observation_date",
    "value",
    "release_time",
    "vintage_date",
    "source",
    "last_updated",
]


def _get_token() -> str:
    load_dotenv(ROOT / ".env", override=False)
    token = os.getenv("TSANGHI_API_KEY", "").strip()
    if not token:
        raise RuntimeError("缺少 TSANGHI_API_KEY：请在项目根目录 .env 中配置")
    return token


def _parse_tsanghi_ref(series_id: str) -> tuple[str, str, str | None]:
    """Parse series_id like 'tsanghi:index/USA/GSPC' or 'tsanghi:forex/EURUSD'.

    Returns (category, ticker, country_code).
    - index: category='index', country_code='USA', ticker='GSPC'
    - forex: category='forex', country_code=None, ticker='EURUSD'
    """
    if ":" not in series_id:
        raise ValueError(f"tsanghi series_id 需为 tsanghi:category/...，收到: {series_id}")
    _, path = series_id.split(":", 1)
    parts = path.split("/")
    if parts[0] == "index" and len(parts) == 3:
        return "index", parts[2], parts[1]
    if parts[0] == "forex" and len(parts) == 2:
        return "forex", parts[1], None
    raise ValueError(f"无法解析 tsanghi series_id: {series_id}")


class TsanghiFetcher:
    def __init__(self, token: str | None = None) -> None:
        self.token = token or _get_token()
        self._session = requests.Session()
        self._session.trust_env = False

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

        for spec in specs:
            start_date = pd.Timestamp(
                (start_by_series or {}).get(spec.id, default_start)
            ).normalize()
            if on_progress is not None:
                on_progress(
                    done, total,
                    f"正在获取：{spec.name}（{spec.series_id}，自 {start_date.date()}）",
                )
            try:
                frame = self._fetch_one(spec, start_date, end_date, fetched_at)
                if frame.empty:
                    errors[spec.id] = "tsanghi 返回为空"
                else:
                    rows.append(frame)
            except Exception as exc:
                errors[spec.id] = f"{type(exc).__name__}: {exc}"
            done += 1

        observations = (
            pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=COLUMNS)
        )
        return FetchResult(observations[COLUMNS], errors)

    def _fetch_one(
        self,
        spec: SeriesSpec,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        fetched_at: datetime,
    ) -> pd.DataFrame:
        category, ticker, country_code = _parse_tsanghi_ref(spec.series_id)
        url = self._build_url(category, ticker, country_code, start_date, end_date)
        resp = self._session.get(url, timeout=30)
        resp.raise_for_status()
        payload = resp.json()
        if payload.get("code") != 200:
            raise RuntimeError(f"tsanghi API 错误: {payload.get('msg', 'unknown')}")
        data = payload.get("data")
        if not data:
            return pd.DataFrame(columns=COLUMNS)
        frame = pd.DataFrame(data)
        return self._normalize(frame, spec, start_date, end_date, fetched_at)

    def _build_url(
        self,
        category: str,
        ticker: str,
        country_code: str | None,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
    ) -> str:
        if category == "index":
            base = f"{BASE_URL}/index/{country_code}/daily"
        else:
            base = f"{BASE_URL}/forex/daily"
        params = (
            f"?token={self.token}&ticker={ticker}"
            f"&start_date={start_date.date().isoformat()}"
            f"&end_date={end_date.date().isoformat()}"
            f"&order=1"
        )
        return base + params

    @staticmethod
    def _normalize(
        raw: pd.DataFrame,
        spec: SeriesSpec,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        fetched_at: datetime,
    ) -> pd.DataFrame:
        frame = raw.copy()
        frame["observation_date"] = pd.to_datetime(frame["date"], errors="coerce")
        frame["value"] = pd.to_numeric(frame["close"], errors="coerce")
        frame = frame.dropna(subset=["observation_date", "value"])
        frame = frame[
            frame["observation_date"].between(start_date, end_date, inclusive="both")
        ]
        frame = frame.sort_values("observation_date").drop_duplicates(
            "observation_date", keep="last"
        )
        frame["series_id"] = spec.id
        frame["release_time"] = frame["observation_date"]
        frame["vintage_date"] = frame["observation_date"].dt.date
        frame["source"] = f"tsanghi:{spec.series_id.split(':', 1)[1]}"
        frame["last_updated"] = fetched_at
        return frame[COLUMNS]


def fetch_tsanghi_observations(
    specs: list[SeriesSpec],
    end: str | date | None = None,
    years: int = 5,
    on_progress: ProgressCallback | None = None,
    start_by_series: Mapping[str, pd.Timestamp] | None = None,
) -> FetchResult:
    if on_progress is not None:
        on_progress(0, max(len(specs), 1), "正在初始化 tsanghi 数据源…")
    return TsanghiFetcher().fetch(
        specs,
        end=end,
        years=years,
        on_progress=on_progress,
        start_by_series=start_by_series,
    )


def fetch_tsanghi_realtime(specs: list[SeriesSpec]) -> pd.DataFrame:
    """Fetch intraday realtime quotes via /daily/realtime endpoint."""
    fetcher = TsanghiFetcher()
    rows: list[dict[str, Any]] = []
    for spec in specs:
        try:
            category, ticker, country_code = _parse_tsanghi_ref(spec.series_id)
            if category == "index":
                base = f"{BASE_URL}/index/{country_code}/daily/realtime"
            else:
                base = f"{BASE_URL}/forex/daily/realtime"
            url = f"{base}?token={fetcher.token}&ticker={ticker}"
            resp = fetcher._session.get(url, timeout=15)
            resp.raise_for_status()
            payload = resp.json()
            data = payload.get("data") or []
            if not data:
                continue
            row = data[0]
            rows.append({
                "series_id": spec.id,
                "name": spec.name,
                "date": row.get("date", ""),
                "open": row.get("open"),
                "high": row.get("high"),
                "low": row.get("low"),
                "close": row.get("close"),
            })
        except Exception:
            continue
    if not rows:
        return pd.DataFrame(columns=["series_id", "name", "date", "open", "high", "low", "close"])
    return pd.DataFrame(rows)
