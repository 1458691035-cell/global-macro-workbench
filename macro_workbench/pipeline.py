from __future__ import annotations

import hashlib
from datetime import date

import numpy as np
import pandas as pd

from .models import SeriesSpec
from .storage import MacroStore


PERIODS = {
    "daily": (1, 5, 21, 63, "B"),
    "weekly": (1, 1, 4, 13, "W-FRI"),
    "monthly": (1, 1, 3, 12, "ME"),
    "quarterly": (1, 1, 1, 4, "QE"),
}


def generate_demo_observations(
    specs: list[SeriesSpec], end: str | date | None = None, years: int = 5
) -> pd.DataFrame:
    end_date = pd.Timestamp(end or date.today()).normalize()
    start_date = end_date - pd.DateOffset(years=years)
    rows: list[dict[str, object]] = []
    for spec in specs:
        dates = pd.date_range(start_date, end_date, freq=PERIODS[spec.frequency][4])
        rng = np.random.default_rng(int(hashlib.sha256(spec.id.encode()).hexdigest()[:8], 16))
        base = 3.0 if spec.unit in {"percent", "ratio"} else 100.0
        if spec.unit == "zscore":
            base = 0.0
        scale = 0.015 if base <= 3 else 0.008
        values = (
            base + np.cumsum(rng.normal(0, scale, len(dates)))
            if base <= 3
            else base * np.exp(np.cumsum(rng.normal(0, scale, len(dates))))
        )
        lag = {"daily": 0, "weekly": 3, "monthly": 14, "quarterly": 30}[spec.frequency]
        for observed, value in zip(dates, values, strict=True):
            released = observed + pd.Timedelta(days=lag)
            if released <= end_date:
                rows.append(
                    {
                        "series_id": spec.id,
                        "observation_date": observed.date(),
                        "value": float(value),
                        "release_time": released.to_pydatetime(),
                        "vintage_date": released.date(),
                        "source": f"demo:{spec.source}",
                        "last_updated": end_date.to_pydatetime(),
                    }
                )
    return pd.DataFrame(rows)


def _transform(values: pd.Series, spec: SeriesSpec) -> pd.Series:
    one, _, _, annual, _ = PERIODS[spec.frequency]
    if spec.transform == "return":
        return values.pct_change(one, fill_method=None) * 100
    if spec.transform == "yoy":
        return values.pct_change(annual, fill_method=None) * 100
    if spec.transform in {"change", "mom_change"}:
        return values.diff(one)
    if spec.transform == "zscore":
        return (values - values.rolling(60, min_periods=8).mean()) / values.rolling(
            60, min_periods=8
        ).std().replace(0, np.nan)
    return values


def calculate_snapshot(
    store: MacroStore, specs: list[SeriesSpec], as_of: str | date
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    as_of_date = pd.Timestamp(as_of).normalize()
    signals: list[dict[str, object]] = []
    quality: list[dict[str, object]] = []
    for spec in specs:
        history = store.query(
            """
            SELECT observation_date, value, vintage_date, release_time
            FROM raw_observations
            WHERE series_id = ? AND CAST(release_time AS DATE) <= CAST(? AS DATE)
              AND vintage_date <= CAST(? AS DATE)
            QUALIFY vintage_date = MAX(vintage_date) OVER (PARTITION BY observation_date)
            ORDER BY observation_date
            """,
            [spec.id, str(as_of_date.date()), str(as_of_date.date())],
        )
        if history.empty:
            quality.append(_quality_row(spec.id, as_of_date, "missing", -1, "尚未发布或未抓取"))
            continue
        values = history.set_index(pd.to_datetime(history.observation_date)).value.astype(float)
        transformed = _transform(values, spec)
        one, week, month, quarter, _ = PERIODS[spec.frequency]
        mean = transformed.rolling(60, min_periods=8).mean()
        std = transformed.rolling(60, min_periods=8).std().replace(0, np.nan)
        zscore = (transformed - mean) / std
        age = (as_of_date - values.index[-1]).days
        anomaly = bool(abs(zscore.iloc[-1]) > 3) if pd.notna(zscore.iloc[-1]) else False
        status = "stale" if age > spec.staleness_days else ("anomaly" if anomaly else "ok")
        quality.append(
            _quality_row(
                spec.id,
                as_of_date,
                status,
                age,
                "数据过期" if status == "stale" else ("异常跳变" if anomaly else "正常"),
                anomaly,
            )
        )
        with np.errstate(invalid="ignore"):
            returns = values.pct_change(fill_method=None).tail(63)
            volatility = (
                float(returns.std(skipna=True) * np.sqrt(252))
                if returns.notna().any()
                else np.nan
            )
        release_times = pd.to_datetime(history.release_time)
        signals.append(
            {
                "series_id": spec.id,
                "as_of_date": as_of_date.date(),
                "value": values.iloc[-1],
                "transformed_value": transformed.iloc[-1],
                "level": zscore.iloc[-1],
                "momentum": _diff(transformed, quarter),
                "surprise": np.nan,
                "percentile": transformed.rank(pct=True).iloc[-1] * 100,
                "change_1d": _pct(values, one),
                "change_1w": _pct(values, week),
                "change_1m": _pct(values, month),
                "change_3m": _pct(values, quarter),
                "volatility": volatility,
                "trend_zscore": zscore.iloc[-1],
                "vintage_date": history.vintage_date.iloc[-1],
                "release_time": release_times.iloc[-1].to_pydatetime(),
                "prev_release_time": (
                    release_times.iloc[-2].to_pydatetime()
                    if len(release_times) > 1
                    else None
                ),
            }
        )
    signal_frame = pd.DataFrame(signals)
    quality_frame = pd.DataFrame(quality)
    regime = _regime(signal_frame, specs, as_of_date)
    return signal_frame, quality_frame, regime


def _quality_row(
    series_id: str,
    as_of: pd.Timestamp,
    status: str,
    age: int,
    message: str,
    anomaly: bool = False,
) -> dict[str, object]:
    return {
        "series_id": series_id,
        "as_of_date": as_of.date(),
        "status": status,
        "age_days": age,
        "missing_count": int(status == "missing"),
        "anomaly": anomaly,
        "cross_source_gap": np.nan,
        "message": message,
    }


def _pct(values: pd.Series, periods: int) -> float:
    return (
        np.nan
        if len(values) <= periods or values.iloc[-periods - 1] == 0
        else float((values.iloc[-1] / values.iloc[-periods - 1] - 1) * 100)
    )


def _diff(values: pd.Series, periods: int) -> float:
    return (
        np.nan
        if len(values) <= periods
        else float(values.iloc[-1] - values.iloc[-periods - 1])
    )


def _regime(
    signals: pd.DataFrame, specs: list[SeriesSpec], as_of: pd.Timestamp
) -> pd.DataFrame:
    empty = {
        "as_of_date": as_of.date(),
        "growth": 0.0,
        "inflation": 0.0,
        "liquidity": 0.0,
        "risk_appetite": 0.0,
        "regime": "Slowdown",
        "confidence": 0.0,
    }
    if signals.empty or "series_id" not in signals.columns:
        return pd.DataFrame([empty])
    metadata = {spec.id: spec for spec in specs}
    work = signals.copy()
    work["module"] = work.series_id.map(lambda value: metadata[value].module)
    work["direction"] = work.series_id.map(lambda value: metadata[value].direction)
    growth = float(np.nan_to_num(work.loc[work.module == "growth", "momentum"].median()))
    inflation = float(np.nan_to_num(work.loc[work.module == "inflation", "momentum"].median()))
    policy = work[work.module == "policy"].copy()
    if policy.empty:
        liquidity = 0.0
    else:
        policy["signed"] = policy.trend_zscore * policy.direction.map(
            {"easing": 1, "tightening": -1}
        ).fillna(0)
        liquidity = float(np.nan_to_num(policy.signed.median()))
    risk = work[work.direction.isin(["risk_on", "risk_off"])].copy()
    if risk.empty:
        risk_appetite = 0.0
    else:
        risk["signed"] = risk.trend_zscore * risk.direction.map(
            {"risk_on": 1, "risk_off": -1}
        )
        risk_appetite = float(np.nan_to_num(risk.signed.median()))
    label = (
        "Goldilocks"
        if growth >= 0 and inflation < 0
        else "Reflation"
        if growth >= 0
        else "Stagflation"
        if inflation >= 0
        else "Slowdown"
    )
    confidence = float(work[["level", "momentum"]].notna().all(axis=1).mean())
    return pd.DataFrame(
        [
            {
                "as_of_date": as_of.date(),
                "growth": growth,
                "inflation": inflation,
                "liquidity": liquidity,
                "risk_appetite": risk_appetite,
                "regime": label,
                "confidence": confidence,
            }
        ]
    )


def run_pipeline(
    store: MacroStore,
    specs: list[SeriesSpec],
    as_of: str | date,
    observations: pd.DataFrame | None = None,
    *,
    replace: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    store.register_catalog(specs)
    if observations is not None and not observations.empty:
        if replace:
            store.replace_series_observations(observations)
        else:
            store.upsert_observations(observations)
    signals, quality, regime = calculate_snapshot(store, specs, as_of)
    snapshot_date = str(pd.Timestamp(as_of).date())
    store.connection.execute(
        "DELETE FROM derived_signals WHERE as_of_date = CAST(? AS DATE)", [snapshot_date]
    )
    store.connection.execute(
        "DELETE FROM quality_status WHERE as_of_date = CAST(? AS DATE)", [snapshot_date]
    )
    store.upsert_frame("derived_signals", signals)
    store.upsert_frame("quality_status", quality)
    store.upsert_frame("regime_snapshots", regime)
    return signals, quality, regime
