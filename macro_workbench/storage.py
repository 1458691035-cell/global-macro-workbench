from __future__ import annotations

from datetime import date
from pathlib import Path

import duckdb
import pandas as pd

from .models import SeriesSpec


SCHEMA = """
CREATE TABLE IF NOT EXISTS series_catalog (
    id VARCHAR PRIMARY KEY, name VARCHAR, module VARCHAR, source VARCHAR,
    source_series_id VARCHAR, asset_proxy VARCHAR, frequency VARCHAR, unit VARCHAR,
    transform VARCHAR, direction VARCHAR, staleness_days INTEGER
);
CREATE TABLE IF NOT EXISTS raw_observations (
    series_id VARCHAR, observation_date DATE, value DOUBLE, release_time TIMESTAMP,
    vintage_date DATE, source VARCHAR, last_updated TIMESTAMP,
    PRIMARY KEY (series_id, observation_date, vintage_date)
);
CREATE TABLE IF NOT EXISTS derived_signals (
    series_id VARCHAR, as_of_date DATE, value DOUBLE, transformed_value DOUBLE,
    level DOUBLE, momentum DOUBLE, surprise DOUBLE, percentile DOUBLE,
    change_1d DOUBLE, change_1w DOUBLE, change_1m DOUBLE, change_3m DOUBLE,
    volatility DOUBLE, trend_zscore DOUBLE, vintage_date DATE,
    release_time TIMESTAMP, prev_release_time TIMESTAMP,
    PRIMARY KEY (series_id, as_of_date, vintage_date)
);
CREATE TABLE IF NOT EXISTS quality_status (
    series_id VARCHAR, as_of_date DATE, status VARCHAR, age_days INTEGER,
    missing_count INTEGER, anomaly BOOLEAN, cross_source_gap DOUBLE, message VARCHAR,
    PRIMARY KEY (series_id, as_of_date)
);
CREATE TABLE IF NOT EXISTS regime_snapshots (
    as_of_date DATE PRIMARY KEY, growth DOUBLE, inflation DOUBLE, liquidity DOUBLE,
    risk_appetite DOUBLE, regime VARCHAR, confidence DOUBLE
);
CREATE TABLE IF NOT EXISTS events (
    event_time TIMESTAMP, region VARCHAR, event VARCHAR, importance INTEGER,
    consensus VARCHAR, trigger_up VARCHAR, trigger_down VARCHAR
);
CREATE TABLE IF NOT EXISTS memo_drafts (
    as_of_date DATE PRIMARY KEY, generated_at TIMESTAMP, content VARCHAR
);
"""


class MacroStore:
    def __init__(self, path: str | Path = "data/macro.duckdb") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = duckdb.connect(str(self.path))
        self.connection.execute(SCHEMA)
        self._migrate()

    def _migrate(self) -> None:
        existing = {
            row[0]
            for row in self.connection.execute(
                """
                SELECT column_name
                FROM information_schema.columns
                WHERE table_name = 'derived_signals'
                """
            ).fetchall()
        }
        for column, ddl in (
            ("release_time", "ALTER TABLE derived_signals ADD COLUMN release_time TIMESTAMP"),
            (
                "prev_release_time",
                "ALTER TABLE derived_signals ADD COLUMN prev_release_time TIMESTAMP",
            ),
        ):
            if column not in existing:
                self.connection.execute(ddl)

    def close(self) -> None:
        self.connection.close()

    def register_catalog(self, specs: list[SeriesSpec]) -> None:
        self.connection.execute("DELETE FROM series_catalog")
        rows = [
            (
                s.id,
                s.name,
                s.module,
                s.source,
                s.series_id,
                s.asset_proxy,
                s.frequency,
                s.unit,
                s.transform,
                s.direction,
                s.staleness_days,
            )
            for s in specs
        ]
        self.connection.executemany(
            """
            INSERT OR REPLACE INTO series_catalog VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )

    def purge_demo_data(self) -> None:
        demo_count = self.connection.execute(
            "SELECT COUNT(*) FROM raw_observations WHERE source LIKE 'demo:%'"
        ).fetchone()[0]
        if not demo_count:
            return
        self.connection.execute("DELETE FROM raw_observations WHERE source LIKE 'demo:%'")
        for table in (
            "derived_signals",
            "quality_status",
            "regime_snapshots",
            "memo_drafts",
        ):
            self.connection.execute(f"DELETE FROM {table}")

    def replace_series_observations(self, frame: pd.DataFrame) -> None:
        if frame.empty:
            return
        series_ids = sorted(set(frame.series_id.tolist()))
        for series_id in series_ids:
            self.connection.execute(
                "DELETE FROM raw_observations WHERE series_id = ?", [series_id]
            )
        self.upsert_frame("raw_observations", frame)

    def upsert_observations(self, frame: pd.DataFrame) -> None:
        """Merge observations without deleting existing history for the series."""
        self.upsert_frame("raw_observations", frame)

    def purge_source_mismatches(self, specs: list[SeriesSpec]) -> int:
        """Drop rows whose stored source prefix conflicts with catalog intent.

        OpenBB-configured series must not keep ``akshare:`` history from old fallbacks.
        """
        deleted = 0
        for spec in specs:
            if spec.source == "openbb":
                result = self.connection.execute(
                    """
                    DELETE FROM raw_observations
                    WHERE series_id = ? AND source LIKE 'akshare:%'
                    RETURNING series_id
                    """,
                    [spec.id],
                ).fetchall()
                deleted += len(result)
            elif spec.source == "akshare":
                result = self.connection.execute(
                    """
                    DELETE FROM raw_observations
                    WHERE series_id = ? AND source LIKE 'openbb:%'
                    RETURNING series_id
                    """,
                    [spec.id],
                ).fetchall()
                deleted += len(result)
        return deleted

    def latest_observation_dates(self) -> dict[str, date]:
        frame = self.query(
            """
            SELECT series_id, MAX(observation_date) AS latest_date
            FROM raw_observations
            GROUP BY series_id
            """
        )
        if frame.empty:
            return {}
        return {
            row.series_id: pd.Timestamp(row.latest_date).date()
            for row in frame.itertuples(index=False)
        }

    def normalize_akshare_vintages(self) -> int:
        """Point-in-time replay needs vintage <= as_of; AKShare fetch dates break that."""
        changed = self.connection.execute(
            """
            SELECT COUNT(*) FROM raw_observations
            WHERE source LIKE 'akshare:%'
              AND (
                vintage_date <> observation_date
                OR (series_id, observation_date) IN (
                    SELECT series_id, observation_date
                    FROM raw_observations
                    WHERE source LIKE 'akshare:%'
                    GROUP BY 1, 2
                    HAVING COUNT(*) > 1
                )
              )
            """
        ).fetchone()[0]
        if not changed:
            return 0
        self.connection.execute(
            """
            CREATE OR REPLACE TEMP TABLE akshare_normalized AS
            SELECT
                series_id,
                observation_date,
                value,
                release_time,
                observation_date AS vintage_date,
                source,
                last_updated
            FROM raw_observations
            WHERE source LIKE 'akshare:%'
            QUALIFY ROW_NUMBER() OVER (
                PARTITION BY series_id, observation_date
                ORDER BY last_updated DESC
            ) = 1
            """
        )
        self.connection.execute("DELETE FROM raw_observations WHERE source LIKE 'akshare:%'")
        self.connection.execute(
            """
            INSERT INTO raw_observations
            SELECT * FROM akshare_normalized
            """
        )
        return int(changed)

    def upsert_frame(self, table: str, frame: pd.DataFrame) -> None:
        if frame.empty:
            return
        allowed = {
            "raw_observations",
            "derived_signals",
            "quality_status",
            "regime_snapshots",
            "events",
            "memo_drafts",
        }
        if table not in allowed:
            raise ValueError(f"不允许写入表 {table}")
        columns = list(frame.columns)
        joined = ", ".join(columns)
        placeholders = ", ".join("?" for _ in columns)
        self.connection.executemany(
            f"INSERT OR REPLACE INTO {table} ({joined}) VALUES ({placeholders})",
            list(frame.itertuples(index=False, name=None)),
        )

    def query(self, sql: str, parameters: list[object] | None = None) -> pd.DataFrame:
        return self.connection.execute(sql, parameters or []).fetchdf()

    def latest_signals(self, as_of_date: str | None = None) -> pd.DataFrame:
        date_filter = "CURRENT_DATE" if as_of_date is None else "CAST(? AS DATE)"
        params = [] if as_of_date is None else [as_of_date]
        return self.query(
            f"""
            SELECT d.*, c.name, c.module, c.source, c.source_series_id, c.asset_proxy,
                   c.frequency, c.unit, c.transform, c.direction, q.status, q.message
            FROM derived_signals d
            JOIN series_catalog c ON c.id = d.series_id
            LEFT JOIN quality_status q USING (series_id, as_of_date)
            WHERE d.as_of_date = (
                SELECT MAX(as_of_date) FROM derived_signals WHERE as_of_date <= {date_filter}
            )
            ORDER BY c.module, ABS(COALESCE(d.trend_zscore, 0)) DESC
            """,
            params,
        )

    def series_history(self, series_id: str, as_of_date: str | None = None) -> pd.DataFrame:
        cutoff = as_of_date or "9999-12-31"
        return self.query(
            """
            SELECT observation_date, value, vintage_date, release_time
            FROM raw_observations
            WHERE series_id = ? AND observation_date <= CAST(? AS DATE)
            QUALIFY vintage_date = MAX(vintage_date) OVER (PARTITION BY observation_date)
            ORDER BY observation_date
            """,
            [series_id, cutoff],
        )

    def export_parquet(self, directory: str | Path = "data/parquet") -> None:
        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        for table in ("raw_observations", "derived_signals", "quality_status", "regime_snapshots"):
            destination = str((target / f"{table}.parquet").resolve()).replace("'", "''")
            self.connection.execute(f"COPY {table} TO '{destination}' (FORMAT PARQUET)")
