from __future__ import annotations

import os
from pathlib import Path

import duckdb
import pandas as pd

from macro_workbench.paths import (
    DEFAULT_DATA_DIR,
    db_path,
    ensure_writable_runtime,
    parquet_dir,
)
from macro_workbench.storage import MacroStore


def _clear_path_env() -> None:
    for key in (
        "MACRO_DATA_DIR",
        "MACRO_DB_PATH",
        "OPENBB_USER_SETTINGS_PATH",
        "OPENBB_CACHE_DIR",
    ):
        os.environ.pop(key, None)


def test_ensure_writable_runtime_uses_tmp_when_data_readonly(
    tmp_path: Path, monkeypatch
) -> None:
    _clear_path_env()
    ro = tmp_path / "data"
    ro.mkdir()
    src_db = ro / "macro.duckdb"
    con = duckdb.connect(str(src_db))
    con.execute("CREATE TABLE marker(x INT); INSERT INTO marker VALUES (42)")
    con.close()
    ro.chmod(0o555)

    tmp_data = tmp_path / "macro_data"
    tmp_openbb = tmp_path / "openbb_cache"
    monkeypatch.setattr("macro_workbench.paths.DEFAULT_DATA_DIR", ro)
    monkeypatch.setattr("macro_workbench.paths.TMP_DATA_DIR", tmp_data)
    monkeypatch.setattr("macro_workbench.paths.TMP_OPENBB_DIR", tmp_openbb)

    data = ensure_writable_runtime()
    assert data == tmp_data
    assert Path(os.environ["MACRO_DB_PATH"]) == tmp_data / "macro.duckdb"
    assert (tmp_data / "macro.duckdb").exists()

    store = MacroStore(db_path())
    try:
        store.connection.execute("CREATE TABLE IF NOT EXISTS t(i INT)")
        store.connection.execute("INSERT INTO t VALUES (1)")
        frame = pd.DataFrame(
            {
                "series_id": ["x"],
                "observation_date": [pd.Timestamp("2026-07-01").date()],
                "value": [1.0],
                "release_time": [pd.Timestamp("2026-07-01")],
                "vintage_date": [pd.Timestamp("2026-07-01").date()],
                "source": ["test"],
                "last_updated": [pd.Timestamp("2026-07-01")],
            }
        )
        store.upsert_observations(frame)
        target = parquet_dir()
        assert target.is_relative_to(tmp_data)
        store.export_parquet(target)
        assert (target / "raw_observations.parquet").exists()
    finally:
        store.close()
        ro.chmod(0o755)
        _clear_path_env()


def test_parquet_dir_prefers_existing_env(tmp_path: Path, monkeypatch) -> None:
    _clear_path_env()
    custom = tmp_path / "custom_data"
    monkeypatch.setenv("MACRO_DATA_DIR", str(custom))
    monkeypatch.setattr("macro_workbench.paths.DEFAULT_DATA_DIR", tmp_path / "unused")
    assert parquet_dir() == custom / "parquet"
    assert custom.exists()
    _clear_path_env()


def test_ensure_writable_runtime_disables_openbb_autobuild(monkeypatch) -> None:
    _clear_path_env()
    monkeypatch.delenv("OPENBB_AUTO_BUILD", raising=False)
    ensure_writable_runtime()
    assert os.environ["OPENBB_AUTO_BUILD"] == "0"
    _clear_path_env()


def test_default_data_dir_constant() -> None:
    assert DEFAULT_DATA_DIR.name == "data"
