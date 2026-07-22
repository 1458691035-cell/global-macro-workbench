"""Writable data paths for local runs and Streamlit Community Cloud.

Streamlit Cloud mounts the repo read-only except ``/tmp``. DuckDB needs a
writable parent for WAL files, and parquet exports must not target ``data/``.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = ROOT / "data"
TMP_DATA_DIR = Path("/tmp/macro_data")
TMP_OPENBB_DIR = Path("/tmp/openbb_cache")


def ensure_writable_runtime() -> Path:
    """Redirect DB/OpenBB caches to ``/tmp`` when ``data/`` is not writable.

    Sets ``MACRO_DATA_DIR``, ``MACRO_DB_PATH``, and OpenBB cache env vars.
    Also disables OpenBB auto-build (it tries to write ``.build.lock`` into
    site-packages, which fails on Streamlit Community Cloud).
    Safe to call multiple times. Returns the effective data directory.
    """
    # Must be set before any ``import openbb``; site-packages is read-only on Cloud.
    os.environ.setdefault("OPENBB_AUTO_BUILD", "0")

    existing = os.environ.get("MACRO_DATA_DIR")
    if existing:
        data_dir = Path(existing)
        data_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("MACRO_DB_PATH", str(data_dir / "macro.duckdb"))
        return data_dir

    if os.access(DEFAULT_DATA_DIR, os.W_OK):
        os.environ.setdefault("MACRO_DATA_DIR", str(DEFAULT_DATA_DIR))
        os.environ.setdefault("MACRO_DB_PATH", str(DEFAULT_DATA_DIR / "macro.duckdb"))
        return DEFAULT_DATA_DIR

    TMP_DATA_DIR.mkdir(parents=True, exist_ok=True)
    src_db = DEFAULT_DATA_DIR / "macro.duckdb"
    dst_db = TMP_DATA_DIR / "macro.duckdb"
    if src_db.exists() and not dst_db.exists():
        shutil.copy2(src_db, dst_db)

    TMP_OPENBB_DIR.mkdir(parents=True, exist_ok=True)
    os.environ["MACRO_DATA_DIR"] = str(TMP_DATA_DIR)
    os.environ["MACRO_DB_PATH"] = str(dst_db)
    # Point OpenBB user/home caches at /tmp (HOME may also be constrained).
    os.environ.setdefault("OPENBB_USER_SETTINGS_PATH", str(TMP_OPENBB_DIR))
    os.environ.setdefault("OPENBB_CACHE_DIR", str(TMP_OPENBB_DIR))
    os.environ.setdefault("XDG_CACHE_HOME", str(TMP_OPENBB_DIR / "xdg_cache"))
    return TMP_DATA_DIR


def data_dir() -> Path:
    """Return the writable data directory (may be under ``/tmp``)."""
    return ensure_writable_runtime()


def db_path() -> Path:
    """Return the DuckDB path, preferring ``MACRO_DB_PATH``."""
    ensure_writable_runtime()
    return Path(os.environ["MACRO_DB_PATH"])


def parquet_dir() -> Path:
    """Return a writable parquet export directory."""
    target = data_dir() / "parquet"
    target.mkdir(parents=True, exist_ok=True)
    return target
