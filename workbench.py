from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Streamlit Cloud has a read-only filesystem except /tmp.
# Redirect OpenBB cache and make a writable copy of the DuckDB file.
if not os.access(ROOT / "data", os.W_OK):
    tmp_data = Path("/tmp/macro_data")
    tmp_data.mkdir(parents=True, exist_ok=True)
    src_db = ROOT / "data" / "macro.duckdb"
    dst_db = tmp_data / "macro.duckdb"
    if src_db.exists() and not dst_db.exists():
        shutil.copy2(src_db, dst_db)
    os.environ["MACRO_DB_PATH"] = str(dst_db)

    tmp_openbb = Path("/tmp/openbb_cache")
    tmp_openbb.mkdir(parents=True, exist_ok=True)
    os.environ["OPENBB_USER_SETTINGS_PATH"] = str(tmp_openbb)
    os.environ["OPENBB_CACHE_DIR"] = str(tmp_openbb)

from macro_workbench.streamlit_app import run


if __name__ == "__main__":
    run()
