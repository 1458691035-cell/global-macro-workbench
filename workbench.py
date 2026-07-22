from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from macro_workbench.paths import ensure_writable_runtime

# Streamlit Cloud has a read-only filesystem except /tmp.
ensure_writable_runtime()

from macro_workbench.streamlit_app import run


if __name__ == "__main__":
    run()
