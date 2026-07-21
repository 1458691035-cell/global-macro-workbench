from macro_workbench.paths import ensure_writable_runtime

ensure_writable_runtime()

from macro_workbench.streamlit_app import run

run()
