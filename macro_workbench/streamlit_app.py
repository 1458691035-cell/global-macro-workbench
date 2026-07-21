from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from .data_router import fetch_all_observations
from .memo import generate_memo
from .models import MODULE_NAMES, SeriesSpec, load_series
from .paths import db_path, ensure_writable_runtime, parquet_dir
from .pipeline import run_pipeline
from .storage import MacroStore
from .tsanghi_source import fetch_tsanghi_realtime


ROOT = Path(__file__).resolve().parents[1]

CHANGE_COLUMNS: dict[str, list[tuple[str, str]]] = {
    "daily": [("change_1d", "change_1d"), ("change_1w", "change_1w"), ("change_1m", "change_1m")],
    "weekly": [("change_1d", "周变化"), ("change_1m", "月变化")],
    "monthly": [("change_1d", "月变化"), ("change_3m", "年变化")],
    "quarterly": [("change_1d", "季变化"), ("change_3m", "年变化")],
}
BASE_COLS_BEFORE = ["name", "value", "level", "momentum", "surprise", "percentile"]
BASE_COLS_AFTER = ["trend_zscore", "release_time", "prev_release_time", "status"]
FREQ_LABELS = {"daily": "日度", "weekly": "周度", "monthly": "月度", "quarterly": "季度"}


def run() -> None:
    ensure_writable_runtime()
    st.set_page_config(page_title="全球宏观策略工作台", page_icon="🌐", layout="wide")
    st.title("全球宏观策略工作台")
    specs = load_series(ROOT / "config" / "series.yaml")
    store = MacroStore(db_path())

    if st.sidebar.button("实时更新", width="stretch", type="primary"):
        st.session_state["pending_realtime"] = True
    if st.sidebar.button("增量更新", width="stretch"):
        st.session_state["pending_update"] = "incremental"
    if st.sidebar.button("全量刷新", width="stretch"):
        st.session_state["pending_update"] = "full"
    st.sidebar.caption(
        "数据路由 v3：美/全球经济 FRED，行情 tsanghi，中国宏观 AKShare；"
        "增量≈30天，全量≈5年。"
    )

    with st.sidebar.expander("单指标插入"):
        spec_labels = {s.id: s.name for s in specs}
        picked = st.multiselect("选择指标", list(spec_labels.keys()), format_func=spec_labels.get)
        if st.button("拉取并写入", width="stretch", disabled=not picked):
            st.session_state["pending_insert"] = picked

    if st.sidebar.button("清除上次错误提示", width="stretch"):
        for key in ("update_errors", "update_message", "update_level", "pending_update"):
            st.session_state.pop(key, None)
        st.rerun()

    if st.session_state.pop("pending_realtime", None):
        realtime_ids = {
            "china_stocks", "hstech", "star_index", "chinext",
            "sp500", "gold", "dxy", "usdcny",
        }
        rt_specs = [s for s in specs if s.id in realtime_ids]
        with st.sidebar:
            st.subheader("实时行情")
            with st.spinner("正在获取 tsanghi 实时数据…"):
                rt = fetch_tsanghi_realtime(rt_specs)
            if rt.empty:
                st.warning("未获取到实时数据。")
            else:
                st.dataframe(
                    rt[["name", "close", "date"]].rename(
                        columns={"name": "品种", "close": "最新价", "date": "日期"}
                    ),
                    hide_index=True,
                    column_config={
                        "最新价": st.column_config.NumberColumn(format="%.2f"),
                    },
                )

    if mode := st.session_state.pop("pending_update", None):
        try:
            _execute_update(specs, store, mode=mode)
        except Exception as exc:  # noqa: BLE001 — surface any update failure in UI
            st.session_state["update_level"] = "error"
            st.session_state["update_message"] = f"更新失败：{exc}"
        st.rerun()

    if picked_ids := st.session_state.pop("pending_insert", None):
        try:
            _execute_insert(specs, store, picked_ids)
        except Exception as exc:  # noqa: BLE001 — surface any insert failure in UI
            st.session_state["update_level"] = "error"
            st.session_state["update_message"] = f"插入失败：{exc}"
        st.rerun()

    if message := st.session_state.pop("update_message", None):
        level = st.session_state.pop("update_level", "success")
        getattr(st.sidebar, level)(message)
    if errors := st.session_state.pop("update_errors", None):
        with st.sidebar.expander(f"{len(errors)} 条指标未更新"):
            for series_id, error in errors.items():
                st.caption(f"`{series_id}`：{error}")

    dates = store.query("SELECT DISTINCT as_of_date FROM derived_signals ORDER BY as_of_date DESC")
    if dates.empty:
        st.info("请在左侧增量更新或全量刷新，或运行 `python -m macro_workbench.main_cli update`。")
        return
    selected = st.sidebar.selectbox("快照日期", dates.as_of_date.dt.date.astype(str).tolist())
    signals = store.latest_signals(selected)
    regime = store.query("SELECT * FROM regime_snapshots WHERE as_of_date = ?", [selected]).iloc[0]
    columns = st.columns(6)
    for column, label, value in (
        (columns[0], "Regime", regime.regime),
        (columns[1], "增长", f"{regime.growth:+.2f}"),
        (columns[2], "通胀", f"{regime.inflation:+.2f}"),
        (columns[3], "流动性", f"{regime.liquidity:+.2f}"),
        (columns[4], "风险偏好", f"{regime.risk_appetite:+.2f}"),
        (columns[5], "置信度", f"{regime.confidence:.0%}"),
    ):
        column.metric(label, value)
    tabs = st.tabs(list(MODULE_NAMES.values()))
    for tab, (module, title) in zip(tabs, MODULE_NAMES.items(), strict=True):
        with tab:
            frame = signals[signals.module == module]
            if module == "events":
                _events(store, frame, selected)
            else:
                _module(store, frame, selected, title)
    st.divider()
    memo = generate_memo(store, selected)
    st.download_button("下载 Memo", memo, file_name=f"macro-memo-{selected}.md")
    st.markdown(memo)


def _execute_update(
    specs: list[SeriesSpec],
    store: MacroStore,
    *,
    mode: str = "incremental",
    lookback_days: int = 30,
    years: int = 5,
) -> None:
    """在主区域显示实时进度；按增量/全量模式拉取并写入。"""
    replace = mode == "full"
    title = "全量刷新" if replace else f"增量更新（回看 {lookback_days} 天）"
    st.subheader(title)
    progress_bar = st.progress(0, text="准备拉取…")
    current = st.empty()
    log = st.container(height=240, autoscroll=True)

    def on_progress(done: int, total: int, message: str) -> None:
        ratio = min(done / total, 1.0) if total else 1.0
        progress_bar.progress(ratio, text=f"{done}/{total}")
        current.markdown(f"**当前：** {message}")
        log.caption(message)

    current.markdown("**当前：** 开始连接数据源…")
    purged = store.purge_source_mismatches(specs)
    if purged:
        log.caption(f"已清理 {purged} 条错误来源历史（如 OpenBB 序列上的 akshare 残留）")
    latest_dates = store.latest_observation_dates()
    result = fetch_all_observations(
        specs,
        years=years,
        on_progress=on_progress,
        mode="full" if replace else "incremental",
        lookback_days=lookback_days,
        latest_dates=latest_dates,
    )

    if result.observations.empty:
        st.session_state["update_level"] = "error"
        st.session_state["update_message"] = "未返回任何有效数据。"
        st.session_state["update_errors"] = result.errors
        current.markdown("**当前：** 未返回任何有效数据")
        return

    current.markdown("**当前：** 正在计算信号与导出…")
    progress_bar.progress(1.0, text=f"{len(specs)}/{len(specs)}")
    store.purge_demo_data()
    store.normalize_akshare_vintages()
    run_pipeline(
        store,
        specs,
        pd.Timestamp.today().date(),
        result.observations,
        replace=replace,
    )
    export_note = _safe_export_parquet(store)
    count = result.observations.series_id.nunique()
    verb = "全量覆盖" if replace else "增量合并"
    st.session_state["update_level"] = "success"
    message = f"{verb} {count}/{len(specs)} 条指标"
    if export_note:
        message = f"{message}（{export_note}）"
    st.session_state["update_message"] = message
    if result.errors:
        st.session_state["update_errors"] = result.errors
    current.markdown(f"**当前：** {message}")


def _execute_insert(specs: list[SeriesSpec], store: MacroStore, picked_ids: list[str]) -> None:
    """拉取选中的指标（全量5年），写入数据库并重算信号。"""
    picked_specs = [s for s in specs if s.id in picked_ids]
    names = "、".join(s.name for s in picked_specs)
    st.subheader(f"单指标插入：{names}")
    progress_bar = st.progress(0, text="准备拉取…")
    current = st.empty()

    def on_progress(done: int, total: int, message: str) -> None:
        progress_bar.progress(min(done / total, 1.0), text=f"{done}/{total}")
        current.markdown(f"**当前：** {message}")

    result = fetch_all_observations(picked_specs, years=5, on_progress=on_progress, mode="full")

    if result.observations.empty:
        st.error("未返回任何有效数据。")
        if result.errors:
            for sid, err in result.errors.items():
                st.caption(f"`{sid}`：{err}")
        return

    store.upsert_observations(result.observations)
    store.register_catalog(specs)
    run_pipeline(store, specs, pd.Timestamp.today().date(), result.observations, replace=False)
    export_note = _safe_export_parquet(store)

    count = result.observations.series_id.nunique()
    rows = len(result.observations)
    progress_bar.progress(1.0, text="完成")
    message = f"已插入 {count} 条指标（共 {rows} 行观测值）"
    if export_note:
        message = f"{message}（{export_note}）"
    st.success(message)
    if result.errors:
        with st.expander(f"{len(result.errors)} 条失败"):
            for sid, err in result.errors.items():
                st.caption(f"`{sid}`：{err}")


def _safe_export_parquet(store: MacroStore) -> str | None:
    """Export parquet to a writable dir; never block a successful DB update."""
    try:
        target = parquet_dir()
        store.export_parquet(target)
        return None
    except OSError as exc:
        return f"Parquet 导出跳过：{exc}"


def _build_display(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    freq = frame.frequency.iloc[0] if "frequency" in frame.columns else "daily"
    change_cols = CHANGE_COLUMNS.get(freq, CHANGE_COLUMNS["daily"])
    cols = BASE_COLS_BEFORE + [src for src, _ in change_cols] + BASE_COLS_AFTER
    rename_map = {src: label for src, label in change_cols}
    rename_map["release_time"] = "本数据公布时间"
    rename_map["prev_release_time"] = "上一条公布时间"
    display = frame[cols].rename(columns=rename_map)
    column_config = {
        label: st.column_config.NumberColumn(label, format="%.2f%%")
        for _, label in change_cols
    }
    column_config["本数据公布时间"] = st.column_config.DatetimeColumn(format="YYYY-MM-DD HH:mm")
    column_config["上一条公布时间"] = st.column_config.DatetimeColumn(format="YYYY-MM-DD HH:mm")
    return display, column_config


def _module(store: MacroStore, frame: pd.DataFrame, selected: str, title: str) -> None:
    st.subheader(title)
    if frame.empty:
        st.warning("该模块无可用信号。")
        return
    freq_order = ["daily", "weekly", "monthly", "quarterly"]
    groups = [(f, frame[frame.frequency == f]) for f in freq_order]
    groups = [(f, g) for f, g in groups if not g.empty]
    show_label = len(groups) > 1
    for freq, group in groups:
        group = group.sort_values("release_time", ascending=False, na_position="last")
        if show_label:
            st.markdown(f"**{FREQ_LABELS[freq]}指标**")
        display, column_config = _build_display(group)
        st.dataframe(display, hide_index=True, column_config=column_config)
    _drill_down(store, frame, selected)


def _drill_down(store: MacroStore, frame: pd.DataFrame, selected: str) -> None:
    labels = frame.set_index("series_id").name.to_dict()
    series_id = st.selectbox(
        "下钻指标",
        frame.series_id.tolist(),
        format_func=labels.get,
        key=f"drill-{frame.module.iloc[0]}",
    )
    signal = frame[frame.series_id == series_id].iloc[0]
    history = store.series_history(series_id, selected)
    left, right = st.columns([2, 1])
    left.plotly_chart(
        px.line(history, x="observation_date", y="value", title=signal["name"]),
        width="stretch",
    )
    right.markdown(
        f"来源：`{signal.source}/{signal.source_series_id}`  \n"
        f"代理：{signal.asset_proxy or '无'}  \n"
        f"频率/单位：{signal.frequency}/{signal.unit}  \n"
        f"转换：{signal.transform}  \n"
        f"本数据公布时间：{signal.get('release_time', '—')}  \n"
        f"上一条公布时间：{signal.get('prev_release_time', '—')}  \n"
        f"Vintage：{signal.vintage_date}  \n"
        f"质量：{signal.status} — {signal.message}"
    )


def _events(store: MacroStore, frame: pd.DataFrame, selected: str) -> None:
    edited = st.data_editor(
        store.query("SELECT * FROM events ORDER BY event_time"),
        num_rows="dynamic",
        width="stretch",
    )
    if st.button("保存事件"):
        store.connection.execute("DELETE FROM events")
        store.upsert_frame("events", edited)
    if not frame.empty:
        _module(store, frame, selected, "收益率曲线与情景锚")
