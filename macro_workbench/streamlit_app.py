from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from .data_router import NO_NEW_VALUE_MSG, fetch_all_observations
from .memo import generate_memo
from .models import MODULE_NAMES, SeriesSpec, load_series
from .pipeline import run_pipeline
from .storage import MacroStore
from .tsanghi_source import fetch_tsanghi_realtime


ROOT = Path(__file__).resolve().parents[1]


def _db_path() -> Path:
    env = os.environ.get("MACRO_DB_PATH")
    if env:
        return Path(env)
    return ROOT / "data" / "macro.duckdb"

CHANGE_COLUMNS: dict[str, list[tuple[str, str]]] = {
    "daily": [("change_1d", "日变化"), ("change_1w", "周变化"), ("change_1m", "月变化")],
    "weekly": [("change_1d", "周变化"), ("change_1m", "月变化")],
    "monthly": [("change_1d", "月变化"), ("change_3m", "年变化")],
    "quarterly": [("change_1d", "季变化"), ("change_3m", "年变化")],
}
# Keep tables scannable: raw/transformed level, recent moves, history rank, freshness.
BASE_COLS_BEFORE = ["name", "value", "transformed_value"]
BASE_COLS_AFTER = ["percentile", "release_time", "prev_release_time", "status"]
DISPLAY_RENAMES = {
    "name": "指标",
    "value": "最新值",
    "transformed_value": "变换值",
    "percentile": "历史分位",
    "release_time": "本数据公布时间",
    "prev_release_time": "上一条公布时间",
    "status": "状态",
}
FREQ_LABELS = {"daily": "日度", "weekly": "周度", "monthly": "月度", "quarterly": "季度"}
# First-page cross-asset layout: thematic blocks instead of one long table.
CROSS_ASSET_GROUPS: list[tuple[str, tuple[str, ...]]] = [
    (
        "股票与 VIX",
        (
            "sp500",
            "euro_stocks",
            "japan_stocks",
            "china_stocks",
            "hstech",
            "chinext",
            "star_index",
            "vix",
        ),
    ),
    ("利率与信用", ("us_2y", "us_10y", "us_30y", "hy_spread", "ig_spread")),
    ("汇率与美元", ("dxy", "em_usd", "eurusd", "usdjpy", "usdcny")),
    ("黄金与比特币", ("gold", "bitcoin")),
    ("铜", ("copper",)),
    ("原油与成品油", ("oil", "gas_gulf")),
]


def run() -> None:
    st.set_page_config(page_title="全球宏观策略工作台", page_icon="🌐", layout="wide")
    st.title("全球宏观策略工作台")
    specs = load_series(ROOT / "config" / "series.yaml")
    store = MacroStore(_db_path())
    store.register_catalog(specs)

    if st.sidebar.button("实时更新", width="stretch", type="primary"):
        st.session_state["pending_realtime"] = True
    if st.sidebar.button("增量更新", width="stretch"):
        st.session_state["pending_update"] = "incremental"
    if st.sidebar.button("全量刷新", width="stretch"):
        st.session_state["pending_update"] = "full"
    st.sidebar.caption(
        "数据路由 v3：美/全球经济 FRED，行情 tsanghi（滞后则 yfinance 补最新点），中国宏观 AKShare；"
        "增量按频率回看（日5/周14/月65天），空窗记「本周期无新值」；全量≈5年。"
    )

    with st.sidebar.expander("单指标插入"):
        spec_labels = {s.id: s.name for s in specs}
        picked = st.multiselect("选择指标", list(spec_labels.keys()), format_func=spec_labels.get)
        if st.button("拉取并写入", width="stretch", disabled=not picked):
            st.session_state["pending_insert"] = picked

    if st.sidebar.button("清除上次错误提示", width="stretch"):
        for key in (
            "update_errors",
            "update_message",
            "update_level",
            "pending_update",
            "realtime_fallback",
        ):
            st.session_state.pop(key, None)
        st.rerun()

    if st.session_state.pop("pending_realtime", None):
        realtime_ids = {
            "china_stocks", "hstech", "star_index", "chinext",
            "sp500", "gold", "usdcny", "euro_stocks",
        }
        rt_specs = [s for s in specs if s.id in realtime_ids]
        with st.spinner("正在获取 tsanghi 实时数据…"):
            st.session_state["realtime_fallback"] = fetch_tsanghi_realtime(rt_specs)

    if mode := st.session_state.pop("pending_update", None):
        _execute_update(specs, store, mode=mode)
        st.rerun()

    if picked_ids := st.session_state.pop("pending_insert", None):
        _execute_insert(specs, store, picked_ids)
        st.rerun()

    if message := st.session_state.pop("update_message", None):
        level = st.session_state.pop("update_level", "success")
        getattr(st.sidebar, level)(message)
    if errors := st.session_state.get("update_errors"):
        soft = {k: v for k, v in errors.items() if v == NO_NEW_VALUE_MSG}
        hard = {k: v for k, v in errors.items() if v != NO_NEW_VALUE_MSG}
        if soft:
            with st.sidebar.expander(f"{len(soft)} 条本周期无新值", expanded=False):
                for series_id, error in soft.items():
                    st.caption(f"`{series_id}`：{error}")
        if hard:
            with st.sidebar.expander(f"{len(hard)} 条指标未更新", expanded=True):
                for series_id, error in hard.items():
                    st.caption(f"`{series_id}`：{error}")
    _render_realtime_sidebar(st.session_state.get("realtime_fallback"))

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
            elif module == "cross_asset":
                _cross_asset(store, frame, selected, title)
            else:
                _module(store, frame, selected, title)
    st.divider()
    memo = generate_memo(store, selected)
    st.download_button("下载 Memo", memo, file_name=f"macro-memo-{selected}.md")
    st.markdown(memo)


def _render_realtime_sidebar(rt: pd.DataFrame | None) -> None:
    if rt is None:
        return
    st.sidebar.subheader("未更新 · 实时行情")
    if rt.empty:
        st.sidebar.warning("未获取到实时数据。")
        return
    st.sidebar.dataframe(
        rt[["name", "close", "date"]].rename(
            columns={"name": "品种", "close": "最新价", "date": "日期"}
        ),
        hide_index=True,
        column_config={
            "最新价": st.column_config.NumberColumn(format="%.2f"),
        },
    )


def _fetch_realtime_for_unupdated(
    specs: list[SeriesSpec], unupdated_ids: set[str]
) -> pd.DataFrame:
    """Pull tsanghi realtime quotes for series that did not update."""
    rt_specs = [s for s in specs if s.id in unupdated_ids and s.source == "tsanghi"]
    if not rt_specs:
        return pd.DataFrame(columns=["series_id", "name", "date", "open", "high", "low", "close"])
    return fetch_tsanghi_realtime(rt_specs)


def _execute_update(
    specs: list[SeriesSpec],
    store: MacroStore,
    *,
    mode: str = "incremental",
    lookback_days: int = 5,
    years: int = 5,
) -> None:
    """在主区域显示实时进度；按增量/全量模式拉取并写入。"""
    replace = mode == "full"
    title = "全量刷新" if replace else "增量更新（日5/周14/月65天）"
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

    updated_ids = (
        set(result.observations["series_id"].astype(str))
        if not result.observations.empty
        else set()
    )
    hard_errors = {
        sid: msg for sid, msg in result.errors.items() if msg != NO_NEW_VALUE_MSG
    }
    unupdated_ids = {s.id for s in specs if s.id not in updated_ids} | set(hard_errors)
    # Soft "no new value" should not trigger realtime fallback.
    unupdated_ids -= {sid for sid, msg in result.errors.items() if msg == NO_NEW_VALUE_MSG}
    if unupdated_ids:
        current.markdown("**当前：** 为未更新行情拉取实时报价…")
        st.session_state["realtime_fallback"] = _fetch_realtime_for_unupdated(
            specs, unupdated_ids
        )
    else:
        st.session_state.pop("realtime_fallback", None)

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
    store.export_parquet(ROOT / "data" / "parquet")
    count = result.observations.series_id.nunique()
    verb = "全量覆盖" if replace else "增量合并"
    st.session_state["update_level"] = "success"
    st.session_state["update_message"] = f"{verb} {count}/{len(specs)} 条指标"
    if result.errors:
        st.session_state["update_errors"] = result.errors
    else:
        st.session_state.pop("update_errors", None)
    current.markdown(f"**当前：** {verb} {count}/{len(specs)} 条指标")


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
    store.export_parquet(ROOT / "data" / "parquet")

    count = result.observations.series_id.nunique()
    rows = len(result.observations)
    progress_bar.progress(1.0, text="完成")
    st.success(f"已插入 {count} 条指标（共 {rows} 行观测值）")
    if result.errors:
        with st.expander(f"{len(result.errors)} 条失败"):
            for sid, err in result.errors.items():
                st.caption(f"`{sid}`：{err}")


def _build_display(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    freq = frame.frequency.iloc[0] if "frequency" in frame.columns else "daily"
    change_cols = CHANGE_COLUMNS.get(freq, CHANGE_COLUMNS["daily"])
    cols = BASE_COLS_BEFORE + [src for src, _ in change_cols] + BASE_COLS_AFTER
    rename_map = {**DISPLAY_RENAMES, **{src: label for src, label in change_cols}}
    display = frame[cols].rename(columns=rename_map)
    column_config = {
        label: st.column_config.NumberColumn(label, format="%.2f%%")
        for _, label in change_cols
    }
    column_config["变换值"] = st.column_config.NumberColumn("变换值", format="%.2f")
    column_config["历史分位"] = st.column_config.NumberColumn("历史分位", format="%.0f")
    column_config["本数据公布时间"] = st.column_config.DatetimeColumn(format="YYYY-MM-DD HH:mm")
    column_config["上一条公布时间"] = st.column_config.DatetimeColumn(format="YYYY-MM-DD HH:mm")
    return display, column_config


def _render_signal_table(frame: pd.DataFrame) -> None:
    """Render one signal table, splitting by frequency only when mixed."""
    if frame.empty:
        return
    freq_order = ["daily", "weekly", "monthly", "quarterly"]
    groups = [(f, frame[frame.frequency == f]) for f in freq_order]
    groups = [(f, g) for f, g in groups if not g.empty]
    show_label = len(groups) > 1
    for freq, group in groups:
        group = group.sort_values("release_time", ascending=False, na_position="last")
        if show_label:
            st.caption(FREQ_LABELS[freq])
        display, column_config = _build_display(group)
        st.dataframe(display, hide_index=True, column_config=column_config)


def _ordered_group(frame: pd.DataFrame, series_ids: tuple[str, ...]) -> pd.DataFrame:
    subset = frame[frame.series_id.isin(series_ids)].copy()
    if subset.empty:
        return subset
    order = {sid: idx for idx, sid in enumerate(series_ids)}
    subset["_order"] = subset.series_id.map(order)
    return subset.sort_values("_order").drop(columns="_order")


def _cross_asset(
    store: MacroStore, frame: pd.DataFrame, selected: str, title: str
) -> None:
    st.subheader(title)
    if frame.empty:
        st.warning("该模块无可用信号。")
        return
    seen: set[str] = set()
    for label, series_ids in CROSS_ASSET_GROUPS:
        group = _ordered_group(frame, series_ids)
        seen.update(series_ids)
        with st.container(border=True):
            st.markdown(f"**{label}**")
            if group.empty:
                st.caption("暂无数据")
            else:
                _render_signal_table(group)
                _drill_down(
                    store,
                    group,
                    selected,
                    key=f"cross-{series_ids[0]}",
                )
    leftover = frame[~frame.series_id.isin(seen)]
    if not leftover.empty:
        with st.container(border=True):
            st.markdown("**其他**")
            _render_signal_table(leftover)
            _drill_down(store, leftover, selected, key="cross-other")


def _module(store: MacroStore, frame: pd.DataFrame, selected: str, title: str) -> None:
    st.subheader(title)
    if frame.empty:
        st.warning("该模块无可用信号。")
        return
    _render_signal_table(frame)
    _drill_down(store, frame, selected, key=frame.module.iloc[0])


def _format_release_time(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "—"
    stamp = pd.Timestamp(value)
    if pd.isna(stamp):
        return "—"
    return stamp.strftime("%Y-%m-%d")


def _drill_down(
    store: MacroStore, frame: pd.DataFrame, selected: str, *, key: str
) -> None:
    """Per-section series picker + history chart."""
    if frame.empty:
        return
    st.caption("下钻与走势")
    labels = frame.set_index("series_id").name.to_dict()
    options = frame.series_id.tolist()
    series_id = st.selectbox(
        "选择指标",
        options,
        format_func=lambda sid: labels.get(sid, sid),
        key=f"drill-{key}",
    )
    signal = frame[frame.series_id == series_id].iloc[0]
    history = store.series_history(series_id, selected)
    left, right = st.columns([2, 1])
    with left:
        if history.empty:
            st.info("该指标暂无历史序列可绘。")
        else:
            st.plotly_chart(
                px.line(
                    history,
                    x="observation_date",
                    y="value",
                    title=str(signal["name"]),
                ),
                width="stretch",
                key=f"chart-{key}-{series_id}",
            )
    proxy = signal["asset_proxy"] if pd.notna(signal["asset_proxy"]) else "无"
    description = signal["description"] if pd.notna(signal.get("description")) else ""
    right.markdown(
        f"**代理**：{proxy}  \n"
        f"**来源**：`{signal['source']}/{signal['source_series_id']}`  \n"
        f"**公布时间**：{_format_release_time(signal.get('release_time'))}  \n"
        f"**说明**：{description or '暂无'}"
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
