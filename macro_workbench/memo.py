from __future__ import annotations

from datetime import date

import pandas as pd

from .storage import MacroStore


SECTIONS = [
    "Overnight",
    "Regime",
    "What changed",
    "What is priced",
    "Positioning",
    "Catalysts",
    "Trade map",
    "Watch list",
]


def generate_memo(store: MacroStore, as_of: str | date) -> str:
    as_of_date = pd.Timestamp(as_of).date()
    signals = store.latest_signals(str(as_of_date))
    regime = store.query(
        "SELECT * FROM regime_snapshots WHERE as_of_date <= ? ORDER BY as_of_date DESC LIMIT 1",
        [str(as_of_date)],
    )
    previous = store.query(
        """
        SELECT series_id, trend_zscore FROM derived_signals
        WHERE as_of_date = (SELECT MAX(as_of_date) FROM derived_signals WHERE as_of_date < ?)
        """,
        [str(as_of_date)],
    )
    if signals.empty or regime.empty:
        raise ValueError("没有可用于生成 memo 的信号快照")
    state = regime.iloc[0]
    market = signals[signals.module == "cross_asset"].copy()
    market["move"] = market[["change_1d", "change_1w"]].abs().max(axis=1)
    changed = signals.merge(
        previous.rename(columns={"trend_zscore": "previous_trend"}), on="series_id", how="left"
    )
    changed["delta"] = (changed.trend_zscore - changed.previous_trend).abs()
    top_changes = changed.nlargest(5, "delta")
    positioning = signals[signals.module == "positioning"].copy()
    positioning["abs_trend"] = positioning.trend_zscore.abs()
    events = store.query(
        """
        SELECT * FROM events WHERE CAST(event_time AS DATE)
        BETWEEN ? AND CAST(? AS DATE) + INTERVAL 7 DAY ORDER BY event_time
        """,
        [str(as_of_date), str(as_of_date)],
    )
    maps = {
        "Goldilocks": ("偏多股票/信用，偏空美元", "增长转负或核心通胀再加速"),
        "Reflation": ("偏多周期品和盈亏平衡通胀", "增长转负或实际利率急升"),
        "Stagflation": ("偏多黄金，降低久期与股票 beta", "通胀动量明确回落"),
        "Slowdown": ("偏多久期，降低周期资产", "增长惊喜转正"),
    }
    expression, invalidation = maps[state.regime]
    lines = [
        f"# 全球宏观每日 Memo — {as_of_date}",
        "",
        "## 一句话结论",
        (
            f"当前偏 {state.regime}；增长 {state.growth:+.2f}、通胀 {state.inflation:+.2f}、"
            f"流动性 {state.liquidity:+.2f}，风险偏好 {state.risk_appetite:+.2f}。"
        ),
        "",
        "## Overnight",
        *[
            f"- {row['name']}：1D {row.change_1d:+.2f}%，1W {row.change_1w:+.2f}%。"
            for _, row in market.nlargest(5, "move").iterrows()
        ],
        "",
        "## Regime",
        f"- {state.regime}，置信度 {state.confidence:.0%}。",
        "",
        "## What changed",
        *(
            [
                f"- {row['name']}：趋势变化幅度 {row.delta:.2f}，动量 {row.momentum:+.2f}。"
                for _, row in top_changes.dropna(subset=["delta"]).iterrows()
            ]
            or ["- 首次快照，尚无前一日可比较。"]
        ),
        "",
        "## What is priced",
        *[
            f"- {row['name']}：趋势 z-score {row.trend_zscore:+.2f}，分位 {row.percentile:.0f}%。"
            for _, row in signals[
                signals.series_id.isin(["breakeven_10y", "term_premium", "dxy", "hy_spread"])
            ].iterrows()
        ],
        "",
        "## Positioning",
        *[
            f"- {row['name']}：趋势 z-score {row.trend_zscore:+.2f}。"
            for _, row in positioning.nlargest(3, "abs_trend").iterrows()
        ],
        "",
        "## Catalysts",
        *(
            [
                f"- {pd.Timestamp(row.event_time):%m-%d} [{row.region}] {row.event}。"
                for _, row in events.iterrows()
            ]
            or ["- 未录入未来 7 天事件；请人工补充共识值。"]
        ),
        "",
        "## Trade map",
        f"- 基准表达：{expression}。",
        f"- 证伪条件：{invalidation}；止损由组合约束人工设定。",
        "",
        "## Watch list",
        *[
            f"- {row['name']}：{row.message}，等待第二数据点确认。"
            for _, row in signals.nlargest(3, "trend_zscore", keep="all").iterrows()
        ],
        "",
        "_自动草稿仅用于研究，需人工确认事件、仓位和止损。_",
    ]
    content = "\n".join(lines)
    store.upsert_frame(
        "memo_drafts",
        pd.DataFrame(
            [{"as_of_date": as_of_date, "generated_at": pd.Timestamp.now(), "content": content}]
        ),
    )
    return content
