from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

from dotenv import load_dotenv

from .data_router import fetch_all_observations
from .memo import generate_memo
from .models import load_series
from .pipeline import run_pipeline
from .storage import MacroStore
from .validation import validate_replay


ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    load_dotenv(ROOT / ".env", override=False)
    parser = argparse.ArgumentParser(description="全球宏观策略工作台")
    parser.add_argument("--db", default=str(ROOT / "data" / "macro.duckdb"))
    commands = parser.add_subparsers(dest="command", required=True)
    update = commands.add_parser("update", help="通过 OpenBB(+AKShare) 更新真实数据")
    update.add_argument("--as-of", default=date.today().isoformat())
    update.add_argument("--years", type=int, default=5, help="全量窗口年数；无历史时增量也用此窗口")
    update.add_argument(
        "--full",
        action="store_true",
        help="全量刷新：重拉 years 窗口并覆盖序列（默认增量合并）",
    )
    update.add_argument(
        "--lookback",
        type=int,
        default=30,
        help="增量回看天数，覆盖近期修订（默认 30）",
    )
    replay = commands.add_parser("validate")
    replay.add_argument("--end")
    replay.add_argument("--days", type=int, default=10)
    replay.add_argument("--output", default=str(ROOT / "data" / "validation-report.md"))
    memo = commands.add_parser("memo")
    memo.add_argument("--as-of", default=date.today().isoformat())
    memo.add_argument("--output")
    args = parser.parse_args(argv)
    specs = load_series(ROOT / "config" / "series.yaml")
    store = MacroStore(args.db)
    try:
        if args.command == "update":
            mode = "full" if args.full else "incremental"
            latest_dates = store.latest_observation_dates()
            purged = store.purge_source_mismatches(specs)
            if purged:
                print(f"已清理 {purged} 条错误来源历史（OpenBB/AKShare 交叉污染）。")
            print(
                f"更新模式：{'全量刷新' if args.full else f'增量（回看 {args.lookback} 天）'}；"
                f"库中已有 {len(latest_dates)}/{len(specs)} 条序列历史。"
            )
            result = fetch_all_observations(
                specs,
                args.as_of,
                args.years,
                mode=mode,
                lookback_days=args.lookback,
                latest_dates=store.latest_observation_dates(),
            )
            if result.observations.empty:
                print("未返回任何有效数据。")
                for series_id, error in result.errors.items():
                    print(f"- {series_id}: {error}")
                return 1
            store.purge_demo_data()
            store.normalize_akshare_vintages()
            run_pipeline(
                store,
                specs,
                args.as_of,
                result.observations,
                replace=args.full,
            )
            store.export_parquet(ROOT / "data" / "parquet")
            succeeded = result.observations.series_id.nunique()
            verb = "覆盖写入" if args.full else "合并写入"
            print(
                f"已{verb} {len(result.observations):,} 条观测，"
                f"{succeeded}/{len(specs)} 条指标更新成功。"
            )
            sources = (
                result.observations.groupby(
                    result.observations.source.str.split(":").str[0]
                )
                .series_id.nunique()
                .to_dict()
            )
            print("来源分布：", sources)
            if result.errors:
                print("未更新指标：")
                for series_id, error in result.errors.items():
                    print(f"- {series_id}: {error}")
        elif args.command == "validate":
            fixed = store.normalize_akshare_vintages()
            if fixed:
                print(f"已校正 {fixed:,} 条 AKShare vintage，使其可按观测日回放。")
            report = validate_replay(store, specs, args.end, args.days)
            output = Path(args.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(report.as_markdown(), encoding="utf-8")
            print(report.as_markdown())
        else:
            content = generate_memo(store, args.as_of)
            if args.output:
                Path(args.output).write_text(content, encoding="utf-8")
            else:
                print(content)
        return 0
    finally:
        store.close()


if __name__ == "__main__":
    raise SystemExit(main())
