# 全球宏观策略工作台

配置驱动、可追溯到原始序列与 vintage 的 Streamlit MVP。数据层以 **OpenBB** 为中间件（FRED 官方宏观 + yfinance 行情），**AKShare** 仅补充中国/亚洲序列。

## 快速开始

```bash
cd /Users/maopengyu/Desktop/global-macro-workbench
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"

# 配置 FRED key（勿提交 .env）
cp .env.example .env
# 编辑 .env：FRED_API_KEY=你的密钥

python -m macro_workbench.main_cli update
python -m macro_workbench.main_cli validate --days 10
python -m streamlit run workbench.py
```

若出现 `No module named 'macro_workbench'`，在已激活的 `.venv` 中重新执行
`python -m pip install -e ".[dev]"`，并始终用 `python -m ...` 启动。

## 数据路由

- OpenBB / FRED：美国利率、增长、通胀、流动性、信用与期限溢价等官方序列
- OpenBB / yfinance：全球股指、汇率、商品、VIX/MOVE（失败时可回退到已映射的 AKShare 接口）
- AKShare：沪深300、中国 PMI、新增信贷、出口同比等

`update` 会写入真实观测、清理 `demo:` 数据，并导出 `data/parquet/`。
面板表格只展示最新值、变换值、近期变化、历史分位与状态（不再展示 level / momentum / surprise / z-score）。

```bash
python -m macro_workbench.main_cli memo --as-of 2026-07-17 --output data/memo.md
```

## 数据与决策链

- `config/series.yaml`：54 条指标的来源、代理、说明、频率、转换、方向含义与过期阈值
- `raw_observations`：原始值、发布日期、观测日期与 vintage；计算时按 `release_time <= as_of`
- `derived_signals` / `quality_status` / `regime_snapshots` / `memo_drafts`：信号、质量、四象限与 memo

## 六模块

1. 跨资产总览  
2. 增长脉冲  
3. 通胀脉冲  
4. 政策与流动性  
5. 定价与仓位  
6. 事件与情景  

## 验证

`python -m macro_workbench.main_cli validate --days 10` 连续回放 10 个交易日，报告覆盖率、健康度、memo 耗时与建议复核指标。
