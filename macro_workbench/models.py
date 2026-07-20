from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


MODULE_NAMES = {
    "cross_asset": "跨资产总览",
    "growth": "增长脉冲",
    "inflation": "通胀脉冲",
    "policy": "政策与流动性",
    "positioning": "定价与仓位",
    "events": "事件与情景",
}


@dataclass(frozen=True)
class SeriesSpec:
    id: str
    name: str
    module: str
    source: str
    series_id: str
    frequency: str
    unit: str
    transform: str
    direction: str
    staleness_days: int
    asset_proxy: str | None = None

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "SeriesSpec":
        unknown_module = value["module"] not in MODULE_NAMES
        if unknown_module:
            raise ValueError(f"未知模块: {value['module']}")
        return cls(**value)


@dataclass(frozen=True)
class Observation:
    series_id: str
    observation_date: str
    value: float
    release_time: datetime
    vintage_date: str
    source: str
    last_updated: datetime


def load_series(path: str | Path) -> list[SeriesSpec]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    specs = [SeriesSpec.from_dict(item) for item in payload["series"]]
    ids = [spec.id for spec in specs]
    if len(ids) != len(set(ids)):
        raise ValueError("series.yaml 中存在重复 id")
    return specs
