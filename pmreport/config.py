from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG: dict[str, Any] = {
    "symbols": {
        "precious_metals": ["GC=F", "SI=F", "PL=F", "PA=F"],
        "industrial_metals": ["HG=F", "ALI=F"],
        "macro": ["DX-Y.NYB", "^TNX", "^TYX"],
    },
    "history_days": 520,
    "news": {"per_symbol": 5},
    "weights": {
        "short": {
            "trend": 0.30,
            "momentum": 0.30,
            "oscillator": 0.15,
            "volatility": 0.10,
            "news_macro": 0.15,
        },
        "medium": {
            "trend": 0.35,
            "momentum": 0.25,
            "oscillator": 0.10,
            "volatility": 0.10,
            "news_macro": 0.20,
        },
        "long": {
            "trend": 0.40,
            "momentum": 0.20,
            "oscillator": 0.05,
            "volatility": 0.15,
            "news_macro": 0.20,
        },
    },
    "thresholds": {"buy": 70, "bullish": 55, "neutral": 45, "bearish": 30},
    "risk": {"volatility_percentile": 0.90, "atr_multiplier": 3.0},
    "report": {"output_dir": "reports", "timezone": "Asia/Shanghai"},
    "database": {"path": "data/market.sqlite"},
    "factors": {
        "path": "data/best_factors.json",
        "population": 120,
        "generations": 8,
        "max_depth": 4,
        "top_n": 10,
        "seed": 7,
    },
}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    if path is None:
        return deepcopy(DEFAULT_CONFIG)

    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    loaded = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    return _deep_merge(DEFAULT_CONFIG, loaded)


def all_symbols(config: dict[str, Any]) -> list[str]:
    symbols: list[str] = []
    for group in config["symbols"].values():
        if isinstance(group, list):
            symbols.extend(group)
    return symbols
