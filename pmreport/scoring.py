from __future__ import annotations

from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from .indicators import add_indicators, last_value


GROUP_LABELS = {
    "precious_metals": "贵金属",
    "industrial_metals": "工业金属",
    "macro": "宏观指标",
}

SYMBOL_NAMES = {
    "GC=F": "黄金",
    "SI=F": "白银",
    "PL=F": "铂金",
    "PA=F": "钯金",
    "HG=F": "铜",
    "ALI=F": "铝",
    "DX-Y.NYB": "美元指数",
    "^TNX": "10年期美债收益率",
    "^TYX": "30年期美债收益率",
}


def select_data_date(
    bars_by_symbol: dict[str, pd.DataFrame],
    primary_symbols: list[str],
    report_date: date,
) -> date:
    latest_dates: list[date] = []
    for symbol in primary_symbols:
        frame = bars_by_symbol.get(symbol)
        if frame is None or frame.empty:
            continue
        try:
            frame_dates = pd.to_datetime(frame.index).date
            valid = [d for d in frame_dates if d <= report_date]
            if valid:
                latest_dates.append(max(valid))
        except (TypeError, ValueError):
            continue
    if not latest_dates:
        for frame in bars_by_symbol.values():
            if frame is None or frame.empty:
                continue
            try:
                frame_dates = pd.to_datetime(frame.index).date
                valid = [d for d in frame_dates if d <= report_date]
                if valid:
                    latest_dates.append(max(valid))
            except (TypeError, ValueError):
                continue
    return max(latest_dates) if latest_dates else report_date


def _clip(value: float, lower: float = 0.0, upper: float = 100.0) -> float:
    return float(max(lower, min(upper, value)))


def _relation_score(current: float, reference: float) -> float:
    if current is None or reference is None:
        return 50.0
    try:
        current_f = float(current)
        reference_f = float(reference)
    except (TypeError, ValueError):
        return 50.0
    if not np.isfinite(current_f) or not np.isfinite(reference_f) or reference_f == 0:
        return 50.0
    relative = ((current_f - reference_f) / reference_f) * 100.0
    return _clip(50.0 + 50.0 * np.tanh(relative))


def _percentile_of_last(series: pd.Series, fallback: float = 50.0) -> float:
    clean = series.dropna()
    if len(clean) < 5:
        return fallback
    return float(clean.rank(pct=True).iloc[-1] * 100.0)


def _trend_score(metrics: dict[str, Any], horizon: str) -> float:
    if horizon == "short":
        scores = [
            _relation_score(metrics.get("close"), metrics.get("sma5")),
            _relation_score(metrics.get("sma5"), metrics.get("sma20")),
            _relation_score(metrics.get("close"), metrics.get("sma20")),
        ]
    elif horizon == "medium":
        scores = [
            _relation_score(metrics.get("close"), metrics.get("sma20")),
            _relation_score(metrics.get("sma20"), metrics.get("sma50")),
            _relation_score(metrics.get("close"), metrics.get("sma50")),
        ]
    else:
        scores = [
            _relation_score(metrics.get("close"), metrics.get("sma50")),
            _relation_score(metrics.get("sma50"), metrics.get("sma200")),
            _relation_score(metrics.get("close"), metrics.get("sma200")),
        ]
    valid = [score for score in scores if np.isfinite(score)]
    if not valid:
        return 50.0
    return float(np.mean(valid))


def _momentum_score(df: pd.DataFrame, horizon: str) -> float:
    column = {"short": "ret20", "medium": "ret60", "long": "ret250"}[horizon]
    if column not in df.columns:
        return 50.0
    return _percentile_of_last(df[column], fallback=50.0)


def _oscillator_score(metrics: dict[str, Any]) -> float:
    rsi_value = metrics.get("rsi14")
    if rsi_value is None or not np.isfinite(rsi_value):
        return 50.0
    return _clip(100.0 - abs(float(rsi_value) - 60.0) * 1.2)


def _volatility_score(df: pd.DataFrame, horizon: str) -> float:
    column = "vol20" if horizon == "short" else "vol60"
    if column not in df.columns:
        return 50.0
    percentile = _percentile_of_last(df[column], fallback=50.0)
    return 100.0 - percentile


def _macro_score(
    symbol_df: pd.DataFrame,
    macro_bars: dict[str, pd.DataFrame],
) -> float:
    dxy = macro_bars.get("DX-Y.NYB")
    tnx = macro_bars.get("^TNX")
    if dxy is None or dxy.empty or tnx is None or tnx.empty:
        return 50.0

    asset = symbol_df["close"].pct_change(fill_method=None).rename("asset")
    dxy_ret = dxy["close"].pct_change(fill_method=None).rename("dxy")
    tnx_ret = tnx["close"].pct_change(fill_method=None).rename("tnx")
    merged = pd.concat([asset, dxy_ret, tnx_ret], axis=1).dropna()
    if len(merged) < 30:
        return 50.0

    recent = merged.tail(120)
    correlation = recent.corr()

    def _coef(left: str, right: str) -> float:
        value = correlation.loc[left, right]
        return float(value) if pd.notna(value) else 0.0

    corr_dxy = _coef("asset", "dxy")
    corr_tnx = _coef("asset", "tnx")
    recent_dxy = float(recent["dxy"].tail(20).mean())
    recent_tnx = float(recent["tnx"].tail(20).mean())

    # 贵金属通常与美元指数、美债收益率负相关；用符号把宏观逆风/顺风转换成正向贡献。
    contribution = -corr_dxy * recent_dxy - corr_tnx * recent_tnx
    return _clip(50.0 + 50.0 * np.tanh(contribution * 100.0))


def _news_macro_score(
    news_score: float,
    symbol_df: pd.DataFrame,
    macro_bars: dict[str, pd.DataFrame],
) -> float:
    macro = _macro_score(symbol_df, macro_bars)
    return 0.7 * float(news_score) + 0.3 * macro


def _metrics_for_symbol(
    df: pd.DataFrame,
    news_score: float,
    macro_bars: dict[str, pd.DataFrame],
    config: dict[str, Any],
) -> dict[str, Any]:
    if df is None or df.empty:
        return {}

    enriched = add_indicators(df)
    if enriched.empty:
        return {}

    last = enriched.iloc[-1]
    metrics = {
        "close": last_value(last["close"]),
        "sma5": last_value(last["sma5"]),
        "sma10": last_value(last["sma10"]),
        "sma20": last_value(last["sma20"]),
        "sma50": last_value(last["sma50"]),
        "sma200": last_value(last["sma200"]),
        "rsi14": last_value(last["rsi14"]),
        "macd_hist": last_value(last["macd_hist"]),
        "atr_pct": last_value(last["atr_pct"]),
        "ret1": last_value(last["ret1"]),
        "ret5": last_value(last["ret5"]),
        "ret20": last_value(last["ret20"]),
        "ret60": last_value(last["ret60"]),
        "ret120": last_value(last["ret120"]),
        "ret250": last_value(last["ret250"]),
    }
    metrics["news_score"] = float(news_score)
    metrics["scores"] = {}
    metrics["signals"] = {}
    metrics["confidences"] = {}

    horizon_lengths = {"short": 20, "medium": 60, "long": 250}
    risk_high = _is_high_risk(enriched, config)
    metrics["risk_high"] = risk_high

    for horizon, min_bars in horizon_lengths.items():
        if len(enriched) < min_bars:
            metrics["scores"][horizon] = None
            metrics["signals"][horizon] = "数据不足"
            metrics["confidences"][horizon] = None
            continue

        weights = config["weights"][horizon]
        factors = {
            "trend": _trend_score(metrics, horizon),
            "momentum": _momentum_score(enriched, horizon),
            "oscillator": _oscillator_score(metrics),
            "volatility": _volatility_score(enriched, horizon),
            "news_macro": _news_macro_score(news_score, enriched, macro_bars),
        }
        score = sum(factors[key] * weights[key] for key in weights)
        score = _clip(score)
        signal = signal_from_score(score, config)
        if risk_high:
            signal = downgrade_signal(signal)

        metrics["scores"][horizon] = round(score, 1)
        metrics["signals"][horizon] = signal
        metrics["confidences"][horizon] = round(abs(score - 50.0) * 2.0, 1)

    return metrics


def _is_high_risk(df: pd.DataFrame, config: dict[str, Any]) -> bool:
    if "vol20" not in df.columns or "atr_pct" not in df.columns:
        return False
    vol_percentile = _percentile_of_last(df["vol20"], fallback=50.0)
    vol_threshold = config["risk"]["volatility_percentile"] * 100.0

    atr_series = df["atr_pct"].dropna()
    if len(atr_series) >= 20:
        median = float(atr_series.median())
        latest = float(atr_series.iloc[-1])
        atr_ratio = latest / median if median > 0 else 0.0
    else:
        atr_ratio = 0.0

    return (
        vol_percentile >= vol_threshold
        or atr_ratio >= float(config["risk"]["atr_multiplier"])
    )


def signal_from_score(score: float, config: dict[str, Any]) -> str:
    thresholds = config["thresholds"]
    if score >= thresholds["buy"]:
        return "买入/增持"
    if score >= thresholds["bullish"]:
        return "持有/偏多"
    if score >= thresholds["neutral"]:
        return "观望"
    if score >= thresholds["bearish"]:
        return "减仓/偏空"
    return "卖出/规避"


_SIGNAL_ORDER = ["卖出/规避", "减仓/偏空", "观望", "持有/偏多", "买入/增持"]


def downgrade_signal(signal: str) -> str:
    if signal not in _SIGNAL_ORDER:
        return signal
    index = _SIGNAL_ORDER.index(signal)
    return _SIGNAL_ORDER[max(0, index - 1)]


def build_results(
    bars_by_symbol: dict[str, pd.DataFrame],
    news_scores: dict[str, float],
    config: dict[str, Any],
    report_date: date,
    alpha_scores: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    macro_bars = {
        symbol: bars_by_symbol.get(symbol, pd.DataFrame())
        for symbol in config["symbols"]["macro"]
    }

    results: list[dict[str, Any]] = []
    alpha_scores = alpha_scores or {}
    for group, symbols in config["symbols"].items():
        for symbol in symbols:
            df = bars_by_symbol.get(symbol, pd.DataFrame())
            news_score = float(news_scores.get(symbol, 50.0))
            metrics = _metrics_for_symbol(df, news_score, macro_bars, config)
            if symbol in alpha_scores:
                alpha_value = float(alpha_scores[symbol])
                metrics["alpha_factor_score"] = round(alpha_value, 1)
                metrics["alpha_factor_signal"] = factor_signal_from_score(alpha_value)
            results.append(
                {
                    "symbol": symbol,
                    "symbol_name": SYMBOL_NAMES.get(symbol, symbol),
                    "group": group,
                    "group_label": GROUP_LABELS.get(group, group),
                    "report_date": report_date.isoformat(),
                    "metrics": metrics,
                    "has_data": bool(df is not None and not df.empty and metrics),
                }
            )
    return results


def factor_signal_from_score(score: float) -> str:
    if score >= 70:
        return "因子偏多"
    if score <= 30:
        return "因子偏空"
    return "因子中性"
