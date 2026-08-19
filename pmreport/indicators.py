from __future__ import annotations

import numpy as np
import pandas as pd


def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=window).mean()


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False, min_periods=span).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100.0 - (100.0 / (1.0 + rs))
    return out.fillna(50.0)


def macd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    macd_line = ema(close, fast) - ema(close, slow)
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def bollinger(
    close: pd.Series,
    period: int = 20,
    num_std: float = 2.0,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    middle = close.rolling(window=period, min_periods=period).mean()
    std = close.rolling(window=period, min_periods=period).std(ddof=0)
    upper = middle + num_std * std
    lower = middle - num_std * std
    return middle, upper, lower


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    previous_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def realised_volatility(close: pd.Series, window: int, annualise: bool = True) -> pd.Series:
    returns = close.pct_change(fill_method=None)
    factor = np.sqrt(252.0) if annualise else 1.0
    return returns.rolling(window=window, min_periods=window).std(ddof=0) * factor


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add technical indicators and rolling returns to a daily OHLCV frame."""

    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy().sort_index()
    close = out["close"].astype(float)

    out["sma5"] = sma(close, 5)
    out["sma10"] = sma(close, 10)
    out["sma20"] = sma(close, 20)
    out["sma50"] = sma(close, 50)
    out["sma200"] = sma(close, 200)
    out["ema12"] = ema(close, 12)
    out["ema26"] = ema(close, 26)

    out["macd"], out["macd_signal"], out["macd_hist"] = macd(close)
    out["rsi14"] = rsi(close, 14)
    out["boll_mid"], out["boll_upper"], out["boll_lower"] = bollinger(close, 20, 2.0)
    out["atr14"] = atr(out, 14)
    out["atr_pct"] = out["atr14"] / close

    for window in (1, 5, 20, 60, 120, 250):
        out[f"ret{window}"] = close.pct_change(periods=window, fill_method=None)

    out["vol20"] = realised_volatility(close, 20)
    out["vol60"] = realised_volatility(close, 60)
    return out


def last_value(series: pd.Series | float | int | None) -> float:
    if series is None:
        return float("nan")
    if isinstance(series, pd.Series):
        clean = series.dropna()
        return float(clean.iloc[-1]) if not clean.empty else float("nan")
    try:
        return float(series)
    except (TypeError, ValueError):
        return float("nan")
