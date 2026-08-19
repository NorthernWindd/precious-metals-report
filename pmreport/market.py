from __future__ import annotations

import sqlite3
import time
from datetime import datetime, timezone
from typing import Any
from urllib.parse import quote
from zoneinfo import ZoneInfo

import pandas as pd
import requests

from .db import upsert_bars, upsert_news
from .news import analyze_news_item


YAHOO_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/126.0 Safari/537.36"
    ),
    "Accept": "application/json",
}


def _get_json(url: str, params: dict[str, Any] | None = None, timeout: int = 20) -> dict[str, Any]:
    response = requests.get(url, params=params, headers=YAHOO_HEADERS, timeout=timeout)
    response.raise_for_status()
    return response.json()


def fetch_history(symbol: str, period: str = "520d", retries: int = 3) -> pd.DataFrame:
    """Download daily OHLCV bars from Yahoo Finance.

    Returns an empty DataFrame when no data is available.
    """

    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            frame = _fetch_history_direct(symbol, period)
            if not frame.empty:
                return frame
            raise ValueError("Yahoo 返回空行情")
        except Exception as exc:  # pragma: no cover - network dependent
            last_error = exc
            if attempt < retries - 1:
                time.sleep(2.0 ** attempt)

    raise RuntimeError(f"无法获取 {symbol} 行情: {last_error}")


def _fetch_history_direct(symbol: str, period: str) -> pd.DataFrame:
    data = _get_json(
        "https://query1.finance.yahoo.com/v8/finance/chart/" + quote(symbol, safe=""),
        params={"range": period, "interval": "1d"},
    )
    chart = data.get("chart", {})
    if chart.get("error") or not chart.get("result"):
        raise ValueError(chart.get("error") or "无 chart.result")

    result = chart["result"][0]
    timestamps = result.get("timestamp") or []
    if not timestamps:
        raise ValueError("无 timestamp")

    quote_data = result.get("indicators", {}).get("quote", [{}])[0]
    close = quote_data.get("close") or []
    if len(close) != len(timestamps):
        raise ValueError("close 与 timestamp 长度不一致")

    meta = result.get("meta", {})
    timezone_name = meta.get("exchangeTimezoneName") or meta.get("timezone") or "UTC"
    try:
        tz = ZoneInfo(timezone_name)
    except Exception:
        tz = timezone.utc

    dates = (
        pd.to_datetime(timestamps, unit="s", utc=True)
        .tz_convert(tz)
        .tz_localize(None)
        .date
        .astype(str)
    )
    volume = quote_data.get("volume") or [None] * len(timestamps)
    frame = pd.DataFrame(
        {
            "date": dates,
            "open": quote_data.get("open") or [None] * len(timestamps),
            "high": quote_data.get("high") or [None] * len(timestamps),
            "low": quote_data.get("low") or [None] * len(timestamps),
            "close": close,
            "volume": volume,
        }
    )
    return _normalise_history(frame, symbol)


def _fetch_history_yfinance(symbol: str, period: str) -> pd.DataFrame:
    import yfinance as yf

    ticker = yf.Ticker(symbol)
    frame = ticker.history(
        period=period,
        interval="1d",
        auto_adjust=False,
        actions=False,
        progress=False,
    )
    return _normalise_history(frame, symbol)


def _normalise_history(frame: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])

    out = frame.copy()
    out = out.reset_index()
    rename_map: dict[Any, str] = {}
    for column in out.columns:
        col_lower = str(column).lower()
        if col_lower in {"date", "datetime", "timestamp"}:
            rename_map[column] = "date"
        elif col_lower == "open":
            rename_map[column] = "open"
        elif col_lower == "high":
            rename_map[column] = "high"
        elif col_lower == "low":
            rename_map[column] = "low"
        elif col_lower == "close":
            rename_map[column] = "close"
        elif col_lower in {"volume", "vol"}:
            rename_map[column] = "volume"

    if (
        "index" in out.columns
        and not any(value == "date" for value in rename_map.values())
        and pd.api.types.is_datetime64_any_dtype(out["index"])
    ):
        rename_map["index"] = "date"

    out = out.rename(columns=rename_map)
    required = {"date", "close"}
    if not required.issubset(out.columns):
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])

    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.dropna(subset=["date", "close"])
    out["date"] = out["date"].dt.tz_localize(None).dt.date.astype(str)
    columns = ["date", "open", "high", "low", "close", "volume"]
    for column in columns:
        if column not in out.columns:
            out[column] = None
    return out[columns].sort_values("date").reset_index(drop=True)


def ingest_symbols(
    conn: sqlite3.Connection,
    symbols: list[str],
    period: str = "520d",
    verbose: bool = False,
) -> dict[str, pd.DataFrame]:
    bars: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        try:
            frame = fetch_history(symbol, period=period)
            count = upsert_bars(conn, symbol, frame)
            if verbose:
                print(f"[pmreport] {symbol}: {count} 条日线", flush=True)
            bars[symbol] = _history_from_rows(frame)
        except Exception as exc:  # pragma: no cover - network dependent
            print(f"[pmreport] 行情抓取失败 {symbol}: {exc}", flush=True)
            bars[symbol] = pd.DataFrame(
                columns=["date", "open", "high", "low", "close", "volume"]
            )
        time.sleep(2.0)
    return bars


def _history_from_rows(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])
    out = frame.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.dropna(subset=["date"])
    return out.set_index("date").sort_index()


def fetch_raw_news(symbol: str, limit: int = 5, retries: int = 3) -> list[dict[str, Any]]:
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            return _fetch_news_direct(symbol, limit)
        except Exception as exc:  # pragma: no cover - network dependent
            last_error = exc
            if attempt < retries - 1:
                time.sleep(2.0 ** attempt)

    if last_error:
        print(f"[pmreport] 新闻抓取失败 {symbol}: {last_error}")
    return []


def _fetch_news_direct(symbol: str, limit: int) -> list[dict[str, Any]]:
    data = _get_json(
        "https://query1.finance.yahoo.com/v1/finance/search",
        params={
            "q": symbol,
            "newsCount": str(limit),
            "quotesCount": "0",
            "enableFuzzyQuery": "false",
        },
    )
    items: list[dict[str, Any]] = []
    for raw in (data.get("news") or [])[:limit]:
        item = analyze_news_item(raw, symbol)
        if item.get("link"):
            items.append(item)
    return items


def _fetch_news_yfinance(symbol: str, limit: int) -> list[dict[str, Any]]:
    import yfinance as yf

    ticker = yf.Ticker(symbol)
    items: list[dict[str, Any]] = []
    for raw in (ticker.news or [])[:limit]:
        item = analyze_news_item(raw, symbol)
        if item.get("link"):
            items.append(item)
    return items


def fetch_and_ingest_news(
    conn: sqlite3.Connection,
    symbols: list[str],
    limit: int = 5,
    verbose: bool = False,
) -> None:
    for symbol in symbols:
        items = fetch_raw_news(symbol, limit=limit)
        if not items:
            if verbose:
                print(f"[pmreport] {symbol}: 无新闻", flush=True)
            continue
        inserted_at = datetime.now(timezone.utc).isoformat()
        for item in items:
            item["inserted_at"] = inserted_at
        count = upsert_news(conn, symbol, items)
        if verbose:
            print(f"[pmreport] {symbol}: {count} 条新闻", flush=True)
