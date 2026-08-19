from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


SCHEMA = """
CREATE TABLE IF NOT EXISTS daily_bars (
    symbol TEXT NOT NULL,
    date TEXT NOT NULL,
    open REAL,
    high REAL,
    low REAL,
    close REAL,
    volume REAL,
    PRIMARY KEY (symbol, date)
);

CREATE TABLE IF NOT EXISTS news (
    symbol TEXT NOT NULL,
    published_at TEXT,
    title TEXT NOT NULL,
    publisher TEXT,
    link TEXT NOT NULL,
    summary_zh TEXT,
    topic_zh TEXT,
    title_zh TEXT,
    key_points TEXT,
    relevance TEXT,
    sentiment REAL,
    inserted_at TEXT NOT NULL,
    PRIMARY KEY (symbol, link)
);
"""


def connect(db_path: str | Path) -> sqlite3.Connection:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(SCHEMA)
    _migrate_news_columns(conn)
    return conn


def _migrate_news_columns(conn: sqlite3.Connection) -> None:
    existing = {
        row["name"]
        for row in conn.execute("PRAGMA table_info(news)").fetchall()
    }
    for column in ("topic_zh", "title_zh", "key_points", "relevance"):
        if column not in existing:
            conn.execute(f"ALTER TABLE news ADD COLUMN {column} TEXT")
    conn.commit()


def _normalise_bars(df: pd.DataFrame, symbol: str) -> list[tuple[Any, ...]]:
    if df is None or df.empty:
        return []

    frame = df.copy()
    frame = frame.reset_index()
    rename_map: dict[str, str] = {}
    for column in frame.columns:
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
        "index" in frame.columns
        and not any(value == "date" for value in rename_map.values())
        and pd.api.types.is_datetime64_any_dtype(frame["index"])
    ):
        rename_map["index"] = "date"

    frame = frame.rename(columns=rename_map)
    required = {"date", "close"}
    if not required.issubset(frame.columns):
        return []

    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame.dropna(subset=["date", "close"])
    frame["date"] = frame["date"].dt.tz_localize(None).dt.date.astype(str)

    rows: list[tuple[Any, ...]] = []
    for _, row in frame.iterrows():
        rows.append(
            (
                symbol,
                row["date"],
                _float_or_none(row.get("open")),
                _float_or_none(row.get("high")),
                _float_or_none(row.get("low")),
                _float_or_none(row.get("close")),
                _float_or_none(row.get("volume")),
            )
        )
    return rows


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def upsert_bars(conn: sqlite3.Connection, symbol: str, df: pd.DataFrame) -> int:
    rows = _normalise_bars(df, symbol)
    if not rows:
        return 0
    conn.executemany(
        """
        INSERT INTO daily_bars(symbol, date, open, high, low, close, volume)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(symbol, date) DO UPDATE SET
            open=excluded.open,
            high=excluded.high,
            low=excluded.low,
            close=excluded.close,
            volume=excluded.volume
        """,
        rows,
    )
    conn.commit()
    return len(rows)


def load_bars(conn: sqlite3.Connection, symbol: str) -> pd.DataFrame:
    frame = pd.read_sql_query(
        """
        SELECT date, open, high, low, close, volume
        FROM daily_bars
        WHERE symbol = ?
        ORDER BY date ASC
        """,
        conn,
        params=(symbol,),
    )
    if frame.empty:
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame.dropna(subset=["date"])
    frame = frame.set_index("date").sort_index()
    return frame


def upsert_news(
    conn: sqlite3.Connection,
    symbol: str,
    items: Iterable[dict[str, Any]],
) -> int:
    count = 0
    for item in items:
        if not item.get("link"):
            continue
        conn.execute(
            """
            INSERT INTO news(
                symbol, published_at, title, publisher, link, summary_zh,
                topic_zh, title_zh, key_points, relevance, sentiment, inserted_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(symbol, link) DO UPDATE SET
                published_at=excluded.published_at,
                title=excluded.title,
                publisher=excluded.publisher,
                summary_zh=excluded.summary_zh,
                topic_zh=excluded.topic_zh,
                title_zh=excluded.title_zh,
                key_points=excluded.key_points,
                relevance=excluded.relevance,
                sentiment=excluded.sentiment,
                inserted_at=excluded.inserted_at
            """,
            (
                symbol,
                item.get("published_at"),
                item.get("title"),
                item.get("publisher"),
                item.get("link"),
                item.get("summary_zh"),
                item.get("topic_zh"),
                item.get("title_zh"),
                item.get("key_points"),
                item.get("relevance"),
                item.get("sentiment"),
                item.get("inserted_at"),
            ),
        )
        count += 1
    conn.commit()
    return count


def load_news(conn: sqlite3.Connection, symbol: str, limit: int | None = None) -> list[dict[str, Any]]:
    sql = """
        SELECT symbol, published_at, title, publisher, link, summary_zh,
               topic_zh, title_zh, key_points, relevance, sentiment, inserted_at
        FROM news
        WHERE symbol = ?
        ORDER BY COALESCE(published_at, inserted_at) DESC
    """
    params: list[Any] = [symbol]
    if limit is not None:
        sql += " LIMIT ?"
        params.append(limit)
    rows = conn.execute(sql, params).fetchall()
    return [dict(row) for row in rows]


def count_bars(conn: sqlite3.Connection, symbol: str) -> int:
    row = conn.execute(
        "SELECT COUNT(*) AS n FROM daily_bars WHERE symbol = ?",
        (symbol,),
    ).fetchone()
    return int(row["n"]) if row else 0
