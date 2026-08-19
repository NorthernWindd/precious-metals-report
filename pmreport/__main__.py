from __future__ import annotations

import argparse
import sys
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from .config import all_symbols, load_config
from .db import connect, load_bars, load_news
from .factor_engine import compute_latest_factor_scores, load_factors
from .market import fetch_and_ingest_news, ingest_symbols
from .news import news_score_for_items
from .report import render_reports
from .scoring import build_results, select_data_date


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="贵金属每日趋势自动报告")
    parser.add_argument("--config", default="config.yaml", help="配置文件路径")
    parser.add_argument(
        "--date",
        help="报告日期，格式 YYYY-MM-DD；默认使用上海时区当天日期",
    )
    parser.add_argument("--output", help="报告输出目录，默认读取配置")
    parser.add_argument("--db", help="SQLite 数据库路径，默认读取配置")
    parser.add_argument(
        "--no-fetch",
        action="store_true",
        help="不从网络抓取行情和新闻，仅使用本地数据库数据",
    )
    parser.add_argument("--verbose", action="store_true", help="输出详细日志")
    return parser.parse_args(argv)


def _parse_date(value: str | None) -> date:
    if value:
        return datetime.strptime(value, "%Y-%m-%d").date()
    return datetime.now(ZoneInfo("Asia/Shanghai")).date()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config(args.config)

    db_path = Path(args.db or config["database"]["path"])
    output_dir = Path(args.output or config["report"]["output_dir"])
    report_date = _parse_date(args.date)

    if args.verbose:
        print(f"[pmreport] 报告日期: {report_date}", file=sys.stderr, flush=True)

    conn = connect(db_path)
    symbols = all_symbols(config)

    try:
        if not args.no_fetch:
            if args.verbose:
                print(f"[pmreport] 抓取 {len(symbols)} 个标的行情...", file=sys.stderr, flush=True)
            ingest_symbols(
                conn=conn,
                symbols=symbols,
                period=f"{config['history_days']}d",
                verbose=args.verbose,
            )
            if args.verbose:
                print("[pmreport] 抓取新闻...", file=sys.stderr, flush=True)
            fetch_and_ingest_news(
                conn=conn,
                symbols=symbols,
                limit=config["news"]["per_symbol"],
                verbose=args.verbose,
            )
    finally:
        conn.close()

    conn = connect(db_path)
    try:
        bars_by_symbol = {
            symbol: load_bars(conn, symbol)
            for symbol in symbols
        }
        news_by_symbol = {
            symbol: load_news(conn, symbol, config["news"]["per_symbol"])
            for symbol in symbols
        }
    finally:
        conn.close()

    primary_symbols = config["symbols"]["precious_metals"]
    data_date = select_data_date(bars_by_symbol, primary_symbols, report_date)

    news_scores = {
        symbol: news_score_for_items(items)
        for symbol, items in news_by_symbol.items()
    }
    factor_path = Path(config["factors"]["path"])
    alpha_scores = (
        compute_latest_factor_scores(
            bars_by_symbol,
            symbols,
            load_factors(factor_path),
        )
        if factor_path.exists()
        else {}
    )
    results = build_results(
        bars_by_symbol=bars_by_symbol,
        news_scores=news_scores,
        config=config,
        report_date=report_date,
        alpha_scores=alpha_scores,
    )

    render_reports(
        results=results,
        bars_by_symbol=bars_by_symbol,
        news_by_symbol=news_by_symbol,
        report_date=report_date,
        data_date=data_date,
        output_dir=output_dir,
        config=config,
    )

    if args.verbose:
        print(
            f"[pmreport] 报告已生成: {output_dir / report_date.isoformat()}",
            file=sys.stderr,
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
