from __future__ import annotations

import argparse
from pathlib import Path

from .config import all_symbols, load_config
from .db import connect, load_bars
from .factor_engine import mine_factors, save_factors


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成贵金属 AlphaGPT 风格因子")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--db", help="SQLite 数据库路径，默认读取配置")
    parser.add_argument("--output", default="data/best_factors.json")
    parser.add_argument("--population", type=int, default=120)
    parser.add_argument("--generations", type=int, default=8)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config(args.config)
    db_path = Path(args.db or config["database"]["path"])

    conn = connect(db_path)
    try:
        symbols = all_symbols(config)
        bars_by_symbol = {symbol: load_bars(conn, symbol) for symbol in symbols}
    finally:
        conn.close()

    factors = mine_factors(
        bars_by_symbol=bars_by_symbol,
        symbols=symbols,
        population_size=args.population,
        generations=args.generations,
        max_depth=args.max_depth,
        top_n=args.top,
        seed=args.seed,
    )
    save_factors(args.output, factors)

    print(f"已生成 {len(factors)} 个因子 -> {args.output}")
    for rank, factor in enumerate(factors, 1):
        print(
            f"#{rank:02d} fitness={factor['fitness']:.2f} "
            f"IC1={factor['ic1_mean']:.3f} IC5={factor['ic5_mean']:.3f} "
            f"tokens={factor['tokens']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

