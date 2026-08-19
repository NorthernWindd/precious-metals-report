import unittest

import numpy as np
import pandas as pd

from pmreport.factor_engine import (
    FEATURE_COUNT,
    build_feature_panel,
    compute_latest_factor_scores,
    execute_formula,
    mine_factors,
)


def make_bars(symbols: list[str]) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(123)
    dates = pd.bdate_range("2025-01-01", periods=180)
    bars: dict[str, pd.DataFrame] = {}
    for index, symbol in enumerate(symbols):
        start = 100.0 + index * 10.0
        returns = rng.normal(0.0002, 0.01, len(dates))
        close = start * np.exp(np.cumsum(returns))
        bars[symbol] = pd.DataFrame(
            {
                "open": close * (1 + rng.normal(0, 0.002, len(dates))),
                "high": close * 1.01,
                "low": close * 0.99,
                "close": close,
                "volume": rng.integers(1000, 3000, len(dates)).astype(float),
            },
            index=dates,
        )
    return bars


class FactorEngineTest(unittest.TestCase):
    def test_build_panel_and_execute_feature(self) -> None:
        symbols = ["A", "B", "C"]
        panel = build_feature_panel(make_bars(symbols), symbols)
        self.assertEqual(panel["feature_tensor"].shape[0], 3)
        self.assertEqual(panel["feature_tensor"].shape[1], FEATURE_COUNT)
        signal = execute_formula([0], panel["feature_tensor"])
        self.assertIsNotNone(signal)
        self.assertEqual(signal.shape[1], panel["feature_tensor"].shape[2])

    def test_mine_and_score_factors(self) -> None:
        symbols = ["A", "B", "C", "D"]
        bars = make_bars(symbols)
        factors = mine_factors(
            bars,
            symbols,
            population_size=6,
            generations=1,
            max_depth=2,
            top_n=2,
        )
        self.assertGreaterEqual(len(factors), 1)
        scores = compute_latest_factor_scores(bars, symbols, factors)
        self.assertEqual(set(scores), set(symbols))


if __name__ == "__main__":
    unittest.main()

