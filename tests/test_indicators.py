import unittest

import numpy as np
import pandas as pd

from pmreport.indicators import add_indicators, rsi, sma


def make_frame(values: list[float]) -> pd.DataFrame:
    dates = pd.bdate_range("2026-01-01", periods=len(values))
    return pd.DataFrame(
        {
            "open": values,
            "high": [value * 1.01 for value in values],
            "low": [value * 0.99 for value in values],
            "close": values,
            "volume": [1000.0] * len(values),
        },
        index=dates,
    )


class IndicatorsTest(unittest.TestCase):
    def test_sma_matches_manual_window(self) -> None:
        series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        result = sma(series, 3)
        self.assertAlmostEqual(result.iloc[-1], 4.0)

    def test_rsi_bounds(self) -> None:
        close = pd.Series(np.linspace(100, 200, 60))
        values = rsi(close, 14).dropna()
        self.assertTrue((values >= 0).all())
        self.assertTrue((values <= 100).all())

    def test_add_indicators_columns(self) -> None:
        frame = make_frame(list(np.linspace(100, 150, 260)))
        out = add_indicators(frame)
        for column in ["sma20", "sma50", "sma200", "rsi14", "macd", "macd_hist", "atr_pct", "vol20", "ret20"]:
            self.assertIn(column, out.columns)


if __name__ == "__main__":
    unittest.main()

