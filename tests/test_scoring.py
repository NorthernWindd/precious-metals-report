import unittest
from datetime import date

import numpy as np
import pandas as pd

from pmreport.config import load_config
from pmreport.scoring import (
    build_results,
    downgrade_signal,
    select_data_date,
    signal_from_score,
)


def make_frame(start: float, end: float, periods: int = 320) -> pd.DataFrame:
    values = np.linspace(start, end, periods)
    dates = pd.bdate_range("2025-01-01", periods=periods)
    return pd.DataFrame(
        {
            "open": values,
            "high": values * 1.01,
            "low": values * 0.99,
            "close": values,
            "volume": [1000.0] * periods,
        },
        index=dates,
    )


class ScoringTest(unittest.TestCase):
    def test_signal_mapping(self) -> None:
        config = load_config()
        self.assertEqual(signal_from_score(80, config), "买入/增持")
        self.assertEqual(signal_from_score(60, config), "持有/偏多")
        self.assertEqual(signal_from_score(50, config), "观望")
        self.assertEqual(signal_from_score(35, config), "减仓/偏空")
        self.assertEqual(signal_from_score(20, config), "卖出/规避")

    def test_downgrade_signal(self) -> None:
        self.assertEqual(downgrade_signal("买入/增持"), "持有/偏多")
        self.assertEqual(downgrade_signal("卖出/规避"), "卖出/规避")

    def test_uptrend_is_bullish(self) -> None:
        config = load_config()
        frame = make_frame(100, 200)
        results = build_results(
            {"GC=F": frame},
            {"GC=F": 55.0},
            config,
            date(2026, 8, 19),
        )
        result = results[0]
        self.assertTrue(result["has_data"])
        self.assertIn(result["metrics"]["signals"]["short"], {"买入/增持", "持有/偏多"})

    def test_downtrend_is_bearish(self) -> None:
        config = load_config()
        frame = make_frame(200, 100)
        results = build_results(
            {"GC=F": frame},
            {"GC=F": 45.0},
            config,
            date(2026, 8, 19),
        )
        result = results[0]
        self.assertIn(result["metrics"]["signals"]["short"], {"减仓/偏空", "卖出/规避"})

    def test_select_data_date_uses_primary(self) -> None:
        older = make_frame(100, 110, 40)
        newer = make_frame(100, 110, 40)
        report_date = date(2026, 8, 19)
        data_date = select_data_date({"GC=F": older, "SI=F": newer}, ["GC=F", "SI=F"], report_date)
        self.assertIsInstance(data_date, date)


if __name__ == "__main__":
    unittest.main()

