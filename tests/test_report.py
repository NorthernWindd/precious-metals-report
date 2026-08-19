import tempfile
import unittest
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from pmreport.config import load_config
from pmreport.report import render_reports
from pmreport.scoring import build_results


class ReportTest(unittest.TestCase):
    def test_render_html_and_markdown(self) -> None:
        config = load_config()
        dates = pd.bdate_range("2025-01-01", periods=320)
        close = np.linspace(100, 160, 320)
        frame = pd.DataFrame(
            {
                "open": close,
                "high": close * 1.01,
                "low": close * 0.99,
                "close": close,
                "volume": [1000.0] * 320,
            },
            index=dates,
        )
        results = build_results(
            {"GC=F": frame},
            {"GC=F": 55.0},
            config,
            date(2026, 8, 19),
        )
        news = {
            "GC=F": [
                {
                    "published_at": "2026-08-18T08:00:00+00:00",
                    "title": "Gold rallies on safe-haven demand",
                    "publisher": "Reuters",
                    "link": "https://example.com/news/1",
                    "summary_zh": "偏多：检测到关键词“上涨”。",
                    "sentiment": 0.7,
                }
            ]
        }
        with tempfile.TemporaryDirectory() as tmp:
            html_path, md_path = render_reports(
                results=results,
                bars_by_symbol={"GC=F": frame},
                news_by_symbol=news,
                report_date=date(2026, 8, 19),
                data_date=date(2026, 8, 18),
                output_dir=Path(tmp),
                config=config,
            )
            self.assertTrue(html_path.exists())
            self.assertTrue(md_path.exists())
            self.assertIn("贵金属每日趋势", html_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()

