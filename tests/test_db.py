import tempfile
import unittest
from pathlib import Path

import pandas as pd

from pmreport.db import connect, count_bars, load_bars, upsert_bars


class DatabaseTest(unittest.TestCase):
    def test_upsert_and_load(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            conn = connect(Path(tmp) / "market.sqlite")
            frame = pd.DataFrame(
                {
                    "date": ["2026-08-17", "2026-08-18"],
                    "open": [100, 101],
                    "high": [102, 103],
                    "low": [99, 100],
                    "close": [101, 102],
                    "volume": [1000, 1100],
                }
            )
            self.assertEqual(upsert_bars(conn, "GC=F", frame), 2)
            self.assertEqual(upsert_bars(conn, "GC=F", frame), 2)
            self.assertEqual(count_bars(conn, "GC=F"), 2)
            loaded = load_bars(conn, "GC=F")
            self.assertEqual(len(loaded), 2)
            self.assertAlmostEqual(loaded["close"].iloc[-1], 102)
            conn.close()


if __name__ == "__main__":
    unittest.main()

