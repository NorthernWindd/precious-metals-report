import unittest

from pmreport.news import analyse_title, news_score_for_items, normalize_news_item


class NewsTest(unittest.TestCase):
    def test_positive_title(self) -> None:
        sentiment, summary, _ = analyse_title("Gold prices rally to a record high on safe-haven demand")
        self.assertGreater(sentiment, 0)
        self.assertIn("偏正面", summary)

    def test_negative_title(self) -> None:
        sentiment, summary, _ = analyse_title("Silver drops as strong dollar and rate hike weigh")
        self.assertLess(sentiment, 0)
        self.assertIn("偏负面", summary)

    def test_neutral_score_without_items(self) -> None:
        self.assertEqual(news_score_for_items([]), 50.0)

    def test_normalize_item(self) -> None:
        raw = {
            "title": "Copper rises on demand",
            "publisher": {"name": "Reuters"},
            "link": "https://example.com/news/1",
            "providerPublishTime": 1700000000,
        }
        item = normalize_news_item(raw, "HG=F")
        self.assertEqual(item["symbol"], "HG=F")
        self.assertGreater(item["sentiment"], 0)


if __name__ == "__main__":
    unittest.main()
