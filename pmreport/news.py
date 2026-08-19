from __future__ import annotations

import json
import math
import re
from datetime import datetime, timezone
from typing import Any

import requests
from bs4 import BeautifulSoup


POSITIVE_PHRASES: dict[str, float] = {
    "record high": 1.5,
    "record production": 1.2,
    "record cash generation": 1.3,
    "beat estimates": 1.2,
    "beats estimates": 1.2,
    "upgraded": 1.0,
    "upgrade": 0.8,
    "outperform": 0.8,
    "surge": 1.4,
    "surges": 1.4,
    "rally": 1.0,
    "rallies": 1.0,
    "gains": 0.8,
    "gain": 0.8,
    "rising demand": 1.0,
    "rises on demand": 1.0,
    "strong demand": 1.2,
    "safe haven": 0.9,
    "safe-haven": 0.9,
    "inflation hedge": 1.0,
    "rate cut": 1.1,
    "dovish": 0.9,
    "stimulus": 0.9,
    "bullish": 0.9,
    "expansion": 0.6,
    "growth": 0.5,
    "moved higher": 1.1,
    "rises": 0.8,
    "rose": 0.8,
    "advance": 0.7,
    "extended their advance": 1.0,
    "weaker dollar": 1.0,
    "lower treasury yields": 1.0,
    "falling yields": 0.9,
    "rate support": 0.8,
}

NEGATIVE_PHRASES: dict[str, float] = {
    "selloff": 1.4,
    "sell-off": 1.4,
    "plunge": 1.5,
    "plunges": 1.5,
    "tumble": 1.4,
    "tumbles": 1.4,
    "bearish": 0.9,
    "rate hike": 1.1,
    "hikes rates": 1.1,
    "hawkish": 0.9,
    "recession": 1.4,
    "weak demand": 1.2,
    "demand destruction": 1.3,
    "strong dollar": 1.0,
    "stronger dollar": 1.0,
    "misses estimates": 1.2,
    "downgrade": 1.0,
    "downgraded": 1.0,
    "slump": 1.0,
    "deficit": 0.7,
    "moved lower": 1.1,
    "slips": 1.0,
    "slips below": 1.2,
    "falls below": 1.2,
    "fell": 1.0,
    "pulls back": 1.0,
    "pullback": 1.0,
    "reduced demand": 1.2,
    "weigh": 0.9,
    "weighs": 0.9,
    "profit-taking": 0.9,
    "profit taking": 0.9,
    "rising treasury yields": 1.1,
    "rising yields": 1.0,
    "stronger oil prices": 0.8,
    "higher oil prices": 0.8,
    "lock in gains": 0.7,
    "declines": 0.7,
    "decline": 0.7,
    "falling": 0.7,
    "falls": 0.7,
    "drop": 0.7,
    "drops": 0.7,
}

NEGATION_TERMS = {
    "not",
    "no",
    "never",
    "without",
    "fails",
    "failed",
    "misses",
    "missed",
}

CATEGORY_RULES: dict[str, set[str]] = {
    "公司/矿业基本面": {
        "earnings",
        "production",
        "cash flow",
        "cash generation",
        "revenue",
        "exploration",
        "drilling",
        "mine",
        "mineral",
        "resource",
        "development",
        "results",
        "guidance",
        "reserves",
    },
    "货币政策/利率": {
        "fed",
        "federal reserve",
        "rate cut",
        "rate hike",
        "interest rate",
        "cpi",
        "inflation",
        "central bank",
        "treasury",
        "yield",
        "dollar",
    },
    "地缘政治/避险": {
        "war",
        "sanction",
        "tariff",
        "conflict",
        "geopolitical",
        "safe haven",
        "middle east",
        "russia",
        "ukraine",
        "israel",
        "china",
    },
    "供需基本面": {
        "demand",
        "supply",
        "inventory",
        "deficit",
        "surplus",
        "output",
        "imports",
        "exports",
        "stockpiles",
    },
    "市场情绪/价格": {
        "rally",
        "selloff",
        "record",
        "bull",
        "bear",
        "price",
        "volatility",
        "market",
        "outlook",
    },
}

CATEGORY_IMPACT = {
    "公司/矿业基本面": "属于公司或矿业项目层面的信息，对现货贵金属价格的直接影响通常有限，更多反映个股和行业经营情况。",
    "货币政策/利率": "货币政策与利率预期是贵金属的核心驱动之一；宽松、降息或弱美元通常利多，紧缩、加息或强美元通常利空。",
    "地缘政治/避险": "地缘冲突和避险情绪可能推升黄金、白银等避险资产，但需要结合事件持续性和市场实际反应判断。",
    "供需基本面": "供需变化会直接作用于金属价格；供应紧张或需求走强偏利多，供应增加或需求疲弱偏利空。",
    "市场情绪/价格": "主要反映价格趋势与短期市场情绪，通常需要结合技术面和资金面做进一步确认。",
}


def _lower(text: str) -> str:
    return text.lower()


def _clean_text(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _extract_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        return str(value.get("content") or value.get("title") or value.get("text") or "")
    return str(value)


def _publisher(raw: dict[str, Any]) -> str:
    publisher = raw.get("publisher")
    if isinstance(publisher, dict):
        return str(publisher.get("name") or publisher.get("content") or "")
    return str(publisher or "Yahoo Finance")


def _link(raw: dict[str, Any]) -> str:
    link = raw.get("link")
    if isinstance(link, dict):
        link = link.get("url") or link.get("content")
    if isinstance(link, str) and link.startswith("//"):
        return "https:" + link
    return str(link or "")


def _published_at(raw: dict[str, Any]) -> str:
    timestamp = raw.get("providerPublishTime") or raw.get("publishTime")
    if not timestamp:
        return datetime.now(timezone.utc).isoformat()
    try:
        return datetime.fromtimestamp(float(timestamp), tz=timezone.utc).isoformat()
    except (TypeError, ValueError, OSError):
        return datetime.now(timezone.utc).isoformat()


def _title(raw: dict[str, Any]) -> str:
    title = raw.get("title")
    if isinstance(title, dict):
        return _extract_text(title)
    return _extract_text(title)


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.replace("\n", " "))
    return [part.strip() for part in parts if len(part.strip()) > 20]


def _sentiment_signal(text: str) -> tuple[float, float, list[str]]:
    sentences = _split_sentences(text)
    if not sentences:
        sentences = [text]

    positive = 0.0
    negative = 0.0
    evidence: list[str] = []

    for sentence in sentences:
        lowered = sentence.lower()
        for phrase, weight in POSITIVE_PHRASES.items():
            if phrase not in lowered:
                continue
            prefix = lowered.split(phrase, 1)[0]
            negated = any(term in prefix.split()[-4:] for term in NEGATION_TERMS)
            if negated:
                negative += weight
                evidence.append(f"-{phrase}")
            else:
                positive += weight
                evidence.append(phrase)
        for phrase, weight in NEGATIVE_PHRASES.items():
            if phrase not in lowered:
                continue
            prefix = lowered.split(phrase, 1)[0]
            negated = any(term in prefix.split()[-4:] for term in NEGATION_TERMS)
            if negated:
                positive += weight
                evidence.append(f"+{phrase}")
            else:
                negative += weight
                evidence.append(phrase)

    total = positive + negative
    if total == 0:
        return 0.0, 0.0, []
    net = positive - negative
    sentiment = math.tanh(net / (total * 0.6))
    confidence = min(1.0, total / 3.0)
    unique_evidence = sorted(set(evidence))
    return float(sentiment), float(confidence), unique_evidence


def _detect_category(text: str) -> str:
    lowered = _lower(text)
    best_category = "市场情绪/价格"
    best_score = -1
    for category, keywords in CATEGORY_RULES.items():
        score = sum(1 for keyword in keywords if keyword in lowered)
        if score > best_score:
            best_score = score
            best_category = category
    return best_category


def _stance(sentiment: float) -> str:
    if sentiment > 0.15:
        return "偏正面"
    if sentiment < -0.15:
        return "偏负面"
    return "方向不明确"


def _relevance(raw: dict[str, Any], category: str) -> str:
    related = raw.get("relatedTickers") or []
    if category == "公司/矿业基本面":
        return "中低"
    if category in {"货币政策/利率", "供需基本面"}:
        return "高"
    if category == "地缘政治/避险":
        return "中高"
    if related:
        return "中"
    return "中"


def _extract_key_sentence(body: str) -> str:
    sentences = _split_sentences(body)
    stop_words = {
        "cookie",
        "privacy",
        "newsletter",
        "subscribe",
        "advertising",
        "terms of use",
        "accessibility",
        "yahoo",
    }
    for sentence in sentences:
        lowered = sentence.lower()
        if any(word in lowered for word in stop_words):
            continue
        if len(sentence) >= 45:
            return sentence[:500]
    return sentences[0][:500] if sentences else ""


def _fetch_article_text(link: str, timeout: int = 12) -> str:
    if not link:
        return ""
    try:
        response = requests.get(
            link,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/126.0 Safari/537.36"
                )
            },
            timeout=timeout,
        )
        response.raise_for_status()
    except Exception:
        return ""

    soup = BeautifulSoup(response.text, "html.parser")
    article_body = ""
    for script in soup.select('script[type="application/ld+json"]'):
        try:
            data = json.loads(script.string or "")
            items = data if isinstance(data, list) else [data]
            for item in items:
                article_body = item.get("articleBody") or item.get("description") or article_body
                if article_body:
                    break
        except Exception:
            continue
        if article_body:
            break

    if not article_body:
        meta = soup.find("meta", attrs={"property": "og:description"}) or soup.find(
            "meta", attrs={"name": "description"}
        )
        article_body = meta.get("content", "") if meta else ""

    if not article_body or len(article_body) < 80:
        paragraphs = soup.select("article p, .caas-body p")
        article_body = " ".join(p.get_text(" ", strip=True) for p in paragraphs)

    return _clean_text(article_body)[:6000]


def analyse_title(title: str) -> tuple[float, str, list[str]]:
    sentiment, _confidence, evidence = _sentiment_signal(title)
    category = _detect_category(title)
    stance = _stance(sentiment)
    evidence_text = "、".join(evidence[:6]) if evidence else "无明确方向证据"
    summary = (
        f"主题：{category}；情绪：{stance}。"
        f"检测到的关键信号：{evidence_text}。"
    )
    return sentiment, summary, evidence


def analyse_article(title: str, body: str) -> dict[str, Any]:
    combined = f"{title}. {title}. {body}"
    sentiment, confidence, evidence = _sentiment_signal(combined)
    category = _detect_category(combined)
    key_sentence = _extract_key_sentence(body)
    return {
        "sentiment": sentiment,
        "confidence": confidence,
        "evidence": evidence,
        "category": category,
        "key_sentence": key_sentence,
    }


def _analysis_summary(
    title: str,
    body: str,
    category: str,
    sentiment: float,
    key_sentence: str,
) -> str:
    stance = _stance(sentiment)
    impact = CATEGORY_IMPACT.get(category, "")
    if body:
        article_point = key_sentence or "未提取到可用正文要点"
        return (
            f"主题：{category}；情绪：{stance}。{impact}"
            f" 标题：{title}。正文要点：{article_point}"
        )
    return f"主题：{category}；情绪：{stance}。{impact} 标题：{title}。"


def normalize_news_item(raw: dict[str, Any], symbol: str) -> dict[str, Any]:
    title = _title(raw)
    link = _link(raw)
    if not title or not link:
        return {}
    sentiment, summary_zh, _ = analyse_title(title)
    category = _detect_category(title)
    return {
        "symbol": symbol,
        "published_at": _published_at(raw),
        "title": title,
        "publisher": _publisher(raw),
        "link": link,
        "summary_zh": summary_zh,
        "topic_zh": category,
        "title_zh": title,
        "key_points": f"标题：{title}",
        "relevance": _relevance(raw, category),
        "sentiment": sentiment,
    }


def analyze_news_item(raw: dict[str, Any], symbol: str) -> dict[str, Any]:
    title = _title(raw)
    link = _link(raw)
    if not title or not link:
        return {}

    body = _fetch_article_text(link)
    if body:
        analysis = analyse_article(title, body)
        sentiment = analysis["sentiment"]
        category = analysis["category"]
        key_sentence = analysis["key_sentence"]
        summary_zh = _analysis_summary(title, body, category, sentiment, key_sentence)
        key_points = key_sentence or f"标题：{title}"
    else:
        sentiment, _, _ = analyse_title(title)
        category = _detect_category(title)
        summary_zh = _analysis_summary(title, "", category, sentiment, "")
        key_points = f"标题：{title}"

    return {
        "symbol": symbol,
        "published_at": _published_at(raw),
        "title": title,
        "publisher": _publisher(raw),
        "link": link,
        "summary_zh": summary_zh,
        "topic_zh": category,
        "title_zh": title,
        "key_points": key_points,
        "relevance": _relevance(raw, category),
        "sentiment": sentiment,
    }


def news_score_for_items(items: list[dict[str, Any]]) -> float:
    if not items:
        return 50.0
    sentiments = [float(item.get("sentiment") or 0.0) for item in items]
    average = sum(sentiments) / len(sentiments)
    return max(0.0, min(100.0, 50.0 + average * 50.0))
