from __future__ import annotations

import base64
import re
from datetime import date, datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from jinja2 import Environment

from .indicators import add_indicators


def _safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_") or "symbol"


def _fmt_number(value: Any, digits: int = 2) -> str:
    if value is None:
        return "—"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    if number != number:
        return "—"
    return f"{number:.{digits}f}"


def _fmt_percent(value: Any, digits: int = 2) -> str:
    text = _fmt_number(value, digits=digits)
    return "—" if text == "—" else f"{text}%"


def _short_datetime(value: str | None) -> str:
    if not value:
        return "—"
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).strftime("%Y-%m-%d %H:%M")
    except ValueError:
        return value[:16]


def _sentiment_label(sentiment: Any) -> str:
    try:
        value = float(sentiment)
    except (TypeError, ValueError):
        return "中性"
    if value > 0.1:
        return "偏多"
    if value < -0.1:
        return "偏空"
    return "中性"


def create_chart(symbol: str, df: Any, output_path: Path) -> None:
    if df is None or df.empty:
        return
    frame = add_indicators(df)
    if frame.empty:
        return
    frame = frame.tail(250)

    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(8.0, 3.2))
    ax.plot(frame.index, frame["close"], label="收盘价", linewidth=1.6, color="#1f3b57")
    for column, label, color in [
        ("sma20", "SMA20", "#e0a100"),
        ("sma50", "SMA50", "#c24a3f"),
        ("sma200", "SMA200", "#4a7c59"),
    ]:
        if column in frame.columns:
            ax.plot(frame.index, frame[column], label=label, linewidth=1.0, color=color, alpha=0.85)
    ax.set_title(symbol, fontsize=12)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=110)
    plt.close(fig)


def _build_rows(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result in results:
        metrics = result.get("metrics") or {}
        signals = metrics.get("signals") or {}
        scores = metrics.get("scores") or {}
        confidences = metrics.get("confidences") or {}
        rows.append(
            {
                "symbol": result["symbol"],
                "symbol_name": result.get("symbol_name", result["symbol"]),
                "display_label": (
                    f"{result.get('symbol_name', result['symbol'])} {result['symbol']}"
                ),
                "group": result["group_label"],
                "has_data": result["has_data"],
                "latest_price": _fmt_number(metrics.get("close")),
                "chg_1d": _fmt_percent(metrics.get("ret1")),
                "chg_5d": _fmt_percent(metrics.get("ret5")),
                "chg_20d": _fmt_percent(metrics.get("ret20")),
                "short_signal": signals.get("short", "数据不足"),
                "short_score": _fmt_number(scores.get("short"), 1),
                "short_conf": _fmt_number(confidences.get("short"), 1),
                "medium_signal": signals.get("medium", "数据不足"),
                "medium_score": _fmt_number(scores.get("medium"), 1),
                "medium_conf": _fmt_number(confidences.get("medium"), 1),
                "long_signal": signals.get("long", "数据不足"),
                "long_score": _fmt_number(scores.get("long"), 1),
                "long_conf": _fmt_number(confidences.get("long"), 1),
                "rsi14": _fmt_number(metrics.get("rsi14"), 1),
                "macd_hist": _fmt_number(metrics.get("macd_hist"), 4),
                "atr_pct": _fmt_percent(metrics.get("atr_pct"), 2),
                "sma20": _fmt_number(metrics.get("sma20")),
                "sma50": _fmt_number(metrics.get("sma50")),
                "sma200": _fmt_number(metrics.get("sma200")),
                "risk_high": bool(metrics.get("risk_high")),
                "alpha_factor_score": _fmt_number(metrics.get("alpha_factor_score"), 1),
                "alpha_factor_signal": metrics.get("alpha_factor_signal", "—"),
            }
        )
    return rows


HTML_TEMPLATE = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{{ report_title }}</title>
  <style>
    :root { color-scheme: light; --ink: #18212f; --muted: #64748b; --line: #dbe3ec; --card: #f8fafc; --accent: #1f3b57; }
    * { box-sizing: border-box; }
    body { margin: 0; padding: 32px 18px 60px; background: #f4f6f8; color: var(--ink); font-family: "Microsoft YaHei", "Segoe UI", Arial, sans-serif; line-height: 1.55; }
    .wrap { max-width: 1180px; margin: 0 auto; }
    header { background: var(--accent); color: #fff; border-radius: 14px; padding: 22px 24px; }
    header h1 { margin: 0 0 8px; font-size: 25px; }
    header .meta { color: #dbe7f5; font-size: 14px; }
    .notice { margin: 14px 0; padding: 12px 14px; border-left: 4px solid #b3402f; background: #fff4f2; border-radius: 8px; color: #7a2c20; font-weight: 600; }
    section { background: #fff; border: 1px solid var(--line); border-radius: 14px; padding: 18px 20px; margin-top: 18px; }
    h2 { margin: 0 0 12px; font-size: 18px; border-bottom: 1px solid var(--line); padding-bottom: 8px; }
    table { width: 100%; border-collapse: collapse; font-size: 13px; }
    th, td { padding: 8px 7px; border-bottom: 1px solid #e6ebf1; text-align: left; vertical-align: top; }
    th { background: #f1f5f9; color: #334155; font-weight: 700; white-space: nowrap; }
    .buy { color: #b33a2b; font-weight: 700; }
    .hold { color: #b47620; font-weight: 700; }
    .neutral { color: #64748b; font-weight: 700; }
    .sell { color: #1f7a4d; font-weight: 700; }
    .risk { color: #b3402f; font-weight: 700; }
    .chart { width: 100%; height: auto; margin: 12px 0; border: 1px solid var(--line); border-radius: 10px; }
    ul { padding-left: 20px; }
    li { margin-bottom: 8px; }
    a { color: #1f5c8b; text-decoration: none; }
    a:hover { text-decoration: underline; }
    @media (max-width: 720px) { table { font-size: 12px; } th, td { padding: 6px 4px; } }
  </style>
</head>
<body>
  <div class="wrap">
    <header>
      <h1>贵金属每日趋势自动报告</h1>
      <div class="meta">报告日期：{{ report_date }} · 数据时点：{{ data_date }} · 生成时间：{{ generated_at }} · 时区：Asia/Shanghai</div>
    </header>
    <div class="notice">仅供技术研究，不构成投资建议。自动买卖信号基于历史价格、技术指标与新闻关键词，可能出现误判。</div>

    <section>
      <h2>综合结论</h2>
      <table>
        <thead><tr><th>标的</th><th>分类</th><th>最新价</th><th>1日</th><th>5日</th><th>20日</th><th>短线</th><th>中线</th><th>长线</th><th>风险</th><th>因子倾向</th></tr></thead>
        <tbody>
          {% for row in rows %}
          <tr>
            <td><strong>{{ row.display_label }}</strong></td>
            <td>{{ row.group }}</td>
            <td>{{ row.latest_price }}</td>
            <td>{{ row.chg_1d }}</td>
            <td>{{ row.chg_5d }}</td>
            <td>{{ row.chg_20d }}</td>
            <td class="{{ signal_class(row.short_signal) }}">{{ row.short_signal }}（{{ row.short_score }}）</td>
            <td class="{{ signal_class(row.medium_signal) }}">{{ row.medium_signal }}（{{ row.medium_score }}）</td>
            <td class="{{ signal_class(row.long_signal) }}">{{ row.long_signal }}（{{ row.long_score }}）</td>
            <td>{% if row.risk_high %}<span class="risk">高风险</span>{% else %}正常{% endif %}</td>
            <td>{{ row.alpha_factor_signal }}（{{ row.alpha_factor_score }}）</td>
          </tr>
          {% endfor %}
        </tbody>
      </table>
    </section>

    {% for row in rows %}
    <section>
      <h2>{{ row.display_label }} · {{ row.group }}</h2>
      <p>
        最新价 {{ row.latest_price }}，20日涨幅 {{ row.chg_20d }}，
        短线 {{ row.short_signal }}（{{ row.short_score }}），
        中线 {{ row.medium_signal }}（{{ row.medium_score }}），
        长线 {{ row.long_signal }}（{{ row.long_score }}），
        因子倾向 {{ row.alpha_factor_signal }}（{{ row.alpha_factor_score }}）。
      </p>
      <table>
        <tr><th>RSI14</th><th>MACD柱</th><th>ATR%</th><th>SMA20</th><th>SMA50</th><th>SMA200</th></tr>
        <tr><td>{{ row.rsi14 }}</td><td>{{ row.macd_hist }}</td><td>{{ row.atr_pct }}</td><td>{{ row.sma20 }}</td><td>{{ row.sma50 }}</td><td>{{ row.sma200 }}</td></tr>
      </table>
      {% if chart_images.get(row.symbol) %}
        <img class="chart" src="{{ chart_images.get(row.symbol) }}" alt="{{ row.symbol }} chart">
      {% endif %}
      {% set symbol_news = news_by_symbol.get(row.symbol, []) %}
      {% if symbol_news %}
      <h2 style="margin-top:14px;">新闻摘要</h2>
      <ul>
        {% for item in symbol_news %}
        <li>
          <strong>[{{ item.sentiment_label }} / {{ item.topic_zh }} / 相关度：{{ item.relevance }}]</strong> {{ item.summary_zh }} <a href="{{ item.link }}" target="_blank">{{ item.title }}</a><br>
          <small>{{ item.publisher }} · {{ item.published_at }}</small>
        </li>
        {% endfor %}
      </ul>
      {% endif %}
    </section>
    {% endfor %}

    <section>
      <h2>方法与说明</h2>
      <p>数据来自 Yahoo Finance。短线/中线/长线分别使用约 20/60/250 个交易日窗口，综合趋势、动量、RSI、波动率和新闻/宏观因子生成 0–100 分。分数 ≥70 为买入/增持，55–69 为持有/偏多，45–54 为观望，30–44 为减仓/偏空，低于 30 为卖出/规避。高风险时信号自动降一档。AlphaGPT 风格因子由“特征 + 算子公式”生成，按历史 IC 和多空收益筛选，最新截面分位映射为因子偏多/中性/偏空。</p>
    </section>
  </div>
</body>
</html>
"""


MD_TEMPLATE = """# 贵金属每日趋势自动报告

> 报告日期：{{ report_date }}  
> 数据时点：{{ data_date }}  
> 生成时间：{{ generated_at }}  
> 时区：Asia/Shanghai

**仅供技术研究，不构成投资建议。**

## 综合结论

| 标的 | 分类 | 最新价 | 1日 | 5日 | 20日 | 短线 | 中线 | 长线 | 风险 | 因子倾向 |
|---|---:|---:|---:|---:|---|---|---|---|---|
{% for row in rows %}| {{ row.display_label }} | {{ row.group }} | {{ row.latest_price }} | {{ row.chg_1d }} | {{ row.chg_5d }} | {{ row.chg_20d }} | {{ row.short_signal }}（{{ row.short_score }}） | {{ row.medium_signal }}（{{ row.medium_score }}） | {{ row.long_signal }}（{{ row.long_score }}） | {% if row.risk_high %}高风险{% else %}正常{% endif %} | {{ row.alpha_factor_signal }}（{{ row.alpha_factor_score }}） |
{% endfor %}
{% for row in rows %}
## {{ row.display_label }} · {{ row.group }}

最新价 {{ row.latest_price }}，20日涨幅 {{ row.chg_20d }}。短线 {{ row.short_signal }}（{{ row.short_score }}），中线 {{ row.medium_signal }}（{{ row.medium_score }}），长线 {{ row.long_signal }}（{{ row.long_score }}），因子倾向 {{ row.alpha_factor_signal }}（{{ row.alpha_factor_score }}）。

| RSI14 | MACD柱 | ATR% | SMA20 | SMA50 | SMA200 |
|---:|---:|---:|---:|---:|---:|
| {{ row.rsi14 }} | {{ row.macd_hist }} | {{ row.atr_pct }} | {{ row.sma20 }} | {{ row.sma50 }} | {{ row.sma200 }} |

{% if chart_rel.get(row.symbol) %}![{{ row.symbol }}]({{ chart_rel.get(row.symbol) }})

{% endif %}{% set symbol_news = news_by_symbol.get(row.symbol, []) %}{% if symbol_news %}### 新闻摘要
{% for item in symbol_news %}- **[{{ item.sentiment_label }} / {{ item.topic_zh }} / 相关度：{{ item.relevance }}]** {{ item.summary_zh }} [{{ item.title }}]({{ item.link }})（{{ item.publisher }}，{{ item.published_at }}）
{% endfor %}{% endif %}
{% endfor %}
## 方法与说明

数据来自 Yahoo Finance。短线/中线/长线分别使用约 20/60/250 个交易日窗口，综合趋势、动量、RSI、波动率和新闻/宏观因子生成 0–100 分。分数 ≥70 为买入/增持，55–69 为持有/偏多，45–54 为观望，30–44 为减仓/偏空，低于 30 为卖出/规避。高风险时信号自动降一档。AlphaGPT 风格因子由“特征 + 算子公式”生成，按历史 IC 和多空收益筛选，最新截面分位映射为因子偏多/中性/偏空。
"""


def _signal_class(signal: str) -> str:
    if "买入" in signal:
        return "buy"
    if "持有" in signal:
        return "hold"
    if "卖出" in signal or "规避" in signal or "减仓" in signal:
        return "sell"
    return "neutral"


def _prepare_news(news_by_symbol: dict[str, list[dict[str, Any]]]) -> dict[str, list[dict[str, Any]]]:
    prepared: dict[str, list[dict[str, Any]]] = {}
    for symbol, items in news_by_symbol.items():
        prepared[symbol] = [
            {
                "title": item.get("title", ""),
                "title_zh": item.get("title_zh") or item.get("title", ""),
                "publisher": item.get("publisher", "Yahoo Finance"),
                "link": item.get("link", "#"),
                "published_at": _short_datetime(item.get("published_at")),
                "summary_zh": item.get("summary_zh") or "",
                "topic_zh": item.get("topic_zh") or "未分类",
                "key_points": item.get("key_points") or "",
                "relevance": item.get("relevance") or "中",
                "sentiment_label": _sentiment_label(item.get("sentiment")),
            }
            for item in items
        ]
    return prepared


def render_reports(
    results: list[dict[str, Any]],
    bars_by_symbol: dict[str, Any],
    news_by_symbol: dict[str, list[dict[str, Any]]],
    report_date: date,
    data_date: date,
    output_dir: Path,
    config: dict[str, Any],
) -> tuple[Path, Path]:
    env = Environment()
    env.globals["signal_class"] = _signal_class

    report_dir = output_dir / report_date.isoformat()
    charts_dir = report_dir / "charts"
    report_dir.mkdir(parents=True, exist_ok=True)
    charts_dir.mkdir(parents=True, exist_ok=True)

    rows = _build_rows(results)
    prepared_news = _prepare_news(news_by_symbol)
    chart_rel: dict[str, str] = {}
    chart_images: dict[str, str] = {}

    for result in results:
        symbol = result["symbol"]
        frame = bars_by_symbol.get(symbol)
        if result["has_data"] and frame is not None and not frame.empty:
            filename = f"{_safe_filename(symbol)}.png"
            path = charts_dir / filename
            create_chart(symbol, frame, path)
            if path.exists():
                chart_rel[symbol] = f"charts/{filename}"
                chart_images[symbol] = "data:image/png;base64," + base64.b64encode(
                    path.read_bytes()
                ).decode("ascii")

    context = {
        "report_title": f"贵金属每日趋势报告 {report_date.isoformat()}",
        "report_date": report_date.isoformat(),
        "data_date": data_date.isoformat(),
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "rows": rows,
        "news_by_symbol": prepared_news,
        "chart_rel": chart_rel,
        "chart_images": chart_images,
    }

    html_path = report_dir / f"metals-report-{report_date.isoformat()}.html"
    md_path = report_dir / f"metals-report-{report_date.isoformat()}.md"
    html_path.write_text(env.from_string(HTML_TEMPLATE).render(**context), encoding="utf-8")
    md_path.write_text(env.from_string(MD_TEMPLATE).render(**context), encoding="utf-8")
    return html_path, md_path
