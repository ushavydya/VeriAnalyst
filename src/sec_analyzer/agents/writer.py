"""Writer agent — synthesises extracted data, critic feedback, news, and market data."""
from __future__ import annotations

import json

from langfuse import Langfuse
from langfuse.types import TraceContext

from sec_analyzer.agents.critic import Critique
from sec_analyzer.agents.extractor import ExtractedData
from sec_analyzer.agents.market_agent import MarketSummary
from sec_analyzer.agents.news_agent import NewsSummary
from sec_analyzer.gateway import LLMGateway, Message, get_gateway

_SYSTEM_PROMPT = """\
You are a senior investment analyst writing a professional research report.
Your audience is a sophisticated investor who wants clarity and insight, not padding.
Write in Markdown. Be direct. Use numbers wherever available. No disclaimers or filler.

CRITICAL: Base every claim strictly on the extracted metrics, data quality issues, filing
excerpts, and market/news data provided in the user message. Do NOT invent product names,
events, competitor comparisons, industry benchmarks, analyst targets, or any figures that
are not explicitly present in the input data. If information is not in the provided data,
omit it rather than speculate."""

_MAX_SECTION_CHARS = 3_000  # chars per section excerpt included in the prompt


def _format_metrics(metrics: dict[str, object]) -> str:
    """Return a Markdown table of extracted metrics."""
    labels = {
        "revenue": "Revenue (M USD)",
        "gross_profit": "Gross Profit (M USD)",
        "operating_income": "Operating Income (M USD)",
        "net_income": "Net Income (M USD)",
        "eps_basic": "EPS Basic (USD)",
        "eps_diluted": "EPS Diluted (USD)",
        "total_assets": "Total Assets (M USD)",
        "total_liabilities": "Total Liabilities (M USD)",
        "total_equity": "Total Equity (M USD)",
        "cash_and_equivalents": "Cash & Equivalents (M USD)",
        "fiscal_year_end": "Fiscal Year End",
    }
    rows = []
    for key, label in labels.items():
        if key in metrics:
            val = metrics[key]
            if isinstance(val, float):
                rows.append(f"| {label} | {val:,.1f} |")
            else:
                rows.append(f"| {label} | {val} |")
    if not rows:
        return "_No metrics extracted._"
    return "| Metric | Value |\n|---|---|\n" + "\n".join(rows)


def _format_news(news: NewsSummary | None) -> str:
    if news is None or not news.articles:
        return "_No recent news available._"
    lines = [f"Sentiment: **{news.sentiment_label}**"
             + (f" ({news.sentiment_score:+.2f})" if news.sentiment_score is not None else "")]
    if news.narrative:
        lines.append(f"\n{news.narrative}")
    lines.append("")
    for a in news.articles[:5]:
        lines.append(f"- [{a.headline}]({a.url}) — {a.source}")
    return "\n".join(lines)


def _format_market(market: MarketSummary | None) -> str:
    if market is None or market.quote is None:
        return "_No market data available._"
    lines = []
    if market.quote:
        lines.append(f"| Current Price | ${market.quote.price:,.2f} |")
        lines.append(f"| Intraday Change | {market.quote.change_pct:+.2f}% |")
    if market.ratios:
        if market.ratios.pe_ratio is not None:
            lines.append(f"| P/E Ratio | {market.ratios.pe_ratio:.1f}x |")
        if market.ratios.week_52_high is not None:
            lines.append(f"| 52-Week High | ${market.ratios.week_52_high:,.2f} |")
        if market.ratios.week_52_low is not None:
            lines.append(f"| 52-Week Low | ${market.ratios.week_52_low:,.2f} |")
        if market.ratios.beta is not None:
            lines.append(f"| Beta | {market.ratios.beta:.2f} |")
    pct_vs_high = market.price_vs_52w_high_pct()
    if pct_vs_high is not None:
        lines.append(f"| vs 52W High | {pct_vs_high:+.1f}% |")
    if not lines:
        return "_No market data available._"
    return "| Metric | Value |\n|---|---|\n" + "\n".join(lines)


def _build_user_prompt(
    data: ExtractedData,
    critique: Critique,
    news: NewsSummary | None = None,
    market: MarketSummary | None = None,
) -> str:
    metrics_table = _format_metrics(data.metrics)

    sections_text = ""
    for name in ("business", "mda", "risk_factors"):
        body = data.sections.get(name, "")[:_MAX_SECTION_CHARS]
        if body:
            sections_text += f"\n\n### {name.upper()}\n{body}"

    issues_block = (
        "\n".join(f"- {i}" for i in critique.issues)
        if critique.issues
        else "None identified."
    )

    confidence_pct = f"{critique.confidence:.0%}"

    fiscal_year = data.metrics.get("fiscal_year_end") or data.filed_date
    # Derive FY label from the year in the fiscal_year_end date (e.g. "2025-09-27" → "FY2025")
    fiscal_year_label = f"FY{str(fiscal_year)[:4]}" if fiscal_year else "FY unknown"
    has_intelligence = news is not None or market is not None
    intelligence_section = ""
    if has_intelligence:
        intelligence_section = f"""
## Current Market Data
{_format_market(market)}

## Recent News & Sentiment
{_format_news(news)}
"""

    sections_note = """
Include these sections:
1. **Executive Summary** — 2-3 sentences on the company and filing period
2. **Financial Highlights** — key metrics with brief commentary
3. **Business Overview** — what the company does, main segments
4. **Key Risks** — top risks from the filing"""

    if has_intelligence:
        sections_note += """
5. **Investment Intelligence** — synthesise current price action, valuation ratios, and recent news sentiment into a forward-looking view. Note any divergence between the annual filing fundamentals and the current market picture.
6. **Data Confidence Note** — flag any quality issues if confidence < 80%"""
    else:
        sections_note += """
5. **Data Confidence Note** — flag any quality issues if confidence < 80%"""

    word_count = "600-800" if has_intelligence else "500-700"

    return f"""Ticker: {data.ticker}
Fiscal year: {fiscal_year_label} (ending {fiscal_year})
SEC filing date: {data.filed_date}
Data confidence: {confidence_pct} — {critique.summary}

## Extracted Financials
{metrics_table}

## Data Quality Issues
{issues_block}
{intelligence_section}
## Filing Excerpts
{sections_text}

---

Write a structured Markdown investment report for {data.ticker} using the data above.
{sections_note}

Keep the total length to roughly {word_count} words."""


async def write_report(
    data: ExtractedData,
    critique: Critique,
    langfuse: Langfuse,
    trace_context: TraceContext,
    *,
    gateway: LLMGateway | None = None,
    news: NewsSummary | None = None,
    market: MarketSummary | None = None,
) -> str:
    """Produce a Markdown investment report from extraction + critique + intelligence."""
    gw = gateway or get_gateway()

    with langfuse.start_as_current_observation(
        name="writer",
        as_type="span",
        trace_context=trace_context,
        input={"ticker": data.ticker, "confidence": critique.confidence,
               "has_news": news is not None, "has_market": market is not None},
    ) as span:

        user_prompt = _build_user_prompt(data, critique, news=news, market=market)
        messages: list[Message] = [{"role": "user", "content": user_prompt}]
        response = await gw.complete(messages, system=_SYSTEM_PROMPT, max_tokens=2048)

        report = response.text.strip()

        # Ensure the report has a top-level heading even if the LLM omitted it
        if not report.startswith("#"):
            fy_end = data.metrics.get("fiscal_year_end") or data.filed_date
            fy_label = f"FY{str(fy_end)[:4]}" if fy_end else "FY unknown"
            report = f"# {data.ticker} — 10-K Analysis ({fy_label})\n\n{report}"

        span.update(output={
            "report_length": len(report),
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens,
        })

    return report
