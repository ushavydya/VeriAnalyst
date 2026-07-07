"""Comparison agent — analyses two or more companies side-by-side.

Runs extraction for each ticker in parallel, then uses the LLM to produce
a structured comparative report highlighting relative strengths, weaknesses,
and key differences.
"""
from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass

from langfuse import Langfuse
from langfuse.types import TraceContext

from sec_analyzer.agents.extractor import ExtractedData, extract
from sec_analyzer.agents.retriever import SECRetriever
from sec_analyzer.cache.sqlite_cache import SQLiteCache
from sec_analyzer.gateway import LLMGateway, Message, get_gateway

_SYSTEM_PROMPT = """\
You are a senior investment analyst writing a comparative research report.
Your audience is a sophisticated investor. Be direct, use numbers, highlight
meaningful differences. Write in Markdown. No disclaimers or filler."""

_METRIC_LABELS: dict[str, str] = {
    "revenue":             "Revenue (M USD)",
    "gross_profit":        "Gross Profit (M USD)",
    "operating_income":    "Operating Income (M USD)",
    "net_income":          "Net Income (M USD)",
    "eps_diluted":         "Diluted EPS (USD)",
    "total_assets":        "Total Assets (M USD)",
    "total_equity":        "Total Equity (M USD)",
    "cash_and_equivalents":"Cash & Equivalents (M USD)",
}


@dataclass
class ComparisonResult:
    tickers: list[str]
    extractions: list[ExtractedData]
    report: str


# ── Prompt builders ───────────────────────────────────────────────────────────

def _metrics_table(extractions: list[ExtractedData]) -> str:
    """Render a Markdown table with one column per ticker."""
    tickers = [e.ticker for e in extractions]
    header = "| Metric | " + " | ".join(tickers) + " |"
    divider = "|---|" + "---|" * len(tickers)
    rows = [header, divider]

    for key, label in _METRIC_LABELS.items():
        cells = []
        for e in extractions:
            val = e.metrics.get(key)
            if val is None:
                cells.append("—")
            elif key == "eps_diluted":
                cells.append(f"${val:.2f}")
            else:
                cells.append(f"{val:,.0f}")
        rows.append(f"| {label} | " + " | ".join(cells) + " |")

    return "\n".join(rows)


def _build_prompt(extractions: list[ExtractedData]) -> str:
    table = _metrics_table(extractions)

    summaries = ""
    for e in extractions:
        fy = e.metrics.get("fiscal_year_end") or e.filed_date
        summary = e.sections.get("summary", "No summary available.")[:1_500]
        summaries += f"\n\n### {e.ticker} (FY {fy})\n{summary}"

    tickers_str = " vs ".join(e.ticker for e in extractions)

    return f"""{tickers_str} — Comparative Analysis

## Verified Financial Metrics
{table}

## Company Summaries
{summaries}

---

Write a structured Markdown comparative report. Include:
1. **Executive Summary** — 2-3 sentences on who these companies are and why comparing them matters
2. **Financial Comparison** — revenue scale, profitability margins, EPS, balance sheet strength; call out the leader in each dimension with numbers
3. **Business Model Differences** — what each company does differently and how it affects the financials
4. **Risk Comparison** — key risks that differ between the companies
5. **Verdict** — which company looks stronger on fundamentals and why, in 2-3 sentences

Keep the total length to roughly 600-800 words."""


# ── Public API ────────────────────────────────────────────────────────────────

async def compare(
    tickers: list[str],
    cache: SQLiteCache,
    langfuse: Langfuse,
    *,
    gateway: LLMGateway | None = None,
) -> ComparisonResult:
    """Fetch, extract, and compare *tickers* in parallel."""
    gw = gateway or get_gateway()
    retriever = SECRetriever(cache=cache, langfuse=langfuse)
    trace_id = uuid.uuid4().hex
    tc = TraceContext(trace_id=trace_id)

    with langfuse.start_as_current_observation(
        name="comparison",
        as_type="span",
        trace_context=tc,
        input={"tickers": tickers},
    ) as span:

        # Fetch all filings in parallel
        filings = await asyncio.gather(
            *[retriever.fetch_10k(ticker.upper(), tc) for ticker in tickers]
        )

        # Extract all in parallel
        extractions: list[ExtractedData] = await asyncio.gather(
            *[
                extract(
                    ticker=f.ticker,
                    filed_date=f.filed_date,
                    document_text=f.read_text(),
                    langfuse=langfuse,
                    trace_context=tc,
                    xbrl_metrics=f.xbrl_metrics,
                    gateway=gw,
                )
                for f in filings
            ]
        )

        # LLM comparative report
        messages: list[Message] = [{"role": "user", "content": _build_prompt(extractions)}]
        response = await gw.complete(messages, system=_SYSTEM_PROMPT, max_tokens=2048)
        report = response.text.strip()
        if not report.startswith("#"):
            tickers_str = " vs ".join(e.ticker for e in extractions)
            report = f"# {tickers_str} — Comparative Analysis\n\n{report}"

        span.update(output={
            "tickers": tickers,
            "report_length": len(report),
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens,
        })

    langfuse.flush()
    return ComparisonResult(
        tickers=[e.ticker for e in extractions],
        extractions=extractions,
        report=report,
    )
