"""Writer agent — synthesises extracted data and critic feedback into a Markdown report."""
from __future__ import annotations

import json

from langfuse import Langfuse
from langfuse.types import TraceContext

from sec_analyzer.agents.critic import Critique
from sec_analyzer.agents.extractor import ExtractedData
from sec_analyzer.gateway import LLMGateway, Message, get_gateway

_SYSTEM_PROMPT = """\
You are a senior investment analyst writing a professional research report.
Your audience is a sophisticated investor who wants clarity and insight, not padding.
Write in Markdown. Be direct. Use numbers wherever available. No disclaimers or filler."""

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


def _build_user_prompt(data: ExtractedData, critique: Critique) -> str:
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
    return f"""Ticker: {data.ticker}
Fiscal year end: {fiscal_year}
SEC filing date: {data.filed_date}
Data confidence: {confidence_pct} — {critique.summary}

## Extracted Financials
{metrics_table}

## Data Quality Issues
{issues_block}

## Filing Excerpts
{sections_text}

---

Write a structured Markdown investment report for {data.ticker} using the data above.
Include these sections:
1. **Executive Summary** — 2-3 sentences on the company and filing period
2. **Financial Highlights** — key metrics with brief commentary
3. **Business Overview** — what the company does, main segments
4. **Key Risks** — top risks from the filing
5. **Data Confidence Note** — flag any quality issues if confidence < 80%

Keep the total length to roughly 500-700 words."""


async def write_report(
    data: ExtractedData,
    critique: Critique,
    langfuse: Langfuse,
    trace_context: TraceContext,
    *,
    gateway: LLMGateway | None = None,
) -> str:
    """Produce a Markdown investment report from extraction + critique."""
    gw = gateway or get_gateway()

    with langfuse.start_as_current_observation(
        name="writer",
        as_type="span",
        trace_context=trace_context,
        input={"ticker": data.ticker, "confidence": critique.confidence},
    ) as span:

        user_prompt = _build_user_prompt(data, critique)
        messages: list[Message] = [{"role": "user", "content": user_prompt}]
        response = await gw.complete(messages, system=_SYSTEM_PROMPT, max_tokens=2048)

        report = response.text.strip()

        # Ensure the report has a top-level heading even if the LLM omitted it
        if not report.startswith("#"):
            fiscal_year = data.metrics.get("fiscal_year_end") or data.filed_date
            report = f"# {data.ticker} — 10-K Analysis (FY {fiscal_year})\n\n{report}"

        span.update(output={
            "report_length": len(report),
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens,
        })

    return report
