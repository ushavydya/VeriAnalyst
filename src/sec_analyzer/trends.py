"""Multi-year trend analysis — runs extract on each historical 10-K filing."""
from __future__ import annotations

import uuid
from dataclasses import dataclass

from langfuse import Langfuse
from langfuse.types import TraceContext

from sec_analyzer.agents.extractor import ExtractedData, extract
from sec_analyzer.agents.retriever import SECRetriever
from sec_analyzer.cache.sqlite_cache import SQLiteCache


@dataclass
class YearlyMetrics:
    filed_date: str
    fiscal_year_end: str | None
    metrics: dict[str, object]


async def fetch_trends(
    ticker: str,
    *,
    years: int = 5,
    cache: SQLiteCache,
    langfuse: Langfuse,
    gateway=None,
) -> list[YearlyMetrics]:
    """Fetch and extract metrics for the last *years* 10-K filings."""
    retriever = SECRetriever(cache=cache, langfuse=langfuse)
    trace_id = uuid.uuid4().hex
    tc = TraceContext(trace_id=trace_id)

    filings = await retriever.fetch_10k_history(ticker, tc, years=years)

    results: list[YearlyMetrics] = []
    for filing in filings:
        data: ExtractedData = await extract(
            ticker=ticker,
            filed_date=filing.filed_date,
            document_text=filing.read_text(),
            langfuse=langfuse,
            trace_context=tc,
            gateway=gateway,
        )
        results.append(YearlyMetrics(
            filed_date=filing.filed_date,
            fiscal_year_end=data.metrics.get("fiscal_year_end"),
            metrics=data.metrics,
        ))

    langfuse.flush()
    return results
