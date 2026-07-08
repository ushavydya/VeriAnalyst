"""LangGraph pipeline: Retriever + Intelligence → Extractor → Critic → Writer.

Node execution order:
  retriever_node       — fetch 10-K filing
  intelligence_node    — fetch news + market data in parallel (runs after retriever)
  extractor_node       — extract financial metrics from filing text
  critic_node          — validate extracted metrics
  writer_node          — synthesise report from all sources

All nodes are instrumented with Langfuse spans grouped under one parent trace.
"""
from __future__ import annotations

import asyncio
import dataclasses
import json
import uuid
from typing import TypedDict

from langgraph.graph import END, StateGraph
from langfuse import Langfuse
from langfuse.types import TraceContext

from sec_analyzer.agents.critic import Critique, critique
from sec_analyzer.agents.extractor import ExtractedData, extract
from sec_analyzer.agents.market_agent import MarketSummary, fetch_market_data
from sec_analyzer.agents.news_agent import NewsSummary, fetch_news
from sec_analyzer.agents.retriever import SECRetriever
from sec_analyzer.agents.writer import write_report
from sec_analyzer.cache.sqlite_cache import SQLiteCache


class PipelineState(TypedDict):
    ticker: str
    trace_id: str
    # populated by retriever_node
    filing_content: str | None
    filing_cik: str | None
    filing_date: str | None
    xbrl_metrics: dict | None
    cache_hit: bool
    # populated by intelligence_node
    news_summary_json: str | None
    market_summary_json: str | None
    # populated by extractor / critic / writer
    extracted_data_json: str | None
    critique_json: str | None
    report: str | None
    error: str | None


def build_pipeline(
    cache: SQLiteCache,
    langfuse: Langfuse,
    *,
    enable_intelligence: bool = True,
) -> "CompiledGraph":
    """Construct and compile the LangGraph pipeline."""
    retriever = SECRetriever(cache=cache, langfuse=langfuse)

    # Lazily import providers only when intelligence is enabled so the pipeline
    # still works without FINNHUB_API_KEY when intelligence is disabled.
    if enable_intelligence:
        from sec_analyzer.providers import get_market_data_provider, get_news_provider
        try:
            _news_provider = get_news_provider()
            _market_provider = get_market_data_provider()
            _intel_available = True
        except (EnvironmentError, ValueError):
            _intel_available = False
    else:
        _intel_available = False

    # ── Node functions ────────────────────────────────────────────────────────

    async def retriever_node(state: PipelineState) -> dict:
        tc = TraceContext(trace_id=state["trace_id"])
        try:
            result = await retriever.fetch_10k(state["ticker"], tc)
            return {
                "filing_content": result.read_text(),
                "filing_cik": result.cik,
                "filing_date": result.filed_date,
                "xbrl_metrics": result.xbrl_metrics,
                "cache_hit": result.cache_hit,
                "error": None,
            }
        except Exception as exc:
            return {"error": str(exc)}

    async def intelligence_node(state: PipelineState) -> dict:
        if state.get("error") or not _intel_available:
            return {"news_summary_json": None, "market_summary_json": None}
        tc = TraceContext(trace_id=state["trace_id"])
        ticker = state["ticker"]

        news_result, market_result = await asyncio.gather(
            fetch_news(ticker, _news_provider, cache, langfuse, tc),
            fetch_market_data(
                ticker, _market_provider, cache, langfuse, tc,
                xbrl_metrics=state.get("xbrl_metrics"),
            ),
            return_exceptions=True,
        )

        news_json = None
        if isinstance(news_result, NewsSummary):
            news_json = json.dumps({
                "ticker": news_result.ticker,
                "date": news_result.date,
                "articles": [dataclasses.asdict(a) for a in news_result.articles],
                "sentiment_score": news_result.sentiment_score,
                "narrative": news_result.narrative,
                "cache_hit": news_result.cache_hit,
            })

        market_json = None
        if isinstance(market_result, MarketSummary):
            market_json = json.dumps({
                "ticker": market_result.ticker,
                "quote": dataclasses.asdict(market_result.quote) if market_result.quote else None,
                "ratios": dataclasses.asdict(market_result.ratios) if market_result.ratios else None,
                "history_bar_count": len(market_result.history.bars) if market_result.history else 0,
                "cache_hits": market_result.cache_hits,
                "pe_computed": market_result.pe_computed,
            })

        return {"news_summary_json": news_json, "market_summary_json": market_json}

    async def extractor_node(state: PipelineState) -> dict:
        if state.get("error"):
            return {}
        if not state.get("filing_content"):
            return {"error": "No filing content retrieved — document may be missing from cache"}
        tc = TraceContext(trace_id=state["trace_id"])
        data = await extract(
            ticker=state["ticker"],
            filed_date=state.get("filing_date", ""),
            document_text=state["filing_content"],
            langfuse=langfuse,
            trace_context=tc,
            xbrl_metrics=state.get("xbrl_metrics"),
        )
        return {"extracted_data_json": json.dumps(data.__dict__)}

    async def critic_node(state: PipelineState) -> dict:
        if state.get("error"):
            return {}
        tc = TraceContext(trace_id=state["trace_id"])
        raw = json.loads(state["extracted_data_json"] or "{}")
        data = ExtractedData(**raw)
        result = await critique(
            data,
            langfuse=langfuse,
            trace_context=tc,
            market_json=state.get("market_summary_json"),
            news_json=state.get("news_summary_json"),
        )
        return {"critique_json": json.dumps(result.__dict__)}

    async def writer_node(state: PipelineState) -> dict:
        if state.get("error"):
            return {"report": f"Pipeline failed: {state['error']}"}
        tc = TraceContext(trace_id=state["trace_id"])
        data = ExtractedData(**json.loads(state["extracted_data_json"] or "{}"))
        crit = Critique(**json.loads(state["critique_json"] or "{}"))

        news = _deserialise_news(state.get("news_summary_json"))
        market = _deserialise_market(state.get("market_summary_json"))

        report = await write_report(
            data, crit, langfuse=langfuse, trace_context=tc,
            news=news, market=market,
        )
        langfuse.create_score(
            trace_id=state["trace_id"],
            name="data-confidence",
            value=crit.confidence,
            comment=crit.summary,
        )
        langfuse.flush()
        return {"report": report}

    # ── Graph assembly ────────────────────────────────────────────────────────

    graph: StateGraph = StateGraph(PipelineState)
    graph.add_node("retriever", retriever_node)
    graph.add_node("intelligence", intelligence_node)
    graph.add_node("extractor", extractor_node)
    graph.add_node("critic", critic_node)
    graph.add_node("writer", writer_node)

    graph.set_entry_point("retriever")
    graph.add_edge("retriever", "intelligence")
    graph.add_edge("intelligence", "extractor")
    graph.add_edge("extractor", "critic")
    graph.add_edge("critic", "writer")
    graph.add_edge("writer", END)

    return graph.compile()


def initial_state(ticker: str) -> PipelineState:
    """Return a fresh PipelineState for *ticker* with a new trace ID."""
    return PipelineState(
        ticker=ticker.upper(),
        trace_id=uuid.uuid4().hex,
        filing_content=None,
        filing_cik=None,
        filing_date=None,
        xbrl_metrics=None,
        cache_hit=False,
        news_summary_json=None,
        market_summary_json=None,
        extracted_data_json=None,
        critique_json=None,
        report=None,
        error=None,
    )


# ── Deserialisation helpers ───────────────────────────────────────────────────

def _deserialise_news(json_str: str | None) -> NewsSummary | None:
    if not json_str:
        return None
    from sec_analyzer.providers.base import NewsArticle
    d = json.loads(json_str)
    return NewsSummary(
        ticker=d["ticker"],
        date=d["date"],
        articles=[NewsArticle(**a) for a in d.get("articles", [])],
        sentiment_score=d.get("sentiment_score"),
        narrative=d.get("narrative"),
        cache_hit=d.get("cache_hit", False),
    )


def _deserialise_market(json_str: str | None) -> MarketSummary | None:
    if not json_str:
        return None
    from sec_analyzer.providers.base import Quote, Ratios
    d = json.loads(json_str)
    quote = Quote(**d["quote"]) if d.get("quote") else None
    ratios = Ratios(**d["ratios"]) if d.get("ratios") else None
    return MarketSummary(
        ticker=d["ticker"],
        quote=quote,
        ratios=ratios,
        cache_hits=d.get("cache_hits", {}),
    )
