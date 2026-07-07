"""LangGraph pipeline: Retriever → Extractor → Critic → Writer.

Every node is instrumented with a Langfuse span that records input,
output, and latency.  The parent trace_id flows through PipelineState
so spans from all nodes are grouped under one trace in the Langfuse UI.
"""
from __future__ import annotations

import json
import uuid
from typing import TypedDict

from langgraph.graph import END, StateGraph
from langfuse import Langfuse
from langfuse.types import TraceContext

from sec_analyzer.agents.critic import Critique, critique
from sec_analyzer.agents.extractor import ExtractedData, extract
from sec_analyzer.agents.retriever import SECRetriever
from sec_analyzer.agents.writer import write_report
from sec_analyzer.cache.sqlite_cache import SQLiteCache


class PipelineState(TypedDict):
    ticker: str
    trace_id: str
    # populated by each node
    filing_content: str | None
    filing_cik: str | None
    filing_date: str | None
    extracted_data_json: str | None   # JSON-serialised ExtractedData
    critique_json: str | None
    report: str | None
    error: str | None
    cache_hit: bool


def build_pipeline(cache: SQLiteCache, langfuse: Langfuse) -> "CompiledGraph":
    """Construct and compile the LangGraph pipeline.

    Returns a runnable graph; call ``await app.ainvoke(state)`` to execute.
    """
    retriever = SECRetriever(cache=cache, langfuse=langfuse)

    # ── Node functions ────────────────────────────────────────────────────────

    async def retriever_node(state: PipelineState) -> dict:
        tc = TraceContext(trace_id=state["trace_id"])
        try:
            result = await retriever.fetch_10k(state["ticker"], tc)
            return {
                "filing_content": result.read_text(),
                "filing_cik": result.cik,
                "filing_date": result.filed_date,
                "cache_hit": result.cache_hit,
                "error": None,
            }
        except Exception as exc:
            return {"error": str(exc)}

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
        )
        return {"extracted_data_json": json.dumps(data.__dict__)}

    async def critic_node(state: PipelineState) -> dict:
        if state.get("error"):
            return {}
        tc = TraceContext(trace_id=state["trace_id"])
        raw = json.loads(state["extracted_data_json"] or "{}")
        data = ExtractedData(**raw)
        result = await critique(data, langfuse=langfuse, trace_context=tc)
        return {"critique_json": json.dumps(result.__dict__)}

    async def writer_node(state: PipelineState) -> dict:
        if state.get("error"):
            return {"report": f"Pipeline failed: {state['error']}"}
        tc = TraceContext(trace_id=state["trace_id"])
        data = ExtractedData(**json.loads(state["extracted_data_json"] or "{}"))
        crit = Critique(**json.loads(state["critique_json"] or "{}"))
        report = await write_report(data, crit, langfuse=langfuse, trace_context=tc)
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
    graph.add_node("extractor", extractor_node)
    graph.add_node("critic", critic_node)
    graph.add_node("writer", writer_node)

    graph.set_entry_point("retriever")
    graph.add_edge("retriever", "extractor")
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
        extracted_data_json=None,
        critique_json=None,
        report=None,
        error=None,
        cache_hit=False,
    )
