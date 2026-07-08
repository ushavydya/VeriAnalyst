"""Tests for the LangGraph pipeline (orchestration/graph.py).

All external I/O is mocked — no network calls, no LLM calls.
Tests verify state transitions, error propagation, and JSON serialisation.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from sec_analyzer.agents.news_agent import NewsSummary
from sec_analyzer.agents.market_agent import MarketSummary
from sec_analyzer.cache.sqlite_cache import SQLiteCache
from sec_analyzer.orchestration.graph import (
    _deserialise_market,
    _deserialise_news,
    build_pipeline,
    initial_state,
)
from sec_analyzer.providers.base import NewsArticle, Quote, Ratios


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_langfuse():
    lf = MagicMock()
    span = MagicMock()
    span.__enter__ = MagicMock(return_value=span)
    span.__exit__ = MagicMock(return_value=False)
    span.update = MagicMock()
    lf.start_as_current_observation.return_value = span
    lf.create_score = MagicMock()
    lf.flush = MagicMock()
    return lf


@pytest.fixture()
async def cache(tmp_path: Path) -> SQLiteCache:
    c = SQLiteCache(db_path=str(tmp_path / "cache.db"), docs_dir=str(tmp_path / "docs"))
    async with c:
        yield c


# ── initial_state ─────────────────────────────────────────────────────────────

def test_initial_state_sets_ticker():
    state = initial_state("aapl")
    assert state["ticker"] == "AAPL"


def test_initial_state_all_none():
    state = initial_state("MSFT")
    assert state["filing_content"] is None
    assert state["news_summary_json"] is None
    assert state["market_summary_json"] is None
    assert state["report"] is None
    assert state["error"] is None


def test_initial_state_unique_trace_ids():
    s1 = initial_state("AAPL")
    s2 = initial_state("AAPL")
    assert s1["trace_id"] != s2["trace_id"]


# ── _deserialise_news ─────────────────────────────────────────────────────────

def test_deserialise_news_none_returns_none():
    assert _deserialise_news(None) is None


def test_deserialise_news_empty_json_string_returns_none():
    assert _deserialise_news("") is None


def test_deserialise_news_round_trip():
    original = NewsSummary(
        ticker="AAPL",
        date="2026-07-08",
        articles=[NewsArticle("Headline", "Summary", "http://x", "Reuters", "2026-07-08")],
        sentiment_score=0.4,
        narrative="Positive sentiment prevailed.",
        cache_hit=False,
    )
    payload = json.dumps({
        "ticker": original.ticker,
        "date": original.date,
        "articles": [
            {"headline": a.headline, "summary": a.summary, "url": a.url,
             "source": a.source, "published_at": a.published_at}
            for a in original.articles
        ],
        "sentiment_score": original.sentiment_score,
        "narrative": original.narrative,
        "cache_hit": original.cache_hit,
    })
    restored = _deserialise_news(payload)
    assert restored is not None
    assert restored.ticker == "AAPL"
    assert restored.sentiment_score == pytest.approx(0.4)
    assert restored.narrative == "Positive sentiment prevailed."
    assert len(restored.articles) == 1
    assert restored.articles[0].headline == "Headline"


def test_deserialise_news_missing_narrative_is_none():
    payload = json.dumps({
        "ticker": "AAPL", "date": "2026-07-08",
        "articles": [], "sentiment_score": None, "cache_hit": False,
        # no "narrative" key
    })
    restored = _deserialise_news(payload)
    assert restored.narrative is None


# ── _deserialise_market ───────────────────────────────────────────────────────

def test_deserialise_market_none_returns_none():
    assert _deserialise_market(None) is None


def test_deserialise_market_round_trip():
    payload = json.dumps({
        "ticker": "AAPL",
        "quote": {"ticker": "AAPL", "price": 210.0, "change_pct": 0.5,
                  "volume": 50_000_000, "market_cap": None, "as_of": "2026-07-08T14:00:00+00:00"},
        "ratios": {"ticker": "AAPL", "pe_ratio": 33.0, "pb_ratio": 45.0,
                   "week_52_high": 260.0, "week_52_low": 164.0, "beta": 1.2},
        "history_bar_count": 0,
        "cache_hits": {"quote": False, "ratios": False, "history": False},
        "pe_computed": False,
    })
    restored = _deserialise_market(payload)
    assert restored is not None
    assert restored.ticker == "AAPL"
    assert restored.quote.price == pytest.approx(210.0)
    assert restored.ratios.pe_ratio == pytest.approx(33.0)


def test_deserialise_market_null_quote_and_ratios():
    payload = json.dumps({
        "ticker": "TLN",
        "quote": None,
        "ratios": None,
        "history_bar_count": 0,
        "cache_hits": {},
        "pe_computed": False,
    })
    restored = _deserialise_market(payload)
    assert restored is not None
    assert restored.quote is None
    assert restored.ratios is None


# ── Pipeline integration (fully mocked) ──────────────────────────────────────

def _make_filing_result(ticker="AAPL"):
    from sec_analyzer.agents.retriever import FilingResult
    import tempfile

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".txt", mode="w")
    tmp.write("<html><body>Apple Inc. 10-K annual report. Revenue was $416 billion.</body></html>")
    tmp.close()

    return FilingResult(
        ticker=ticker,
        cik="0000320193",
        accession_number="0000320193-25-000123",
        form_type="10-K",
        filed_date="2025-10-31",
        document_url="https://www.sec.gov/Archives/fake.htm",
        document_path=Path(tmp.name),
        cache_hit=False,
        xbrl_metrics={"revenue": 416161.0, "net_income": 112010.0, "eps_diluted": 7.46,
                      "fiscal_year_end": "2025-09-27"},
    )


def _make_gw(responses: list):
    """Return a mock gateway whose complete() cycles through *responses* in order."""
    from sec_analyzer.gateway.base import ModelResponse

    idx = {"n": 0}

    async def fake_complete(messages, **kwargs):
        resp = responses[min(idx["n"], len(responses) - 1)]
        idx["n"] += 1
        return resp

    gw = MagicMock()
    gw.complete = AsyncMock(side_effect=fake_complete)
    return gw


_EXTRACTOR_RESP = lambda: __import__("sec_analyzer.gateway.base", fromlist=["ModelResponse"]).ModelResponse(
    text='{"revenue": 416161.0, "net_income": 112010.0, "eps_diluted": 7.46, "fiscal_year_end": "2025-09-27"}',
    model="test", input_tokens=100, output_tokens=50,
)
_CRITIC_RESP = lambda: __import__("sec_analyzer.gateway.base", fromlist=["ModelResponse"]).ModelResponse(
    text="Extraction quality is high.", model="test",
)
_WRITER_RESP = lambda: __import__("sec_analyzer.gateway.base", fromlist=["ModelResponse"]).ModelResponse(
    text="# AAPL — 10-K Analysis (FY2025)\n\n## Executive Summary\nApple had a strong year.",
    model="test", input_tokens=200, output_tokens=100,
)

# Patch all four agent modules that call get_gateway()
_GW_PATCHES = [
    "sec_analyzer.agents.extractor.get_gateway",
    "sec_analyzer.agents.critic.get_gateway",
    "sec_analyzer.agents.writer.get_gateway",
]


async def test_pipeline_runs_end_to_end(cache):
    """Full pipeline with mocked retriever, LLM, and no intelligence layer."""
    filing = _make_filing_result()
    lf = _make_langfuse()
    gw = _make_gw([_EXTRACTOR_RESP(), _CRITIC_RESP(), _WRITER_RESP()])

    with patch("sec_analyzer.agents.retriever.SECRetriever.fetch_10k", return_value=filing):
        with patch(_GW_PATCHES[0], return_value=gw), \
             patch(_GW_PATCHES[1], return_value=gw), \
             patch(_GW_PATCHES[2], return_value=gw):
            pipeline = build_pipeline(cache, lf, enable_intelligence=False)
            state = initial_state("AAPL")
            result = await pipeline.ainvoke(state)

    assert result["error"] is None
    assert result["report"] is not None
    assert "AAPL" in result["report"]
    assert result["extracted_data_json"] is not None
    metrics = json.loads(result["extracted_data_json"])["metrics"]
    assert metrics["revenue"] == pytest.approx(416161.0)


async def test_pipeline_error_propagates_gracefully(cache):
    """When retriever raises, pipeline sets error and skips remaining nodes."""
    lf = _make_langfuse()

    with patch(
        "sec_analyzer.agents.retriever.SECRetriever.fetch_10k",
        side_effect=Exception("EDGAR unreachable"),
    ):
        pipeline = build_pipeline(cache, lf, enable_intelligence=False)
        state = initial_state("AAPL")
        result = await pipeline.ainvoke(state)

    assert result["error"] == "EDGAR unreachable"
    assert result["report"] is not None
    assert "Pipeline failed" in result["report"]


async def test_pipeline_intelligence_disabled_leaves_json_none(cache):
    """With enable_intelligence=False, news/market JSON stays None."""
    filing = _make_filing_result()
    lf = _make_langfuse()
    gw = _make_gw([_EXTRACTOR_RESP(), _CRITIC_RESP(), _WRITER_RESP()])

    with patch("sec_analyzer.agents.retriever.SECRetriever.fetch_10k", return_value=filing):
        with patch(_GW_PATCHES[0], return_value=gw), \
             patch(_GW_PATCHES[1], return_value=gw), \
             patch(_GW_PATCHES[2], return_value=gw):
            pipeline = build_pipeline(cache, lf, enable_intelligence=False)
            state = initial_state("AAPL")
            result = await pipeline.ainvoke(state)

    assert result["news_summary_json"] is None
    assert result["market_summary_json"] is None
