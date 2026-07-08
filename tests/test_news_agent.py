"""Unit tests for news_agent.py."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from sec_analyzer.agents.news_agent import NewsSummary, _llm_sentiment, fetch_news
from sec_analyzer.cache.sqlite_cache import SQLiteCache
from sec_analyzer.providers.base import NewsArticle, NewsResult


def _make_langfuse():
    lf = MagicMock()
    span = MagicMock()
    span.__enter__ = MagicMock(return_value=span)
    span.__exit__ = MagicMock(return_value=False)
    lf.start_as_current_observation.return_value = span
    return lf


def _make_provider(articles=None, sentiment=0.3):
    provider = MagicMock()
    provider.fetch_news = AsyncMock(return_value=NewsResult(
        ticker="AAPL",
        articles=articles or [
            NewsArticle("Big news", "Details", "http://x", "Reuters", "2026-07-07T00:00:00+00:00")
        ],
        sentiment_score=sentiment,
    ))
    return provider


def _make_gateway(score=0.4, narrative="Strong results beat estimates. Analysts are bullish."):
    gw = MagicMock()
    gw.complete = AsyncMock(return_value=MagicMock(
        text=json.dumps({"score": score, "narrative": narrative}),
        input_tokens=50, output_tokens=30,
    ))
    return gw


@pytest.fixture()
async def cache(tmp_path: Path) -> SQLiteCache:
    c = SQLiteCache(db_path=str(tmp_path / "cache.db"), docs_dir=str(tmp_path / "docs"))
    async with c:
        yield c


# ── Provider + LLM integration ────────────────────────────────────────────────

async def test_fetch_news_calls_provider_when_no_cache(cache):
    provider = _make_provider()
    gw = _make_gateway()
    result = await fetch_news("AAPL", provider, cache, _make_langfuse(), {}, gateway=gw)
    provider.fetch_news.assert_awaited_once()
    assert result.ticker == "AAPL"
    assert len(result.articles) == 1
    assert result.cache_hit is False


async def test_fetch_news_uses_cache_on_second_call(cache):
    provider = _make_provider()
    gw = _make_gateway()
    await fetch_news("AAPL", provider, cache, _make_langfuse(), {}, gateway=gw)
    result2 = await fetch_news("AAPL", provider, cache, _make_langfuse(), {}, gateway=gw)
    assert provider.fetch_news.await_count == 1  # not called again
    assert result2.cache_hit is True


async def test_fetch_news_provider_sentiment_takes_precedence(cache):
    """When provider returns a sentiment score, it overrides the LLM score."""
    provider = _make_provider(sentiment=0.65)
    gw = _make_gateway(score=0.1)  # LLM would say 0.1 — should be ignored
    result = await fetch_news("AAPL", provider, cache, _make_langfuse(), {}, gateway=gw)
    assert result.sentiment_score == pytest.approx(0.65)


async def test_fetch_news_llm_score_used_when_provider_returns_none(cache):
    """When provider sentiment is None, the LLM score is used."""
    provider = _make_provider(sentiment=None)
    gw = _make_gateway(score=0.55)
    result = await fetch_news("AAPL", provider, cache, _make_langfuse(), {}, gateway=gw)
    assert result.sentiment_score == pytest.approx(0.55)


async def test_fetch_news_narrative_populated(cache):
    provider = _make_provider()
    gw = _make_gateway(narrative="Revenue beat expectations. Management raised guidance.")
    result = await fetch_news("AAPL", provider, cache, _make_langfuse(), {}, gateway=gw)
    assert result.narrative == "Revenue beat expectations. Management raised guidance."


async def test_fetch_news_narrative_cached_and_restored(cache):
    provider = _make_provider()
    gw = _make_gateway(narrative="Analysts are optimistic. Shares rose 5%.")
    await fetch_news("AAPL", provider, cache, _make_langfuse(), {}, gateway=gw)
    result2 = await fetch_news("AAPL", provider, cache, _make_langfuse(), {}, gateway=gw)
    assert result2.cache_hit is True
    assert result2.narrative == "Analysts are optimistic. Shares rose 5%."


async def test_fetch_news_cache_restores_articles(cache):
    articles = [
        NewsArticle("H1", "S1", "http://a", "Reuters", "2026-07-07T00:00:00+00:00"),
        NewsArticle("H2", "S2", "http://b", "Bloomberg", "2026-07-07T01:00:00+00:00"),
    ]
    provider = _make_provider(articles=articles, sentiment=0.1)
    gw = _make_gateway()
    await fetch_news("AAPL", provider, cache, _make_langfuse(), {}, gateway=gw)
    cached = await fetch_news("AAPL", provider, cache, _make_langfuse(), {}, gateway=gw)
    assert len(cached.articles) == 2
    assert cached.articles[0].headline == "H1"
    assert cached.articles[1].source == "Bloomberg"


async def test_fetch_news_llm_error_falls_back_gracefully(cache):
    """If LLM call fails, sentiment stays as provider value (or None)."""
    provider = _make_provider(sentiment=None)
    gw = MagicMock()
    gw.complete = AsyncMock(side_effect=Exception("LLM timeout"))
    result = await fetch_news("AAPL", provider, cache, _make_langfuse(), {}, gateway=gw)
    assert result.sentiment_score is None
    assert result.narrative is None


# ── _llm_sentiment unit tests ─────────────────────────────────────────────────

async def test_llm_sentiment_returns_score_and_narrative():
    gw = _make_gateway(score=0.6, narrative="Bullish news dominated. EPS beat expectations.")
    articles = [NewsArticle("Beat", "Q4 EPS beat", "http://x", "Reuters", "2026-07-07")]
    score, narrative = await _llm_sentiment("AAPL", articles, gw)
    assert score == pytest.approx(0.6)
    assert "beat" in narrative.lower()


async def test_llm_sentiment_clamps_score():
    gw = MagicMock()
    gw.complete = AsyncMock(return_value=MagicMock(
        text=json.dumps({"score": 5.0, "narrative": "Extreme bullish."}),
    ))
    articles = [NewsArticle("H", "S", "http://x", "Reuters", "2026-07-07")]
    score, _ = await _llm_sentiment("AAPL", articles, gw)
    assert score == pytest.approx(1.0)


async def test_llm_sentiment_empty_articles_returns_none():
    gw = _make_gateway()
    score, narrative = await _llm_sentiment("AAPL", [], gw)
    assert score is None
    assert narrative is None
    gw.complete.assert_not_awaited()


async def test_llm_sentiment_error_returns_none():
    gw = MagicMock()
    gw.complete = AsyncMock(side_effect=Exception("network error"))
    articles = [NewsArticle("H", "S", "http://x", "Reuters", "2026-07-07")]
    score, narrative = await _llm_sentiment("AAPL", articles, gw)
    assert score is None
    assert narrative is None


# ── NewsSummary helpers ───────────────────────────────────────────────────────

def test_sentiment_label_bullish():
    s = NewsSummary("AAPL", "2026-07-07", sentiment_score=0.5)
    assert s.sentiment_label == "bullish"


def test_sentiment_label_bearish():
    s = NewsSummary("AAPL", "2026-07-07", sentiment_score=-0.3)
    assert s.sentiment_label == "bearish"


def test_sentiment_label_neutral():
    s = NewsSummary("AAPL", "2026-07-07", sentiment_score=0.1)
    assert s.sentiment_label == "neutral"


def test_sentiment_label_none_is_neutral():
    s = NewsSummary("AAPL", "2026-07-07", sentiment_score=None)
    assert s.sentiment_label == "neutral"


def test_news_summary_narrative_field():
    s = NewsSummary("AAPL", "2026-07-07", narrative="Revenue beat. Guidance raised.")
    assert s.narrative == "Revenue beat. Guidance raised."
