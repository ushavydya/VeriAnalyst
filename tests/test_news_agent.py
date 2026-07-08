"""Unit tests for news_agent.py."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from sec_analyzer.agents.news_agent import NewsSummary, fetch_news
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


@pytest.fixture()
async def cache(tmp_path: Path) -> SQLiteCache:
    c = SQLiteCache(db_path=str(tmp_path / "cache.db"), docs_dir=str(tmp_path / "docs"))
    async with c:
        yield c


async def test_fetch_news_calls_provider_when_no_cache(cache):
    provider = _make_provider()
    result = await fetch_news("AAPL", provider, cache, _make_langfuse(), {})
    provider.fetch_news.assert_awaited_once()
    assert result.ticker == "AAPL"
    assert len(result.articles) == 1
    assert result.cache_hit is False


async def test_fetch_news_uses_cache_on_second_call(cache):
    provider = _make_provider()
    await fetch_news("AAPL", provider, cache, _make_langfuse(), {})
    result2 = await fetch_news("AAPL", provider, cache, _make_langfuse(), {})
    assert provider.fetch_news.await_count == 1  # not called again
    assert result2.cache_hit is True


async def test_fetch_news_sentiment_score_preserved(cache):
    provider = _make_provider(sentiment=0.65)
    result = await fetch_news("AAPL", provider, cache, _make_langfuse(), {})
    assert result.sentiment_score == pytest.approx(0.65)


async def test_fetch_news_null_sentiment_preserved(cache):
    provider = _make_provider(sentiment=None)
    result = await fetch_news("AAPL", provider, cache, _make_langfuse(), {})
    assert result.sentiment_score is None


async def test_fetch_news_cache_restores_articles(cache):
    articles = [
        NewsArticle("H1", "S1", "http://a", "Reuters", "2026-07-07T00:00:00+00:00"),
        NewsArticle("H2", "S2", "http://b", "Bloomberg", "2026-07-07T01:00:00+00:00"),
    ]
    provider = _make_provider(articles=articles, sentiment=0.1)
    await fetch_news("AAPL", provider, cache, _make_langfuse(), {})
    cached = await fetch_news("AAPL", provider, cache, _make_langfuse(), {})
    assert len(cached.articles) == 2
    assert cached.articles[0].headline == "H1"
    assert cached.articles[1].source == "Bloomberg"


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
