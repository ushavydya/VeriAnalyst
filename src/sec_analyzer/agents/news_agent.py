"""News Agent — fetches recent news and sentiment for a ticker with caching."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

from langfuse import Langfuse
from langfuse.types import TraceContext

from sec_analyzer.cache.sqlite_cache import SQLiteCache
from sec_analyzer.providers.base import NewsArticle, NewsProvider, NewsResult


@dataclass
class NewsSummary:
    ticker: str
    date: str                         # YYYY-MM-DD the data was fetched for
    articles: list[NewsArticle] = field(default_factory=list)
    sentiment_score: float | None = None   # -1.0 (bearish) → +1.0 (bullish)
    cache_hit: bool = False

    @property
    def sentiment_label(self) -> str:
        if self.sentiment_score is None:
            return "neutral"
        if self.sentiment_score >= 0.2:
            return "bullish"
        if self.sentiment_score <= -0.2:
            return "bearish"
        return "neutral"


async def fetch_news(
    ticker: str,
    provider: NewsProvider,
    cache: SQLiteCache,
    langfuse: Langfuse,
    trace_context: TraceContext,
    *,
    max_articles: int = 20,
) -> NewsSummary:
    """Fetch recent news for *ticker*, using the cache when available."""
    today = datetime.now(tz=timezone.utc).date().isoformat()

    with langfuse.start_as_current_observation(
        name="news_agent",
        as_type="span",
        trace_context=trace_context,
        input={"ticker": ticker, "date": today},
    ) as span:

        cached = await cache.get_news(ticker, today)
        if cached is not None:
            articles = [NewsArticle(**a) for a in cached["articles"]]
            summary = NewsSummary(
                ticker=ticker,
                date=today,
                articles=articles,
                sentiment_score=cached["sentiment_score"],
                cache_hit=True,
            )
            span.update(output={"cache_hit": True, "article_count": len(articles)})
            return summary

        result: NewsResult = await provider.fetch_news(ticker, max_articles=max_articles)

        articles_dicts = [
            {
                "headline": a.headline,
                "summary": a.summary,
                "url": a.url,
                "source": a.source,
                "published_at": a.published_at,
            }
            for a in result.articles
        ]
        await cache.store_news(ticker, today, articles_dicts, result.sentiment_score)

        span.update(output={
            "cache_hit": False,
            "article_count": len(result.articles),
            "sentiment_score": result.sentiment_score,
        })

    return NewsSummary(
        ticker=ticker,
        date=today,
        articles=result.articles,
        sentiment_score=result.sentiment_score,
        cache_hit=False,
    )
