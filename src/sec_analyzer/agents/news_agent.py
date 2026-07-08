"""News Agent — fetches recent news and sentiment for a ticker with caching.

Sentiment is computed in two stages:
1. Provider score (e.g. Finnhub buzz/sentiment) — fast, pre-computed, often unavailable on free tier.
2. LLM score + narrative — reads article headlines/summaries and returns a score in [-1.0, 1.0]
   plus a 2-sentence narrative of the news themes. Always runs when articles are available.

The provider score is used when present; the LLM score fills in when it is None.
The narrative is always LLM-generated and stored in the cache alongside the score.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone

from langfuse import Langfuse
from langfuse.types import TraceContext

from sec_analyzer.cache.sqlite_cache import SQLiteCache
from sec_analyzer.gateway import LLMGateway, Message, get_gateway
from sec_analyzer.providers.base import NewsArticle, NewsProvider, NewsResult

_SENTIMENT_SYSTEM = """\
You are a financial news analyst. Given a list of recent news headlines and summaries
for a stock, output ONLY a JSON object with these fields:
  "score": float from -1.0 (very bearish) to +1.0 (very bullish), 0.0 for neutral
  "narrative": exactly 2 sentences summarising the main news themes and their market relevance

Be precise about the score:
  -1.0 to -0.5 : strongly negative (major scandal, bankruptcy risk, missed earnings, regulatory action)
  -0.5 to -0.2 : mildly negative (guidance cut, competitive pressure, management uncertainty)
  -0.2 to +0.2 : neutral (routine updates, mixed signals)
  +0.2 to +0.5 : mildly positive (beat estimates, product launch, new contract)
  +0.5 to +1.0 : strongly positive (record results, major deal, significant upgrade)

Reply with JSON only — no markdown fences."""


def _articles_prompt(ticker: str, articles: list[NewsArticle]) -> str:
    lines = [f"Ticker: {ticker}", f"Articles ({len(articles)}):", ""]
    for i, a in enumerate(articles, 1):
        lines.append(f"{i}. [{a.source}] {a.headline}")
        if a.summary:
            lines.append(f"   {a.summary[:200]}")
    return "\n".join(lines)


async def _llm_sentiment(
    ticker: str,
    articles: list[NewsArticle],
    gateway: LLMGateway,
) -> tuple[float | None, str | None]:
    """Return (score, narrative) by asking the LLM to analyse the article list."""
    if not articles:
        return None, None
    prompt = _articles_prompt(ticker, articles)
    messages: list[Message] = [{"role": "user", "content": prompt}]
    try:
        response = await gateway.complete(
            messages, system=_SENTIMENT_SYSTEM, max_tokens=256, json_mode=True
        )
        data = json.loads(response.text)
        score = float(data["score"])
        score = max(-1.0, min(1.0, score))  # clamp to valid range
        narrative = str(data.get("narrative", "")).strip() or None
        return score, narrative
    except Exception:
        return None, None


@dataclass
class NewsSummary:
    ticker: str
    date: str                          # YYYY-MM-DD the data was fetched for
    articles: list[NewsArticle] = field(default_factory=list)
    sentiment_score: float | None = None    # -1.0 (bearish) → +1.0 (bullish)
    narrative: str | None = None            # LLM-generated 2-sentence news summary
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
    gateway: LLMGateway | None = None,
) -> NewsSummary:
    """Fetch recent news for *ticker*, using the cache when available.

    After fetching articles, runs an LLM pass to compute a sentiment score
    (falling back to the provider score when LLM fails) and a narrative summary.
    Both are stored in the cache so subsequent cache hits include them.
    """
    today = datetime.now(tz=timezone.utc).date().isoformat()
    gw = gateway or get_gateway()

    with langfuse.start_as_current_observation(
        name="news_agent",
        as_type="span",
        trace_context=trace_context,
        input={"ticker": ticker, "date": today},
    ) as span:

        cached = await cache.get_news(ticker, today)
        if cached is not None:
            articles = [NewsArticle(**a) for a in cached["articles"]]
            narrative = cached.get("narrative")
            sentiment_score = cached["sentiment_score"]

            # Back-fill LLM sentiment + narrative when the cache entry predates this feature
            if articles and (narrative is None or sentiment_score is None):
                llm_score, narrative = await _llm_sentiment(ticker, articles, gw)
                if sentiment_score is None:
                    sentiment_score = llm_score
                # Update cache with enriched data
                await cache.store_news(
                    ticker, today,
                    [{"headline": a.headline, "summary": a.summary, "url": a.url,
                      "source": a.source, "published_at": a.published_at} for a in articles],
                    sentiment_score, narrative,
                )

            summary = NewsSummary(
                ticker=ticker,
                date=today,
                articles=articles,
                sentiment_score=sentiment_score,
                narrative=narrative,
                cache_hit=True,
            )
            span.update(output={"cache_hit": True, "article_count": len(articles),
                                "backfilled_sentiment": narrative is not None})
            return summary

        result: NewsResult = await provider.fetch_news(ticker, max_articles=max_articles)

        # LLM sentiment + narrative from article content
        llm_score, narrative = await _llm_sentiment(ticker, result.articles, gw)

        # Provider score takes precedence when available; LLM fills the gap
        final_score = result.sentiment_score if result.sentiment_score is not None else llm_score

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
        await cache.store_news(
            ticker, today, articles_dicts, final_score, narrative
        )

        span.update(output={
            "cache_hit": False,
            "article_count": len(result.articles),
            "provider_sentiment": result.sentiment_score,
            "llm_sentiment": llm_score,
            "final_sentiment": final_score,
            "has_narrative": narrative is not None,
        })

    return NewsSummary(
        ticker=ticker,
        date=today,
        articles=result.articles,
        sentiment_score=final_score,
        narrative=narrative,
        cache_hit=False,
    )
