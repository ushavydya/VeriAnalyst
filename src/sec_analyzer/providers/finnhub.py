"""Finnhub implementation of NewsProvider and MarketDataProvider.

Finnhub free tier covers:
  - Company news:       GET /company-news
  - Sentiment:          GET /news-sentiment
  - Quote:             GET /quote
  - Basic financials:  GET /stock/metric  (ratios, 52-week high/low, beta)
  - Candles (OHLCV):   GET /stock/candle

Rate limit: 60 API calls/minute on free tier.
API key:    set FINNHUB_API_KEY in .env
"""
from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone

import httpx

from sec_analyzer.providers.base import (
    MarketDataProvider,
    MarketHistory,
    NewsArticle,
    NewsProvider,
    NewsResult,
    PriceBar,
    Quote,
    Ratios,
)

_BASE = "https://finnhub.io/api/v1"

# Period string → number of calendar days to look back for news/candles
_PERIOD_DAYS: dict[str, int] = {
    "1y": 365,
    "6m": 182,
    "3m": 91,
    "1m": 30,
}


class FinnhubProvider(NewsProvider, MarketDataProvider):
    """Single class that satisfies both provider interfaces using the Finnhub API."""

    def __init__(self, api_key: str) -> None:
        self._key = api_key
        self._headers = {"X-Finnhub-Token": api_key}

    # ── NewsProvider ──────────────────────────────────────────────────────────

    async def fetch_news(
        self,
        ticker: str,
        *,
        max_articles: int = 20,
    ) -> NewsResult:
        today = datetime.now(tz=timezone.utc).date()
        from_date = (today - timedelta(days=7)).isoformat()
        to_date = today.isoformat()

        async with httpx.AsyncClient(headers=self._headers, timeout=15.0) as client:
            news_resp, sentiment_resp = await _gather(
                client.get(f"{_BASE}/company-news", params={
                    "symbol": ticker.upper(),
                    "from": from_date,
                    "to": to_date,
                }),
                client.get(f"{_BASE}/news-sentiment", params={"symbol": ticker.upper()}),
            )

        articles: list[NewsArticle] = []
        for item in (news_resp.json() or [])[:max_articles]:
            articles.append(NewsArticle(
                headline=item.get("headline", ""),
                summary=item.get("summary", ""),
                url=item.get("url", ""),
                source=item.get("source", ""),
                published_at=datetime.fromtimestamp(
                    item.get("datetime", 0), tz=timezone.utc
                ).isoformat(),
            ))

        sentiment_score: float | None = None
        sentiment_data = sentiment_resp.json() if sentiment_resp.status_code == 200 else {}
        buzz = sentiment_data.get("buzz", {})
        if buzz:
            # Finnhub returns bearishPercent / bullishPercent (0–1 each)
            bull = buzz.get("bullishPercent", 0.0)
            bear = buzz.get("bearishPercent", 0.0)
            if bull + bear > 0:
                sentiment_score = round(bull - bear, 4)  # -1.0 → +1.0

        return NewsResult(
            ticker=ticker.upper(),
            articles=articles,
            sentiment_score=sentiment_score,
        )

    # ── MarketDataProvider ────────────────────────────────────────────────────

    async def fetch_quote(self, ticker: str) -> Quote:
        async with httpx.AsyncClient(headers=self._headers, timeout=15.0) as client:
            resp = await client.get(f"{_BASE}/quote", params={"symbol": ticker.upper()})
        resp.raise_for_status()
        data = resp.json()

        return Quote(
            ticker=ticker.upper(),
            price=float(data.get("c", 0)),          # current price
            change_pct=float(data.get("dp", 0)),    # % change
            volume=int(data.get("v", 0) or 0),
            market_cap=None,                        # not in /quote; available via metrics
            as_of=datetime.fromtimestamp(
                data.get("t", time.time()), tz=timezone.utc
            ).isoformat(),
        )

    async def fetch_ratios(self, ticker: str) -> Ratios:
        async with httpx.AsyncClient(headers=self._headers, timeout=15.0) as client:
            resp = await client.get(
                f"{_BASE}/stock/metric",
                params={"symbol": ticker.upper(), "metric": "all"},
            )
        resp.raise_for_status()
        m = resp.json().get("metric", {})

        return Ratios(
            ticker=ticker.upper(),
            pe_ratio=_float(m.get("peNormalizedAnnual") or m.get("peTTM")),
            pb_ratio=_float(m.get("pbAnnual") or m.get("pbQuarterly")),
            week_52_high=_float(m.get("52WeekHigh")),
            week_52_low=_float(m.get("52WeekLow")),
            beta=_float(m.get("beta")),
        )

    async def fetch_history(
        self,
        ticker: str,
        *,
        period: str = "1y",
    ) -> MarketHistory:
        days = _PERIOD_DAYS.get(period, 365)
        now = int(time.time())
        from_ts = now - days * 86_400

        async with httpx.AsyncClient(headers=self._headers, timeout=15.0) as client:
            resp = await client.get(
                f"{_BASE}/stock/candle",
                params={
                    "symbol": ticker.upper(),
                    "resolution": "D",   # daily bars
                    "from": from_ts,
                    "to": now,
                },
            )
        resp.raise_for_status()
        data = resp.json()

        bars: list[PriceBar] = []
        if data.get("s") == "ok":
            timestamps = data.get("t", [])
            for i, ts in enumerate(timestamps):
                bars.append(PriceBar(
                    date=datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d"),
                    open=float(data["o"][i]),
                    high=float(data["h"][i]),
                    low=float(data["l"][i]),
                    close=float(data["c"][i]),
                    volume=int(data["v"][i]),
                ))

        return MarketHistory(ticker=ticker.upper(), period=period, bars=bars)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _float(val: object) -> float | None:
    try:
        return float(val) if val is not None else None  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


async def _gather(*coros):
    """Run coroutines concurrently and return results in order."""
    import asyncio
    return await asyncio.gather(*coros)
