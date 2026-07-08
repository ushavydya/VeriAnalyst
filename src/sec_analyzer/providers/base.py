"""Abstract interfaces and shared dataclasses for news and market data providers.

All agents and cache code depend only on these types — never on a concrete
provider (Finnhub, NewsAPI, yfinance, etc.).  Swapping a provider means adding
a new file under providers/ and updating the factory in __init__.py.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


# ── Shared dataclasses ────────────────────────────────────────────────────────

@dataclass
class NewsArticle:
    headline:     str
    summary:      str
    url:          str
    source:       str
    published_at: str   # ISO-8601 datetime string


@dataclass
class NewsResult:
    ticker:          str
    articles:        list[NewsArticle]
    sentiment_score: float | None   # -1.0 (bearish) → +1.0 (bullish); None if unavailable


@dataclass
class Quote:
    ticker:     str
    price:      float
    change_pct: float        # intraday change %
    volume:     int
    market_cap: float | None  # USD
    as_of:      str           # ISO-8601 datetime string


@dataclass
class Ratios:
    ticker:       str
    pe_ratio:     float | None
    pb_ratio:     float | None
    week_52_high: float | None
    week_52_low:  float | None
    beta:         float | None


@dataclass
class PriceBar:
    date:   str    # YYYY-MM-DD
    open:   float
    high:   float
    low:    float
    close:  float
    volume: int


@dataclass
class MarketHistory:
    ticker: str
    period: str              # '1y', '6m', etc.
    bars:   list[PriceBar] = field(default_factory=list)


# ── Abstract provider interfaces ──────────────────────────────────────────────

class NewsProvider(ABC):
    """Fetch news articles and sentiment for a ticker."""

    @abstractmethod
    async def fetch_news(
        self,
        ticker: str,
        *,
        max_articles: int = 20,
    ) -> NewsResult:
        """Return recent news articles and an aggregate sentiment score."""


class MarketDataProvider(ABC):
    """Fetch real-time and historical market data for a ticker."""

    @abstractmethod
    async def fetch_quote(self, ticker: str) -> Quote:
        """Return the latest price quote."""

    @abstractmethod
    async def fetch_ratios(self, ticker: str) -> Ratios:
        """Return key valuation ratios."""

    @abstractmethod
    async def fetch_history(
        self,
        ticker: str,
        *,
        period: str = "1y",
    ) -> MarketHistory:
        """Return daily OHLCV bars for *period* (e.g. '1y', '6m', '3m')."""
