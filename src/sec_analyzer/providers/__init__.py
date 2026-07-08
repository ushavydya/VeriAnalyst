"""Data provider factory — returns the configured NewsProvider / MarketDataProvider.

Configure via environment variables:
    NEWS_PROVIDER=finnhub          (default)
    MARKET_DATA_PROVIDER=finnhub   (default)
    FINNHUB_API_KEY=<your key>

Adding a new provider:
    1. Create providers/<name>.py implementing NewsProvider and/or MarketDataProvider
    2. Add an elif branch in get_news_provider() / get_market_data_provider()
    3. No other files need to change.
"""
from __future__ import annotations

import os

from sec_analyzer.providers.base import MarketDataProvider, NewsProvider


def get_news_provider() -> NewsProvider:
    name = os.environ.get("NEWS_PROVIDER", "finnhub").lower()
    if name == "finnhub":
        from sec_analyzer.providers.finnhub import FinnhubProvider
        return FinnhubProvider(api_key=_require("FINNHUB_API_KEY"))
    raise ValueError(f"Unknown NEWS_PROVIDER: {name!r}. Supported: finnhub")


def get_market_data_provider() -> MarketDataProvider:
    name = os.environ.get("MARKET_DATA_PROVIDER", "finnhub").lower()
    if name == "finnhub":
        from sec_analyzer.providers.finnhub import FinnhubProvider
        return FinnhubProvider(api_key=_require("FINNHUB_API_KEY"))
    raise ValueError(f"Unknown MARKET_DATA_PROVIDER: {name!r}. Supported: finnhub")


def _require(key: str) -> str:
    val = os.environ.get(key, "")
    if not val:
        raise EnvironmentError(f"{key} is not set. Add it to your .env file.")
    return val
