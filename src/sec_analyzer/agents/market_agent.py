"""Market Data Agent — fetches price, ratios, and history for a ticker with caching."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from langfuse import Langfuse
from langfuse.types import TraceContext

from sec_analyzer.cache.sqlite_cache import SQLiteCache
from sec_analyzer.providers.base import MarketDataProvider, MarketHistory, PriceBar, Quote, Ratios


@dataclass
class MarketSummary:
    ticker: str
    quote: Quote | None = None
    ratios: Ratios | None = None
    history: MarketHistory | None = None
    cache_hits: dict[str, bool] = field(default_factory=dict)
    pe_computed: bool = False  # True when pe_ratio was derived from XBRL, not the provider

    @property
    def price(self) -> float | None:
        return self.quote.price if self.quote else None

    @property
    def change_pct(self) -> float | None:
        return self.quote.change_pct if self.quote else None

    @property
    def pe_ratio(self) -> float | None:
        return self.ratios.pe_ratio if self.ratios else None

    @property
    def week_52_high(self) -> float | None:
        return self.ratios.week_52_high if self.ratios else None

    @property
    def week_52_low(self) -> float | None:
        return self.ratios.week_52_low if self.ratios else None

    @property
    def beta(self) -> float | None:
        return self.ratios.beta if self.ratios else None

    def price_vs_52w_high_pct(self) -> float | None:
        """How far below the 52-week high the current price is (negative = below)."""
        if self.quote and self.ratios and self.ratios.week_52_high:
            return round((self.quote.price / self.ratios.week_52_high - 1) * 100, 2)
        return None


async def fetch_market_data(
    ticker: str,
    provider: MarketDataProvider,
    cache: SQLiteCache,
    langfuse: Langfuse,
    trace_context: TraceContext,
    *,
    history_period: str = "1y",
    xbrl_metrics: dict | None = None,
) -> MarketSummary:
    """Fetch quote, ratios, and price history for *ticker* with caching."""
    with langfuse.start_as_current_observation(
        name="market_agent",
        as_type="span",
        trace_context=trace_context,
        input={"ticker": ticker, "history_period": history_period},
    ) as span:

        quote, ratios, history, cache_hits = await _fetch_all(
            ticker, provider, cache, history_period
        )

        # Compute P/E from live price + XBRL eps when provider doesn't return one.
        # Use the live quote price (not cached ratios) so the value reflects today's
        # price; do NOT store this back into the ratios cache — the cache holds
        # provider data only, and a computed P/E would become stale as price moves.
        pe_computed = False
        if ratios is not None and ratios.pe_ratio is None and quote is not None:
            filled = _fill_computed_pe(ratios, quote.price, xbrl_metrics)
            if filled.pe_ratio is not None:
                ratios = filled
                pe_computed = True

        span.update(output={
            "cache_hits": cache_hits,
            "price": quote.price if quote else None,
            "pe_ratio": ratios.pe_ratio if ratios else None,
            "pe_computed": pe_computed,
            "history_bars": len(history.bars) if history else 0,
        })

    return MarketSummary(
        ticker=ticker,
        quote=quote,
        ratios=ratios,
        history=history,
        cache_hits=cache_hits,
        pe_computed=pe_computed,
    )


def _fill_computed_pe(ratios: Ratios, price: float, xbrl_metrics: dict | None) -> Ratios:
    """Return a new Ratios with pe_ratio filled from price / eps_diluted if available."""
    if not xbrl_metrics:
        return ratios
    eps = xbrl_metrics.get("eps_diluted")
    if not eps or eps <= 0:
        return ratios
    computed_pe = round(price / eps, 2)
    return Ratios(
        ticker=ratios.ticker,
        pe_ratio=computed_pe,
        pb_ratio=ratios.pb_ratio,
        week_52_high=ratios.week_52_high,
        week_52_low=ratios.week_52_low,
        beta=ratios.beta,
    )


async def _fetch_all(
    ticker: str,
    provider: MarketDataProvider,
    cache: SQLiteCache,
    history_period: str,
) -> tuple[Quote | None, Ratios | None, MarketHistory | None, dict[str, bool]]:
    quote, ratios, history = await asyncio.gather(
        _get_quote(ticker, provider, cache),
        _get_ratios(ticker, provider, cache),
        _get_history(ticker, provider, cache, history_period),
    )

    cache_hits = {
        "quote": isinstance(quote, _CacheHit),
        "ratios": isinstance(ratios, _CacheHit),
        "history": isinstance(history, _CacheHit),
    }
    return (
        quote.value if isinstance(quote, _CacheHit) else quote,
        ratios.value if isinstance(ratios, _CacheHit) else ratios,
        history.value if isinstance(history, _CacheHit) else history,
        cache_hits,
    )


class _CacheHit:
    """Thin wrapper to distinguish cache-hit vs live-fetch returns."""
    def __init__(self, value):
        self.value = value


async def _get_quote(ticker: str, provider: MarketDataProvider, cache: SQLiteCache):
    cached = await cache.get_market_data(ticker, "quote", "current")
    if cached is not None:
        q = cached
        return _CacheHit(Quote(
            ticker=q["ticker"],
            price=q["price"],
            change_pct=q["change_pct"],
            volume=q["volume"],
            market_cap=q.get("market_cap"),
            as_of=q["as_of"],
        ))
    try:
        quote = await provider.fetch_quote(ticker)
        await cache.store_market_data(ticker, "quote", "current", {
            "ticker": quote.ticker,
            "price": quote.price,
            "change_pct": quote.change_pct,
            "volume": quote.volume,
            "market_cap": quote.market_cap,
            "as_of": quote.as_of,
        })
        return quote
    except Exception:
        return None


async def _get_ratios(ticker: str, provider: MarketDataProvider, cache: SQLiteCache):
    cached = await cache.get_market_data(ticker, "ratios", "current")
    if cached is not None:
        r = cached
        return _CacheHit(Ratios(
            ticker=r["ticker"],
            pe_ratio=r.get("pe_ratio"),
            pb_ratio=r.get("pb_ratio"),
            week_52_high=r.get("week_52_high"),
            week_52_low=r.get("week_52_low"),
            beta=r.get("beta"),
        ))
    try:
        ratios = await provider.fetch_ratios(ticker)
        await cache.store_market_data(ticker, "ratios", "current", {
            "ticker": ratios.ticker,
            "pe_ratio": ratios.pe_ratio,
            "pb_ratio": ratios.pb_ratio,
            "week_52_high": ratios.week_52_high,
            "week_52_low": ratios.week_52_low,
            "beta": ratios.beta,
        })
        return ratios
    except Exception:
        return None


async def _get_history(
    ticker: str, provider: MarketDataProvider, cache: SQLiteCache, period: str
):
    cached = await cache.get_market_data(ticker, "history", period)
    if cached is not None:
        bars = [PriceBar(**b) for b in cached.get("bars", [])]
        return _CacheHit(MarketHistory(ticker=cached["ticker"], period=cached["period"], bars=bars))
    try:
        history = await provider.fetch_history(ticker, period=period)
        await cache.store_market_data(ticker, "history", period, {
            "ticker": history.ticker,
            "period": history.period,
            "bars": [
                {"date": b.date, "open": b.open, "high": b.high,
                 "low": b.low, "close": b.close, "volume": b.volume}
                for b in history.bars
            ],
        })
        return history
    except Exception:
        return None
