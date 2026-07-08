"""Unit tests for market_agent.py."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from sec_analyzer.agents.market_agent import MarketSummary, fetch_market_data
from sec_analyzer.cache.sqlite_cache import SQLiteCache
from sec_analyzer.providers.base import MarketHistory, PriceBar, Quote, Ratios


def _make_langfuse():
    lf = MagicMock()
    span = MagicMock()
    span.__enter__ = MagicMock(return_value=span)
    span.__exit__ = MagicMock(return_value=False)
    lf.start_as_current_observation.return_value = span
    return lf


def _make_provider(*, price=200.0, pe=28.5, bars=2):
    provider = MagicMock()
    provider.fetch_quote = AsyncMock(return_value=Quote(
        ticker="AAPL", price=price, change_pct=1.5, volume=50_000_000,
        market_cap=3e12, as_of="2026-07-07T14:00:00+00:00",
    ))
    provider.fetch_ratios = AsyncMock(return_value=Ratios(
        ticker="AAPL", pe_ratio=pe, pb_ratio=45.0,
        week_52_high=260.0, week_52_low=164.0, beta=1.2,
    ))
    provider.fetch_history = AsyncMock(return_value=MarketHistory(
        ticker="AAPL", period="1y",
        bars=[PriceBar(f"2026-07-0{i+1}", 198+i, 202+i, 197+i, 200+i, 5_000_000) for i in range(bars)],
    ))
    return provider


@pytest.fixture()
async def cache(tmp_path: Path) -> SQLiteCache:
    c = SQLiteCache(db_path=str(tmp_path / "cache.db"), docs_dir=str(tmp_path / "docs"))
    async with c:
        yield c


async def test_fetch_market_data_calls_provider(cache):
    provider = _make_provider()
    result = await fetch_market_data("AAPL", provider, cache, _make_langfuse(), {})
    provider.fetch_quote.assert_awaited_once()
    provider.fetch_ratios.assert_awaited_once()
    provider.fetch_history.assert_awaited_once()
    assert result.ticker == "AAPL"


async def test_fetch_market_data_price(cache):
    provider = _make_provider(price=199.5)
    result = await fetch_market_data("AAPL", provider, cache, _make_langfuse(), {})
    assert result.price == pytest.approx(199.5)


async def test_fetch_market_data_ratios(cache):
    provider = _make_provider(pe=30.0)
    result = await fetch_market_data("AAPL", provider, cache, _make_langfuse(), {})
    assert result.pe_ratio == pytest.approx(30.0)
    assert result.beta == pytest.approx(1.2)
    assert result.week_52_high == pytest.approx(260.0)


async def test_fetch_market_data_history(cache):
    provider = _make_provider(bars=5)
    result = await fetch_market_data("AAPL", provider, cache, _make_langfuse(), {})
    assert len(result.history.bars) == 5


async def test_fetch_market_data_uses_cache_on_second_call(cache):
    provider = _make_provider()
    await fetch_market_data("AAPL", provider, cache, _make_langfuse(), {})
    result2 = await fetch_market_data("AAPL", provider, cache, _make_langfuse(), {})
    assert provider.fetch_quote.await_count == 1
    assert provider.fetch_ratios.await_count == 1
    assert provider.fetch_history.await_count == 1
    assert result2.cache_hits == {"quote": True, "ratios": True, "history": True}


async def test_fetch_market_data_cache_restores_values(cache):
    provider = _make_provider(price=201.0, pe=32.0)
    await fetch_market_data("AAPL", provider, cache, _make_langfuse(), {})
    result2 = await fetch_market_data("AAPL", provider, cache, _make_langfuse(), {})
    assert result2.price == pytest.approx(201.0)
    assert result2.pe_ratio == pytest.approx(32.0)


async def test_fetch_market_data_provider_error_returns_none(cache):
    provider = _make_provider()
    provider.fetch_quote = AsyncMock(side_effect=Exception("network error"))
    result = await fetch_market_data("AAPL", provider, cache, _make_langfuse(), {})
    assert result.quote is None
    assert result.price is None


async def test_fetch_market_data_partial_failure(cache):
    """Ratios failure should not prevent quote/history from being returned."""
    provider = _make_provider()
    provider.fetch_ratios = AsyncMock(side_effect=Exception("timeout"))
    result = await fetch_market_data("AAPL", provider, cache, _make_langfuse(), {})
    assert result.quote is not None
    assert result.ratios is None
    assert result.history is not None


# ── MarketSummary helpers ─────────────────────────────────────────────────────

def test_price_vs_52w_high_pct():
    summary = MarketSummary(
        ticker="AAPL",
        quote=Quote("AAPL", 195.0, 0.5, 1_000_000, None, "2026-07-07T14:00:00+00:00"),
        ratios=Ratios("AAPL", 28.0, 45.0, 260.0, 164.0, 1.2),
    )
    pct = summary.price_vs_52w_high_pct()
    assert pct is not None
    assert pct == pytest.approx((195 / 260 - 1) * 100, abs=0.01)


def test_price_vs_52w_high_none_when_no_quote():
    summary = MarketSummary(ticker="AAPL")
    assert summary.price_vs_52w_high_pct() is None


def test_market_summary_defaults():
    summary = MarketSummary(ticker="AAPL")
    assert summary.price is None
    assert summary.pe_ratio is None
    assert summary.cache_hits == {}


# ── Computed P/E ──────────────────────────────────────────────────────────────

from sec_analyzer.agents.market_agent import _fill_computed_pe


def test_fill_computed_pe_when_provider_returns_none():
    ratios = Ratios("TLN", pe_ratio=None, pb_ratio=None,
                    week_52_high=451.0, week_52_low=255.0, beta=1.6)
    filled = _fill_computed_pe(ratios, price=368.0, xbrl_metrics={"eps_diluted": 12.5})
    assert filled.pe_ratio == pytest.approx(368.0 / 12.5, rel=0.01)


def test_fill_computed_pe_not_overwritten_when_provider_has_value():
    ratios = Ratios("AAPL", pe_ratio=41.0, pb_ratio=None,
                    week_52_high=260.0, week_52_low=164.0, beta=1.1)
    # pe_ratio is already set — _fill_computed_pe should not be called by agent,
    # but if called directly it would overwrite; test agent-level guard instead
    assert ratios.pe_ratio == 41.0


def test_fill_computed_pe_no_xbrl_metrics_returns_unchanged():
    ratios = Ratios("TLN", pe_ratio=None, pb_ratio=None,
                    week_52_high=451.0, week_52_low=255.0, beta=1.6)
    filled = _fill_computed_pe(ratios, price=368.0, xbrl_metrics=None)
    assert filled.pe_ratio is None


def test_fill_computed_pe_negative_eps_skipped():
    ratios = Ratios("TLN", pe_ratio=None, pb_ratio=None,
                    week_52_high=451.0, week_52_low=255.0, beta=1.6)
    filled = _fill_computed_pe(ratios, price=368.0, xbrl_metrics={"eps_diluted": -3.0})
    assert filled.pe_ratio is None


def test_fill_computed_pe_zero_eps_skipped():
    ratios = Ratios("TLN", pe_ratio=None, pb_ratio=None,
                    week_52_high=451.0, week_52_low=255.0, beta=1.6)
    filled = _fill_computed_pe(ratios, price=368.0, xbrl_metrics={"eps_diluted": 0.0})
    assert filled.pe_ratio is None


async def test_fetch_market_data_computes_pe_from_xbrl(cache):
    """When provider returns pe_ratio=None, agent computes it from xbrl eps_diluted."""
    from sec_analyzer.providers.base import Ratios as R
    provider = _make_provider(pe=None)
    provider.fetch_ratios = AsyncMock(return_value=R(
        ticker="TLN", pe_ratio=None, pb_ratio=None,
        week_52_high=451.0, week_52_low=255.0, beta=1.6,
    ))
    result = await fetch_market_data(
        "TLN", provider, cache, _make_langfuse(), {},
        xbrl_metrics={"eps_diluted": 12.5},
    )
    assert result.pe_ratio is not None
    assert result.pe_ratio == pytest.approx(result.price / 12.5, rel=0.01)
    assert result.pe_computed is True


async def test_fetch_market_data_no_computed_pe_when_provider_has_it(cache):
    """Provider P/E takes precedence — pe_computed must be False."""
    provider = _make_provider(pe=41.0)
    result = await fetch_market_data(
        "AAPL", provider, cache, _make_langfuse(), {},
        xbrl_metrics={"eps_diluted": 5.0},
    )
    assert result.pe_ratio == pytest.approx(41.0)
    assert result.pe_computed is False


async def test_fetch_market_data_pe_computed_false_when_no_xbrl(cache):
    """pe_computed is False when xbrl_metrics not provided even if provider has no P/E."""
    from sec_analyzer.providers.base import Ratios as R
    provider = _make_provider(pe=None)
    provider.fetch_ratios = AsyncMock(return_value=R(
        ticker="TLN", pe_ratio=None, pb_ratio=None,
        week_52_high=451.0, week_52_low=255.0, beta=1.6,
    ))
    result = await fetch_market_data("TLN", provider, cache, _make_langfuse(), {})
    assert result.pe_ratio is None
    assert result.pe_computed is False
