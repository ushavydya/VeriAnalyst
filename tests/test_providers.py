"""Unit tests for providers/base.py interfaces and providers/__init__.py factory."""
from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

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


# ── ABC enforcement ────────────────────────────────────────────────────────────

def test_news_provider_is_abstract():
    with pytest.raises(TypeError):
        NewsProvider()  # type: ignore[abstract]


def test_market_data_provider_is_abstract():
    with pytest.raises(TypeError):
        MarketDataProvider()  # type: ignore[abstract]


def test_concrete_news_provider_must_implement_fetch_news():
    class Incomplete(NewsProvider):
        pass

    with pytest.raises(TypeError):
        Incomplete()


def test_concrete_market_provider_must_implement_all_methods():
    class Incomplete(MarketDataProvider):
        async def fetch_quote(self, ticker): ...

    with pytest.raises(TypeError):
        Incomplete()


# ── Dataclass defaults ────────────────────────────────────────────────────────

def test_market_history_bars_default_empty():
    h = MarketHistory(ticker="AAPL", period="1y")
    assert h.bars == []


def test_news_result_fields():
    r = NewsResult(ticker="AAPL", articles=[], sentiment_score=0.5)
    assert r.ticker == "AAPL"
    assert r.sentiment_score == 0.5


def test_quote_fields():
    q = Quote(ticker="AAPL", price=200.0, change_pct=1.5, volume=1_000_000,
               market_cap=3e12, as_of="2026-07-07T14:00:00+00:00")
    assert q.price == 200.0
    assert q.market_cap == 3e12


def test_price_bar_fields():
    bar = PriceBar(date="2026-07-07", open=198.0, high=202.0, low=197.0, close=200.0, volume=5_000_000)
    assert bar.close == 200.0


def test_ratios_nullable_fields():
    r = Ratios(ticker="AAPL", pe_ratio=None, pb_ratio=None,
               week_52_high=None, week_52_low=None, beta=None)
    assert r.pe_ratio is None


# ── Factory: get_news_provider ────────────────────────────────────────────────

def test_get_news_provider_finnhub(monkeypatch):
    monkeypatch.setenv("NEWS_PROVIDER", "finnhub")
    monkeypatch.setenv("FINNHUB_API_KEY", "test-key")
    from sec_analyzer.providers import get_news_provider
    from sec_analyzer.providers.finnhub import FinnhubProvider
    provider = get_news_provider()
    assert isinstance(provider, FinnhubProvider)


def test_get_news_provider_default_is_finnhub(monkeypatch):
    monkeypatch.delenv("NEWS_PROVIDER", raising=False)
    monkeypatch.setenv("FINNHUB_API_KEY", "test-key")
    from sec_analyzer.providers import get_news_provider
    from sec_analyzer.providers.finnhub import FinnhubProvider
    provider = get_news_provider()
    assert isinstance(provider, FinnhubProvider)


def test_get_news_provider_missing_key_raises(monkeypatch):
    monkeypatch.setenv("NEWS_PROVIDER", "finnhub")
    monkeypatch.delenv("FINNHUB_API_KEY", raising=False)
    from sec_analyzer.providers import get_news_provider
    with pytest.raises(EnvironmentError, match="FINNHUB_API_KEY"):
        get_news_provider()


def test_get_news_provider_unknown_raises(monkeypatch):
    monkeypatch.setenv("NEWS_PROVIDER", "newsapi")
    monkeypatch.setenv("FINNHUB_API_KEY", "test-key")
    from sec_analyzer.providers import get_news_provider
    with pytest.raises(ValueError, match="newsapi"):
        get_news_provider()


def test_get_market_data_provider_finnhub(monkeypatch):
    monkeypatch.setenv("MARKET_DATA_PROVIDER", "finnhub")
    monkeypatch.setenv("FINNHUB_API_KEY", "test-key")
    from sec_analyzer.providers import get_market_data_provider
    from sec_analyzer.providers.finnhub import FinnhubProvider
    provider = get_market_data_provider()
    assert isinstance(provider, FinnhubProvider)


def test_get_market_data_provider_unknown_raises(monkeypatch):
    monkeypatch.setenv("MARKET_DATA_PROVIDER", "yfinance")
    monkeypatch.setenv("FINNHUB_API_KEY", "test-key")
    from sec_analyzer.providers import get_market_data_provider
    with pytest.raises(ValueError, match="yfinance"):
        get_market_data_provider()


# ── FinnhubProvider (mocked HTTP) ─────────────────────────────────────────────

@pytest.fixture()
def finnhub():
    from sec_analyzer.providers.finnhub import FinnhubProvider
    return FinnhubProvider(api_key="test-key")


def _mock_response(json_data: object, status_code: int = 200):
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data
    resp.raise_for_status = MagicMock()
    return resp


async def test_fetch_news_parses_articles(finnhub):
    news_json = [
        {"headline": "AAPL rises", "summary": "Details", "url": "http://x", "source": "Reuters", "datetime": 1751900000}
    ]
    sentiment_json = {"buzz": {"bullishPercent": 0.7, "bearishPercent": 0.2}}

    with patch("sec_analyzer.providers.finnhub._gather", new=AsyncMock(
        return_value=[_mock_response(news_json), _mock_response(sentiment_json)]
    )):
        result = await finnhub.fetch_news("AAPL")

    assert result.ticker == "AAPL"
    assert len(result.articles) == 1
    assert result.articles[0].headline == "AAPL rises"
    assert result.sentiment_score == pytest.approx(0.5, abs=0.01)


async def test_fetch_news_no_sentiment_when_buzz_empty(finnhub):
    with patch("sec_analyzer.providers.finnhub._gather", new=AsyncMock(
        return_value=[_mock_response([]), _mock_response({})]
    )):
        result = await finnhub.fetch_news("AAPL")

    assert result.sentiment_score is None


async def test_fetch_news_limits_articles(finnhub):
    news_json = [{"headline": f"h{i}", "summary": "", "url": "", "source": "", "datetime": 0} for i in range(30)]
    with patch("sec_analyzer.providers.finnhub._gather", new=AsyncMock(
        return_value=[_mock_response(news_json), _mock_response({})]
    )):
        result = await finnhub.fetch_news("AAPL", max_articles=5)

    assert len(result.articles) == 5


async def test_fetch_quote_parses_fields(finnhub):
    quote_json = {"c": 200.5, "dp": 1.2, "v": 50_000_000, "t": 1751900000}
    resp = _mock_response(quote_json)
    with patch("httpx.AsyncClient") as mock_client:
        mock_client.return_value.__aenter__ = AsyncMock(return_value=mock_client.return_value)
        mock_client.return_value.__aexit__ = AsyncMock(return_value=False)
        mock_client.return_value.get = AsyncMock(return_value=resp)
        result = await finnhub.fetch_quote("AAPL")

    assert result.ticker == "AAPL"
    assert result.price == pytest.approx(200.5)
    assert result.change_pct == pytest.approx(1.2)
    assert result.volume == 50_000_000


async def test_fetch_ratios_parses_fields(finnhub):
    metric_json = {"metric": {"peNormalizedAnnual": 28.5, "pbAnnual": 45.0, "52WeekHigh": 260.0, "52WeekLow": 164.0, "beta": 1.2}}
    resp = _mock_response(metric_json)
    with patch("httpx.AsyncClient") as mock_client:
        mock_client.return_value.__aenter__ = AsyncMock(return_value=mock_client.return_value)
        mock_client.return_value.__aexit__ = AsyncMock(return_value=False)
        mock_client.return_value.get = AsyncMock(return_value=resp)
        result = await finnhub.fetch_ratios("AAPL")

    assert result.pe_ratio == pytest.approx(28.5)
    assert result.week_52_high == pytest.approx(260.0)
    assert result.beta == pytest.approx(1.2)


async def test_fetch_history_parses_bars(finnhub):
    candle_json = {
        "s": "ok",
        "t": [1751900000, 1751986400],
        "o": [198.0, 199.0],
        "h": [202.0, 203.0],
        "l": [197.0, 198.0],
        "c": [200.0, 201.0],
        "v": [5_000_000, 4_500_000],
    }
    resp = _mock_response(candle_json)
    with patch("httpx.AsyncClient") as mock_client:
        mock_client.return_value.__aenter__ = AsyncMock(return_value=mock_client.return_value)
        mock_client.return_value.__aexit__ = AsyncMock(return_value=False)
        mock_client.return_value.get = AsyncMock(return_value=resp)
        result = await finnhub.fetch_history("AAPL", period="1y")

    assert result.ticker == "AAPL"
    assert result.period == "1y"
    assert len(result.bars) == 2
    assert result.bars[0].close == pytest.approx(200.0)


async def test_fetch_history_no_data_returns_empty_bars(finnhub):
    candle_json = {"s": "no_data"}
    resp = _mock_response(candle_json)
    with patch("httpx.AsyncClient") as mock_client:
        mock_client.return_value.__aenter__ = AsyncMock(return_value=mock_client.return_value)
        mock_client.return_value.__aexit__ = AsyncMock(return_value=False)
        mock_client.return_value.get = AsyncMock(return_value=resp)
        result = await finnhub.fetch_history("AAPL")

    assert result.bars == []
