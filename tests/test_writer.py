"""Tests for the writer agent (mocked — no real LLM calls)."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from sec_analyzer.agents.critic import Critique
from sec_analyzer.agents.extractor import ExtractedData
from sec_analyzer.agents.market_agent import MarketSummary
from sec_analyzer.agents.news_agent import NewsSummary
from sec_analyzer.agents.writer import _format_metrics, _format_news, write_report
from sec_analyzer.gateway.base import ModelResponse
from sec_analyzer.providers.base import NewsArticle, Quote, Ratios


# ── _format_metrics ───────────────────────────────────────────────────────────

def test_format_metrics_known_keys():
    table = _format_metrics({"revenue": 394328.0, "net_income": 96995.0})
    assert "Revenue" in table
    assert "394,328.0" in table


def test_format_metrics_skips_unknown_keys():
    table = _format_metrics({"unknown_field": 999.0, "revenue": 1.0})
    assert "unknown_field" not in table


def test_format_metrics_empty():
    assert "_No metrics extracted._" in _format_metrics({})


def test_format_metrics_fiscal_year_end_as_string():
    table = _format_metrics({"fiscal_year_end": "2023-09-30", "revenue": 1.0})
    assert "2023-09-30" in table


def test_format_metrics_empty_returns_placeholder():
    assert "_No metrics extracted._" in _format_metrics({})


def test_format_metrics_includes_revenue_in_millions():
    table = _format_metrics({"revenue": 100_000.0})
    assert "100,000.0" in table


def test_format_metrics_eps_shows_two_decimal_places():
    table = _format_metrics({"eps_diluted": 6.13})
    assert "6.1" in table


def test_format_metrics_skips_missing_keys():
    table = _format_metrics({"revenue": 1.0})
    assert "EPS" not in table


# ── _format_news ──────────────────────────────────────────────────────────────

def _make_news(sentiment=0.4, narrative=None, articles=None) -> NewsSummary:
    arts = articles or [
        NewsArticle("AAPL beats Q4 estimates", "EPS of $1.46 beat.", "http://r.co", "Reuters", "2026-07-07"),
    ]
    return NewsSummary("AAPL", "2026-07-07", articles=arts,
                       sentiment_score=sentiment, narrative=narrative)


def test_format_news_none_returns_placeholder():
    assert "_No recent news available._" in _format_news(None)


def test_format_news_empty_articles_returns_placeholder():
    news = NewsSummary("AAPL", "2026-07-07", articles=[])
    assert "_No recent news available._" in _format_news(news)


def test_format_news_shows_sentiment_label():
    result = _format_news(_make_news(sentiment=0.5))
    assert "bullish" in result.lower()


def test_format_news_shows_sentiment_score():
    result = _format_news(_make_news(sentiment=0.5))
    assert "+0.50" in result


def test_format_news_shows_narrative_when_present():
    result = _format_news(_make_news(narrative="Strong results lifted sentiment. Guidance was raised."))
    assert "Strong results lifted sentiment" in result


def test_format_news_no_narrative_when_absent():
    result = _format_news(_make_news(narrative=None))
    # Should not crash and should still show articles
    assert "AAPL beats Q4 estimates" in result


def test_format_news_shows_headlines():
    result = _format_news(_make_news())
    assert "AAPL beats Q4 estimates" in result


def test_format_news_limits_to_five_articles():
    arts = [
        NewsArticle(f"Headline {i}", "", f"http://{i}", "Reuters", "2026-07-07")
        for i in range(10)
    ]
    result = _format_news(_make_news(articles=arts))
    # Only first 5 shown
    assert "Headline 4" in result
    assert "Headline 5" not in result


# ── write_report (integration, mocked gateway) ───────────────────────────────

def _make_langfuse_mock():
    span = MagicMock()
    span.update = MagicMock()
    ctx = AsyncMock()
    ctx.__aenter__ = AsyncMock(return_value=span)
    ctx.__aexit__ = AsyncMock(return_value=False)
    lf = MagicMock()
    lf.start_as_current_observation = MagicMock(return_value=ctx)
    return lf


def _sample_data(fy_end="2025-09-27") -> ExtractedData:
    return ExtractedData(
        ticker="AAPL",
        filed_date="2025-10-31",
        sections={
            "business": "Apple designs and sells iPhones, Macs, and services.",
            "mda": "Net sales were $394.3B, up 2% year-over-year.",
            "risk_factors": "Competition, supply chain disruptions, regulatory changes.",
        },
        metrics={"revenue": 416161.0, "net_income": 112010.0, "eps_diluted": 7.46,
                 "fiscal_year_end": fy_end},
    )


def _sample_critique() -> Critique:
    return Critique(
        ticker="AAPL",
        confidence=0.88,
        issues=[],
        summary="Extraction looks accurate and complete.",
    )


async def test_write_report_returns_string():
    lf = _make_langfuse_mock()
    tc = MagicMock()
    fake_resp = ModelResponse(
        text="# AAPL — 10-K Analysis\n\n## Executive Summary\nApple had a great year.",
        model="qwen2.5:7b",
        input_tokens=500,
        output_tokens=200,
    )
    gw = AsyncMock()
    gw.complete = AsyncMock(return_value=fake_resp)

    report = await write_report(_sample_data(), _sample_critique(), lf, tc, gateway=gw)

    assert isinstance(report, str)
    assert len(report) > 0
    assert report.startswith("#")


async def test_write_report_adds_heading_when_missing():
    lf = _make_langfuse_mock()
    tc = MagicMock()
    fake_resp = ModelResponse(
        text="Apple had a great fiscal year with record revenue.",
        model="qwen2.5:7b",
    )
    gw = AsyncMock()
    gw.complete = AsyncMock(return_value=fake_resp)

    report = await write_report(_sample_data(), _sample_critique(), lf, tc, gateway=gw)

    assert report.startswith("# AAPL")


async def test_write_report_passes_ticker_and_confidence_to_langfuse():
    lf = _make_langfuse_mock()
    tc = MagicMock()
    fake_resp = ModelResponse(text="# Report", model="qwen2.5:7b")
    gw = AsyncMock()
    gw.complete = AsyncMock(return_value=fake_resp)

    await write_report(_sample_data(), _sample_critique(), lf, tc, gateway=gw)

    call_kwargs = lf.start_as_current_observation.call_args.kwargs
    assert call_kwargs["input"]["ticker"] == "AAPL"
    assert call_kwargs["input"]["confidence"] == 0.88


async def test_write_report_fiscal_year_label_correct():
    """Prompt must say FY2025 for a filing with fiscal_year_end=2025-09-27."""
    lf = _make_langfuse_mock()
    tc = MagicMock()
    fake_resp = ModelResponse(text="# Report", model="qwen2.5:7b")
    gw = AsyncMock()
    gw.complete = AsyncMock(return_value=fake_resp)

    await write_report(_sample_data(fy_end="2025-09-27"), _sample_critique(), lf, tc, gateway=gw)

    prompt_text = gw.complete.call_args.args[0][0]["content"]
    assert "FY2025" in prompt_text
    # Must NOT say FY26 (common LLM mistake when inferring from calendar year)
    assert "FY26" not in prompt_text and "FY2026" not in prompt_text


async def test_write_report_fallback_heading_uses_fy_label():
    """When LLM omits heading, the injected heading uses FY2025 not raw date."""
    lf = _make_langfuse_mock()
    tc = MagicMock()
    fake_resp = ModelResponse(text="Some report without a heading.", model="qwen2.5:7b")
    gw = AsyncMock()
    gw.complete = AsyncMock(return_value=fake_resp)

    report = await write_report(_sample_data(fy_end="2025-09-27"), _sample_critique(), lf, tc, gateway=gw)

    assert "FY2025" in report
    assert "FY26" not in report


async def test_write_report_with_news_includes_has_news_flag():
    lf = _make_langfuse_mock()
    tc = MagicMock()
    fake_resp = ModelResponse(text="# Report", model="qwen2.5:7b")
    gw = AsyncMock()
    gw.complete = AsyncMock(return_value=fake_resp)

    news = _make_news(narrative="Positive news dominated coverage.")
    await write_report(_sample_data(), _sample_critique(), lf, tc, gateway=gw, news=news)

    call_kwargs = lf.start_as_current_observation.call_args.kwargs
    assert call_kwargs["input"]["has_news"] is True


async def test_write_report_narrative_appears_in_prompt():
    """The LLM prompt should include the news narrative when available."""
    lf = _make_langfuse_mock()
    tc = MagicMock()
    fake_resp = ModelResponse(text="# Report", model="qwen2.5:7b")
    gw = AsyncMock()
    gw.complete = AsyncMock(return_value=fake_resp)

    news = _make_news(narrative="Record EPS drove a rally. Analysts raised targets.")
    await write_report(_sample_data(), _sample_critique(), lf, tc, gateway=gw, news=news)

    prompt_text = gw.complete.call_args.args[0][0]["content"]
    assert "Record EPS drove a rally" in prompt_text


async def test_write_report_with_market_passes_flag():
    lf = _make_langfuse_mock()
    tc = MagicMock()
    fake_resp = ModelResponse(text="# Report", model="qwen2.5:7b")
    gw = AsyncMock()
    gw.complete = AsyncMock(return_value=fake_resp)

    market = MarketSummary(
        ticker="AAPL",
        quote=Quote("AAPL", 210.0, 0.5, 50_000_000, None, "2026-07-08T14:00:00+00:00"),
        ratios=Ratios("AAPL", 33.0, 45.0, 260.0, 164.0, 1.2),
    )
    await write_report(_sample_data(), _sample_critique(), lf, tc, gateway=gw, market=market)

    call_kwargs = lf.start_as_current_observation.call_args.kwargs
    assert call_kwargs["input"]["has_market"] is True
