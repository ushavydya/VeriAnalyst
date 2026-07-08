"""Tests for the critic agent (mocked — no real LLM calls)."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from sec_analyzer.agents.critic import (
    Critique,
    _divergence_checks,
    _market_checks,
    _rule_checks,
    critique,
)
from sec_analyzer.agents.extractor import ExtractedData
from sec_analyzer.gateway.base import ModelResponse


# ── _rule_checks ──────────────────────────────────────────────────────────────

def test_rule_checks_clean_passes():
    metrics = {
        "revenue": 394328.0,
        "gross_profit": 169148.0,
        "net_income": 96995.0,
        "eps_basic": 6.16,
        "eps_diluted": 6.13,
        "total_assets": 352583.0,
        "total_liabilities": 290437.0,
        "total_equity": 62146.0,
    }
    assert _rule_checks(metrics) == []


def test_rule_checks_missing_required():
    issues = _rule_checks({"gross_profit": 100.0})
    assert any("revenue" in i for i in issues)
    assert any("net_income" in i for i in issues)


def test_rule_checks_gross_profit_exceeds_revenue():
    issues = _rule_checks({"revenue": 100.0, "net_income": 10.0, "gross_profit": 200.0})
    assert any("Gross profit" in i for i in issues)


def test_rule_checks_diluted_eps_exceeds_basic():
    issues = _rule_checks({"revenue": 1.0, "net_income": 0.1, "eps_basic": 1.00, "eps_diluted": 1.50})
    assert any("Diluted EPS" in i for i in issues)


def test_rule_checks_balance_sheet_mismatch():
    issues = _rule_checks({
        "revenue": 100.0, "net_income": 10.0,
        "total_assets": 500.0,
        "total_liabilities": 100.0,
        "total_equity": 100.0,  # 100+100=200 ≠ 500
    })
    assert any("balance" in i.lower() for i in issues)


# ── net_income threshold (tightened to 75% of revenue) ───────────────────────

def test_rule_checks_net_income_exceeds_75pct_revenue():
    # 80% net margin should fire
    issues = _rule_checks({"revenue": 100.0, "net_income": 80.0})
    assert any("75%" in i for i in issues)


def test_rule_checks_net_income_below_75pct_passes():
    # 74% net margin is high but should not fire
    issues = _rule_checks({"revenue": 100.0, "net_income": 74.0})
    assert not any("75%" in i for i in issues)


def test_rule_checks_large_loss_fires():
    # Large losses (negative net income > 75% of revenue) should also fire
    issues = _rule_checks({"revenue": 100.0, "net_income": -80.0})
    assert any("75%" in i for i in issues)


def test_rule_checks_nvda_like_margins_pass():
    # NVDA FY2026: net_income=120067, revenue=215938 → 55.6% margin → should NOT flag
    issues = _rule_checks({"revenue": 215938.0, "net_income": 120067.0})
    assert not any("75%" in i for i in issues)


# ── critique (integration, mocked gateway) ───────────────────────────────────

def _make_langfuse_mock():
    span = MagicMock()
    span.update = MagicMock()
    ctx = AsyncMock()
    ctx.__aenter__ = AsyncMock(return_value=span)
    ctx.__aexit__ = AsyncMock(return_value=False)
    lf = MagicMock()
    lf.start_as_current_observation = MagicMock(return_value=ctx)
    return lf


def _good_extracted_data() -> ExtractedData:
    return ExtractedData(
        ticker="AAPL",
        filed_date="2023-10-27",
        sections={"mda": "Revenue increased 10% driven by iPhone sales."},
        metrics={
            "revenue": 394328.0,
            "net_income": 96995.0,
            "eps_basic": 6.16,
            "eps_diluted": 6.13,
        },
    )


async def test_critique_returns_critique_object():
    lf = _make_langfuse_mock()
    tc = MagicMock()
    # LLM now returns a plain-text summary sentence, not JSON
    fake_resp = ModelResponse(
        text="Extraction looks complete with all required metrics present.",
        model="qwen2.5:7b",
        input_tokens=100,
        output_tokens=20,
    )
    gw = AsyncMock()
    gw.complete = AsyncMock(return_value=fake_resp)

    result = await critique(_good_extracted_data(), lf, tc, gateway=gw)

    assert isinstance(result, Critique)
    assert result.ticker == "AAPL"
    assert 0.0 <= result.confidence <= 1.0
    assert isinstance(result.issues, list)
    assert result.summary


async def test_critique_penalises_confidence_when_rules_fire():
    lf = _make_langfuse_mock()
    tc = MagicMock()

    fake_resp = ModelResponse(text="Gross profit exceeds revenue — likely extraction error.", model="qwen2.5:7b")
    gw = AsyncMock()
    gw.complete = AsyncMock(return_value=fake_resp)

    bad_data = ExtractedData(
        ticker="BAD",
        filed_date="2023-01-01",
        sections={"mda": "Some text."},
        metrics={"revenue": 100.0, "net_income": 10.0, "gross_profit": 999.0},
    )
    result = await critique(bad_data, lf, tc, gateway=gw)

    assert result.confidence <= 0.75   # penalised by rule violation
    assert any("Gross profit" in i for i in result.issues)


async def test_critique_survives_bad_llm_output():
    lf = _make_langfuse_mock()
    tc = MagicMock()
    fake_resp = ModelResponse(text="", model="qwen2.5:7b")
    gw = AsyncMock()
    gw.complete = AsyncMock(return_value=fake_resp)

    result = await critique(_good_extracted_data(), lf, tc, gateway=gw)
    assert isinstance(result, Critique)  # no crash even with empty LLM output


# ── _market_checks ────────────────────────────────────────────────────────────

import json as _json


def _mkt(price=200.0, pe=28.5, pb=45.0, hi=260.0, lo=164.0, beta=1.2) -> str:
    return _json.dumps({
        "quote": {"price": price, "change_pct": 1.0, "volume": 1_000_000},
        "ratios": {"pe_ratio": pe, "pb_ratio": pb, "week_52_high": hi, "week_52_low": lo, "beta": beta},
    })


def test_market_checks_clean_passes():
    assert _market_checks(_mkt()) == []


def test_market_checks_none_input_passes():
    assert _market_checks(None) == []


def test_market_checks_non_positive_price():
    issues = _market_checks(_mkt(price=0.0))
    assert any("non-positive" in i for i in issues)


def test_market_checks_implausibly_high_pe():
    issues = _market_checks(_mkt(pe=600.0))
    assert any("implausibly high" in i for i in issues)


def test_market_checks_negative_pe_flagged():
    issues = _market_checks(_mkt(pe=-5.0))
    assert any("negative P/E" in i for i in issues)


def test_market_checks_beta_out_of_range():
    issues = _market_checks(_mkt(beta=7.0))
    assert any("beta" in i for i in issues)


def test_market_checks_52w_low_exceeds_high():
    issues = _market_checks(_mkt(hi=100.0, lo=200.0))
    assert any("52-week low" in i and "52-week high" in i for i in issues)


def test_market_checks_price_above_52w_high():
    issues = _market_checks(_mkt(price=300.0, hi=260.0, lo=164.0))
    assert any("52-week high" in i for i in issues)


# ── _divergence_checks ────────────────────────────────────────────────────────

def _news(score: float) -> str:
    return _json.dumps({"sentiment_score": score, "articles": []})


def test_divergence_bearish_sentiment_profitable_company():
    metrics = {"revenue": 100.0, "net_income": 10.0}
    signals = _divergence_checks(metrics, None, _news(-0.5))
    assert any("bearish" in s for s in signals)


def test_divergence_bullish_sentiment_loss_making():
    metrics = {"revenue": 100.0, "net_income": -20.0}
    signals = _divergence_checks(metrics, None, _news(0.6))
    assert any("bullish" in s for s in signals)


def test_divergence_neutral_sentiment_no_signal():
    metrics = {"revenue": 100.0, "net_income": 10.0}
    signals = _divergence_checks(metrics, None, _news(0.1))
    assert signals == []


def test_divergence_near_52w_low_profitable():
    metrics = {"revenue": 100.0, "net_income": 5.0}
    mkt = _json.dumps({
        "quote": {"price": 165.0},
        "ratios": {"week_52_high": 260.0, "week_52_low": 164.0},
    })
    signals = _divergence_checks(metrics, mkt, None)
    assert any("52-week low" in s for s in signals)


def test_divergence_no_market_no_news_no_signals():
    metrics = {"revenue": 100.0, "net_income": 10.0}
    assert _divergence_checks(metrics, None, None) == []


# ── critique passes market/news through to issues ─────────────────────────────

async def test_critique_includes_market_warnings_in_issues():
    lf = _make_langfuse_mock()
    tc = MagicMock()
    fake_resp = ModelResponse(text="Metrics look complete.", model="qwen2.5:7b")
    gw = AsyncMock()
    gw.complete = AsyncMock(return_value=fake_resp)

    bad_market = _mkt(price=0.0)  # triggers non-positive price warning
    result = await critique(_good_extracted_data(), lf, tc, gateway=gw, market_json=bad_market)

    assert any("non-positive" in i for i in result.issues)


async def test_critique_market_warnings_do_not_penalise_confidence():
    """Market warnings are informational — they must not reduce the confidence score."""
    lf = _make_langfuse_mock()
    tc = MagicMock()
    fake_resp = ModelResponse(text="Good extraction.", model="qwen2.5:7b")
    gw = AsyncMock()
    gw.complete = AsyncMock(return_value=fake_resp)

    bad_market = _mkt(pe=999.0)  # implausibly high P/E warning
    result_with = await critique(_good_extracted_data(), lf, tc, gateway=gw, market_json=bad_market)
    result_without = await critique(_good_extracted_data(), lf, tc, gateway=gw)

    assert result_with.confidence == result_without.confidence


async def test_critique_includes_divergence_signals_in_issues():
    lf = _make_langfuse_mock()
    tc = MagicMock()
    fake_resp = ModelResponse(text="Good extraction.", model="qwen2.5:7b")
    gw = AsyncMock()
    gw.complete = AsyncMock(return_value=fake_resp)

    bearish_news = _news(-0.5)  # bearish while company is profitable → divergence
    result = await critique(_good_extracted_data(), lf, tc, gateway=gw, news_json=bearish_news)

    assert any("bearish" in i for i in result.issues)
