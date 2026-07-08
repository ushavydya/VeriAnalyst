"""Tests for the critic agent (mocked — no real LLM calls)."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from sec_analyzer.agents.critic import Critique, _rule_checks, critique
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
