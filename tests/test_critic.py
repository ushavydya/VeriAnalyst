"""Tests for the critic agent (mocked — no real LLM calls)."""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from sec_analyzer.agents.critic import Critique, _parse_llm_response, _rule_checks, critique
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


# ── _parse_llm_response ───────────────────────────────────────────────────────

def test_parse_llm_response_valid():
    payload = json.dumps({
        "confidence": 0.85,
        "issues": ["Revenue figure seems slightly low vs prior year"],
        "summary": "Extraction looks mostly accurate.",
    })
    conf, issues, summary = _parse_llm_response(payload)
    assert conf == 0.85
    assert len(issues) == 1
    assert "accurate" in summary


def test_parse_llm_response_clamps_confidence():
    payload = json.dumps({"confidence": 1.5, "issues": [], "summary": "Great."})
    conf, _, _ = _parse_llm_response(payload)
    assert conf == 1.0


def test_parse_llm_response_strips_fences():
    payload = "```json\n{\"confidence\": 0.9, \"issues\": [], \"summary\": \"OK\"}\n```"
    conf, issues, summary = _parse_llm_response(payload)
    assert conf == 0.9


def test_parse_llm_response_bad_json_returns_defaults():
    conf, issues, _ = _parse_llm_response("I cannot provide that.")
    assert conf == 0.5
    assert issues  # contains a "unparseable" notice


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
