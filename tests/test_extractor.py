"""Tests for the extractor agent (mocked — no real LLM calls)."""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from sec_analyzer.agents.extractor import (
    ExtractedData,
    _format_metrics_for_prompt,
    _html_to_text,
    _parse_metrics,
    _split_sections,
    extract,
)
from sec_analyzer.gateway.base import ModelResponse


# ── _html_to_text ─────────────────────────────────────────────────────────────

def test_html_to_text_removes_tags():
    result = _html_to_text("<p>Hello <b>world</b></p>")
    assert "Hello" in result
    assert "world" in result
    assert "<" not in result


def test_html_to_text_preserves_table_structure():
    html = "<table><tr><td>Revenue</td><td>394,328</td></tr></table>"
    result = _html_to_text(html)
    assert "Revenue" in result
    assert "394,328" in result
    # cells should be tab-separated on same line
    assert "Revenue\t394,328" in result


# ── _split_sections ───────────────────────────────────────────────────────────

_FAKE_10K = """
<div>Some preamble text.</div>
<div>Item 1. Business</div><div>We sell widgets globally.</div>
<div>Item 1A. Risk Factors</div><div>Interest rates may rise.</div>
<div>Item 7. Management&#8217;s Discussion and Analysis</div>
<div>Revenue increased 10% year-over-year.</div>
<div>Item 8. Financial Statements and Supplementary Data</div>
<table><tr><td>Total assets</td><td>500</td></tr></table>
"""


def test_split_sections_finds_expected_keys():
    sections = _split_sections(_FAKE_10K)
    assert "business" in sections
    assert "risk_factors" in sections
    assert "mda" in sections
    assert "financials" in sections


def test_split_sections_content_is_truncated():
    long_doc = _FAKE_10K + ("x" * 20_000)
    sections = _split_sections(long_doc)
    for body in sections.values():
        assert len(body) <= 8_000


def test_split_sections_fallback_for_plain_text():
    sections = _split_sections("<div>no item headers here</div>")
    assert "raw" in sections


# ── _parse_metrics ────────────────────────────────────────────────────────────

def test_parse_metrics_valid_json():
    payload = json.dumps({"revenue": 394328.0, "net_income": 96995.0, "fiscal_year_end": "2023-09-30"})
    result = _parse_metrics(payload)
    assert result["revenue"] == 394328.0
    assert result["fiscal_year_end"] == "2023-09-30"


def test_parse_metrics_strips_markdown_fences():
    payload = "```json\n{\"revenue\": 100.0}\n```"
    assert _parse_metrics(payload)["revenue"] == 100.0


def test_parse_metrics_drops_null_values():
    payload = json.dumps({"revenue": 100.0, "eps_diluted": None})
    result = _parse_metrics(payload)
    assert "eps_diluted" not in result


def test_parse_metrics_invalid_json_returns_empty():
    assert _parse_metrics("not json at all") == {}


# ── extract (integration, mocked gateway) ────────────────────────────────────

def _make_langfuse_mock():
    span = MagicMock()
    span.update = MagicMock()

    ctx_mgr = MagicMock()
    ctx_mgr.__enter__ = MagicMock(return_value=span)
    ctx_mgr.__exit__ = MagicMock(return_value=False)

    lf = MagicMock()
    lf.start_as_current_observation = MagicMock(return_value=ctx_mgr)
    return lf


# Fake HTML 10-K that includes a simple financial table for rule-based parsing
_FAKE_10K_WITH_TABLES = _FAKE_10K + """
<table>
<tr><td>Total net sales</td><td>394,328</td></tr>
<tr><td>Net income</td><td>96,995</td></tr>
<tr><td>Diluted</td><td>6.16</td></tr>
</table>
"""


async def test_extract_populates_sections_and_metrics():
    lf = _make_langfuse_mock()
    trace_ctx = MagicMock()

    # LLM is called for qualitative summary only; metrics come from rule-based parser
    fake_response = ModelResponse(
        text="Apple designs iPhones and related services.", model="qwen2.5:7b",
        input_tokens=200, output_tokens=50,
    )
    gw = AsyncMock()
    gw.complete = AsyncMock(return_value=fake_response)

    result = await extract(
        ticker="AAPL",
        filed_date="2023-10-27",
        document_text=_FAKE_10K_WITH_TABLES,
        langfuse=lf,
        trace_context=trace_ctx,
        gateway=gw,
    )

    assert isinstance(result, ExtractedData)
    assert result.ticker == "AAPL"
    assert result.metrics["revenue"] == 394328.0
    assert result.metrics["net_income"] == 96995.0
    assert "business" in result.sections
    assert "summary" in result.sections


async def test_extract_falls_back_gracefully_on_bad_llm_output():
    lf = _make_langfuse_mock()
    trace_ctx = MagicMock()

    # LLM returns junk for the summary — should not crash
    bad_response = ModelResponse(text="", model="qwen2.5:7b")
    gw = AsyncMock()
    gw.complete = AsyncMock(return_value=bad_response)

    result = await extract(
        ticker="MSFT",
        filed_date="2023-06-30",
        document_text=_FAKE_10K,   # no financial table → empty metrics, not a crash
        langfuse=lf,
        trace_context=trace_ctx,
        gateway=gw,
    )

    assert result.ticker == "MSFT"
    assert isinstance(result.metrics, dict)  # empty or partial, but no crash


# ── _format_metrics_for_prompt ────────────────────────────────────────────────

def test_format_metrics_includes_revenue_in_millions():
    out = _format_metrics_for_prompt({"revenue": 52017.0}, "2025-12-31")
    assert "$52,017M" in out


def test_format_metrics_eps_shows_two_decimal_places():
    out = _format_metrics_for_prompt({"eps_diluted": 4.73}, "2025-12-31")
    assert "$4.73" in out


def test_format_metrics_skips_missing_keys():
    out = _format_metrics_for_prompt({"revenue": 52017.0}, "2025-12-31")
    assert "Net Income" not in out


def test_format_metrics_empty_returns_placeholder():
    out = _format_metrics_for_prompt({}, "2025-12-31")
    assert "no verified metrics" in out


# ── extract with xbrl_metrics ─────────────────────────────────────────────────

_XBRL_METRICS = {
    "revenue": 52017.0,
    "net_income": 10053.0,
    "eps_diluted": 4.73,
    "total_assets": 61802.0,
    "fiscal_year_end": "2025-12-31",
}


async def test_extract_uses_xbrl_metrics_when_provided():
    """When xbrl_metrics are passed, they should be used directly without HTML parsing."""
    lf = _make_langfuse_mock()
    trace_ctx = MagicMock()

    fake_response = ModelResponse(
        text="Uber grew revenue significantly in FY2025.",
        model="qwen2.5:7b",
        input_tokens=300,
        output_tokens=60,
    )
    gw = AsyncMock()
    gw.complete = AsyncMock(return_value=fake_response)

    result = await extract(
        ticker="UBER",
        filed_date="2026-02-13",
        document_text=_FAKE_10K,
        langfuse=lf,
        trace_context=trace_ctx,
        gateway=gw,
        xbrl_metrics=_XBRL_METRICS,
    )

    # XBRL values should be used directly
    assert result.metrics["revenue"] == 52017.0
    assert result.metrics["net_income"] == 10053.0
    assert result.metrics["eps_diluted"] == 4.73


async def test_extract_xbrl_metrics_appear_in_llm_prompt():
    """Verified XBRL metrics must be injected into the LLM prompt."""
    lf = _make_langfuse_mock()
    trace_ctx = MagicMock()

    fake_response = ModelResponse(text="Summary.", model="qwen2.5:7b")
    gw = AsyncMock()
    gw.complete = AsyncMock(return_value=fake_response)

    await extract(
        ticker="UBER",
        filed_date="2026-02-13",
        document_text=_FAKE_10K,
        langfuse=lf,
        trace_context=trace_ctx,
        gateway=gw,
        xbrl_metrics=_XBRL_METRICS,
    )

    # The user message sent to the LLM should contain the verified metric values
    call_args = gw.complete.call_args
    messages = call_args[0][0]
    user_content = messages[0]["content"]
    assert "52,017" in user_content   # revenue injected
    assert "4.73" in user_content     # EPS injected


async def test_extract_falls_back_to_html_when_no_xbrl():
    """Without xbrl_metrics, rule-based HTML table parsing is used."""
    lf = _make_langfuse_mock()
    trace_ctx = MagicMock()

    fake_response = ModelResponse(text="Summary.", model="qwen2.5:7b")
    gw = AsyncMock()
    gw.complete = AsyncMock(return_value=fake_response)

    result = await extract(
        ticker="AAPL",
        filed_date="2024-11-01",
        document_text=_FAKE_10K_WITH_TABLES,
        langfuse=lf,
        trace_context=trace_ctx,
        gateway=gw,
        xbrl_metrics=None,  # no XBRL — should fall back to HTML
    )

    assert result.metrics.get("revenue") == 394328.0
