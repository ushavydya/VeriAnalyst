"""Tests for the writer agent (mocked — no real LLM calls)."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from sec_analyzer.agents.critic import Critique
from sec_analyzer.agents.extractor import ExtractedData
from sec_analyzer.agents.writer import _format_metrics, write_report
from sec_analyzer.gateway.base import ModelResponse


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


def _sample_data() -> ExtractedData:
    return ExtractedData(
        ticker="AAPL",
        filed_date="2023-10-27",
        sections={
            "business": "Apple designs and sells iPhones, Macs, and services.",
            "mda": "Net sales were $394.3B, up 2% year-over-year.",
            "risk_factors": "Competition, supply chain disruptions, regulatory changes.",
        },
        metrics={"revenue": 394328.0, "net_income": 96995.0, "eps_diluted": 6.13},
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
