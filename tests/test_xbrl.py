"""Unit tests for the XBRL fact fetcher — no network calls."""
from __future__ import annotations

import pytest

from sec_analyzer.xbrl import _pick_entry, fetch_xbrl_facts


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _revenue_concept(entries: list[dict]) -> dict:
    """Wrap a list of entries in a us-gaap concept dict."""
    return {"units": {"USD": entries}}


def _entry(val: float, end: str, filed: str, form: str = "10-K") -> dict:
    return {"val": val, "end": end, "filed": filed, "form": form}


# ── _pick_entry ───────────────────────────────────────────────────────────────

def test_pick_entry_returns_most_recent_fiscal_year():
    concept = _revenue_concept([
        _entry(37_281_000_000, "2023-12-31", "2024-02-14"),
        _entry(43_978_000_000, "2024-12-31", "2025-02-13"),
        _entry(52_017_000_000, "2025-12-31", "2026-02-13"),
    ])
    entry = _pick_entry(concept, fy_year=None, unit_key="USD")
    assert entry["val"] == 52_017_000_000


def test_pick_entry_filters_by_fy_year():
    concept = _revenue_concept([
        _entry(43_978_000_000, "2024-12-31", "2025-02-13"),
        _entry(52_017_000_000, "2025-12-31", "2026-02-13"),
    ])
    entry = _pick_entry(concept, fy_year="2024", unit_key="USD")
    assert entry["val"] == 43_978_000_000


def test_pick_entry_ignores_non_10k_forms():
    concept = _revenue_concept([
        _entry(99_999_999, "2025-09-30", "2025-11-01", form="10-Q"),
        _entry(52_017_000_000, "2025-12-31", "2026-02-13", form="10-K"),
    ])
    entry = _pick_entry(concept, fy_year=None, unit_key="USD")
    assert entry["val"] == 52_017_000_000


def test_pick_entry_prefers_latest_amendment():
    """Two 10-K entries for the same fiscal year — pick the most recently filed."""
    concept = _revenue_concept([
        _entry(52_000_000_000, "2025-12-31", "2026-02-13"),   # original
        _entry(52_017_000_000, "2025-12-31", "2026-03-01"),   # amended
    ])
    entry = _pick_entry(concept, fy_year=None, unit_key="USD")
    assert entry["val"] == 52_017_000_000


def test_pick_entry_returns_none_when_no_match():
    concept = _revenue_concept([
        _entry(52_017_000_000, "2025-12-31", "2026-02-13"),
    ])
    assert _pick_entry(concept, fy_year="2020", unit_key="USD") is None


def test_pick_entry_returns_none_for_empty_concept():
    assert _pick_entry({}, fy_year=None, unit_key="USD") is None


def test_pick_entry_returns_none_for_wrong_unit():
    concept = _revenue_concept([_entry(52_017_000_000, "2025-12-31", "2026-02-13")])
    assert _pick_entry(concept, fy_year=None, unit_key="USD/shares") is None


# ── fetch_xbrl_facts ─────────────────────────────────────────────────────────

def _make_raw(concepts: dict) -> dict:
    """Wrap concepts in the EDGAR company facts envelope."""
    return {"facts": {"us-gaap": concepts}}


def test_fetch_xbrl_facts_scales_to_millions():
    raw = _make_raw({
        "Revenues": _revenue_concept([
            _entry(52_017_000_000, "2025-12-31", "2026-02-13"),
        ]),
    })
    metrics = fetch_xbrl_facts("0001543151", raw=raw)
    assert metrics["revenue"] == pytest.approx(52_017.0)


def test_fetch_xbrl_facts_eps_not_scaled():
    raw = _make_raw({
        "EarningsPerShareDiluted": {"units": {"USD/shares": [
            _entry(4.73, "2025-12-31", "2026-02-13"),
        ]}},
    })
    metrics = fetch_xbrl_facts("0001543151", raw=raw)
    assert metrics["eps_diluted"] == pytest.approx(4.73)


def test_fetch_xbrl_facts_tries_fallback_concepts():
    """If primary revenue concept missing, falls back to Revenues."""
    raw = _make_raw({
        "Revenues": _revenue_concept([
            _entry(52_017_000_000, "2025-12-31", "2026-02-13"),
        ]),
        # RevenueFromContractWithCustomerExcludingAssessedTax intentionally absent
    })
    metrics = fetch_xbrl_facts("0001543151", raw=raw)
    assert "revenue" in metrics


def test_fetch_xbrl_facts_includes_fiscal_year_end():
    raw = _make_raw({
        "NetIncomeLoss": _revenue_concept([
            _entry(10_053_000_000, "2025-12-31", "2026-02-13"),
        ]),
    })
    metrics = fetch_xbrl_facts("0001543151", raw=raw)
    assert metrics.get("fiscal_year_end") == "2025-12-31"


def test_fetch_xbrl_facts_filters_by_fiscal_year():
    raw = _make_raw({
        "Revenues": _revenue_concept([
            _entry(43_978_000_000, "2024-12-31", "2025-02-13"),
            _entry(52_017_000_000, "2025-12-31", "2026-02-13"),
        ]),
    })
    metrics = fetch_xbrl_facts("0001543151", fiscal_year_end="2024", raw=raw)
    assert metrics["revenue"] == pytest.approx(43_978.0)


def test_fetch_xbrl_facts_empty_raw_returns_empty():
    metrics = fetch_xbrl_facts("0001543151", raw={"facts": {}})
    assert metrics == {}


def test_fetch_xbrl_facts_raises_without_raw():
    with pytest.raises(ValueError, match="raw="):
        fetch_xbrl_facts("0001543151")
