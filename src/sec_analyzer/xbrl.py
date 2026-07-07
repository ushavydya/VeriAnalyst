"""XBRL fact fetcher — pulls verified financial metrics from SEC EDGAR.

Uses the EDGAR company facts API:
  https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json

Returns metrics in millions USD (same unit as the rest of the pipeline),
except EPS which is in USD per share.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx

_COMPANYFACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"

# us-gaap concept → (metric_key, unit, scale_to_millions)
# Some concepts have multiple candidate names; we try them in order.
_CONCEPT_MAP: list[tuple[str, list[str], str, bool]] = [
    # (metric_key, candidate_concepts, unit, divide_by_1M)
    ("revenue", [
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "Revenues",
        "SalesRevenueNet",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
    ], "USD", True),
    ("gross_profit", ["GrossProfit"], "USD", True),
    ("operating_income", ["OperatingIncomeLoss"], "USD", True),
    ("net_income", ["NetIncomeLoss"], "USD", True),
    ("eps_basic", ["EarningsPerShareBasic"], "USD/shares", False),
    ("eps_diluted", ["EarningsPerShareDiluted"], "USD/shares", False),
    ("total_assets", ["Assets"], "USD", True),
    ("total_liabilities", ["Liabilities"], "USD", True),
    ("total_equity", [
        "StockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    ], "USD", True),
    ("cash_and_equivalents", [
        "CashAndCashEquivalentsAtCarryingValue",
        "CashCashEquivalentsAndShortTermInvestments",
    ], "USD", True),
]


def fetch_xbrl_facts(
    cik: str,
    fiscal_year_end: str | None = None,
    *,
    raw: dict[str, Any] | None = None,
) -> dict[str, float]:
    """Extract metrics from XBRL data.

    *fiscal_year_end* — YYYY or YYYY-MM-DD of the fiscal year end. If None,
    returns the most recently filed 10-K annual values regardless of year.

    Pass *raw* to avoid a second HTTP call if you already have the JSON.
    Otherwise call :func:`download_xbrl_facts` first.
    """
    if raw is None:
        raise ValueError("Pass raw= from download_xbrl_facts()")

    us_gaap = raw.get("facts", {}).get("us-gaap", {})
    fy_year = fiscal_year_end[:4] if fiscal_year_end else None

    metrics: dict[str, float] = {}
    fiscal_year_end: str | None = None

    for metric_key, concepts, unit_key, scale in _CONCEPT_MAP:
        for concept in concepts:
            entry = _pick_entry(us_gaap.get(concept, {}), fy_year, unit_key)
            if entry is not None:
                val = float(entry["val"])
                metrics[metric_key] = val / 1_000_000 if scale else val
                if fiscal_year_end is None:
                    fiscal_year_end = entry.get("end")
                break

    if fiscal_year_end:
        metrics["fiscal_year_end"] = fiscal_year_end  # type: ignore[assignment]

    return metrics


async def download_xbrl_facts(cik: str, headers: dict[str, str]) -> dict[str, Any]:
    """Fetch the full company facts JSON from EDGAR for *cik*."""
    url = _COMPANYFACTS_URL.format(cik=cik.lstrip("0"))
    # EDGAR accepts zero-padded or bare CIK in the URL
    padded_url = _COMPANYFACTS_URL.format(cik=cik)
    async with httpx.AsyncClient(headers=headers, timeout=30.0) as client:
        for attempt_url in (padded_url, url):
            resp = await client.get(attempt_url)
            if resp.status_code == 200:
                return resp.json()
        resp.raise_for_status()
    return {}  # unreachable


# ── Helpers ───────────────────────────────────────────────────────────────────

def _pick_entry(concept_data: dict, fy_year: str | None, unit_key: str) -> dict | None:
    """Return the best matching XBRL entry dict for a single concept.

    If *fy_year* is given, filters to that fiscal year end (YYYY).
    If None, returns the entry from the most recent fiscal year.
    """
    units = concept_data.get("units", {})
    entries: list[dict] = units.get(unit_key, [])

    candidates = [
        e for e in entries
        if e.get("form") == "10-K" and e.get("val") is not None
    ]
    if fy_year:
        candidates = [e for e in candidates if e.get("end", "").startswith(fy_year)]

    if not candidates:
        return None

    # Sort by fiscal year end descending, then filed date (for amendments)
    candidates.sort(key=lambda e: (e.get("end", ""), e.get("filed", "")), reverse=True)
    return candidates[0]
