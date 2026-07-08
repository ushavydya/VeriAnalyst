"""Trends eval — verifies that fetch_trends returns distinct, plausible data per year.

Usage:
    python evals/eval_trends.py              # default tickers
    python evals/eval_trends.py UBER AAPL    # specific tickers

Checks per ticker:
  - fiscal_year_end values are strictly decreasing (no year repeated)
  - revenue values are not all identical (XBRL accession filtering works)
  - at least `years` filings are returned
  - revenue is positive for each year

Results are logged to Langfuse and printed to stdout.
"""
from __future__ import annotations

import asyncio
import os
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

from langfuse import Langfuse

from sec_analyzer.cache.sqlite_cache import SQLiteCache
from sec_analyzer.trends import fetch_trends

_DEFAULT_TICKERS = ["UBER", "AAPL", "MSFT"]
_YEARS = 3


async def _eval_ticker(ticker: str, langfuse: Langfuse, trace_id: str) -> dict:
    async with SQLiteCache() as cache:
        results = await fetch_trends(ticker, years=_YEARS, cache=cache, langfuse=langfuse)

    issues: list[str] = []

    if len(results) < _YEARS:
        issues.append(f"Expected {_YEARS} filings, got {len(results)}")

    # fiscal_year_end must be strictly decreasing (no repeated year)
    fy_ends = [r.fiscal_year_end for r in results if r.fiscal_year_end]
    if len(fy_ends) != len(set(fy_ends)):
        issues.append(f"Duplicate fiscal_year_end values: {fy_ends}")

    if fy_ends != sorted(fy_ends, reverse=True):
        issues.append(f"fiscal_year_end not strictly decreasing: {fy_ends}")

    # Revenue values must not all be identical (catches the "all same year" bug)
    revenues = [r.metrics.get("revenue") for r in results if r.metrics.get("revenue")]
    if revenues and len(set(revenues)) == 1:
        issues.append(f"All {len(revenues)} filings have identical revenue={revenues[0]} — XBRL year filtering broken")

    # Revenue must be positive for every year
    for r in results:
        rev = r.metrics.get("revenue")
        if rev is not None and rev <= 0:
            issues.append(f"Non-positive revenue {rev} for fiscal_year_end={r.fiscal_year_end}")

    passed = len(issues) == 0
    score = 1.0 if passed else 0.0

    langfuse.create_score(
        trace_id=trace_id,
        name=f"eval.trends.{ticker.lower()}",
        value=score,
        comment="; ".join(issues) if issues else "all checks passed",
    )

    return {
        "ticker": ticker,
        "passed": passed,
        "issues": issues,
        "filings": [
            {"filed": r.filed_date, "fy_end": r.fiscal_year_end, "revenue": r.metrics.get("revenue")}
            for r in results
        ],
    }


def _print_results(results: list[dict]) -> None:
    print("\n" + "=" * 70)
    print(f"{'Ticker':<8} {'Pass':^6}  Detail")
    print("-" * 70)
    for r in results:
        mark = "✓" if r["passed"] else "✗"
        detail = "OK" if r["passed"] else "; ".join(r["issues"])
        print(f"{r['ticker']:<8} {mark:^6}  {detail}")
        for f in r["filings"]:
            rev_str = f"${f['revenue']:,.0f}M" if f["revenue"] else "—"
            print(f"         filed={f['filed']}  fy_end={f['fy_end']}  revenue={rev_str}")
    print("=" * 70)
    passed = sum(1 for r in results if r["passed"])
    print(f"Overall: {passed}/{len(results)} tickers fully passed\n")


async def run_trends_evals(tickers: list[str]) -> list[dict]:
    langfuse = Langfuse(
        public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
        secret_key=os.environ["LANGFUSE_SECRET_KEY"],
        host=os.environ.get("LANGFUSE_HOST", "http://localhost:3000"),
    )

    trace_id = uuid.uuid4().hex
    print(f"Eval trace: {os.environ.get('LANGFUSE_HOST', 'http://localhost:3000')}/traces/{trace_id}\n")

    results = []
    for ticker in tickers:
        print(f"  Evaluating {ticker}…", end=" ", flush=True)
        try:
            result = await _eval_ticker(ticker, langfuse, trace_id)
            results.append(result)
            print("✓" if result["passed"] else "✗")
        except Exception as e:
            print(f"ERROR: {e}")

    langfuse.flush()
    _print_results(results)
    return results


if __name__ == "__main__":
    tickers = [t.upper() for t in sys.argv[1:]] if sys.argv[1:] else _DEFAULT_TICKERS
    asyncio.run(run_trends_evals(tickers))
