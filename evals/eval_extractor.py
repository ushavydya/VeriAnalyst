"""Extractor eval — compares extracted metrics against golden dataset.

Usage:
    python evals/eval_extractor.py              # all tickers
    python evals/eval_extractor.py AAPL MSFT    # specific tickers

Results are logged to Langfuse as scores on a dedicated eval trace.
A summary table is printed to stdout.
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

from dataclasses import dataclass

from langfuse import Langfuse
from langfuse.types import TraceContext

from sec_analyzer.agents.extractor import extract
from sec_analyzer.agents.retriever import SECRetriever
from sec_analyzer.cache.sqlite_cache import SQLiteCache
from evals.golden_dataset import GOLDEN, GoldenRecord


@dataclass
class FieldResult:
    field: str
    expected: float
    actual: float | None
    passed: bool
    pct_error: float | None   # None if actual is missing


@dataclass
class TickerResult:
    ticker: str
    fiscal_year_match: bool
    field_results: list[FieldResult]

    @property
    def passed(self) -> bool:
        return self.fiscal_year_match and all(r.passed for r in self.field_results)

    @property
    def score(self) -> float:
        """Fraction of checks that passed (0–1)."""
        checks = [self.fiscal_year_match] + [r.passed for r in self.field_results]
        return sum(checks) / len(checks) if checks else 0.0


def _check_metric(
    field: str,
    expected: float,
    actual: float | None,
    tolerance: float,
) -> FieldResult:
    if actual is None:
        return FieldResult(field=field, expected=expected, actual=None, passed=False, pct_error=None)
    pct_err = abs(actual - expected) / max(abs(expected), 1.0)
    return FieldResult(
        field=field,
        expected=expected,
        actual=actual,
        passed=pct_err <= tolerance,
        pct_error=round(pct_err * 100, 1),
    )


async def _eval_ticker(
    record: GoldenRecord,
    retriever: SECRetriever,
    langfuse: Langfuse,
    trace_id: str,
) -> TickerResult:
    tc = TraceContext(trace_id=trace_id)

    filing = await retriever.fetch_10k(record.ticker, tc)

    # Extract — pass XBRL metrics so eval tests the full pipeline
    data = await extract(
        ticker=record.ticker,
        filed_date=filing.filed_date,
        document_text=filing.read_text(),
        langfuse=langfuse,
        trace_context=tc,
        xbrl_metrics=filing.xbrl_metrics,
    )

    # Check fiscal year
    fy = str(data.metrics.get("fiscal_year_end", ""))
    fiscal_year_match = record.fiscal_year_end in fy

    # Check each metric
    field_results = [
        _check_metric(field, expected, data.metrics.get(field), record.tolerance)
        for field, expected in record.metrics.items()
    ]

    result = TickerResult(
        ticker=record.ticker,
        fiscal_year_match=fiscal_year_match,
        field_results=field_results,
    )

    # Log score to Langfuse
    langfuse.create_score(
        trace_id=trace_id,
        name=f"eval.extractor.{record.ticker.lower()}",
        value=result.score,
        comment=_summary_comment(result),
    )

    return result


def _summary_comment(result: TickerResult) -> str:
    failures = []
    if not result.fiscal_year_match:
        failures.append("fiscal_year_end mismatch")
    for r in result.field_results:
        if not r.passed:
            actual_str = f"{r.actual:,.1f}" if r.actual is not None else "missing"
            failures.append(f"{r.field}: expected {r.expected:,.1f}, got {actual_str} ({r.pct_error}% err)")
    return "; ".join(failures) if failures else "all checks passed"


def _print_table(results: list[TickerResult]) -> None:
    print("\n" + "=" * 70)
    print(f"{'Ticker':<8} {'FY':^5} {'Score':^7}  Field results")
    print("-" * 70)
    for r in results:
        fy_mark = "✓" if r.fiscal_year_match else "✗"
        score_str = f"{r.score:.0%}"
        fields = "  ".join(
            f"{f.field}={'✓' if f.passed else f'✗({f.pct_error}%)'}"
            for f in r.field_results
        )
        print(f"{r.ticker:<8} {fy_mark:^5} {score_str:^7}  {fields}")
    print("=" * 70)
    overall = sum(r.score for r in results) / len(results) if results else 0
    passed = sum(1 for r in results if r.passed)
    print(f"Overall: {passed}/{len(results)} tickers fully passed  |  Avg score: {overall:.0%}\n")


async def run_evals(tickers: list[str] | None = None) -> list[TickerResult]:
    langfuse = Langfuse(
        public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
        secret_key=os.environ["LANGFUSE_SECRET_KEY"],
        host=os.environ.get("LANGFUSE_HOST", "http://localhost:3000"),
    )

    records = [r for r in GOLDEN if tickers is None or r.ticker in tickers]
    if not records:
        print(f"No golden records found for: {tickers}")
        return []

    trace_id = uuid.uuid4().hex
    print(f"Eval trace: {os.environ.get('LANGFUSE_HOST', 'http://localhost:3000')}/traces/{trace_id}\n")

    results: list[TickerResult] = []
    async with SQLiteCache() as cache:
        retriever = SECRetriever(cache=cache, langfuse=langfuse)
        for record in records:
            print(f"  Evaluating {record.ticker}…", end=" ", flush=True)
            try:
                result = await _eval_ticker(record, retriever, langfuse, trace_id)
                results.append(result)
                print("✓" if result.passed else "✗")
            except Exception as e:
                print(f"ERROR: {e}")

    langfuse.flush()
    _print_table(results)
    return results


if __name__ == "__main__":
    tickers = sys.argv[1:] or None
    asyncio.run(run_evals([t.upper() for t in tickers] if tickers else None))
