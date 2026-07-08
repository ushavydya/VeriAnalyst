"""Ground-truth financial metrics for eval regression testing.

Values are fetched from SEC EDGAR XBRL (company facts API) — the same
authoritative source as the 10-K filing itself. Run `python evals/golden_dataset.py`
to refresh all records from EDGAR.
"""
from __future__ import annotations

import asyncio
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class GoldenRecord:
    ticker: str
    fiscal_year_end: str        # YYYY — checked as substring match against extracted value
    metrics: dict[str, float]   # expected values (millions USD, except EPS)
    tolerance: float = 0.02     # 2% relative tolerance


# ── Golden dataset ────────────────────────────────────────────────────────────
# Sourced from SEC EDGAR XBRL company facts API.
# Refresh with: python evals/golden_dataset.py

GOLDEN: list[GoldenRecord] = [
    GoldenRecord(
        ticker="AAPL",
        fiscal_year_end="2025",  # FY ended September 27, 2025
        metrics={
            "revenue":       416161.0,
            "net_income":    112010.0,
            "eps_diluted":       7.46,
            "total_assets":  364980.0,
        },
    ),
    GoldenRecord(
        ticker="MSFT",
        fiscal_year_end="2025",  # FY ended June 30, 2025
        metrics={
            "revenue":       281724.0,
            "net_income":    101832.0,
            "eps_diluted":      13.64,
        },
    ),
    GoldenRecord(
        ticker="UBER",
        fiscal_year_end="2025",  # FY ended December 31, 2025
        metrics={
            "revenue":        52017.0,
            "net_income":     10053.0,
            "eps_diluted":        4.73,
        },
    ),
    GoldenRecord(
        ticker="NVDA",
        fiscal_year_end="2026",  # FY ended January 25, 2026
        metrics={
            "revenue":       215938.0,
            "net_income":    120067.0,
            "eps_diluted":       4.90,
            "total_assets":  206803.0,
        },
    ),
    GoldenRecord(
        ticker="AMD",
        fiscal_year_end="2025",  # FY ended December 27, 2025
        metrics={
            "revenue":        34639.0,
            "net_income":      4335.0,
            "eps_diluted":       2.65,
            "total_assets":   76926.0,
        },
    ),
    GoldenRecord(
        ticker="JPM",
        fiscal_year_end="2025",  # FY ended December 31, 2025
        metrics={
            "revenue":       182447.0,
            "net_income":     57048.0,
            "eps_diluted":      20.02,
            "total_assets": 4424900.0,
        },
    ),
]


# ── XBRL refresh ─────────────────────────────────────────────────────────────

async def _refresh() -> None:
    """Fetch current XBRL values for all golden records and print a comparison."""
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env")

    from sec_analyzer.cache.sqlite_cache import SQLiteCache
    from sec_analyzer.agents.retriever import SECRetriever
    from langfuse import Langfuse
    from langfuse.types import TraceContext
    import uuid

    langfuse = Langfuse(
        public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
        secret_key=os.environ["LANGFUSE_SECRET_KEY"],
        host=os.environ.get("LANGFUSE_HOST", "http://localhost:3000"),
    )

    print(f"\n{'Ticker':<8} {'Field':<22} {'Golden':>12} {'XBRL':>12} {'Match':>6}")
    print("-" * 65)

    async with SQLiteCache() as cache:
        retriever = SECRetriever(cache=cache, langfuse=langfuse)
        tc = TraceContext(trace_id=uuid.uuid4().hex)

        for record in GOLDEN:
            # fetch_10k resolves CIK and returns the latest accession — use
            # the real accession as the XBRL cache key, and pass fiscal_year_end
            # as the year filter so we compare the right filing year.
            filing = await retriever.fetch_10k(record.ticker, tc)
            xbrl = await retriever.fetch_xbrl(
                filing.cik, filing.accession_number, tc,
                fiscal_year=record.fiscal_year_end,
            )

            for field_name, golden_val in record.metrics.items():
                xbrl_val = xbrl.get(field_name)
                if xbrl_val is None:
                    match = "MISSING"
                else:
                    pct = abs(xbrl_val - golden_val) / max(abs(golden_val), 1) * 100
                    match = f"✓ ({pct:.1f}%)" if pct <= record.tolerance * 100 else f"✗ ({pct:.1f}%)"
                xbrl_str = f"{xbrl_val:,.2f}" if xbrl_val is not None else "—"
                print(f"{record.ticker:<8} {field_name:<22} {golden_val:>12,.2f} {xbrl_str:>12} {match:>6}")

    langfuse.flush()
    print()


if __name__ == "__main__":
    asyncio.run(_refresh())
