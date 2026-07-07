"""End-to-end example: run the full VeriAnalyst pipeline for a ticker.

Usage:
    python examples/analyze_ticker.py AAPL
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

# Allow running from the repo root without installing
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv

load_dotenv()

from sec_analyzer.cache.sqlite_cache import SQLiteCache
from sec_analyzer.observability.langfuse_setup import get_langfuse_client
from sec_analyzer.orchestration.graph import build_pipeline, initial_state


async def main(ticker: str) -> None:
    langfuse = get_langfuse_client()

    async with SQLiteCache() as cache:
        pipeline = build_pipeline(cache=cache, langfuse=langfuse)
        state = initial_state(ticker)

        print(f"Running pipeline for {ticker}  (trace_id={state['trace_id']})")
        result = await pipeline.ainvoke(state)

    if result.get("error"):
        print(f"\n[ERROR] {result['error']}")
        sys.exit(1)

    cache_status = "HIT" if result.get("cache_hit") else "MISS"
    print(f"Cache: {cache_status}  |  CIK: {result.get('filing_cik')}  |  Filed: {result.get('filing_date')}\n")
    print(result.get("report", "(no report generated)"))
    host = getattr(langfuse, "_base_url", "http://localhost:3000")
    print(f"\nTrace: {host}/traces/{state['trace_id']}")


if __name__ == "__main__":
    ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    asyncio.run(main(ticker))
