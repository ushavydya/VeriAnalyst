"""Compare extractor + critic output with thinking on vs off for the same ticker.

Usage:
    python examples/compare_thinking.py AAPL
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv
load_dotenv()

from unittest.mock import MagicMock
from sec_analyzer.agents.extractor import extract
from sec_analyzer.agents.critic import critique
from sec_analyzer.cache.sqlite_cache import SQLiteCache
from sec_analyzer.agents.retriever import SECRetriever
from sec_analyzer.gateway.ollama_backend import OllamaGateway


def _noop_langfuse():
    """Minimal no-op Langfuse so we don't need a running server."""
    span = MagicMock()
    span.update = MagicMock()
    ctx = MagicMock()
    ctx.__enter__ = MagicMock(return_value=span)
    ctx.__exit__ = MagicMock(return_value=False)
    lf = MagicMock()
    lf.start_as_current_observation = MagicMock(return_value=ctx)
    return lf


class ThinkingOllamaGateway(OllamaGateway):
    """OllamaGateway with thinking enabled and a larger token budget."""
    async def complete(self, messages, *, system=None, max_tokens=4096):
        import httpx
        from sec_analyzer.gateway.base import ModelResponse

        payload_messages = []
        if system:
            payload_messages.append({"role": "system", "content": system})
        payload_messages.extend({"role": m["role"], "content": m["content"]} for m in messages)

        payload = {
            "model": self._model,
            "messages": payload_messages,
            "stream": False,
            "think": True,
            "options": {"num_predict": max_tokens * 6},  # room for reasoning chain
        }

        async with httpx.AsyncClient(timeout=300.0) as client:
            response = await client.post(f"{self._base_url}/api/chat", json=payload)
            response.raise_for_status()
            data = response.json()

        text = data.get("message", {}).get("content", "")
        return ModelResponse(text=text, model=self._model)


async def run_pipeline(ticker: str, gateway, label: str, document_text: str, filed_date: str):
    lf = _noop_langfuse()
    tc = MagicMock()

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")

    extracted = await extract(
        ticker=ticker,
        filed_date=filed_date,
        document_text=document_text,
        langfuse=lf,
        trace_context=tc,
        gateway=gateway,
    )
    print(f"\nSections found: {list(extracted.sections.keys())}")
    print(f"Metrics extracted ({len(extracted.metrics)}):")
    for k, v in extracted.metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:,.2f}")
        else:
            print(f"  {k}: {v}")

    crit = await critique(extracted, lf, tc, gateway=gateway)
    print(f"\nCritic confidence: {crit.confidence:.0%}")
    print(f"Issues: {crit.issues or 'none'}")
    print(f"Summary: {crit.summary}")

    return extracted, crit


async def main(ticker: str):
    # Fetch the filing once, reuse for both runs
    print(f"Fetching latest 10-K for {ticker} …")
    lf = _noop_langfuse()
    async with SQLiteCache() as cache:
        retriever = SECRetriever(cache=cache, langfuse=lf)
        tc = MagicMock()
        filing = await retriever.fetch_10k(ticker, tc)
        document_text = filing.read_text()
        filed_date = filing.filed_date
        print(f"Filing date: {filed_date}  |  doc length: {len(document_text):,} chars")

    no_think_gw = OllamaGateway()   # think=False (current default)
    think_gw = ThinkingOllamaGateway()

    data_off, crit_off = await run_pipeline(ticker, no_think_gw, "THINKING OFF", document_text, filed_date)
    data_on, crit_on = await run_pipeline(ticker, think_gw, "THINKING ON", document_text, filed_date)

    # Side-by-side diff
    print(f"\n{'='*60}")
    print("  COMPARISON")
    print(f"{'='*60}")
    all_keys = sorted(set(data_off.metrics) | set(data_on.metrics))
    if not all_keys:
        print("Neither run extracted any metrics — check the filing text.")
    else:
        print(f"{'Metric':<30} {'Think OFF':>15} {'Think ON':>15}")
        print("-" * 62)
        for k in all_keys:
            v_off = data_off.metrics.get(k, "—")
            v_on = data_on.metrics.get(k, "—")
            flag = " ◄ differs" if v_off != v_on else ""
            fmt = lambda v: f"{v:,.2f}" if isinstance(v, float) else str(v)
            print(f"{k:<30} {fmt(v_off):>15} {fmt(v_on):>15}{flag}")

    print(f"\n{'Confidence':<30} {crit_off.confidence:>14.0%} {crit_on.confidence:>14.0%}")


if __name__ == "__main__":
    ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    asyncio.run(main(ticker))
