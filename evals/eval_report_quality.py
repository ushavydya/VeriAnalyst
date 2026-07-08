"""LLM-as-judge eval — scores report quality on a rubric.

Usage:
    python evals/eval_report_quality.py AAPL
    python evals/eval_report_quality.py AAPL UBER MSFT

Rubric dimensions (each 0–1):
  - factual_grounding  : claims are supported by the extracted metrics
  - completeness       : covers all key financial dimensions
  - fiscal_year_accuracy: correct fiscal year cited throughout
  - no_hallucination   : no invented figures or events
  - clarity            : well-structured, readable prose

Each dimension is scored and logged to Langfuse. An aggregate score is also posted.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

from langfuse import Langfuse
from langfuse.types import TraceContext

from sec_analyzer.agents.extractor import extract
from sec_analyzer.agents.retriever import SECRetriever
from sec_analyzer.agents.writer import write_report
from sec_analyzer.agents.critic import critique
from sec_analyzer.cache.sqlite_cache import SQLiteCache
from sec_analyzer.gateway import get_gateway


_JUDGE_SYSTEM = """\
You are an expert financial analyst evaluating the quality of an AI-generated
10-K analysis report. Score each dimension from 0.0 to 1.0 with one decimal place.

Scoring rubric:
- factual_grounding   (1.0 = every claim matches the provided metrics; 0.0 = major mismatches)
- completeness        (1.0 = covers revenue, profitability, EPS, balance sheet, risks; 0.0 = major gaps)
- fiscal_year_accuracy(1.0 = correct fiscal year cited consistently; 0.0 = wrong or missing year)
- no_hallucination    (1.0 = no invented numbers or events; 0.0 = clear fabrications)
- clarity             (1.0 = well-structured and readable; 0.0 = confusing or incoherent)

Reply with JSON only — no markdown fences, no commentary:
{
  "factual_grounding": <float>,
  "completeness": <float>,
  "fiscal_year_accuracy": <float>,
  "no_hallucination": <float>,
  "clarity": <float>,
  "reasoning": "<one sentence explaining the lowest score>"
}"""


def _judge_prompt(ticker: str, metrics: dict, report: str) -> str:
    metrics_block = json.dumps(
        {k: v for k, v in metrics.items() if isinstance(v, (int, float, str))},
        indent=2,
    )
    return (
        f"Ticker: {ticker}\n\n"
        f"Extracted metrics (ground truth):\n{metrics_block}\n\n"
        f"Report to evaluate:\n{report}"
    )


async def _eval_report(ticker: str, langfuse: Langfuse, trace_id: str) -> dict:
    tc = TraceContext(trace_id=trace_id)
    gw = get_gateway()

    async with SQLiteCache() as cache:
        retriever = SECRetriever(cache=cache, langfuse=langfuse)
        filing = await retriever.fetch_10k(ticker, tc)
        data = await extract(
            ticker=ticker,
            filed_date=filing.filed_date,
            document_text=filing.read_text(),
            langfuse=langfuse,
            trace_context=tc,
            xbrl_metrics=filing.xbrl_metrics,
        )
        crit = await critique(data, langfuse=langfuse, trace_context=tc)
        report = await write_report(data, crit, langfuse=langfuse, trace_context=tc)

    # LLM judge
    messages = [{"role": "user", "content": _judge_prompt(ticker, data.metrics, report)}]
    response = await gw.complete(
        messages,
        system=_judge_system_safe(),
        max_tokens=512,
        json_mode=True,
    )

    scores = _parse_scores(response.text)

    # Log each dimension to Langfuse
    for dimension, value in scores.items():
        if dimension == "reasoning":
            continue
        langfuse.create_score(
            trace_id=trace_id,
            name=f"eval.report.{ticker.lower()}.{dimension}",
            value=float(value),
        )

    # Aggregate score
    numeric = {k: v for k, v in scores.items() if k != "reasoning" and isinstance(v, (int, float))}
    aggregate = sum(numeric.values()) / len(numeric) if numeric else 0.0
    langfuse.create_score(
        trace_id=trace_id,
        name=f"eval.report.{ticker.lower()}.aggregate",
        value=aggregate,
        comment=scores.get("reasoning", ""),
    )

    return {"ticker": ticker, "scores": scores, "aggregate": aggregate}


def _judge_system_safe() -> str:
    return _JUDGE_SYSTEM


def _parse_scores(text: str) -> dict:
    raw = text.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"reasoning": f"Failed to parse judge response: {text[:200]}"}


def _print_results(results: list[dict]) -> None:
    dims = ["factual_grounding", "completeness", "fiscal_year_accuracy", "no_hallucination", "clarity"]
    print("\n" + "=" * 75)
    header = f"{'Ticker':<8}" + "".join(f"{d[:8]:^10}" for d in dims) + f"{'Aggregate':^10}"
    print(header)
    print("-" * 75)
    for r in results:
        scores = r["scores"]
        row = f"{r['ticker']:<8}"
        for d in dims:
            val = scores.get(d)
            row += f"{val:^10.1f}" if isinstance(val, (int, float)) else f"{'N/A':^10}"
        row += f"{r['aggregate']:^10.1f}"
        print(row)
    print("=" * 75)
    if results:
        avg = sum(r["aggregate"] for r in results) / len(results)
        print(f"Mean aggregate score: {avg:.2f}\n")


async def run_report_evals(tickers: list[str]) -> list[dict]:
    langfuse = Langfuse(
        public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
        secret_key=os.environ["LANGFUSE_SECRET_KEY"],
        host=os.environ.get("LANGFUSE_HOST", "http://localhost:3000"),
    )

    trace_id = uuid.uuid4().hex
    print(f"Eval trace: {os.environ.get('LANGFUSE_HOST', 'http://localhost:3000')}/traces/{trace_id}\n")

    results = []
    for ticker in tickers:
        print(f"  Judging {ticker}…", end=" ", flush=True)
        try:
            result = await _eval_report(ticker, langfuse, trace_id)
            results.append(result)
            print(f"aggregate={result['aggregate']:.1f}")
        except Exception as e:
            print(f"ERROR: {e}")

    langfuse.flush()
    _print_results(results)
    return results


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python evals/eval_report_quality.py TICKER [TICKER ...]")
        sys.exit(1)
    tickers = [t.upper() for t in sys.argv[1:]]
    asyncio.run(run_report_evals(tickers))
