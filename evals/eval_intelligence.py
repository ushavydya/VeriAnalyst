"""Investment Intelligence eval — verifies market data, news, critic signals, and report synthesis.

Usage:
    python evals/eval_intelligence.py              # default tickers
    python evals/eval_intelligence.py AAPL MSFT    # specific tickers

Requires FINNHUB_API_KEY to be set; skips gracefully if missing.

Checks per ticker:
  Market data:
    - price > 0
    - P/E ratio in plausible range (0–500) if present
    - 52W high >= 52W low
    - history has at least 100 daily bars (WARN on free tier)
  News:
    - at least 1 article returned
    - LLM-computed sentiment score in [-1.0, 1.0] when articles present
    - LLM-generated narrative present when articles present
    - cache hit on second call (TTL not yet expired)
    - narrative preserved from cache on second call
  Critic:
    - market and news data flow through to critic_node without crashing
  Report:
    - "Investment Intelligence" section present in report when data is available
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

from sec_analyzer.cache.sqlite_cache import SQLiteCache

_DEFAULT_TICKERS = ["AAPL", "MSFT"]


# ── Market data checks ────────────────────────────────────────────────────────

def _check_market(market_json: str | None, ticker: str) -> list[str]:
    issues: list[str] = []
    if not market_json:
        issues.append("No market data returned (FINNHUB_API_KEY may not be set)")
        return issues

    d = json.loads(market_json)
    quote = d.get("quote") or {}
    ratios = d.get("ratios") or {}

    price = quote.get("price")
    if price is None:
        issues.append("quote.price is missing")
    elif price <= 0:
        issues.append(f"quote.price is non-positive: {price}")

    pe = ratios.get("pe_ratio")
    if pe is not None and (pe < 0 or pe > 500):
        issues.append(f"pe_ratio={pe:.1f} is outside plausible range (0–500)")

    hi = ratios.get("week_52_high")
    lo = ratios.get("week_52_low")
    if hi is not None and lo is not None and lo > hi:
        issues.append(f"52W low ({lo}) > 52W high ({hi})")

    bar_count = d.get("history_bar_count", 0)
    if bar_count == 0:
        # Finnhub free tier does not provide /stock/candle — treat as a warning
        issues.append(f"WARN: history returned 0 bars (Finnhub free-tier candle endpoint limitation)")
    elif bar_count < 100:
        issues.append(f"history has only {bar_count} bars (expected ≥100 for 1y)")

    return issues


# ── News checks ───────────────────────────────────────────────────────────────

def _check_news(news_json: str | None, ticker: str) -> list[str]:
    issues: list[str] = []
    if not news_json:
        issues.append("No news data returned (FINNHUB_API_KEY may not be set)")
        return issues

    d = json.loads(news_json)
    articles = d.get("articles", [])
    if not articles:
        issues.append(f"No articles returned for {ticker} (may be a Finnhub free-tier limit)")
        return issues

    # Sentiment score — LLM-computed so should always be present when articles exist
    score = d.get("sentiment_score")
    if score is None:
        issues.append(f"sentiment_score is None despite {len(articles)} articles — LLM sentiment failed")
    elif not (-1.0 <= score <= 1.0):
        issues.append(f"sentiment_score={score} is outside [-1, 1]")

    # Narrative — LLM-generated summary, always expected when articles are present
    narrative = d.get("narrative")
    if not narrative:
        issues.append("LLM-generated narrative is missing despite articles being present")

    return issues


# ── Report intelligence section check ────────────────────────────────────────

def _check_report_has_intelligence(report: str | None, has_market: bool, has_news: bool) -> list[str]:
    issues: list[str] = []
    if not report:
        issues.append("No report produced")
        return issues
    if has_market or has_news:
        if "Investment Intelligence" not in report:
            issues.append("Report is missing the 'Investment Intelligence' section despite data being available")
    return issues


# ── Critic issues check ───────────────────────────────────────────────────────

def _check_critic_issues(critique_json: str | None) -> list[str]:
    """Verify critique_json is well-formed; not a content correctness check."""
    issues: list[str] = []
    if not critique_json:
        issues.append("No critique produced")
        return issues
    try:
        crit = json.loads(critique_json)
        if "confidence" not in crit:
            issues.append("Critique missing 'confidence' field")
        if not isinstance(crit.get("issues"), list):
            issues.append("Critique 'issues' field is not a list")
    except Exception as e:
        issues.append(f"Critique JSON parse error: {e}")
    return issues


# ── Per-ticker eval ───────────────────────────────────────────────────────────

async def _eval_ticker(ticker: str, langfuse: Langfuse, trace_id: str) -> dict:
    from sec_analyzer.orchestration.graph import build_pipeline, initial_state

    async with SQLiteCache() as cache:
        pipeline = build_pipeline(cache, langfuse, enable_intelligence=True)
        state = initial_state(ticker)
        result = await pipeline.ainvoke(state)

    issues: list[str] = []

    market_json = result.get("market_summary_json")
    news_json = result.get("news_summary_json")
    report = result.get("report", "")
    critique_json = result.get("critique_json")

    issues += _check_market(market_json, ticker)
    issues += _check_news(news_json, ticker)
    issues += _check_report_has_intelligence(
        report,
        has_market=market_json is not None,
        has_news=news_json is not None,
    )
    issues += _check_critic_issues(critique_json)

    # Second run — verify cache hit for market + news
    async with SQLiteCache() as cache:
        pipeline2 = build_pipeline(cache, langfuse, enable_intelligence=True)
        state2 = initial_state(ticker)
        result2 = await pipeline2.ainvoke(state2)

    if result2.get("market_summary_json"):
        mkt2 = json.loads(result2["market_summary_json"])
        cache_hits = mkt2.get("cache_hits", {})
        if not cache_hits.get("quote"):
            issues.append("Market quote was not served from cache on second run")
        if not cache_hits.get("ratios"):
            issues.append("Market ratios were not served from cache on second run")

    if result2.get("news_summary_json"):
        news2 = json.loads(result2["news_summary_json"])
        if not news2.get("cache_hit"):
            issues.append("News was not served from cache on second run")
        # Narrative must survive the cache round-trip
        if news2.get("articles") and not news2.get("narrative"):
            issues.append("News narrative was not preserved in cache on second run")

    hard_issues = [i for i in issues if not i.startswith("WARN:")]
    passed = len(hard_issues) == 0
    score = 1.0 if passed else max(0.0, 1.0 - len(hard_issues) * 0.2)

    langfuse.create_score(
        trace_id=trace_id,
        name=f"eval.intelligence.{ticker.lower()}",
        value=score,
        comment="; ".join(issues) if issues else "all checks passed",
    )

    news_d = json.loads(news_json) if news_json else {}
    return {
        "ticker": ticker,
        "passed": passed,
        "score": score,
        "issues": issues,
        "has_market": market_json is not None,
        "has_news": news_json is not None,
        "market_bar_count": json.loads(market_json).get("history_bar_count", 0) if market_json else 0,
        "article_count": len(news_d.get("articles", [])),
        "sentiment_score": news_d.get("sentiment_score"),
        "has_narrative": bool(news_d.get("narrative")),
        "has_intelligence_section": "Investment Intelligence" in (report or ""),
        "critic_issues_count": len(json.loads(critique_json).get("issues", [])) if critique_json else 0,
    }


# ── Reporting ─────────────────────────────────────────────────────────────────

def _print_results(results: list[dict]) -> None:
    print("\n" + "=" * 80)
    print(f"{'Ticker':<8} {'Pass':^5} {'Score':^6}  Detail")
    print("-" * 80)
    for r in results:
        mark = "✓" if r["passed"] else "✗"
        detail = "OK" if r["passed"] else "; ".join(r["issues"][:2])
        print(f"{r['ticker']:<8} {mark:^5} {r['score']:^6.2f}  {detail}")
        sentiment_str = f"{r['sentiment_score']:+.2f}" if r['sentiment_score'] is not None else "n/a"
        print(
            f"         market={'yes' if r['has_market'] else 'no'} ({r['market_bar_count']} bars)  "
            f"news={'yes' if r['has_news'] else 'no'} "
            f"({r['article_count']} articles, sentiment={sentiment_str}, "
            f"narrative={'yes' if r['has_narrative'] else 'no'})  "
            f"intel_section={'yes' if r['has_intelligence_section'] else 'no'}  "
            f"critic_issues={r['critic_issues_count']}"
        )
    print("=" * 80)
    passed = sum(1 for r in results if r["passed"])
    print(f"Overall: {passed}/{len(results)} tickers fully passed\n")


# ── Entry point ───────────────────────────────────────────────────────────────

async def run_intelligence_evals(tickers: list[str]) -> list[dict]:
    finnhub_key = os.environ.get("FINNHUB_API_KEY", "")
    if not finnhub_key:
        print("WARNING: FINNHUB_API_KEY is not set — market/news checks will flag as missing data")

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
            results.append({"ticker": ticker, "passed": False, "score": 0.0,
                            "issues": [str(e)], "has_market": False, "has_news": False,
                            "market_bar_count": 0, "article_count": 0, "sentiment_score": None,
                            "has_narrative": False, "has_intelligence_section": False,
                            "critic_issues_count": 0})

    langfuse.flush()
    _print_results(results)
    return results


if __name__ == "__main__":
    tickers = [t.upper() for t in sys.argv[1:]] if sys.argv[1:] else _DEFAULT_TICKERS
    asyncio.run(run_intelligence_evals(tickers))
