"""Critic eval — verifies rule checks, market checks, and divergence signals.

Usage:
    python evals/eval_critic.py

Unlike other evals this does NOT call an LLM — it tests the rule-based layers
(_rule_checks, _market_checks, _divergence_checks) with synthetic inputs and
checks that the right signals are raised (and only those signals).

Checks:
  Rule checks:
    - missing required metrics flagged
    - non-positive revenue flagged
    - gross profit > revenue flagged
    - net income > 75% of revenue flagged
    - diluted EPS > basic EPS flagged
    - balance sheet imbalance > 5% flagged
    - clean data produces no issues
  Market checks:
    - non-positive price flagged
    - P/E > 500 flagged
    - negative P/E flagged
    - beta > ±5 flagged
    - 52W low > 52W high flagged
    - price > 52W high by >5% flagged
    - clean market data produces no warnings
  Divergence checks:
    - bearish sentiment on profitable company flagged
    - bullish sentiment on loss-making company flagged
    - price near 52W low on profitable company flagged
    - no signal on neutral sentiment
    - no signal when news missing

Results are logged to Langfuse and printed to stdout.
"""
from __future__ import annotations

import json
import os
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

from langfuse import Langfuse

from sec_analyzer.agents.critic import _divergence_checks, _market_checks, _rule_checks


# ── Helpers ──────────────────────────────────────────────────────────────────

def _market_json(price=200.0, pe=25.0, beta=1.2, hi=260.0, lo=164.0) -> str:
    return json.dumps({
        "quote": {"price": price},
        "ratios": {"pe_ratio": pe, "beta": beta, "week_52_high": hi, "week_52_low": lo},
    })


def _news_json(score: float | None) -> str:
    return json.dumps({"sentiment_score": score})


def _profitable_metrics() -> dict:
    return {"revenue": 100_000, "net_income": 10_000}


def _loss_metrics() -> dict:
    return {"revenue": 100_000, "net_income": -5_000}


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str


# ── Individual checks ─────────────────────────────────────────────────────────

def _run_checks() -> list[CheckResult]:
    results: list[CheckResult] = []

    def ok(name: str, cond: bool, detail: str = "") -> None:
        results.append(CheckResult(name=name, passed=cond, detail=detail))

    # ── Rule checks ──────────────────────────────────────────────────────────

    issues = _rule_checks({})
    ok("rule: missing required metrics",
       any("Missing" in i for i in issues),
       f"issues={issues}")

    issues = _rule_checks({"revenue": -1, "net_income": 1_000})
    ok("rule: non-positive revenue",
       any("non-positive" in i for i in issues),
       f"issues={issues}")

    issues = _rule_checks({"revenue": 100, "net_income": 5, "gross_profit": 110})
    ok("rule: gross profit exceeds revenue",
       any("Gross profit" in i for i in issues),
       f"issues={issues}")

    issues = _rule_checks({"revenue": 100_000, "net_income": 80_000})
    ok("rule: net income > 75% revenue",
       any("75%" in i for i in issues),
       f"issues={issues}")

    issues = _rule_checks({"revenue": 100, "net_income": 10,
                           "eps_diluted": 5.0, "eps_basic": 4.5})
    ok("rule: diluted EPS > basic EPS",
       any("Diluted EPS" in i for i in issues),
       f"issues={issues}")

    issues = _rule_checks({
        "revenue": 100, "net_income": 10,
        "total_assets": 1_000, "total_liabilities": 600, "total_equity": 300,
    })
    ok("rule: balance sheet imbalance",
       any("balance" in i.lower() for i in issues),
       f"issues={issues}")

    issues = _rule_checks({
        "revenue": 100_000, "net_income": 10_000,
        "gross_profit": 40_000,
        "eps_diluted": 2.0, "eps_basic": 2.1,
        "total_assets": 900, "total_liabilities": 600, "total_equity": 300,
    })
    ok("rule: clean data → no issues",
       issues == [],
       f"issues={issues}")

    # ── Market checks ────────────────────────────────────────────────────────

    warnings = _market_checks(_market_json(price=-5))
    ok("market: non-positive price",
       any("non-positive" in w for w in warnings),
       f"warnings={warnings}")

    warnings = _market_checks(_market_json(pe=600))
    ok("market: P/E > 500",
       any("500" in w or "implausibly high" in w for w in warnings),
       f"warnings={warnings}")

    warnings = _market_checks(_market_json(pe=-10))
    ok("market: negative P/E",
       any("negative P/E" in w for w in warnings),
       f"warnings={warnings}")

    warnings = _market_checks(_market_json(beta=7))
    ok("market: beta > 5",
       any("beta" in w.lower() for w in warnings),
       f"warnings={warnings}")

    warnings = _market_json(hi=100, lo=200)
    w2 = _market_checks(warnings)
    ok("market: 52W low > 52W high",
       any("52-week low" in w and "exceeds" in w for w in w2),
       f"warnings={w2}")

    warnings = _market_checks(_market_json(price=290, hi=260))
    ok("market: price > 52W high by >5%",
       any("exceeds 52-week high" in w or "not in sync" in w for w in warnings),
       f"warnings={warnings}")

    warnings = _market_checks(_market_json())
    ok("market: clean data → no warnings",
       warnings == [],
       f"warnings={warnings}")

    warnings = _market_checks(None)
    ok("market: None input → no warnings",
       warnings == [],
       f"warnings={warnings}")

    # ── Divergence checks ────────────────────────────────────────────────────

    signals = _divergence_checks(
        _profitable_metrics(),
        _market_json(),
        _news_json(-0.5),  # bearish on profitable
    )
    ok("divergence: bearish sentiment on profitable company",
       any("bearish" in s for s in signals),
       f"signals={signals}")

    signals = _divergence_checks(
        _loss_metrics(),
        _market_json(),
        _news_json(0.6),  # bullish on loss-maker
    )
    ok("divergence: bullish sentiment on loss-making company",
       any("bullish" in s for s in signals),
       f"signals={signals}")

    # Price in bottom 15% of 52W range with positive net income
    signals = _divergence_checks(
        _profitable_metrics(),
        _market_json(price=165, lo=160, hi=260),  # pct_from_low ≈ 5%
        None,
    )
    ok("divergence: price near 52W low on profitable company",
       any("52-week low" in s for s in signals),
       f"signals={signals}")

    signals = _divergence_checks(
        _profitable_metrics(),
        _market_json(),
        _news_json(0.1),  # neutral sentiment
    )
    ok("divergence: neutral sentiment → no signal",
       not any("bearish" in s or "bullish" in s for s in signals),
       f"signals={signals}")

    signals = _divergence_checks(_profitable_metrics(), _market_json(), None)
    ok("divergence: missing news → no sentiment signal",
       not any("bearish" in s or "bullish" in s for s in signals),
       f"signals={signals}")

    return results


# ── Reporting ─────────────────────────────────────────────────────────────────

def _print_table(results: list[CheckResult]) -> None:
    passed = sum(1 for r in results if r.passed)
    total = len(results)
    print("\n" + "=" * 72)
    print(f"{'Check':<50} {'Result':^6}")
    print("-" * 72)
    for r in results:
        mark = "✓" if r.passed else "✗"
        print(f"{r.name:<50} {mark:^6}")
        if not r.passed:
            print(f"  DETAIL: {r.detail}")
    print("=" * 72)
    print(f"Overall: {passed}/{total} checks passed\n")


def run_evals() -> list[CheckResult]:
    langfuse = Langfuse(
        public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
        secret_key=os.environ["LANGFUSE_SECRET_KEY"],
        host=os.environ.get("LANGFUSE_HOST", "http://localhost:3000"),
    )

    trace_id = uuid.uuid4().hex
    print(f"Eval trace: {os.environ.get('LANGFUSE_HOST', 'http://localhost:3000')}/traces/{trace_id}\n")

    results = _run_checks()

    passed = sum(1 for r in results if r.passed)
    total = len(results)
    score = passed / total if total else 0.0

    langfuse.create_score(
        trace_id=trace_id,
        name="eval.critic.rule_checks",
        value=score,
        comment=f"{passed}/{total} critic rule/market/divergence checks passed",
    )
    langfuse.flush()

    _print_table(results)
    return results


if __name__ == "__main__":
    results = run_evals()
    failed = [r for r in results if not r.passed]
    sys.exit(1 if failed else 0)
