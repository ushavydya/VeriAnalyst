"""Critic agent — validates extracted financial data and assigns a confidence score."""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from langfuse import Langfuse
from langfuse.types import TraceContext

from sec_analyzer.agents.extractor import ExtractedData
from sec_analyzer.gateway import LLMGateway, Message, get_gateway

# ── Required metrics — extraction is considered incomplete without these ───────

_REQUIRED_METRICS = {"revenue", "net_income"}

_SYSTEM_PROMPT = """\
You are a senior financial analyst reviewing extracted SEC 10-K metrics.
Write one concise sentence summarising the overall data quality based ONLY on
the rule-based issues listed. Do not invent new issues, do not comment on
filing dates or compare to historical norms you don't have access to.
Reply with plain text only — no JSON, no headers."""


@dataclass
class Critique:
    ticker: str
    confidence: float  # 0–1; reflects 10-K extraction quality only
    issues: list[str] = field(default_factory=list)
    summary: str = ""


# ── Rule-based sanity checks: 10-K extraction ────────────────────────────────

def _rule_checks(metrics: dict[str, object]) -> list[str]:
    """Return rule-violation strings for extracted 10-K metrics (empty = all clear)."""
    issues: list[str] = []
    m = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}

    missing = _REQUIRED_METRICS - m.keys()
    if missing:
        issues.append(f"Missing required metrics: {', '.join(sorted(missing))}")

    if "revenue" in m and m["revenue"] <= 0:
        issues.append("Revenue is non-positive — likely an extraction error")

    if "gross_profit" in m and "revenue" in m:
        if m["gross_profit"] > m["revenue"]:
            issues.append("Gross profit exceeds revenue — impossible")

    if "net_income" in m and "revenue" in m:
        # Net income above 75% of revenue is a strong signal of extraction error;
        # even exceptional businesses (e.g. NVDA at ~55%) stay below this bound
        if abs(m["net_income"]) > m["revenue"] * 0.75:
            issues.append("Net income magnitude exceeds 75% of revenue — verify extraction")

    if "eps_diluted" in m and "eps_basic" in m:
        # Diluted EPS is always ≤ basic EPS in magnitude (more shares = lower per-share)
        if m["eps_diluted"] > m["eps_basic"] + 0.01:
            issues.append("Diluted EPS exceeds basic EPS — logically impossible")

    if "total_assets" in m and "total_liabilities" in m and "total_equity" in m:
        implied_assets = m["total_liabilities"] + m["total_equity"]
        pct_err = abs(m["total_assets"] - implied_assets) / max(abs(m["total_assets"]), 1)
        if pct_err > 0.05:
            issues.append(
                f"Balance sheet does not balance: assets={m['total_assets']:.0f}, "
                f"liabilities+equity={implied_assets:.0f} (>{pct_err:.0%} diff)"
            )

    return issues


# ── Rule-based sanity checks: market data ────────────────────────────────────

def _market_checks(market_json: str | None) -> list[str]:
    """Return warnings for implausible market data values (empty = all clear)."""
    if not market_json:
        return []
    try:
        d = json.loads(market_json)
    except Exception:
        return ["Market data JSON could not be parsed"]

    warnings: list[str] = []
    quote = d.get("quote") or {}
    ratios = d.get("ratios") or {}

    price = quote.get("price")
    if price is not None and price <= 0:
        warnings.append(f"Market data: price is non-positive ({price}) — data may be stale or erroneous")

    pe = ratios.get("pe_ratio")
    if pe is not None:
        if pe > 500:
            warnings.append(f"Market data: P/E ratio {pe:.0f}x is implausibly high — verify provider data")
        elif pe < 0:
            warnings.append(f"Market data: negative P/E ({pe:.1f}x) — company may be loss-making; verify against net_income")

    beta = ratios.get("beta")
    if beta is not None and abs(beta) > 5:
        warnings.append(f"Market data: beta {beta:.2f} is outside normal range (±5) — data quality uncertain")

    hi = ratios.get("week_52_high")
    lo = ratios.get("week_52_low")
    if hi is not None and lo is not None and lo > hi:
        warnings.append(f"Market data: 52-week low ({lo}) exceeds 52-week high ({hi}) — data error")

    if price is not None and hi is not None and price > hi * 1.05:
        warnings.append(
            f"Market data: current price ${price:.2f} exceeds 52-week high ${hi:.2f} "
            "by >5% — data may not be in sync"
        )

    return warnings


# ── Divergence signals: fundamentals vs. current market ──────────────────────

def _divergence_checks(
    metrics: dict[str, object],
    market_json: str | None,
    news_json: str | None,
) -> list[str]:
    """Flag notable divergences between 10-K fundamentals, price action, and sentiment."""
    signals: list[str] = []
    m = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}

    # Sentiment vs. fundamentals
    if news_json:
        try:
            news = json.loads(news_json)
            score = news.get("sentiment_score")
            if score is not None and "net_income" in m and "revenue" in m:
                profitable = m["net_income"] > 0
                if score <= -0.3 and profitable:
                    signals.append(
                        f"Divergence: news sentiment is bearish ({score:+.2f}) despite profitable fundamentals "
                        "— may indicate near-term headwinds not reflected in the annual filing"
                    )
                elif score >= 0.3 and m["net_income"] < 0:
                    signals.append(
                        f"Divergence: news sentiment is bullish ({score:+.2f}) despite reported net loss "
                        "— market may be pricing in a turnaround"
                    )
        except Exception:
            pass

    # Price vs. 52-week range — flag if stock is near 52W low while fundamentals are strong
    if market_json:
        try:
            mkt = json.loads(market_json)
            quote = mkt.get("quote") or {}
            ratios = mkt.get("ratios") or {}
            price = quote.get("price")
            lo = ratios.get("week_52_low")
            hi = ratios.get("week_52_high")
            if price and lo and hi and (hi - lo) > 0:
                pct_from_low = (price - lo) / (hi - lo)
                if pct_from_low < 0.15 and m.get("net_income", 0) > 0:
                    signals.append(
                        f"Divergence: stock is near its 52-week low (bottom {pct_from_low:.0%} of range) "
                        "despite positive net income — potential value opportunity or deteriorating outlook"
                    )
        except Exception:
            pass

    return signals


# ── LLM critique ──────────────────────────────────────────────────────────────

def _build_summary_prompt(data: ExtractedData, rule_issues: list[str]) -> str:
    metrics_block = json.dumps({k: v for k, v in data.metrics.items()}, indent=2)
    issues_text = (
        "\n".join(f"- {i}" for i in rule_issues)
        if rule_issues else "None."
    )
    return (
        f"Ticker: {data.ticker} | Metrics extracted: {len(data.metrics)}\n"
        f"Extracted values:\n{metrics_block}\n\n"
        f"Rule-based issues found:\n{issues_text}\n\n"
        f"Write one sentence summarising the extraction quality."
    )


def _confidence_from_rules(rule_issues: list[str], metrics: dict) -> float:
    """Derive a confidence score from 10-K rule checks and metric completeness.

    Market/news warnings and divergence signals are informational and do NOT
    penalise this score — they appear in issues but aren't extraction errors.
    """
    numeric_keys = {k for k, v in metrics.items() if isinstance(v, (int, float))}
    missing = _REQUIRED_METRICS - numeric_keys

    if missing == _REQUIRED_METRICS:  # nothing extracted at all
        return 0.0
    if missing:                       # partial extraction
        base = 0.5
    else:
        base = 0.9

    # Each rule violation knocks 0.15 off; warnings/divergences don't count here
    penalty = min(len(rule_issues) * 0.15, base)
    return round(max(0.0, base - penalty), 2)


# ── Public API ────────────────────────────────────────────────────────────────

async def critique(
    data: ExtractedData,
    langfuse: Langfuse,
    trace_context: TraceContext,
    *,
    gateway: LLMGateway | None = None,
    market_json: str | None = None,
    news_json: str | None = None,
) -> Critique:
    """Validate extracted data + optional market/news inputs; return a Critique."""
    gw = gateway or get_gateway()

    with langfuse.start_as_current_observation(
        name="critic",
        as_type="span",
        trace_context=trace_context,
        input={
            "ticker": data.ticker,
            "metrics_count": len(data.metrics),
            "has_market": market_json is not None,
            "has_news": news_json is not None,
        },
    ) as span:

        rule_issues = _rule_checks(data.metrics)
        confidence = _confidence_from_rules(rule_issues, data.metrics)

        # Market data and divergence checks are informational — appended to issues
        # but do not affect the confidence score (which measures extraction quality)
        market_warnings = _market_checks(market_json)
        divergence_signals = _divergence_checks(data.metrics, market_json, news_json)
        all_issues = rule_issues + market_warnings + divergence_signals

        # LLM provides only a one-sentence summary of extraction quality
        user_prompt = _build_summary_prompt(data, rule_issues)
        messages: list[Message] = [{"role": "user", "content": user_prompt}]
        response = await gw.complete(messages, system=_SYSTEM_PROMPT, max_tokens=200)
        lines = response.text.strip().splitlines()
        summary = lines[0] if lines else "No summary available."

        result = Critique(
            ticker=data.ticker,
            confidence=confidence,
            issues=all_issues,
            summary=summary,
        )

        span.update(output={
            "confidence": result.confidence,
            "rule_issues": rule_issues,
            "market_warnings": market_warnings,
            "divergence_signals": divergence_signals,
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens,
        })

    return result
