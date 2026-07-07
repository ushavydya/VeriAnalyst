"""Critic agent — validates extracted financial data and assigns a confidence score."""
from __future__ import annotations

import json
import re
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

_MAX_SECTION_CHARS = 4_000  # chars of filing text sent to the LLM


@dataclass
class Critique:
    ticker: str
    confidence: float  # 0–1
    issues: list[str] = field(default_factory=list)
    summary: str = ""


# ── Rule-based sanity checks ─────────────────────────────────────────────────

def _rule_checks(metrics: dict[str, object]) -> list[str]:
    """Return a list of rule-violation strings (empty = all clear)."""
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
        # Net income > revenue is theoretically possible via one-time gains but very rare
        if abs(m["net_income"]) > m["revenue"] * 2:
            issues.append("Net income magnitude is more than 2× revenue — verify extraction")

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
    """Derive a confidence score purely from rule checks and metric completeness."""
    required = {"revenue", "net_income"}
    missing = required - metrics.keys()

    if missing == required:          # nothing extracted at all
        return 0.0
    if missing:                      # partial extraction
        base = 0.5
    else:
        base = 0.9

    # Each rule violation knocks 0.15 off
    penalty = min(len(rule_issues) * 0.15, base)
    return round(max(0.0, base - penalty), 2)


def _parse_llm_response(text: str) -> tuple[float, list[str], str]:
    """Return (confidence, issues, summary) from the LLM JSON reply.

    Kept for backward-compatibility with tests.
    """
    raw = text.strip()
    raw = re.sub(r"^```[a-z]*\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw).strip()

    try:
        data = json.loads(raw)
        confidence = max(0.0, min(1.0, float(data.get("confidence", 0.5))))
        issues = [str(i) for i in data.get("issues", [])]
        summary = str(data.get("summary", ""))
        return confidence, issues, summary
    except (json.JSONDecodeError, TypeError, ValueError):
        return 0.5, ["LLM returned unparseable critique"], "Could not parse LLM response."


# ── Public API ────────────────────────────────────────────────────────────────

async def critique(
    data: ExtractedData,
    langfuse: Langfuse,
    trace_context: TraceContext,
    *,
    gateway: LLMGateway | None = None,
) -> Critique:
    """Validate *data* with rule checks + LLM review and return a Critique."""
    gw = gateway or get_gateway()

    with langfuse.start_as_current_observation(
        name="critic",
        as_type="span",
        trace_context=trace_context,
        input={"ticker": data.ticker, "metrics_count": len(data.metrics)},
    ) as span:

        rule_issues = _rule_checks(data.metrics)
        confidence = _confidence_from_rules(rule_issues, data.metrics)

        # LLM provides only a one-sentence summary; does not affect confidence
        user_prompt = _build_summary_prompt(data, rule_issues)
        messages: list[Message] = [{"role": "user", "content": user_prompt}]
        response = await gw.complete(messages, system=_SYSTEM_PROMPT, max_tokens=200)
        lines = response.text.strip().splitlines()
        summary = lines[0] if lines else "No summary available."

        result = Critique(
            ticker=data.ticker,
            confidence=confidence,
            issues=rule_issues,
            summary=summary,
        )

        span.update(output={
            "confidence": result.confidence,
            "issues": result.issues,
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens,
        })

    return result
