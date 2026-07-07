"""Extractor agent — parses 10-K text and extracts structured financial data via LLM."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from bs4 import BeautifulSoup, Tag
from langfuse import Langfuse
from langfuse.types import TraceContext

from sec_analyzer.gateway import LLMGateway, Message, get_gateway

# ── Section detection ─────────────────────────────────────────────────────────

# Matched against the *raw HTML* before stripping so we can locate section
# boundaries, then we parse each slice with BeautifulSoup.
_SECTION_PATTERNS: list[tuple[str, str]] = [
    ("business",      r"item\s+1[.\s]+business"),
    ("risk_factors",  r"item\s+1a[.\s]+risk\s+factors"),
    ("mda",           r"item\s+7[.\s]+management"),
    ("financials",    r"item\s+8[.\s]+financial\s+statements"),
]

_MAX_DOC_CHARS = 1_500_000  # full filing is ~1.5 MB; keep all of it
_MAX_SECTION_CHARS = 12_000 # chars per section sent to LLM

# ── Metric schema (used in the LLM prompt) ───────────────────────────────────

_METRICS_SCHEMA: dict[str, str] = {
    "revenue":             "float | null — total net revenue / net sales (millions USD)",
    "gross_profit":        "float | null — gross profit (millions USD)",
    "operating_income":    "float | null — operating income / loss (millions USD)",
    "net_income":          "float | null — net income attributable to common shareholders (millions USD)",
    "eps_basic":           "float | null — basic earnings per share (USD)",
    "eps_diluted":         "float | null — diluted earnings per share (USD)",
    "total_assets":        "float | null — total assets (millions USD)",
    "total_liabilities":   "float | null — total liabilities (millions USD)",
    "total_equity":        "float | null — total stockholders' equity (millions USD)",
    "cash_and_equivalents":"float | null — cash and cash equivalents (millions USD)",
    "fiscal_year_end":     "str  | null — fiscal year end date, ISO format YYYY-MM-DD",
}

_SYSTEM_PROMPT = """\
You are a senior financial analyst. Summarise the company's business model,
recent financial performance, and key risks in 2-3 concise paragraphs based
on the filing excerpts provided. Focus on facts already in the text."""


# ── HTML / text helpers ───────────────────────────────────────────────────────

def _html_to_text(html: str) -> str:
    """Convert an HTML fragment to readable text, preserving table structure."""
    soup = BeautifulSoup(html, "html.parser")

    # Remove script/style noise
    for tag in soup(["script", "style", "head"]):
        tag.decompose()

    parts: list[str] = []
    for element in soup.descendants:
        if not isinstance(element, Tag):
            continue
        name = element.name

        if name == "table":
            parts.append(_table_to_text(element))
        elif name in ("p", "div", "span") and element.find_parent("table") is None:
            text = element.get_text(" ", strip=True)
            if text:
                parts.append(text)
        elif name in ("h1", "h2", "h3", "h4"):
            text = element.get_text(" ", strip=True)
            if text:
                parts.append(f"\n## {text}\n")

    result = "\n".join(parts)
    result = re.sub(r"\n{3,}", "\n\n", result)
    return result.strip()


def _table_to_text(table: Tag) -> str:
    """Render an HTML table as tab-separated rows, skipping empty ones."""
    rows: list[str] = []
    for tr in table.find_all("tr"):
        cells = [td.get_text(" ", strip=True) for td in tr.find_all(["td", "th"])]
        cells = [c for c in cells if c]  # drop blank cells
        if cells:
            rows.append("\t".join(cells))
    return "\n".join(rows)


def _split_sections(html: str) -> dict[str, str]:
    """Parse the full HTML document, then split into named 10-K sections."""
    # Parse once — extract clean text with table structure preserved
    full_text = _html_to_text(html[:_MAX_DOC_CHARS])

    # Locate section boundaries in the *clean text* (skips TOC href matches)
    hits: list[tuple[str, int]] = []
    for name, pattern in _SECTION_PATTERNS:
        # Skip the first match if it lands before 5% into the doc (likely TOC)
        min_pos = len(full_text) // 20
        for m in re.finditer(pattern, full_text, re.IGNORECASE):
            if m.start() >= min_pos:
                hits.append((name, m.start()))
                break  # take first non-TOC match for each section

    hits.sort(key=lambda x: x[1])

    if not hits:
        return {"raw": full_text[:_MAX_SECTION_CHARS * 2]}

    sections: dict[str, str] = {}
    for i, (name, start) in enumerate(hits):
        end = hits[i + 1][1] if i + 1 < len(hits) else len(full_text)
        text = full_text[start:end].strip()
        sections[name] = text[:_MAX_SECTION_CHARS]

    # Always add financial tables directly — they're more reliable than
    # section-boundary slicing for numeric data extraction.
    fin_tables = _extract_financial_tables(html[:_MAX_DOC_CHARS])
    if fin_tables:
        sections["financial_tables"] = fin_tables[:_MAX_SECTION_CHARS]

    return sections


def _prioritise_numeric_content(text: str) -> str:
    """Reorder lines so rows with dollar/number values come first."""
    lines = text.splitlines()
    numeric = [l for l in lines if re.search(r"\d[\d,]*(?:\.\d+)?", l)]
    non_numeric = [l for l in lines if l not in numeric]
    reordered = numeric + non_numeric
    return "\n".join(reordered)


# Income-statement / balance-sheet anchor terms — if a table contains any of
# these it is almost certainly a financial statement table.
_FINANCIAL_TABLE_ANCHORS = re.compile(
    r"net sales|total net sales|net revenue|total revenue|"
    r"net income|total assets|total liabilities|earnings per share",
    re.IGNORECASE,
)


def _extract_financial_tables(html: str) -> str:
    """Find and return income-statement / balance-sheet tables as readable text."""
    soup = BeautifulSoup(html[:_MAX_DOC_CHARS], "html.parser")
    found: list[str] = []
    for table in soup.find_all("table"):
        text = table.get_text(" ")
        if _FINANCIAL_TABLE_ANCHORS.search(text) and re.search(r"\d[\d,]{2,}", text):
            found.append(_table_to_text(table))
            if len("\n\n".join(found)) >= _MAX_SECTION_CHARS:
                break
    return "\n\n".join(found)


# ── Public API ────────────────────────────────────────────────────────────────

@dataclass
class ExtractedData:
    ticker: str
    filed_date: str
    sections: dict[str, str] = field(default_factory=dict)
    metrics: dict[str, object] = field(default_factory=dict)


async def extract(
    ticker: str,
    filed_date: str,
    document_text: str,
    langfuse: Langfuse,
    trace_context: TraceContext,
    *,
    gateway: LLMGateway | None = None,
) -> ExtractedData:
    """Parse *document_text* and extract structured financial data via the LLM gateway."""
    gw = gateway or get_gateway()

    with langfuse.start_as_current_observation(
        name="extractor",
        as_type="span",
        trace_context=trace_context,
        input={"ticker": ticker, "doc_length": len(document_text or "")},
    ) as span:

        sections = _split_sections(document_text)

        # ── Metric extraction: rule-based from tab-separated financial table ──
        fin_table = sections.get("financial_tables", "")
        metrics = _extract_metrics_from_table(fin_table) if fin_table else {}

        # ── Qualitative summary: LLM summarises business + MD&A + risk text ──
        qual_text = "\n\n".join(
            sections[k] for k in ("business", "mda", "risk_factors") if k in sections
        )[:_MAX_SECTION_CHARS]
        user_prompt = (
            f"Summarise the following 10-K sections for {ticker} "
            f"(filed {filed_date}) in 2-3 paragraphs:\n\n{qual_text}"
        )
        messages: list[Message] = [{"role": "user", "content": user_prompt}]
        response = await gw.complete(messages, system=_SYSTEM_PROMPT, max_tokens=1024)
        sections["summary"] = response.text.strip()

        result = ExtractedData(
            ticker=ticker,
            filed_date=filed_date,
            sections=sections,
            metrics=metrics,
        )

        span.update(output={
            "sections": list(sections.keys()),
            "metrics_found": list(metrics.keys()),
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens,
        })

    return result


def _parse_metrics(llm_text: str) -> dict[str, object]:
    """Extract the JSON object from the LLM reply, stripping any markdown fences.

    Kept for backward-compatibility with tests; not used in the main pipeline.
    """
    raw = llm_text.strip()
    raw = re.sub(r"^```[a-z]*\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw).strip()

    try:
        data: dict = json.loads(raw)
    except json.JSONDecodeError:
        return {}

    known_keys = set(_METRICS_SCHEMA.keys())
    if not (known_keys & data.keys()):
        for v in data.values():
            if isinstance(v, dict) and (known_keys & v.keys()):
                data = v
                break

    cleaned: dict[str, object] = {}
    for key, value in data.items():
        if value is None:
            continue
        if key == "fiscal_year_end":
            cleaned[key] = str(value)
        else:
            try:
                cleaned[key] = float(value)
            except (TypeError, ValueError):
                pass

    return cleaned


# ── Rule-based metric extraction from tab-separated financial table text ──────

# Each entry: (metric_key, list of row-label patterns to try)
_METRIC_PATTERNS: list[tuple[str, list[str]]] = [
    ("revenue",             ["total net sales", "total net revenue", "total revenue", "net sales"]),
    ("gross_profit",        ["gross margin", "gross profit"]),
    ("operating_income",    ["operating income"]),
    ("net_income",          ["net income"]),
    ("eps_basic",           ["basic"]),
    ("eps_diluted",         ["diluted"]),
    ("total_assets",        ["total assets"]),
    ("total_liabilities",   ["total liabilities"]),
    ("total_equity",        ["total stockholders", "total shareholders", "total equity"]),
    ("cash_and_equivalents",["cash and cash equivalents"]),
]

_NUMBER_RE = re.compile(r"\(?([\d,]+(?:\.\d+)?)\)?")


def _first_number(text: str) -> float | None:
    """Return the first parseable number in *text*, treating parentheses as negative."""
    m = _NUMBER_RE.search(text)
    if not m:
        return None
    value = float(m.group(1).replace(",", ""))
    if text.lstrip().startswith("("):
        value = -value
    return value


def _extract_metrics_from_table(table_text: str) -> dict[str, object]:
    """Parse tab-separated financial table rows and return a metrics dict."""
    metrics: dict[str, object] = {}
    lines = table_text.splitlines()

    for metric_key, patterns in _METRIC_PATTERNS:
        for pattern in patterns:
            for line in lines:
                if re.search(pattern, line, re.IGNORECASE):
                    # Take the first numeric column (most recent year)
                    cells = [c.strip() for c in line.split("\t")]
                    for cell in cells[1:]:  # skip the label column
                        val = _first_number(cell)
                        if val is not None and val != 0:
                            metrics[metric_key] = val
                            break
                    if metric_key in metrics:
                        break
            if metric_key in metrics:
                break

    # Fiscal year end: find all month-day-year dates and take the most recent.
    # Table headers often list two years (current + prior); we want the largest year.
    _DATE_PAT = re.compile(
        r"(january|february|march|april|may|june|july|august|september|october|november|december)"
        r"\s+\d{1,2},?\s+(\d{4})",
        re.IGNORECASE,
    )
    date_matches = _DATE_PAT.findall(table_text)
    if date_matches:
        # findall returns (month, year) tuples; pick the tuple with the largest year
        best = max(date_matches, key=lambda t: int(t[1]))
        # Re-search for the full matched string of the best year
        best_m = re.search(
            rf"{re.escape(best[0])}\s+\d{{1,2}},?\s+{best[1]}", table_text, re.IGNORECASE
        )
        if best_m:
            metrics["fiscal_year_end"] = best_m.group(0).strip()

    return metrics
