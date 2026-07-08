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
on the filing excerpts provided. Focus on facts already in the text.

When verified financial metrics are provided, you MUST cite those exact figures.
Do not invent or estimate any numbers — use only the verified metrics supplied."""


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
    xbrl_metrics: dict[str, float] | None = None,
) -> ExtractedData:
    """Parse *document_text* and return structured financial data.

    If *xbrl_metrics* are provided (from EDGAR XBRL), they are used directly
    as the authoritative metrics. The LLM only produces a qualitative narrative.
    Otherwise falls back to rule-based HTML table parsing.
    """
    gw = gateway or get_gateway()

    with langfuse.start_as_current_observation(
        name="extractor",
        as_type="span",
        trace_context=trace_context,
        input={"ticker": ticker, "doc_length": len(document_text or ""), "xbrl": xbrl_metrics is not None},
    ) as span:

        sections = _split_sections(document_text)

        # ── Metrics: XBRL (preferred) or rule-based HTML fallback ────────────
        if xbrl_metrics:
            metrics: dict[str, object] = dict(xbrl_metrics)
            metrics_source = "xbrl"
        else:
            fin_table = sections.get("financial_tables", "")
            metrics = _extract_metrics_from_table(fin_table) if fin_table else {}
            metrics_source = "html"

        # ── Qualitative summary: LLM reads business/MD&A/risk narrative ──────
        qual_text = "\n\n".join(
            sections[k] for k in ("business", "mda", "risk_factors") if k in sections
        )[:_MAX_SECTION_CHARS]

        metrics_block = _format_metrics_for_prompt(metrics, filed_date)
        user_prompt = (
            f"Verified financial metrics for {ticker} (fiscal year ending {filed_date}):\n"
            f"{metrics_block}\n\n"
            f"Summarise the following 10-K sections in 2-3 paragraphs. "
            f"You must use the verified metrics above when citing any figures:\n\n{qual_text}"
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
            "metrics_source": metrics_source,
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens,
        })

    return result


def _format_metrics_for_prompt(metrics: dict[str, object], filed_date: str) -> str:
    """Format verified metrics as a readable block for injection into the LLM prompt."""
    _LABELS = {
        "revenue": "Total Revenue",
        "gross_profit": "Gross Profit",
        "operating_income": "Operating Income",
        "net_income": "Net Income",
        "eps_basic": "Basic EPS",
        "eps_diluted": "Diluted EPS",
        "total_assets": "Total Assets",
        "total_liabilities": "Total Liabilities",
        "total_equity": "Total Stockholders' Equity",
        "cash_and_equivalents": "Cash & Equivalents",
        "fiscal_year_end": "Fiscal Year End",
    }
    lines = []
    for key, label in _LABELS.items():
        val = metrics.get(key)
        if val is None:
            continue
        if key in ("eps_basic", "eps_diluted"):
            lines.append(f"  {label}: ${val:.2f}")
        elif isinstance(val, float):
            lines.append(f"  {label}: ${val:,.0f}M")
        else:
            lines.append(f"  {label}: {val}")
    return "\n".join(lines) if lines else "  (no verified metrics available)"


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


_YEAR_RE = re.compile(r"^\d{4}$")
# Matches separator/label cells that don't carry financial values.
# Second alternative catches numeric percentage cells like "18%" or "(5%)" —
# growth-rate columns that appear between year columns in some 10-K tables.
_SKIP_CELLS = re.compile(r"^[\$%—\-\s]*$|^-?\(?\d[\d,.]*\)?\s*%$")


def _find_current_year_slot(lines: list[str]) -> int | None:
    """Scan header rows and return the 0-based *numeric slot* of the most recent year.

    Numeric slot counts non-skip cells before the best-year column, mirroring
    exactly how _value_at_slot iterates data rows.

    Example header row cells: ['', '2024', 'Growth%', '2025']
    → best_abs=3; non-skip cells before index 3: '2024', 'Growth%' → slot 2? No —
    we mirror _value_at_slot which skips cells matching _SKIP_CELLS.
    '2024' is a number → slot 0; 'Growth%' matches _SKIP_CELLS → skipped; '2025' → slot 1.
    Returns 1 ✓
    """
    for line in lines[:20]:
        cells = [c.strip() for c in line.split("\t")]
        year_cells = [(i, int(c)) for i, c in enumerate(cells) if _YEAR_RE.match(c)]
        if len(year_cells) >= 2:
            best_abs = max(year_cells, key=lambda t: t[1])[0]
            # Count non-skip cells before best_abs (same logic as _value_at_slot)
            slot = sum(
                1 for i, c in enumerate(cells[1:], start=1)
                if i < best_abs and not _SKIP_CELLS.match(c) and _first_number(c) is not None
            )
            return slot
    return None


def _value_at_slot(cells: list[str], slot: int) -> float | None:
    """Return the number at numeric *slot* in cells (skipping label + separator cells)."""
    current_slot = 0
    for cell in cells[1:]:  # skip label
        if _SKIP_CELLS.match(cell):
            continue
        val = _first_number(cell)
        if val is not None:
            if current_slot == slot:
                return val  # zero is a valid financial metric (e.g. breakeven)
            current_slot += 1
    # Fallback: first number in the row (including zero)
    for cell in cells[1:]:
        val = _first_number(cell)
        if val is not None:
            return val
    return None


def _extract_metrics_from_table(table_text: str) -> dict[str, object]:
    """Parse tab-separated financial table rows and return a metrics dict."""
    metrics: dict[str, object] = {}
    lines = table_text.splitlines()

    current_year_slot = _find_current_year_slot(lines)

    for metric_key, patterns in _METRIC_PATTERNS:
        for pattern in patterns:
            for line in lines:
                if re.search(pattern, line, re.IGNORECASE):
                    cells = [c.strip() for c in line.split("\t")]
                    if current_year_slot is not None:
                        val = _value_at_slot(cells, current_year_slot)
                    else:
                        val = next(
                            (v for c in cells[1:] if (v := _first_number(c)) is not None and v != 0),
                            None,
                        )
                    if val is not None:
                        metrics[metric_key] = val
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
