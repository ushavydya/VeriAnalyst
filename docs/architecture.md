# VeriAnalyst — Architecture

## Pipeline Overview

```
SEC EDGAR XBRL ──► exact metrics (revenue, EPS, assets…)
                          │
SEC EDGAR HTML ──► LLM qualitative summary (business, MD&A, risks)
                          │
                    ┌─────▼──────┐
                    │  Extractor │
                    └─────┬──────┘
                          │  ExtractedData
                    ┌─────▼──────┐
                    │   Critic   │
                    └─────┬──────┘
                          │  Critique + confidence
                    ┌─────▼──────┐
                    │   Writer   │
                    └─────┬──────┘
                          │  Markdown report
                    Langfuse trace
```

Orchestrated by **LangGraph** (`StateGraph`). Every node is a Python async function; `PipelineState` (a `TypedDict`) flows between them.

---

## Components

### Retriever (`agents/retriever.py`)

1. Resolves ticker → CIK via SEC `company_tickers.json` (cached in SQLite).
2. Fetches filing list from `submissions/CIK{cik}.json`.
3. Downloads the primary 10-K HTM document from EDGAR Archives.
4. Fetches XBRL company facts from `data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json` and attaches verified metrics to `FilingResult.xbrl_metrics`.
5. Two-layer cache: SQLite metadata + filesystem document store (SHA-256 URL-keyed). XBRL JSON is cached separately in `xbrl_facts` table.
6. Enforces 10 req/sec rate limit via async token-bucket.
7. `fetch_10k_history(ticker, years=N)` fetches multiple annual filings for trend analysis.

### XBRL Fetcher (`xbrl.py`)

Fetches the EDGAR company facts JSON and maps us-gaap concepts to pipeline metric keys:

| Metric | us-gaap concept(s) tried in order |
|---|---|
| revenue | `RevenueFromContractWithCustomerExcludingAssessedTax`, `Revenues`, `SalesRevenueNet` |
| net_income | `NetIncomeLoss` |
| eps_diluted | `EarningsPerShareDiluted` |
| total_assets | `Assets` |
| … | … |

Always returns the **most recent 10-K annual value** (sorted by fiscal year end date, then filed date to handle amendments). Values in millions USD; EPS in USD/share.

### Extractor (`agents/extractor.py`)

**Two-source strategy:**

| Source | Used for | Why |
|---|---|---|
| XBRL | All numeric metrics | Authoritative, structured, amendment-aware |
| HTML | Qualitative summary (LLM) | XBRL has no narrative text |

When XBRL metrics are available (always for major filers), rule-based HTML table parsing is skipped entirely. The LLM is given verified XBRL numbers injected into its prompt and instructed to cite only those figures — it cannot hallucinate a number that differs from the SEC filing.

Returns `ExtractedData(ticker, filed_date, sections, metrics)`.

### Critic (`agents/critic.py`)

1. **Rule-based confidence scoring** (no LLM):
   - Base 0.9 if both `revenue` and `net_income` extracted; 0.5 if partial; 0.0 if neither
   - −0.15 per rule violation (gross profit > revenue, diluted EPS > basic EPS, balance sheet mismatch, etc.)
2. LLM call for a one-sentence data quality summary only.
3. Returns `Critique(ticker, confidence, issues, summary)`.

### Writer (`agents/writer.py`)

Synthesises `ExtractedData` + `Critique` into a Markdown investment report via LLM. Sections: Executive Summary, Financial Highlights, Business Overview, Key Risks, Data Confidence Note.

### Comparison Agent (`comparison.py`)

Runs retrieval and extraction for 2–4 tickers **in parallel** using `asyncio.gather`, then passes all extractions to the LLM in a single prompt for a structured comparative report. Sections: Executive Summary, Financial Comparison, Business Model Differences, Risk Comparison, Verdict.

Parallel execution means comparing 3 companies takes roughly the same wall-clock time as analysing 1.

### Trends (`trends.py`)

Runs `fetch_10k_history` + `extract` on each historical filing to build a time series of metrics. Metrics-only — does not call Critic or Writer.

### LLM Gateway (`gateway/`)

Abstract `LLMGateway` interface with two backends:

| Backend | Config |
|---|---|
| `AnthropicBackend` | `LLM_PROVIDER=anthropic`, `ANTHROPIC_API_KEY` |
| `OllamaBackend` | `LLM_PROVIDER=ollama`, `OLLAMA_MODEL=qwen3.5:latest` |

`get_gateway()` reads `LLM_PROVIDER` from env. Both implement `complete(messages, *, system, max_tokens, json_mode) → ModelResponse`.

---

## Observability (Langfuse v3)

Each pipeline run has a shared **trace** (`trace_id = uuid4().hex` in `PipelineState`). Every agent attaches a **span** via `langfuse.start_as_current_observation(...)`:

| Span | Input | Output |
|---|---|---|
| `retriever` | ticker, form_type | cache_hit, cik, filed_date, xbrl_metrics |
| `retriever.xbrl` | cik, fiscal_year | source (cache/edgar), metrics_found |
| `retriever.resolve_cik` | ticker | cik |
| `retriever.find_filing` | cik, form_type | accession, filed_date |
| `retriever.download` | url | path, bytes |
| `extractor` | ticker, doc_length, xbrl=bool | sections, metrics_found, metrics_source, tokens |
| `critic` | ticker, metrics_count | confidence, issues, tokens |
| `writer` | ticker, confidence | report_length, tokens |
| `comparison` | tickers | report_length, tokens |

After the writer completes, `langfuse.create_score(trace_id, name="data-confidence", value=confidence)` attaches the critic score to the trace.

Langfuse runs locally via Docker Compose (6 services: langfuse-web, langfuse-worker, postgres, clickhouse@24.3, redis, minio).

---

## Caching

Four stores, zero TTL (cached until manually cleared):

| Table / Store | Key | Contents |
|---|---|---|
| SQLite `ticker_cik` | ticker | CIK string |
| SQLite `filings` | (ticker, form_type, accession_number) | filing metadata |
| SQLite `xbrl_facts` | (cik, "raw") | raw EDGAR company facts JSON |
| SQLite `documents` + filesystem | SHA-256(URL)[:16] | raw HTM text as `.txt` |

`filings` PK includes `accession_number` so all historical filings coexist; `get_filing()` returns the most recent by `filed_date DESC`.

Default location: `~/.cache/verianalyst/`. Clear with `rm -rf ~/.cache/verianalyst/`.

---

## Evals

Two eval scripts, both posting scores to Langfuse:

### `eval_extractor.py` — Accuracy eval (integration test)

Runs the full retrieval + extraction pipeline for each ticker in the golden dataset and compares extracted metrics against XBRL-sourced ground truth. Tolerance: 2%.

```
Ticker    FY    Score   Field results
AAPL       ✓    100%    revenue=✓  net_income=✓  eps_diluted=✓  total_assets=✓
MSFT       ✓    100%    revenue=✓  net_income=✓  eps_diluted=✓
UBER       ✓    100%    revenue=✓  net_income=✓  eps_diluted=✓
```

### `eval_report_quality.py` — LLM-as-judge

Sends the generated report + extracted metrics to the LLM acting as a judge. Scores five dimensions (0–1): `factual_grounding`, `completeness`, `fiscal_year_accuracy`, `no_hallucination`, `clarity`. Posts each dimension and an aggregate score to Langfuse.

### Unit tests (`tests/`)

72 pytest-asyncio tests covering:
- `test_xbrl.py` — XBRL entry selection, concept fallbacks, scaling, fiscal year filtering
- `test_sqlite_cache.py` — XBRL cache round-trip, filing ordering, CIK lookup
- `test_extractor.py` — HTML parsing, XBRL path, prompt injection, fallback
- `test_retriever.py` — cache hit/miss paths, rate limiter, URL construction
- `test_critic.py`, `test_writer.py`, `test_gateway.py`

All tests run in under 1 second with no network calls.

---

## Web UI (`app/main.py`)

Streamlit app with three tabs:

| Tab | What it does |
|---|---|
| **📄 Analysis** | Full pipeline for one ticker; report + follow-up Q&A grounded in the filing |
| **📈 Trends** | Up to 5 years of filings; summary table + line charts (revenue, net income, EPS, assets, cash) |
| **⚖️ Compare** | 2–4 tickers in parallel; side-by-side metrics table, bar charts, LLM comparative report with Verdict |

Async pipelines run in dedicated threads with fresh event loops to avoid Streamlit runtime conflicts.

---

## Project Layout

```
VeriAnalyst/
├── app/
│   └── main.py                    # Streamlit UI
├── src/sec_analyzer/
│   ├── agents/
│   │   ├── retriever.py           # EDGAR fetcher + XBRL + cache + rate limiter
│   │   ├── extractor.py           # XBRL metrics + LLM qualitative summary
│   │   ├── critic.py              # Rule-based confidence + LLM quality summary
│   │   └── writer.py              # LLM report writer
│   ├── cache/
│   │   └── sqlite_cache.py        # SQLite cache (filings, CIK, XBRL, documents)
│   ├── gateway/
│   │   ├── base.py
│   │   ├── anthropic_backend.py
│   │   └── ollama_backend.py
│   ├── orchestration/
│   │   └── graph.py               # LangGraph StateGraph pipeline
│   ├── comparison.py              # Parallel multi-company comparison
│   ├── trends.py                  # Multi-year metric extraction
│   └── xbrl.py                   # EDGAR XBRL company facts fetcher
├── evals/
│   ├── golden_dataset.py          # XBRL-sourced ground truth
│   ├── eval_extractor.py          # Accuracy eval
│   └── eval_report_quality.py     # LLM-as-judge eval
├── tests/                         # 72 unit tests
├── examples/
├── docs/
│   └── architecture.md
├── docker-compose.yml             # Langfuse v3 self-hosted stack
├── pyproject.toml
├── .env.example
└── README.md
```

---

## Tech Stack

| Layer | Library |
|---|---|
| Orchestration | LangGraph |
| Observability | Langfuse v3 (self-hosted) |
| LLM | Anthropic Claude or Ollama |
| Financial data | SEC EDGAR XBRL + HTML filings |
| HTML parsing | BeautifulSoup4 |
| HTTP | httpx (async) |
| Cache | aiosqlite + filesystem |
| UI | Streamlit |
| Tests | pytest-asyncio |
