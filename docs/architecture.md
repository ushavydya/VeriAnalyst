# VeriAnalyst — Architecture

## Pipeline Overview

```
SEC EDGAR
    │
    ▼
┌───────────┐    ┌───────────┐    ┌────────┐    ┌────────┐
│ Retriever │ ─► │ Extractor │ ─► │ Critic │ ─► │ Writer │
└───────────┘    └───────────┘    └────────┘    └────────┘
      │                │                              │
  SQLite +         Metrics +                     Markdown
 File Cache        Sections                       Report
                 (rule-based)
                      +
                  LLM Summary
```

Orchestrated by **LangGraph** (`StateGraph`). Every node is a Python async function; `PipelineState` (a `TypedDict`) flows between them. After the writer completes, a `data-confidence` score is auto-posted to Langfuse.

---

## Components

### Retriever (`agents/retriever.py`)

1. Resolves ticker → CIK via SEC `company_tickers.json` (cached in SQLite).
2. Fetches filing list from `submissions/CIK{cik}.json`.
3. Downloads the primary 10-K HTM document from EDGAR Archives.
4. Two-layer cache: SQLite metadata + filesystem document store (SHA-256 URL-keyed).
5. Enforces 10 req/sec rate limit via async token-bucket.
6. `fetch_10k_history(ticker, years=5)` fetches multiple annual filings for trend analysis.

### Extractor (`agents/extractor.py`)

1. Parses HTML with **BeautifulSoup4** — preserves table structure as tab-separated rows.
2. **Rule-based metric extraction** — scans financial tables by row label regex; no LLM needed for numbers. Extracts: revenue, gross profit, operating income, net income, EPS (basic/diluted), total assets, liabilities, equity, cash.
3. Picks the **most recent fiscal year** from table headers (avoids reading prior-year comparative columns).
4. LLM call (via gateway) for qualitative summary of Business, MDA, and Risk Factors sections only.
5. Returns `ExtractedData(ticker, filed_date, sections, metrics)`.

### Critic (`agents/critic.py`)

1. **Rule-based confidence scoring** — no LLM involved in scoring:
   - Base 0.9 if both `revenue` and `net_income` extracted; 0.5 if partial; 0.0 if neither
   - −0.15 per rule violation (gross profit > revenue, diluted EPS > basic EPS, balance sheet mismatch, etc.)
2. LLM call for a single-sentence data quality summary only.
3. Returns `Critique(ticker, confidence, issues, summary)`.

### Writer (`agents/writer.py`)

Synthesises `ExtractedData` + `Critique` into a narrative Markdown investment report via LLM. Sections: Executive Summary, Financial Highlights, Business Overview, Key Risks, Data Confidence Note.

### LLM Gateway (`gateway/`)

Abstract `LLMGateway` interface with two backends:

| Backend | Class | Config |
|---|---|---|
| Anthropic | `AnthropicBackend` | `LLM_PROVIDER=anthropic`, `ANTHROPIC_API_KEY` |
| Ollama | `OllamaBackend` | `LLM_PROVIDER=ollama`, `OLLAMA_MODEL=qwen3.5:latest` |

`get_gateway()` reads `LLM_PROVIDER` from env and returns the correct backend. Both implement `complete(messages, *, system, max_tokens, json_mode) → ModelResponse`.

### Trends (`trends.py`)

Runs `fetch_10k_history` + `extract` on each historical filing to build a time series of metrics. Used by the Streamlit **Trends** tab. Does not call the critic or writer — metrics extraction only.

---

## Observability (Langfuse v3)

Each pipeline run has a shared **trace** (`trace_id = uuid4().hex` in `PipelineState`). Every agent attaches a **span** via `langfuse.start_as_current_observation(...)`:

| Span | Input | Output |
|---|---|---|
| `retriever` | ticker, form_type | cache_hit, cik, filed_date |
| `retriever.resolve_cik` | ticker | cik |
| `retriever.find_filing` | cik, form_type | accession, filed_date |
| `retriever.download` | url | path, bytes |
| `extractor` | ticker, doc_length | sections, metrics_found, tokens |
| `critic` | ticker, metrics_count | confidence, issues, tokens |
| `writer` | ticker, confidence | report_length, tokens |

After the writer completes, `langfuse.create_score(trace_id, name="data-confidence", value=confidence)` attaches the critic score to the trace.

Langfuse runs locally via Docker Compose (6 services: langfuse-web, langfuse-worker, postgres, clickhouse, redis, minio).

---

## Caching

Two layers, zero TTL (cached forever until manually cleared):

| Layer | Store | Key | Contents |
|---|---|---|---|
| CIK lookup | SQLite `ticker_cik` | ticker | CIK string |
| Filing metadata | SQLite `filings` | (ticker, form_type) | accession, URL, filed_date |
| Document content | SQLite `documents` + filesystem | SHA-256(URL)[:16] | raw HTM text as `.txt` |

Default location: `~/.cache/verianalyst/`. Clear with `rm -rf ~/.cache/verianalyst/`.

> **Note:** `filings` table uses `PRIMARY KEY (ticker, form_type)` — only the latest filing per ticker is stored in metadata. `fetch_10k_history` bypasses this by looking up older filings directly in the `documents` table by URL.

---

## Web UI (`app/main.py`)

Streamlit app with two tabs:

- **📄 Analysis** — runs the full pipeline for a ticker, renders the report, and allows follow-up Q&A via LLM grounded in the report text
- **📈 Trends** — fetches up to 5 years of filings, extracts metrics from each, and displays a summary table + line charts (revenue, net income, EPS, assets, cash)

Async pipeline runs in a dedicated thread with its own event loop to avoid conflicts with Streamlit's runtime.

---

## Project Layout

```
VeriAnalyst/
├── app/
│   └── main.py                  # Streamlit UI
├── src/sec_analyzer/
│   ├── agents/
│   │   ├── retriever.py         # EDGAR fetcher + cache + rate limiter
│   │   ├── extractor.py         # HTML parser + rule-based metrics + LLM summary
│   │   ├── critic.py            # Rule-based confidence + LLM quality summary
│   │   └── writer.py            # LLM report writer
│   ├── cache/
│   │   └── sqlite_cache.py      # Two-layer cache implementation
│   ├── gateway/
│   │   ├── base.py              # LLMGateway abstract interface
│   │   ├── anthropic_backend.py
│   │   └── ollama_backend.py
│   ├── observability/
│   │   └── langfuse_setup.py
│   ├── orchestration/
│   │   └── graph.py             # LangGraph StateGraph pipeline
│   └── trends.py                # Multi-year metric extraction
├── tests/
│   ├── test_retriever.py
│   ├── test_extractor.py
│   ├── test_critic.py
│   ├── test_writer.py
│   └── test_gateway.py
├── examples/
│   ├── analyze_ticker.py        # CLI smoke test
│   └── compare_thinking.py      # Thinking on vs off comparison
├── evals/
├── docs/
│   └── architecture.md          # this file
├── docker-compose.yml           # Langfuse v3 self-hosted stack
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
| HTML parsing | BeautifulSoup4 |
| HTTP | httpx (async) |
| Cache | aiosqlite + filesystem |
| UI | Streamlit |
| Tests | pytest-asyncio |
