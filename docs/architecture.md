# VeriAnalyst — Architecture

## Pipeline Overview

```
SEC EDGAR XBRL ──► exact metrics (revenue, EPS, assets…)         ┐
SEC EDGAR HTML ──► LLM qualitative summary (business, MD&A, risks)┘
                                                                    │
Finnhub ─────────► news + sentiment + live market data             │
                          │                                         │
                   ┌──────▼──────┐                          ┌──────▼──────┐
                   │ Intelligence │ (parallel)               │  Retriever  │
                   └──────┬──────┘                          └──────┬──────┘
                          │  news_summary_json                     │  filing_content
                          │  market_summary_json                   │  xbrl_metrics
                          └────────────────┬───────────────────────┘
                                           │
                                    ┌──────▼──────┐
                                    │  Extractor  │
                                    └──────┬──────┘
                                           │  ExtractedData
                                    ┌──────▼──────┐
                                    │    Critic   │
                                    └──────┬──────┘
                                           │  Critique + confidence
                                    ┌──────▼──────┐
                                    │    Writer   │
                                    └──────┬──────┘
                                           │  Markdown report
                                     Langfuse trace
```

Orchestrated by **LangGraph** (`StateGraph`). Every node is a Python async function; `PipelineState` (a `TypedDict`) flows between them. The intelligence node runs concurrently with the retriever — both start immediately and the extractor waits for both to complete.

---

## Components

### Retriever (`agents/retriever.py`)

1. Resolves ticker → CIK via SEC `company_tickers.json` (cached in SQLite).
2. Fetches filing list from `submissions/CIK{cik}.json`.
3. Downloads the primary 10-K HTM document from EDGAR Archives.
4. Fetches XBRL company facts from `data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json` and attaches verified metrics to `FilingResult.xbrl_metrics`.
5. Two-layer cache: SQLite metadata + filesystem document store (SHA-256 URL-keyed). XBRL JSON cached in `xbrl_facts` table.
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

### Intelligence Node (`orchestration/graph.py` + `agents/news_agent.py` + `agents/market_agent.py`)

Runs after the retriever in parallel with EDGAR document processing. Uses `asyncio.gather` internally to fetch news and market data concurrently. Degrades gracefully if `FINNHUB_API_KEY` is not set.

**News Agent (`news_agent.py`)**

- Calls `NewsProvider.fetch_news(ticker)` for up to 20 recent headlines.
- Sentiment score (`-1.0` bearish → `+1.0` bullish) comes from Finnhub's buzz/sentiment endpoint.
- 24-hour TTL cache in `news_cache` SQLite table.
- Returns `NewsSummary(ticker, date, articles, sentiment_score, cache_hit)`.

**Market Agent (`market_agent.py`)**

- Fetches quote, ratios, and price history in parallel via `asyncio.gather`.
- Per-stream TTL cache: 15 min (quote), 1 h (ratios), 24 h (history).
- **Computed P/E**: when the provider returns `pe_ratio=None`, falls back to `price / eps_diluted` using the live quote and XBRL metric. Only computed when `eps_diluted > 0` (skips loss-making companies). `MarketSummary.pe_computed=True` flags this path so the UI can show a tooltip.
- Returns `MarketSummary(ticker, quote, ratios, history, cache_hits, pe_computed)`.

**Provider abstraction (`providers/`)**

```
providers/
├── base.py       # NewsProvider + MarketDataProvider ABCs; NewsArticle, Quote, Ratios, MarketHistory dataclasses
├── finnhub.py    # FinnhubProvider implementing both ABCs
└── __init__.py   # get_news_provider() / get_market_data_provider() factory functions
```

Adding a new provider means implementing the two ABCs and updating the factory — no changes to agents.

### Extractor (`agents/extractor.py`)

**Two-source strategy:**

| Source | Used for | Why |
|---|---|---|
| XBRL | All numeric metrics | Authoritative, structured, amendment-aware |
| HTML | Qualitative summary (LLM) | XBRL has no narrative text |

When XBRL metrics are available (always for major filers), rule-based HTML table parsing is skipped entirely. The LLM is given verified XBRL numbers injected into its prompt and instructed to cite only those figures — it cannot hallucinate a number that differs from the SEC filing.

Returns `ExtractedData(ticker, filed_date, sections, metrics)`.

### Critic (`agents/critic.py`)

Three layers of validation:

1. **10-K rule checks** (`_rule_checks`) — affect the confidence score:
   - Missing required metrics (`revenue`, `net_income`)
   - Non-positive revenue
   - Gross profit > revenue
   - `|net_income| > 75%` of revenue
   - Diluted EPS > basic EPS
   - Balance sheet imbalance > 5%
   - Base 0.9 if both required metrics present; 0.5 if partial; 0.0 if neither. −0.15 per violation.

2. **Market data sanity checks** (`_market_checks`) — informational only, do not affect confidence:
   - Non-positive price
   - P/E > 500 or negative
   - Beta outside ±5
   - 52W low > 52W high
   - Current price > 52W high by > 5%

3. **Divergence signals** (`_divergence_checks`) — informational only:
   - Bearish news sentiment (≤ −0.3) on a profitable company
   - Bullish news sentiment (≥ +0.3) on a loss-making company
   - Stock price in bottom 15% of 52W range despite positive net income

Returns `Critique(ticker, confidence, issues, summary)`. Issues list contains all three layers; confidence reflects only extraction quality.

### Writer (`agents/writer.py`)

Synthesises `ExtractedData` + `Critique` + optional `NewsSummary` + `MarketSummary` into a Markdown investment report via LLM.

Sections: Executive Summary, Financial Highlights, Business Overview, Key Risks, Investment Intelligence (when intelligence data is available), Data Confidence Note.

The system prompt explicitly constrains the LLM to use only provided data — no invented product names, competitor comparisons, or figures not present in the input. The fiscal year label (`FY2025`) is derived from `fiscal_year_end` date and passed explicitly to prevent the LLM from misinterpreting it.

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
| `news_agent` | ticker, date | cache_hit, article_count, sentiment_score |
| `market_agent` | ticker, history_period | cache_hits, price, pe_ratio, pe_computed, history_bars |
| `extractor` | ticker, doc_length, xbrl=bool | sections, metrics_found, metrics_source, tokens |
| `critic` | ticker, metrics_count, has_market, has_news | confidence, rule_issues, market_warnings, divergence_signals, tokens |
| `writer` | ticker, confidence, has_news, has_market | report_length, tokens |
| `comparison` | tickers | report_length, tokens |

After the writer completes, `langfuse.create_score(trace_id, name="data-confidence", value=confidence)` attaches the critic score to the trace.

Langfuse runs locally via Docker Compose (6 services: langfuse-web, langfuse-worker, postgres, clickhouse@24.3, redis, minio).

---

## Caching

Six stores, with TTL enforcement for real-time data:

| Table / Store | Key | TTL | Contents |
|---|---|---|---|
| SQLite `ticker_cik` | ticker | none | CIK string |
| SQLite `filings` | (ticker, form_type, accession_number) | none | filing metadata |
| SQLite `xbrl_facts` | (cik, "raw") | none | raw EDGAR company facts JSON |
| SQLite `documents` + filesystem | SHA-256(URL)[:16] | none | raw HTM text as `.txt` |
| SQLite `news_cache` | (ticker, date) | 24 h | articles list + sentiment score |
| SQLite `market_data_cache` | (ticker, data_type, period) | 15 min / 1 h / 24 h | quote / ratios / price history |

`filings` PK includes `accession_number` so all historical filings coexist; `get_filing()` returns the most recent by `filed_date DESC`.

Default location: `~/.cache/verianalyst/`. Clear with `rm -rf ~/.cache/verianalyst/`.

---

## Evals

Five eval scripts, all posting scores to Langfuse:

### `eval_extractor.py` — Accuracy eval (integration test)

Runs the full retrieval + extraction pipeline for each ticker in the golden dataset and compares extracted metrics against XBRL-sourced ground truth. Tolerance: 2%. Golden dataset: AAPL, MSFT, UBER, NVDA, AMD, JPM.

### `eval_trends.py` — Multi-year consistency

Fetches 3 years of 10-K filings for UBER, AAPL, MSFT. Checks: fiscal year values are strictly decreasing (no year repeated), revenue values differ across years (accession filtering works), revenue is positive for each year.

### `eval_intelligence.py` — Intelligence layer integration

End-to-end check with a live Finnhub key: market data present and plausible, news articles returned, cache hit on second call, "Investment Intelligence" section present in the report, critic well-formed. 0 history bars treated as a warning (Finnhub free-tier limitation), not a hard failure.

### `eval_critic.py` — Rule/market/divergence logic (no LLM, fast)

20 deterministic checks against synthetic inputs covering all three critic layers. Verifies that each rule fires on the right input and that clean data produces no false positives.

### `eval_report_quality.py` — LLM-as-judge

Sends the generated report + extracted metrics to the LLM acting as a judge. Scores five dimensions (0–1): `factual_grounding`, `completeness`, `fiscal_year_accuracy`, `no_hallucination`, `clarity`. Posts each dimension and an aggregate score to Langfuse.

### Unit tests (`tests/`)

158 pytest-asyncio tests covering:
- `test_xbrl.py` — XBRL entry selection, concept fallbacks, scaling, fiscal year filtering
- `test_sqlite_cache.py` — cache round-trips, filing ordering, CIK lookup, news/market TTL
- `test_extractor.py` — HTML parsing, XBRL path, prompt injection, fallback
- `test_retriever.py` — cache hit/miss paths, rate limiter, URL construction
- `test_providers.py` — ABC enforcement, factory functions, mocked Finnhub HTTP
- `test_news_agent.py` — cache-first path, provider error handling, sentiment label
- `test_market_agent.py` — parallel fetch, computed P/E (pe_computed flag), partial failure
- `test_critic.py` — rule checks, market checks, divergence signals, integration
- `test_writer.py`, `test_gateway.py`

All tests run in under 2 seconds with no network calls.

---

## Web UI (`app/main.py`)

Streamlit app with four tabs:

| Tab | What it does |
|---|---|
| **📄 Analysis** | Full pipeline for one ticker; report with step-by-step section reveals + follow-up Q&A grounded in the filing |
| **🧠 Intelligence** | Live market metrics table (price, P/E with computed tooltip, 52W range, beta), news sentiment gauge + top headlines, Investment Intelligence synthesis section from the report |
| **📈 Trends** | Up to 5 years of filings; summary table + line charts (revenue, net income, EPS, assets, cash) |
| **⚖️ Compare** | 2–4 tickers in parallel; side-by-side metrics table, bar charts, LLM comparative report with Verdict |

Async pipelines run in dedicated threads with fresh event loops to avoid Streamlit runtime conflicts.

---

## Project Layout

```
VeriAnalyst/
├── app/
│   └── main.py                    # Streamlit UI (4 tabs)
├── src/sec_analyzer/
│   ├── agents/
│   │   ├── retriever.py           # EDGAR fetcher + XBRL + cache + rate limiter
│   │   ├── news_agent.py          # News fetch + TTL cache; NewsSummary dataclass
│   │   ├── market_agent.py        # Parallel quote/ratios/history; computed P/E
│   │   ├── extractor.py           # XBRL metrics + LLM qualitative summary
│   │   ├── critic.py              # Rule + market + divergence checks + LLM summary
│   │   └── writer.py              # LLM report writer (grounded, no hallucination)
│   ├── cache/
│   │   └── sqlite_cache.py        # 6-table SQLite cache with TTL for market/news
│   ├── gateway/
│   │   ├── base.py
│   │   ├── anthropic_backend.py
│   │   └── ollama_backend.py
│   ├── providers/
│   │   ├── base.py                # NewsProvider + MarketDataProvider ABCs + dataclasses
│   │   ├── finnhub.py             # Finnhub implementation
│   │   └── __init__.py            # Provider factory functions
│   ├── orchestration/
│   │   └── graph.py               # LangGraph StateGraph pipeline
│   ├── comparison.py              # Parallel multi-company comparison
│   ├── trends.py                  # Multi-year metric extraction
│   └── xbrl.py                   # EDGAR XBRL company facts fetcher
├── evals/
│   ├── golden_dataset.py          # XBRL-sourced ground truth (6 tickers)
│   ├── eval_extractor.py          # Accuracy eval
│   ├── eval_trends.py             # Multi-year consistency
│   ├── eval_intelligence.py       # Intelligence layer integration
│   ├── eval_critic.py             # Deterministic rule/market/divergence checks
│   └── eval_report_quality.py     # LLM-as-judge eval
├── tests/                         # 158 unit tests
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
| Market data & news | Finnhub (free tier) |
| HTML parsing | BeautifulSoup4 |
| HTTP | httpx (async) |
| Cache | aiosqlite + filesystem |
| UI | Streamlit |
| Tests | pytest-asyncio |
