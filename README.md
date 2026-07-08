# VeriAnalyst

Multi-agent SEC 10-K analysis pipeline. Five agents — **Retriever → Intelligence → Extractor → Critic → Writer** — orchestrated with LangGraph, fully traced in Langfuse, with a Streamlit interface for interactive analysis, multi-year trends, side-by-side comparison, and live market intelligence.

## What it does

| Agent | Role |
|---|---|
| **Retriever** | Fetches 10-K filings from SEC EDGAR with a two-layer cache (SQLite metadata + filesystem documents) and 10 req/sec rate limiting. Fetches verified financial metrics from EDGAR XBRL. |
| **Intelligence** | Runs in parallel with extraction — fetches live market data (price, P/E, 52W range, beta) and recent news with sentiment from Finnhub, both with TTL-based caching. |
| **Extractor** | Uses XBRL facts as authoritative metrics (revenue, net income, EPS, assets…). LLM reads only qualitative sections (Business, MD&A, Risk Factors) with verified numbers injected into the prompt. |
| **Critic** | Rule-based sanity checks on 10-K extraction (balance sheet, EPS ordering, gross margin). Also validates market data (price, P/E, beta range) and flags divergences between fundamentals and current market sentiment. Scores extraction confidence 0–1. |
| **Writer** | Synthesises metrics, critique, news, and market data into a narrative investment report. Grounded strictly in provided data — no external knowledge injected. |

Every agent span is traced in Langfuse. Confidence scores are auto-posted as Langfuse scores after each run.

## Features

- **Chat interface** — enter a ticker, get a full report, then ask follow-up questions grounded in the filing
- **Investment Intelligence tab** — live market metrics (price, P/E, 52W range, beta), news sentiment gauge, top headlines, and an "Investment Intelligence" synthesis section in the report
- **Multi-year trends** — fetch up to 5 years of 10-K filings and chart revenue, net income, EPS, and assets over time
- **Company comparison** — analyse 2–4 companies in parallel and generate a side-by-side comparative report with bar charts and a verdict
- **XBRL-sourced metrics** — numbers come directly from SEC EDGAR XBRL (same source as the filing), eliminating HTML table parsing errors
- **Computed P/E** — when the market data provider returns no P/E (e.g. low-volume tickers), the pipeline derives it from live price ÷ XBRL `eps_diluted`; loss-making companies show "N/M (loss)"
- **Langfuse observability** — per-agent latency, token usage, confidence scores, and trace waterfall at `http://localhost:3000`
- **TTL-aware cache** — SQLite for EDGAR metadata + XBRL facts + market data + news (market data: 15 min–24 h TTLs); filesystem for documents
- **Eval suite** — extractor accuracy, trends consistency, intelligence layer, critic logic, and LLM-as-judge report quality evals; all post scores to Langfuse
- **158 unit tests** — cache, providers, news agent, market agent, critic checks, and full pipeline (mocked LLM + HTTP)
- **Ollama support** — runs fully offline with a local model; swap to Anthropic by setting `LLM_PROVIDER=anthropic`

## Quickstart

### 1. Start Langfuse (self-hosted)

```bash
cd /path/to/VeriAnalyst
docker compose up -d
```

Wait ~30 seconds for migrations, then open `http://localhost:3000` and log in:
- Email: `admin@verianalyst.local`
- Password: `changeme123`

### 2. Configure environment

```bash
cp .env.example .env
```

Edit `.env`:

```env
# Required
SEC_USER_AGENT=YourName your@email.com   # EDGAR fair-use policy

# LLM — choose one
LLM_PROVIDER=ollama                      # local (default)
OLLAMA_MODEL=qwen3.5:latest

# LLM_PROVIDER=anthropic                 # cloud
# ANTHROPIC_API_KEY=sk-ant-...

# Market data + news (free tier at finnhub.io)
FINNHUB_API_KEY=your_key_here

# Langfuse (pre-seeded by docker-compose — no changes needed)
LANGFUSE_HOST=http://localhost:3000
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
```

### 3. Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

### 4. Run the CLI

```bash
python examples/analyze_ticker.py AAPL
```

### 5. Run the web UI

```bash
streamlit run app/main.py
```

Open `http://localhost:8501`:
- **📄 Analysis** — enter a ticker, get a report with step-by-step reveals, ask follow-up questions
- **🧠 Intelligence** — live price/valuation metrics, news sentiment, and synthesis from the report
- **📈 Trends** — multi-year charts for any ticker
- **⚖️ Compare** — enter `UBER, LYFT` or `AAPL, MSFT, GOOGL` for a side-by-side comparison

### 6. Run tests

```bash
pytest
```

### 7. Run evals

```bash
python evals/eval_extractor.py                      # accuracy vs XBRL ground truth
python evals/eval_trends.py                         # multi-year filing consistency
python evals/eval_intelligence.py AAPL              # market data + news + report synthesis
python evals/eval_critic.py                         # rule/market/divergence checks (no LLM)
python evals/eval_report_quality.py UBER AAPL MSFT  # LLM-as-judge
```

## Project layout

```
src/sec_analyzer/
├── agents/
│   ├── retriever.py      # EDGAR fetcher, rate limiter, cache, XBRL integration
│   ├── news_agent.py     # News fetch + caching (Finnhub); NewsSummary dataclass
│   ├── market_agent.py   # Parallel quote/ratios/history fetch; computed P/E
│   ├── extractor.py      # XBRL metrics + LLM qualitative summary
│   ├── critic.py         # Rule checks + market checks + divergence signals + LLM summary
│   └── writer.py         # LLM narrative report writer
├── cache/
│   └── sqlite_cache.py   # SQLite cache: filings, CIK, XBRL, documents, news, market data
├── gateway/
│   ├── base.py           # LLMGateway abstract interface
│   ├── anthropic_backend.py
│   └── ollama_backend.py
├── providers/
│   ├── base.py           # NewsProvider + MarketDataProvider ABCs + dataclasses
│   ├── finnhub.py        # Finnhub implementation (news + market data)
│   └── __init__.py       # Factory: get_news_provider(), get_market_data_provider()
├── orchestration/
│   └── graph.py          # LangGraph StateGraph: retriever→intelligence→extractor→critic→writer
├── comparison.py         # Parallel multi-company comparison agent
├── trends.py             # Multi-year historical metric extraction
└── xbrl.py              # EDGAR XBRL company facts fetcher

app/
└── main.py               # Streamlit UI (Analysis / Intelligence / Trends / Compare tabs)

evals/
├── golden_dataset.py          # XBRL-sourced ground truth for AAPL, MSFT, UBER, NVDA, AMD, JPM
├── eval_extractor.py          # Accuracy eval: extracted metrics vs golden dataset
├── eval_trends.py             # Multi-year filing consistency (distinct years, positive revenue)
├── eval_intelligence.py       # Market data + news + critic + report synthesis
├── eval_critic.py             # Rule/market/divergence checks with synthetic inputs (no LLM)
└── eval_report_quality.py     # LLM-as-judge: factual grounding, completeness, clarity

tests/                    # 158 pytest-asyncio unit tests (mocked LLM + HTTP)
```

## Tech stack

| Layer | Library |
|---|---|
| Orchestration | LangGraph |
| Observability | Langfuse v3 (self-hosted via Docker) |
| LLM | Anthropic Claude or Ollama |
| Financial data | SEC EDGAR XBRL + HTML filings |
| Market data & news | Finnhub (free tier) |
| HTML parsing | BeautifulSoup4 |
| HTTP | httpx (async) |
| Cache | aiosqlite + filesystem |
| UI | Streamlit |
| Tests | pytest-asyncio |

## How metrics are sourced

Numbers (revenue, EPS, assets, etc.) come from **SEC EDGAR XBRL** — the machine-readable financial facts that companies file alongside their 10-K. This is the same authoritative source as the filing itself, but structured JSON rather than HTML tables.

```
EDGAR XBRL  →  exact metrics (revenue, EPS, net income, assets…)
EDGAR HTML  →  LLM qualitative summary (business model, risks, MD&A)
Finnhub     →  live price, P/E, 52W range, beta, recent news + sentiment
Combined    →  report + critique + confidence score + intelligence section
```

The LLM never extracts numbers from HTML — it only reads narrative text and has verified XBRL numbers injected into its prompt, so it cannot cite a figure that differs from the SEC filing.

## Intelligence layer

The intelligence node runs in parallel with EDGAR retrieval and provides two data streams:

**Market data** (via `market_agent.py`):
- Live quote (price, intraday change %, volume, market cap)
- Valuation ratios (P/E, P/B, beta, 52-week high/low)
- Price history (daily bars for 1 year)
- Computed P/E: when the provider returns no P/E, falls back to `price / eps_diluted` from XBRL; loss-making companies (eps ≤ 0) show "N/M (loss)"

**News** (via `news_agent.py`):
- Up to 20 recent headlines with source and URL
- Aggregate sentiment score from Finnhub (–1.0 bearish → +1.0 bullish)
- 24-hour TTL cache

The critic then checks for divergences: bearish news on a profitable company, bullish news on a loss-maker, stock near 52W low despite positive net income. These appear as informational signals in the report but do not penalise the confidence score.

## Caching strategy

Six SQLite tables + filesystem, with TTL enforcement for market and news data:

| Layer | Key | TTL | Contents |
|---|---|---|---|
| `ticker_cik` | ticker | none | CIK string |
| `filings` | (ticker, form_type, accession) | none | filing metadata |
| `xbrl_facts` | (cik, "raw") | none | raw EDGAR company facts JSON |
| `documents` + filesystem | SHA-256(URL)[:16] | none | raw HTM text |
| `news_cache` | (ticker, date) | 24 h | articles + sentiment score |
| `market_data_cache` | (ticker, data_type, period) | 15 min (quote) / 1 h (ratios) / 24 h (history) | quote, ratios, or price history |

Default location: `~/.cache/verianalyst/`. Clear with `rm -rf ~/.cache/verianalyst/`.

## Langfuse scores

| Score | When | Value |
|---|---|---|
| `data-confidence` | After every pipeline run | Critic confidence 0–1 |
| `eval.extractor.{ticker}` | `eval_extractor.py` | Extraction accuracy 0–1 |
| `eval.report.{ticker}.{dim}` | `eval_report_quality.py` | Per-dimension quality 0–1 |
| `eval.critic.rule_checks` | `eval_critic.py` | Fraction of rule/market/divergence checks passed |
| `eval.intelligence.*` | `eval_intelligence.py` | Market data, news, and synthesis checks |

## SEC EDGAR notes

EDGAR requires a descriptive `User-Agent` header (`Name email@domain.com`). Set `SEC_USER_AGENT` in `.env`. The retriever enforces the 10 req/sec fair-use rate limit automatically.

## Finnhub free tier notes

The Finnhub free tier covers news headlines, company sentiment buzz, live quotes, and basic ratios. Some data points are unavailable on the free tier:
- **Sentiment score** requires sufficient article volume — low-coverage tickers may return `null`
- **Price history** (`/stock/candle`) is not available on the free tier — the Intelligence tab shows market snapshot only
- **Volume** may be 0 for low-volume tickers — the UI shows "—" rather than 0
