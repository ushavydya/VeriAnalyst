# VeriAnalyst

Multi-agent SEC 10-K analysis pipeline. Four agents — **Retriever → Extractor → Critic → Writer** — orchestrated with LangGraph, fully traced in Langfuse, with a Streamlit interface for interactive analysis, multi-year trends, and side-by-side company comparison.

## What it does

| Agent | Role |
|---|---|
| **Retriever** | Fetches 10-K filings from SEC EDGAR with a two-layer cache (SQLite metadata + filesystem documents) and 10 req/sec rate limiting. Also fetches verified financial metrics from EDGAR XBRL. |
| **Extractor** | Uses XBRL facts as authoritative metrics (revenue, net income, EPS, assets…). LLM reads only qualitative sections (Business, MD&A, Risk Factors) with verified numbers injected into the prompt. |
| **Critic** | Runs rule-based sanity checks (balance sheet equation, EPS ordering, gross margin bounds) and scores data confidence 0–1. |
| **Writer** | Synthesises metrics, critique, and qualitative sections into a narrative investment report. |

Every agent span is traced in Langfuse. Confidence scores are auto-posted as Langfuse scores after each run.

## Features

- **Chat interface** — enter a ticker, get a full report, then ask follow-up questions grounded in the filing
- **Multi-year trends** — fetch up to 5 years of 10-K filings and chart revenue, net income, EPS, and assets over time
- **Company comparison** — analyse 2–4 companies in parallel and generate a side-by-side comparative report with bar charts and a verdict
- **XBRL-sourced metrics** — numbers come directly from SEC EDGAR XBRL (same source as the filing), eliminating HTML table parsing errors
- **Langfuse observability** — per-agent latency, token usage, confidence scores, and trace waterfall at `http://localhost:3000`
- **Two-layer cache** — SQLite for EDGAR metadata + XBRL facts, filesystem for documents (content-addressed by URL hash); no re-downloads
- **Eval suite** — extractor accuracy evals against XBRL ground truth, LLM-as-judge report quality evals; both post scores to Langfuse
- **Unit tests** — 72 tests covering XBRL parsing, cache methods, extractor logic, and the full pipeline (mocked LLM + HTTP)
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
- **📄 Analysis** — enter a ticker, get a report, ask follow-up questions
- **📈 Trends** — multi-year charts for any ticker
- **⚖️ Compare** — enter `UBER, LYFT` or `AAPL, MSFT, GOOGL` for a side-by-side comparison

### 6. Run tests

```bash
pytest
```

### 7. Run evals

```bash
python evals/eval_extractor.py          # accuracy vs XBRL ground truth
python evals/eval_report_quality.py UBER AAPL MSFT   # LLM-as-judge
```

## Project layout

```
src/sec_analyzer/
├── agents/
│   ├── retriever.py      # EDGAR fetcher, rate limiter, cache, XBRL integration
│   ├── extractor.py      # XBRL metrics + LLM qualitative summary
│   ├── critic.py         # Rule-based confidence scoring + LLM quality summary
│   └── writer.py         # LLM narrative report writer
├── cache/
│   └── sqlite_cache.py   # Two-layer cache: SQLite metadata + filesystem documents
├── gateway/
│   ├── base.py           # LLMGateway abstract interface
│   ├── anthropic_backend.py
│   └── ollama_backend.py
├── orchestration/
│   └── graph.py          # LangGraph StateGraph pipeline
├── comparison.py         # Parallel multi-company comparison agent
├── trends.py             # Multi-year historical metric extraction
└── xbrl.py              # EDGAR XBRL company facts fetcher

app/
└── main.py               # Streamlit UI (Analysis / Trends / Compare tabs)

evals/
├── golden_dataset.py     # XBRL-sourced ground truth for AAPL, MSFT, UBER
├── eval_extractor.py     # Accuracy eval: extracted metrics vs golden dataset
└── eval_report_quality.py# LLM-as-judge: factual grounding, completeness, clarity

tests/                    # 72 pytest-asyncio unit tests (mocked LLM + HTTP)
```

## Tech stack

| Layer | Library |
|---|---|
| Orchestration | LangGraph |
| Observability | Langfuse v3 (self-hosted via Docker) |
| LLM | Anthropic Claude or Ollama |
| Financial data | SEC EDGAR XBRL + HTML filings |
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
Combined    →  report + critique + confidence score
```

The LLM never extracts numbers from HTML — it only reads narrative text and has verified XBRL numbers injected into its prompt, so it cannot cite a figure that differs from the SEC filing.

## Caching strategy

Three SQLite tables + filesystem:

| Layer | Key | Contents |
|---|---|---|
| `ticker_cik` | ticker | CIK string |
| `filings` | (ticker, form_type, accession) | filing metadata |
| `xbrl_facts` | (cik, "raw") | raw EDGAR company facts JSON |
| `documents` + filesystem | SHA-256(URL)[:16] | raw HTM text |

Default location: `~/.cache/verianalyst/`. Clear with `rm -rf ~/.cache/verianalyst/`.

## Langfuse scores

Two types of scores are automatically posted to Langfuse:

| Score | When | Value |
|---|---|---|
| `data-confidence` | After every pipeline run | Critic confidence 0–1 |
| `eval.extractor.{ticker}` | When running `eval_extractor.py` | Extraction accuracy 0–1 |
| `eval.report.{ticker}.{dim}` | When running `eval_report_quality.py` | Per-dimension quality 0–1 |

## SEC EDGAR notes

EDGAR requires a descriptive `User-Agent` header (`Name email@domain.com`). Set `SEC_USER_AGENT` in `.env`. The retriever enforces the 10 req/sec fair-use rate limit automatically.
