# VeriAnalyst

Multi-agent SEC 10-K analysis pipeline. Four agents — **Retriever → Extractor → Critic → Writer** — orchestrated with LangGraph, fully traced in Langfuse, with a Streamlit chat interface for interactive analysis and multi-year trend charts.

## What it does

| Agent | Role |
|---|---|
| **Retriever** | Fetches 10-K filings from SEC EDGAR with a two-layer cache (SQLite metadata + filesystem documents) and 10 req/sec rate limiting |
| **Extractor** | Parses HTML filings with BeautifulSoup, extracts structured financial metrics (revenue, net income, EPS, assets…) via rule-based table parsing, and produces a qualitative summary via LLM |
| **Critic** | Runs rule-based sanity checks (balance sheet equation, EPS ordering, gross margin bounds) and scores data confidence 0–1 |
| **Writer** | Synthesises metrics, critique, and qualitative sections into a narrative investment report |

Every agent span is traced in Langfuse. Confidence scores are auto-posted as Langfuse scores after each run.

## Features

- **Chat interface** — enter a ticker, get a full report, then ask follow-up questions grounded in the filing
- **Multi-year trends** — fetch up to 5 years of 10-K filings and chart revenue, net income, EPS, and assets over time
- **Langfuse observability** — per-agent latency, token usage, confidence scores, and trace waterfall at `http://localhost:3000`
- **Two-layer cache** — SQLite for EDGAR metadata, filesystem for documents (content-addressed by URL hash); no re-downloads
- **Ollama support** — runs fully offline with `qwen3.5:latest`; swap to Anthropic by setting `LLM_PROVIDER=anthropic`

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

Open `http://localhost:8501`. Enter a ticker in the sidebar to run an analysis, then use the chat input for follow-up questions. Switch to the **Trends** tab for multi-year charts.

### 6. Run tests

```bash
pytest
```

## Project layout

```
src/sec_analyzer/
├── agents/
│   ├── retriever.py      # EDGAR fetcher with rate limiting + cache
│   ├── extractor.py      # HTML parser + rule-based metric extraction + LLM summary
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
└── trends.py             # Multi-year historical metric extraction

app/
└── main.py               # Streamlit chat + trends UI

examples/
├── analyze_ticker.py     # CLI end-to-end smoke test
└── compare_thinking.py   # Thinking on vs off comparison

tests/                    # pytest-asyncio unit tests (mocked LLM + HTTP)
```

## Tech stack

| Layer | Library |
|---|---|
| Orchestration | LangGraph |
| Observability | Langfuse v3 (self-hosted via Docker) |
| LLM | Anthropic Claude or Ollama (qwen3.5) |
| HTML parsing | BeautifulSoup4 |
| HTTP | httpx (async) |
| Cache | aiosqlite + filesystem |
| UI | Streamlit |
| Tests | pytest-asyncio |

## Caching strategy

Two layers:

1. **SQLite** (`~/.cache/verianalyst/cache.db`) — stores ticker→CIK mappings, filing metadata (accession number, filed date, URL), and URL→filepath mappings across three tables
2. **Filesystem** (`~/.cache/verianalyst/documents/`) — stores the raw filing text as `<sha256(url)[:16]>.txt`; content-addressed so the same document is never downloaded twice

To clear the cache: `rm -rf ~/.cache/verianalyst/`

## Langfuse scores

After each pipeline run, the critic's confidence score (0–1) is automatically posted to Langfuse as a `data-confidence` score on the trace, visible in the Scores tab and aggregatable across runs in the Dashboard.

## SEC EDGAR notes

EDGAR requires a descriptive `User-Agent` header (`Name email@domain.com`). Set `SEC_USER_AGENT` in `.env`. The retriever enforces the 10 req/sec fair-use rate limit automatically.
