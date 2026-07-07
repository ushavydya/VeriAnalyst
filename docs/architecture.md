# VeriAnalyst — Architecture

## Pipeline Overview

```
SEC EDGAR
    │
    ▼
┌───────────┐    ┌───────────┐    ┌────────┐    ┌────────┐
│ Retriever │ ─► │ Extractor │ ─► │ Critic │ ─► │ Writer │
└───────────┘    └───────────┘    └────────┘    └────────┘
      │                                               │
   SQLite +                                      Markdown
  File Cache                                      Report
```

Orchestrated by **LangGraph** (`StateGraph`). Every node is a Python async function; `PipelineState` (a `TypedDict`) flows between them.

---

## Components

### Retriever (`src/sec_analyzer/agents/retriever.py`)

1. Looks up CIK from SEC `company_tickers.json` (cached in SQLite).
2. Fetches filing list from `submissions/CIK{cik}.json`.
3. Downloads the primary 10-K document from EDGAR Archives.
4. Two-layer cache: SQLite metadata + filesystem document store (SHA-256 keyed).
5. Respects 10 req/sec rate limit via async token-bucket rate limiter.

### Extractor (`src/sec_analyzer/agents/extractor.py`) — *stub*

Parses HTML/XBRL 10-K into structured `ExtractedData` (sections + metrics). Full implementation uses Claude to produce JSON.

### Critic (`src/sec_analyzer/agents/critic.py`) — *stub*

Validates extracted data for consistency; returns a confidence score and issues list. Full implementation calls Claude with a structured critique prompt.

### Writer (`src/sec_analyzer/agents/writer.py`) — *stub*

Synthesises extraction + critique into a narrative investment report (Markdown). Full implementation streams Claude's response.

---

## Observability (Langfuse)

Each pipeline run has a shared **trace** (UUID in `PipelineState.trace_id`). Every LangGraph node attaches a **span** to that trace recording:

- `input` — ticker / document length / metrics count
- `output` — result summary / latency_ms / token cost (when Claude calls are live)
- `latency_ms` — wall-clock milliseconds

Langfuse runs locally; see **Local Setup** below.

---

## Caching

| Layer | Store | Key | Contents |
|---|---|---|---|
| CIK lookup | SQLite `ticker_cik` | ticker | CIK string |
| Filing metadata | SQLite `filings` | (ticker, form_type) | accession, URL, date |
| Document file | SQLite `documents` + filesystem | SHA-256(URL) | raw document text |

Default paths: `~/.cache/verianalyst/cache.db` and `~/.cache/verianalyst/documents/`.

---

## Local Setup

### 1. Langfuse (self-hosted via Docker Compose)

```yaml
# docker-compose.yml  (place at repo root or run standalone)
version: "3.9"
services:
  langfuse:
    image: langfuse/langfuse:latest
    ports:
      - "3000:3000"
    environment:
      - DATABASE_URL=postgresql://postgres:postgres@db:5432/langfuse
      - NEXTAUTH_SECRET=change-me-in-production
      - NEXTAUTH_URL=http://localhost:3000
      - SALT=change-me-in-production
    depends_on:
      - db

  db:
    image: postgres:15
    environment:
      POSTGRES_PASSWORD: postgres
      POSTGRES_DB: langfuse
    volumes:
      - langfuse_pg:/var/lib/postgresql/data

volumes:
  langfuse_pg:
```

```bash
docker compose up -d
# Open http://localhost:3000, create a project, copy the API keys
```

### 2. Environment variables

```bash
cp .env.example .env
# Fill in LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, ANTHROPIC_API_KEY
# Set SEC_USER_AGENT to "YourCompany your@email.com"
```

### 3. Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

### 4. Run

```bash
python examples/analyze_ticker.py AAPL
```

### 5. Tests

```bash
pytest
```

---

## Project Layout

```
VeriAnalyst/
├── src/sec_analyzer/
│   ├── agents/
│   │   ├── retriever.py   # full implementation
│   │   ├── extractor.py   # stub
│   │   ├── critic.py      # stub
│   │   └── writer.py      # stub
│   ├── cache/
│   │   └── sqlite_cache.py
│   ├── observability/
│   │   └── langfuse_setup.py
│   └── orchestration/
│       └── graph.py        # LangGraph pipeline
├── tests/
│   └── test_retriever.py
├── examples/
│   └── analyze_ticker.py
├── evals/
├── docs/
│   └── architecture.md     # this file
├── pyproject.toml
└── .env.example
```
