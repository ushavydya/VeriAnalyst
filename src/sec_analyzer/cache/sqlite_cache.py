"""Two-layer cache: SQLite for filing metadata + filesystem for document text."""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import aiosqlite


@dataclass
class CachedFiling:
    ticker: str
    cik: str
    accession_number: str
    form_type: str
    filed_date: str
    document_url: str
    file_path: str  # absolute path on disk


_SCHEMA = """
CREATE TABLE IF NOT EXISTS ticker_cik (
    ticker      TEXT PRIMARY KEY,
    cik         TEXT NOT NULL,
    cached_at   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS filings (
    ticker             TEXT NOT NULL,
    cik                TEXT NOT NULL,
    accession_number   TEXT NOT NULL,
    form_type          TEXT NOT NULL,
    filed_date         TEXT NOT NULL,
    document_url       TEXT NOT NULL,
    cached_at          TEXT NOT NULL,
    PRIMARY KEY (ticker, form_type, accession_number)
);

CREATE INDEX IF NOT EXISTS filings_ticker_form_date
    ON filings (ticker, form_type, filed_date DESC);

CREATE TABLE IF NOT EXISTS documents (
    url         TEXT PRIMARY KEY,
    file_path   TEXT NOT NULL,
    cached_at   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS xbrl_facts (
    cik              TEXT NOT NULL,
    accession_number TEXT NOT NULL,
    facts_json       TEXT NOT NULL,
    cached_at        TEXT NOT NULL,
    PRIMARY KEY (cik, accession_number)
);

-- One row per ticker per calendar day.  TTL checked in application layer (24h).
-- INSERT OR REPLACE so a manual refresh always gets the freshest data.
CREATE TABLE IF NOT EXISTS news_cache (
    ticker          TEXT NOT NULL,
    cached_date     TEXT NOT NULL,  -- YYYY-MM-DD of the fetch
    articles_json   TEXT NOT NULL,  -- JSON array of NewsArticle dicts
    sentiment_score REAL,           -- -1.0 to +1.0; NULL if unavailable
    cached_at       TEXT NOT NULL,
    PRIMARY KEY (ticker, cached_date)
);

-- One row per (ticker, data_type, period).  TTL varies by data_type:
--   quote   / current → 15 minutes
--   ratios  / current → 1 hour
--   history / 1y|6m   → 24 hours
CREATE TABLE IF NOT EXISTS market_data_cache (
    ticker      TEXT NOT NULL,
    data_type   TEXT NOT NULL,  -- 'quote' | 'ratios' | 'history'
    period      TEXT NOT NULL,  -- 'current' for quote/ratios; '1y','6m' etc for history
    data_json   TEXT NOT NULL,
    cached_at   TEXT NOT NULL,
    PRIMARY KEY (ticker, data_type, period)
);
"""


class SQLiteCache:
    """Async SQLite cache for EDGAR metadata and filesystem paths."""

    def __init__(
        self,
        db_path: str | None = None,
        docs_dir: str | None = None,
    ) -> None:
        default_base = Path.home() / ".cache" / "verianalyst"
        self._db_path = Path(db_path or os.environ.get("CACHE_DB_PATH", "") or default_base / "cache.db")
        self._docs_dir = Path(docs_dir or os.environ.get("CACHE_DOCS_DIR", "") or default_base / "documents")
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._docs_dir.mkdir(parents=True, exist_ok=True)
        self._conn: aiosqlite.Connection | None = None

    async def open(self) -> None:
        self._conn = await aiosqlite.connect(self._db_path)
        self._conn.row_factory = aiosqlite.Row
        await self._conn.executescript(_SCHEMA)
        await self._conn.commit()

    async def close(self) -> None:
        if self._conn:
            await self._conn.close()
            self._conn = None

    async def __aenter__(self) -> "SQLiteCache":
        await self.open()
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()

    # ── CIK lookup ────────────────────────────────────────────────────────────

    async def get_cik(self, ticker: str) -> str | None:
        assert self._conn, "call open() first"
        async with self._conn.execute(
            "SELECT cik FROM ticker_cik WHERE ticker = ?", (ticker.upper(),)
        ) as cur:
            row = await cur.fetchone()
            return row["cik"] if row else None

    async def store_cik(self, ticker: str, cik: str) -> None:
        assert self._conn
        await self._conn.execute(
            "INSERT OR REPLACE INTO ticker_cik (ticker, cik, cached_at) VALUES (?, ?, ?)",
            (ticker.upper(), cik, _now()),
        )
        await self._conn.commit()

    # ── Filing metadata ───────────────────────────────────────────────────────

    async def get_filing(self, ticker: str, form_type: str = "10-K") -> CachedFiling | None:
        """Return the most recently filed cached filing for (ticker, form_type).

        Returns None if no filing found, or if the document file is missing
        from the documents table.
        """
        assert self._conn
        async with self._conn.execute(
            """SELECT f.ticker, f.cik, f.accession_number, f.form_type,
                      f.filed_date, f.document_url, d.file_path
               FROM filings f
               JOIN documents d ON d.url = f.document_url
               WHERE f.ticker = ? AND f.form_type = ?
               ORDER BY f.filed_date DESC
               LIMIT 1""",
            (ticker.upper(), form_type),
        ) as cur:
            row = await cur.fetchone()
            return CachedFiling(**dict(row)) if row else None

    async def store_filing(
        self,
        ticker: str,
        cik: str,
        accession_number: str,
        form_type: str,
        filed_date: str,
        document_url: str,
        file_path: str,
    ) -> None:
        assert self._conn
        await self._conn.execute(
            """INSERT OR IGNORE INTO filings
               (ticker, cik, accession_number, form_type, filed_date,
                document_url, cached_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (ticker.upper(), cik, accession_number, form_type, filed_date, document_url, _now()),
        )
        await self._conn.execute(
            "INSERT OR REPLACE INTO documents (url, file_path, cached_at) VALUES (?, ?, ?)",
            (document_url, file_path, _now()),
        )
        await self._conn.commit()

    # ── Document file path ────────────────────────────────────────────────────

    async def get_document_path(self, url: str) -> str | None:
        assert self._conn
        async with self._conn.execute(
            "SELECT file_path FROM documents WHERE url = ?", (url,)
        ) as cur:
            row = await cur.fetchone()
            return row["file_path"] if row else None

    def document_path_for(self, url: str) -> Path:
        """Return a deterministic filesystem path for caching a URL."""
        digest = hashlib.sha256(url.encode()).hexdigest()[:16]
        return self._docs_dir / f"{digest}.txt"

    # ── XBRL facts ────────────────────────────────────────────────────────────

    async def get_xbrl_facts(self, cik: str, accession_number: str) -> dict | None:
        assert self._conn
        async with self._conn.execute(
            "SELECT facts_json FROM xbrl_facts WHERE cik = ? AND accession_number = ?",
            (cik, accession_number),
        ) as cur:
            row = await cur.fetchone()
            return json.loads(row["facts_json"]) if row else None

    async def store_xbrl_facts(self, cik: str, accession_number: str, facts: dict) -> None:
        assert self._conn
        await self._conn.execute(
            """INSERT OR REPLACE INTO xbrl_facts
               (cik, accession_number, facts_json, cached_at)
               VALUES (?, ?, ?, ?)""",
            (cik, accession_number, json.dumps(facts), _now()),
        )
        await self._conn.commit()

    # ── News cache ────────────────────────────────────────────────────────────

    async def get_news(self, ticker: str, date: str) -> dict | None:
        """Return cached news for (ticker, date) if fresher than 24h, else None."""
        assert self._conn
        async with self._conn.execute(
            "SELECT articles_json, sentiment_score, cached_at FROM news_cache"
            " WHERE ticker = ? AND cached_date = ?",
            (ticker.upper(), date),
        ) as cur:
            row = await cur.fetchone()
        if row is None:
            return None
        if _is_stale(row["cached_at"], hours=24):
            return None
        return {
            "articles": json.loads(row["articles_json"]),
            "sentiment_score": row["sentiment_score"],
        }

    async def store_news(
        self,
        ticker: str,
        date: str,
        articles: list[dict],
        sentiment_score: float | None,
    ) -> None:
        assert self._conn
        await self._conn.execute(
            """INSERT OR REPLACE INTO news_cache
               (ticker, cached_date, articles_json, sentiment_score, cached_at)
               VALUES (?, ?, ?, ?, ?)""",
            (ticker.upper(), date, json.dumps(articles), sentiment_score, _now()),
        )
        await self._conn.commit()

    # ── Market data cache ─────────────────────────────────────────────────────

    _MARKET_TTL_HOURS: dict[str, float] = {
        "quote":   0.25,   # 15 minutes
        "ratios":  1.0,    # 1 hour
        "history": 24.0,   # 24 hours
    }

    async def get_market_data(
        self, ticker: str, data_type: str, period: str
    ) -> dict | None:
        """Return cached market data if within TTL for data_type, else None."""
        assert self._conn
        async with self._conn.execute(
            "SELECT data_json, cached_at FROM market_data_cache"
            " WHERE ticker = ? AND data_type = ? AND period = ?",
            (ticker.upper(), data_type, period),
        ) as cur:
            row = await cur.fetchone()
        if row is None:
            return None
        ttl_hours = self._MARKET_TTL_HOURS.get(data_type, 1.0)
        if _is_stale(row["cached_at"], hours=ttl_hours):
            return None
        return json.loads(row["data_json"])

    async def store_market_data(
        self, ticker: str, data_type: str, period: str, data: dict
    ) -> None:
        assert self._conn
        await self._conn.execute(
            """INSERT OR REPLACE INTO market_data_cache
               (ticker, data_type, period, data_json, cached_at)
               VALUES (?, ?, ?, ?, ?)""",
            (ticker.upper(), data_type, period, json.dumps(data), _now()),
        )
        await self._conn.commit()


def _now() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _is_stale(cached_at: str, *, hours: float) -> bool:
    """Return True if cached_at is older than *hours* from now."""
    age = datetime.now(tz=timezone.utc) - datetime.fromisoformat(cached_at)
    return age.total_seconds() > hours * 3600
