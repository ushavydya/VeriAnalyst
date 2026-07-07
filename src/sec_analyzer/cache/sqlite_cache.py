"""Two-layer cache: SQLite for filing metadata + filesystem for document text."""
from __future__ import annotations

import hashlib
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
    PRIMARY KEY (ticker, form_type)
);

CREATE TABLE IF NOT EXISTS documents (
    url         TEXT PRIMARY KEY,
    file_path   TEXT NOT NULL,
    cached_at   TEXT NOT NULL
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
        assert self._conn
        async with self._conn.execute(
            """SELECT f.ticker, f.cik, f.accession_number, f.form_type,
                      f.filed_date, f.document_url, d.file_path
               FROM filings f
               LEFT JOIN documents d ON d.url = f.document_url
               WHERE f.ticker = ? AND f.form_type = ?""",
            (ticker.upper(), form_type),
        ) as cur:
            row = await cur.fetchone()
            if not row:
                return None
            return CachedFiling(**dict(row))

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
            """INSERT OR REPLACE INTO filings
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

    # ── Document file path ─────────────────────────────────────────────────────

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


def _now() -> str:
    return datetime.now(tz=timezone.utc).isoformat()
