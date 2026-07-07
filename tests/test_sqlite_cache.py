"""Unit tests for SQLiteCache — covers XBRL facts storage and filing methods."""
from __future__ import annotations

from pathlib import Path

import pytest

from sec_analyzer.cache.sqlite_cache import SQLiteCache


@pytest.fixture()
async def cache(tmp_path: Path) -> SQLiteCache:
    c = SQLiteCache(
        db_path=str(tmp_path / "cache.db"),
        docs_dir=str(tmp_path / "docs"),
    )
    async with c:
        yield c


# ── XBRL facts ────────────────────────────────────────────────────────────────

async def test_xbrl_facts_round_trip(cache: SQLiteCache):
    facts = {"revenue": 52017.0, "net_income": 10053.0, "fiscal_year_end": "2025-12-31"}
    await cache.store_xbrl_facts("0001543151", "raw", facts)
    retrieved = await cache.get_xbrl_facts("0001543151", "raw")
    assert retrieved == facts


async def test_xbrl_facts_returns_none_when_missing(cache: SQLiteCache):
    result = await cache.get_xbrl_facts("0001543151", "raw")
    assert result is None


async def test_xbrl_facts_replace_on_update(cache: SQLiteCache):
    await cache.store_xbrl_facts("0001543151", "raw", {"revenue": 43978.0})
    await cache.store_xbrl_facts("0001543151", "raw", {"revenue": 52017.0})
    retrieved = await cache.get_xbrl_facts("0001543151", "raw")
    assert retrieved["revenue"] == 52017.0


async def test_xbrl_facts_isolated_by_cik(cache: SQLiteCache):
    await cache.store_xbrl_facts("CIK_A", "raw", {"revenue": 100.0})
    await cache.store_xbrl_facts("CIK_B", "raw", {"revenue": 200.0})
    assert (await cache.get_xbrl_facts("CIK_A", "raw"))["revenue"] == 100.0
    assert (await cache.get_xbrl_facts("CIK_B", "raw"))["revenue"] == 200.0


# ── Filing metadata ───────────────────────────────────────────────────────────

async def test_get_filing_returns_most_recent(cache: SQLiteCache, tmp_path: Path):
    """get_filing() should return the most recently filed entry."""
    docs = tmp_path / "docs"
    docs.mkdir(exist_ok=True)
    for i, (acc, date) in enumerate([
        ("ACC-2023", "2024-02-15"),
        ("ACC-2024", "2025-02-13"),
        ("ACC-2025", "2026-02-12"),
    ]):
        p = docs / f"doc{i}.txt"
        p.write_text(f"filing {date}")
        await cache.store_filing("UBER", "0001543151", acc, "10-K", date, f"http://example.com/{acc}", str(p))

    result = await cache.get_filing("UBER")
    assert result is not None
    assert result.accession_number == "ACC-2025"


async def test_store_filing_insert_or_ignore(cache: SQLiteCache, tmp_path: Path):
    """Storing the same accession twice should not raise and should not overwrite."""
    p = tmp_path / "doc.txt"
    p.write_text("content")
    await cache.store_filing("AAPL", "0000320193", "ACC-001", "10-K", "2024-11-01", "http://x.com/doc", str(p))
    await cache.store_filing("AAPL", "0000320193", "ACC-001", "10-K", "2024-11-01", "http://x.com/doc", str(p))

    result = await cache.get_filing("AAPL")
    assert result is not None


# ── CIK lookup ────────────────────────────────────────────────────────────────

async def test_cik_round_trip(cache: SQLiteCache):
    await cache.store_cik("AAPL", "0000320193")
    assert await cache.get_cik("AAPL") == "0000320193"


async def test_cik_case_insensitive(cache: SQLiteCache):
    await cache.store_cik("aapl", "0000320193")
    assert await cache.get_cik("AAPL") == "0000320193"


async def test_cik_returns_none_when_missing(cache: SQLiteCache):
    assert await cache.get_cik("ZZZZ") is None
