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


# ── News cache ────────────────────────────────────────────────────────────────

async def test_news_round_trip(cache: SQLiteCache):
    articles = [{"headline": "Big news", "summary": "Details", "url": "http://x", "source": "Reuters", "published_at": "2026-07-07T00:00:00+00:00"}]
    await cache.store_news("AAPL", "2026-07-07", articles, 0.35, "Positive outlook driven by strong earnings.")
    result = await cache.get_news("AAPL", "2026-07-07")
    assert result is not None
    assert result["articles"] == articles
    assert result["sentiment_score"] == pytest.approx(0.35)
    assert result["narrative"] == "Positive outlook driven by strong earnings."


async def test_news_round_trip_without_narrative(cache: SQLiteCache):
    await cache.store_news("AAPL", "2026-07-07", [], 0.1)
    result = await cache.get_news("AAPL", "2026-07-07")
    assert result is not None
    assert result["narrative"] is None


async def test_news_returns_none_when_missing(cache: SQLiteCache):
    assert await cache.get_news("AAPL", "2026-07-07") is None


async def test_news_case_insensitive_ticker(cache: SQLiteCache):
    await cache.store_news("aapl", "2026-07-07", [], 0.1)
    assert await cache.get_news("AAPL", "2026-07-07") is not None


async def test_news_null_sentiment(cache: SQLiteCache):
    await cache.store_news("AAPL", "2026-07-07", [], None)
    result = await cache.get_news("AAPL", "2026-07-07")
    assert result is not None
    assert result["sentiment_score"] is None


async def test_news_ttl_expired(cache: SQLiteCache, monkeypatch):
    """get_news returns None when cached_at is >24h ago."""
    from datetime import datetime, timedelta, timezone
    from sec_analyzer.cache import sqlite_cache as sc

    old_ts = (datetime.now(tz=timezone.utc) - timedelta(hours=25)).isoformat()
    monkeypatch.setattr(sc, "_now", lambda: old_ts)
    await cache.store_news("AAPL", "2026-07-06", [], 0.1)

    monkeypatch.undo()
    assert await cache.get_news("AAPL", "2026-07-06") is None


async def test_news_ttl_fresh(cache: SQLiteCache):
    """get_news returns data when cached within 24h."""
    await cache.store_news("AAPL", "2026-07-07", [{"headline": "ok"}], 0.0)
    assert await cache.get_news("AAPL", "2026-07-07") is not None


# ── Market data cache ─────────────────────────────────────────────────────────

async def test_market_data_round_trip(cache: SQLiteCache):
    data = {"price": 200.0, "change_pct": 1.5}
    await cache.store_market_data("AAPL", "quote", "current", data)
    result = await cache.get_market_data("AAPL", "quote", "current")
    assert result == data


async def test_market_data_returns_none_when_missing(cache: SQLiteCache):
    assert await cache.get_market_data("AAPL", "quote", "current") is None


async def test_market_data_isolated_by_type(cache: SQLiteCache):
    await cache.store_market_data("AAPL", "quote", "current", {"price": 200.0})
    await cache.store_market_data("AAPL", "ratios", "current", {"pe": 30.0})
    assert (await cache.get_market_data("AAPL", "quote", "current"))["price"] == 200.0
    assert (await cache.get_market_data("AAPL", "ratios", "current"))["pe"] == 30.0


async def test_market_data_replace_on_update(cache: SQLiteCache):
    await cache.store_market_data("AAPL", "quote", "current", {"price": 199.0})
    await cache.store_market_data("AAPL", "quote", "current", {"price": 201.0})
    assert (await cache.get_market_data("AAPL", "quote", "current"))["price"] == 201.0


async def test_market_data_quote_ttl_expired(cache: SQLiteCache, monkeypatch):
    """Quote TTL is 15 min; stale entry returns None."""
    from datetime import datetime, timedelta, timezone
    from sec_analyzer.cache import sqlite_cache as sc

    old_ts = (datetime.now(tz=timezone.utc) - timedelta(minutes=20)).isoformat()
    monkeypatch.setattr(sc, "_now", lambda: old_ts)
    await cache.store_market_data("AAPL", "quote", "current", {"price": 199.0})

    monkeypatch.undo()
    assert await cache.get_market_data("AAPL", "quote", "current") is None


async def test_market_data_quote_ttl_fresh(cache: SQLiteCache):
    """Quote cached just now is still fresh."""
    await cache.store_market_data("AAPL", "quote", "current", {"price": 200.0})
    assert await cache.get_market_data("AAPL", "quote", "current") is not None


async def test_market_data_history_ttl_expired(cache: SQLiteCache, monkeypatch):
    """History TTL is 24h."""
    from datetime import datetime, timedelta, timezone
    from sec_analyzer.cache import sqlite_cache as sc

    old_ts = (datetime.now(tz=timezone.utc) - timedelta(hours=25)).isoformat()
    monkeypatch.setattr(sc, "_now", lambda: old_ts)
    await cache.store_market_data("AAPL", "history", "1y", {"bars": []})

    monkeypatch.undo()
    assert await cache.get_market_data("AAPL", "history", "1y") is None


async def test_market_data_ratios_ttl_expired(cache: SQLiteCache, monkeypatch):
    """Ratios TTL is 1h."""
    from datetime import datetime, timedelta, timezone
    from sec_analyzer.cache import sqlite_cache as sc

    old_ts = (datetime.now(tz=timezone.utc) - timedelta(hours=2)).isoformat()
    monkeypatch.setattr(sc, "_now", lambda: old_ts)
    await cache.store_market_data("AAPL", "ratios", "current", {"pe": 30.0})

    monkeypatch.undo()
    assert await cache.get_market_data("AAPL", "ratios", "current") is None
