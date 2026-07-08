"""Tests for SECRetriever — cache paths and HTTP fetch paths."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import httpx
import pytest
import respx

from sec_analyzer.agents.retriever import SECRetriever, _archive_url
from sec_analyzer.cache.sqlite_cache import SQLiteCache

# ── Constants ─────────────────────────────────────────────────────────────────

TICKER = "AAPL"
CIK = "0000320193"
ACCESSION = "0000320193-24-000001"
DOC_NAME = "aapl-20240928.htm"
FILED_DATE = "2024-11-01"
DOC_URL = _archive_url(CIK, ACCESSION, DOC_NAME)
FAKE_DOC = "<html><body>Apple 10-K content</body></html>"

TICKERS_PAYLOAD = {
    "0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
}

SUBMISSIONS_PAYLOAD = {
    "cik": CIK,
    "name": "Apple Inc.",
    "filings": {
        "recent": {
            "accessionNumber": [ACCESSION, "0000320193-23-000001"],
            "form": ["10-K", "10-Q"],
            "filingDate": [FILED_DATE, "2023-08-01"],
            "primaryDocument": [DOC_NAME, "aapl-20230701.htm"],
        }
    },
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_langfuse() -> MagicMock:
    """Return a MagicMock that mimics the Langfuse 4.x API.

    start_as_current_observation returns a synchronous context manager
    (_AgnosticContextManager), so we mock __enter__/__exit__ (not async).
    """
    span = MagicMock()
    span.update = MagicMock()

    class _SyncCM:
        def __enter__(self):
            return span
        def __exit__(self, *_):
            pass

    lf = MagicMock()
    lf.start_as_current_observation.return_value = _SyncCM()
    return lf


def _make_trace_context():
    from langfuse.types import TraceContext
    import uuid
    return TraceContext(trace_id=str(uuid.uuid4()))


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture()
async def cache(tmp_path: Path) -> SQLiteCache:
    c = SQLiteCache(
        db_path=str(tmp_path / "cache.db"),
        docs_dir=str(tmp_path / "docs"),
    )
    async with c:
        yield c


# ── Cache-hit path ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_cache_hit_skips_download(cache: SQLiteCache, tmp_path: Path) -> None:
    """When the latest EDGAR accession matches the cached document, no download is made."""
    doc_path = tmp_path / "docs" / "fake.txt"
    doc_path.parent.mkdir(parents=True, exist_ok=True)
    doc_path.write_text(FAKE_DOC)

    await cache.store_cik(TICKER, CIK)
    await cache.store_filing(
        TICKER, CIK, ACCESSION, "10-K", FILED_DATE, DOC_URL, str(doc_path)
    )

    lf = _make_langfuse()
    retriever = SECRetriever(cache=cache, langfuse=lf)
    tc = _make_trace_context()

    # We always check EDGAR for the latest accession; document download is skipped
    # when the accession matches what's already cached.
    with respx.mock(assert_all_called=False) as mock:
        mock.get(f"https://data.sec.gov/submissions/CIK{CIK}.json").mock(
            return_value=httpx.Response(200, json=SUBMISSIONS_PAYLOAD)
        )
        result = await retriever.fetch_10k(TICKER, tc)

    assert result.cache_hit is True
    assert result.ticker == TICKER
    assert result.cik == CIK
    assert result.read_text() == FAKE_DOC


# ── Full fetch path ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_full_fetch(cache: SQLiteCache) -> None:
    """Cold cache → resolves CIK, finds filing, downloads document."""
    lf = _make_langfuse()
    retriever = SECRetriever(cache=cache, langfuse=lf)
    tc = _make_trace_context()

    with respx.mock(assert_all_called=True) as mock:
        mock.get("https://www.sec.gov/files/company_tickers.json").mock(
            return_value=httpx.Response(200, json=TICKERS_PAYLOAD)
        )
        mock.get(f"https://data.sec.gov/submissions/CIK{CIK}.json").mock(
            return_value=httpx.Response(200, json=SUBMISSIONS_PAYLOAD)
        )
        mock.get(DOC_URL).mock(
            return_value=httpx.Response(200, content=FAKE_DOC.encode())
        )

        result = await retriever.fetch_10k(TICKER, tc)

    assert result.cache_hit is False
    assert result.cik == CIK
    assert result.accession_number == ACCESSION
    assert result.filed_date == FILED_DATE
    assert result.read_text() == FAKE_DOC

    assert await cache.get_cik(TICKER) == CIK
    cached = await cache.get_filing(TICKER)
    assert cached is not None
    assert cached.accession_number == ACCESSION


# ── Partial cache: CIK known ──────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_partial_cache_cik_known(cache: SQLiteCache) -> None:
    """CIK already cached → skip company_tickers.json fetch."""
    await cache.store_cik(TICKER, CIK)

    lf = _make_langfuse()
    retriever = SECRetriever(cache=cache, langfuse=lf)
    tc = _make_trace_context()

    with respx.mock(assert_all_called=True) as mock:
        mock.get(f"https://data.sec.gov/submissions/CIK{CIK}.json").mock(
            return_value=httpx.Response(200, json=SUBMISSIONS_PAYLOAD)
        )
        mock.get(DOC_URL).mock(
            return_value=httpx.Response(200, content=FAKE_DOC.encode())
        )

        result = await retriever.fetch_10k(TICKER, tc)

    assert result.cik == CIK


# ── Unknown ticker ─────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_unknown_ticker_raises(cache: SQLiteCache) -> None:
    lf = _make_langfuse()
    retriever = SECRetriever(cache=cache, langfuse=lf)
    tc = _make_trace_context()

    with respx.mock() as mock:
        mock.get("https://www.sec.gov/files/company_tickers.json").mock(
            return_value=httpx.Response(200, json={"0": {"cik_str": 1, "ticker": "XYZ"}})
        )

        with pytest.raises(ValueError, match="FAKE"):
            await retriever.fetch_10k("FAKE", tc)


# ── Rate-limit helper ──────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_rate_limiter_enforces_interval() -> None:
    """Two back-to-back acquires at 10 req/sec should take ≥ 90 ms."""
    import time
    from sec_analyzer.agents.retriever import _RateLimiter

    rl = _RateLimiter(rate=10.0)
    await rl.acquire()
    t0 = time.monotonic()
    await rl.acquire()
    elapsed_ms = (time.monotonic() - t0) * 1000
    assert elapsed_ms >= 90, f"Expected ≥90 ms gap, got {elapsed_ms:.1f} ms"


# ── Archive URL helper ─────────────────────────────────────────────────────────

def test_archive_url_format() -> None:
    url = _archive_url("0000320193", "0000320193-24-000001", "aapl-20240928.htm")
    assert "320193" in url
    assert "000032019324000001" in url
    assert url.endswith("aapl-20240928.htm")


# ── fetch_10k_history — each filing gets its own year's XBRL data ─────────────

# Two historical 10-K filings with distinct accession numbers
ACC_2024 = "0000320193-24-000001"
ACC_2023 = "0000320193-23-000001"
DOC_2024 = "aapl-20240928.htm"
DOC_2023 = "aapl-20230930.htm"

HISTORY_SUBMISSIONS = {
    "cik": CIK,
    "name": "Apple Inc.",
    "filings": {
        "recent": {
            "accessionNumber": [ACC_2024, ACC_2023],
            "form": ["10-K", "10-K"],
            "filingDate": ["2024-11-01", "2023-11-03"],
            "primaryDocument": [DOC_2024, DOC_2023],
        }
    },
}

# Synthetic XBRL company facts: two distinct revenue values, one per accession
XBRL_FACTS_PAYLOAD = {
    "facts": {
        "us-gaap": {
            "RevenueFromContractWithCustomerExcludingAssessedTax": {
                "units": {
                    "USD": [
                        {
                            "start": "2023-10-01", "end": "2024-09-28",
                            "val": 391035000000, "accn": ACC_2024,
                            "fy": 2024, "fp": "FY", "form": "10-K",
                            "filed": "2024-11-01",
                        },
                        {
                            "start": "2022-10-01", "end": "2023-09-30",
                            "val": 383285000000, "accn": ACC_2023,
                            "fy": 2023, "fp": "FY", "form": "10-K",
                            "filed": "2023-11-03",
                        },
                    ]
                }
            }
        }
    }
}


@pytest.mark.asyncio
async def test_history_each_filing_gets_own_xbrl_year(
    cache: SQLiteCache, tmp_path: Path
) -> None:
    """Each historical filing must return XBRL metrics from its own fiscal year,
    not always the most recent year's data."""
    doc_2024 = tmp_path / "docs" / "doc2024.txt"
    doc_2023 = tmp_path / "docs" / "doc2023.txt"
    doc_2024.parent.mkdir(parents=True, exist_ok=True)
    doc_2024.write_text("Apple FY2024 10-K")
    doc_2023.write_text("Apple FY2023 10-K")

    url_2024 = _archive_url(CIK, ACC_2024, DOC_2024)
    url_2023 = _archive_url(CIK, ACC_2023, DOC_2023)
    await cache.store_cik(TICKER, CIK)
    await cache.store_filing(TICKER, CIK, ACC_2024, "10-K", "2024-11-01", url_2024, str(doc_2024))
    await cache.store_filing(TICKER, CIK, ACC_2023, "10-K", "2023-11-03", url_2023, str(doc_2023))
    # Pre-seed XBRL cache with the full facts payload under both accessions
    await cache.store_xbrl_facts(CIK, ACC_2024, XBRL_FACTS_PAYLOAD)
    await cache.store_xbrl_facts(CIK, ACC_2023, XBRL_FACTS_PAYLOAD)

    lf = _make_langfuse()
    retriever = SECRetriever(cache=cache, langfuse=lf)
    tc = _make_trace_context()

    with respx.mock(assert_all_called=False) as mock:
        mock.get(f"https://data.sec.gov/submissions/CIK{CIK}.json").mock(
            return_value=httpx.Response(200, json=HISTORY_SUBMISSIONS)
        )
        filings = await retriever.fetch_10k_history(TICKER, tc, years=2)

    assert len(filings) == 2

    fy_ends = [f.xbrl_metrics.get("fiscal_year_end") for f in filings]
    revenues = [f.xbrl_metrics.get("revenue") for f in filings]

    # Each filing must have a distinct fiscal year end
    assert fy_ends[0] != fy_ends[1], (
        f"Both filings returned the same fiscal_year_end={fy_ends[0]!r} — "
        "XBRL accession filtering is broken"
    )
    # FY2024 filing is most recent, must have higher revenue
    assert revenues[0] != revenues[1], "Both filings returned identical revenue"
    assert revenues[0] > revenues[1], "Older filing should have lower revenue than newer"
