"""SEC EDGAR Retriever — fetches and caches 10-K filings.

Rate limit: 10 req/sec (EDGAR fair-use policy).
User-Agent header is required; set SEC_USER_AGENT in .env.
"""
from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass
from pathlib import Path

import httpx
from langfuse import Langfuse
from langfuse.types import TraceContext

from sec_analyzer.cache.sqlite_cache import CachedFiling, SQLiteCache
from sec_analyzer.xbrl import download_xbrl_facts, fetch_xbrl_facts

_EDGAR_BASE = "https://data.sec.gov"
_EDGAR_ARCHIVES = "https://www.sec.gov/Archives/edgar/data"
_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"


@dataclass
class FilingResult:
    ticker: str
    cik: str
    accession_number: str
    form_type: str
    filed_date: str
    document_url: str
    document_path: Path
    cache_hit: bool
    xbrl_metrics: dict[str, float] | None = None  # None = not yet fetched

    def read_text(self) -> str:
        return self.document_path.read_text(encoding="utf-8", errors="replace")


class _RateLimiter:
    """Token-bucket style rate limiter; default 10 req/sec for EDGAR."""

    def __init__(self, rate: float = 10.0) -> None:
        self._min_interval = 1.0 / rate
        self._last_call = 0.0
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        async with self._lock:
            now = time.monotonic()
            wait = self._min_interval - (now - self._last_call)
            if wait > 0:
                await asyncio.sleep(wait)
            self._last_call = time.monotonic()


class SECRetriever:
    """Fetches the most recent 10-K for a ticker from SEC EDGAR.

    Cache strategy:
      - SQLite stores filing metadata (ticker → CIK, accession, URL).
      - Filesystem stores the document text keyed by a hash of the URL.
      - On a cache hit neither the SQLite lookup nor the HTTP call incur
        Langfuse latency; both are still surfaced as spans so dashboards show
        hit rate.
    """

    def __init__(
        self,
        cache: SQLiteCache,
        langfuse: Langfuse,
        *,
        rate: float = 10.0,
    ) -> None:
        self._cache = cache
        self._lf = langfuse
        self._rl = _RateLimiter(rate)
        user_agent = os.environ.get("SEC_USER_AGENT", "VeriAnalyst admin@example.com")
        self._headers = {
            "User-Agent": user_agent,
            "Accept-Encoding": "gzip, deflate",
            "Host": "data.sec.gov",
        }

    @property
    def _data_headers(self) -> dict[str, str]:
        return {**self._headers, "Host": "data.sec.gov"}

    # ── Public ────────────────────────────────────────────────────────────────

    async def fetch_xbrl(
        self,
        cik: str,
        fiscal_year: str | None,
        trace_context: TraceContext,
    ) -> dict[str, float]:
        """Return XBRL-sourced metrics for *cik* and *fiscal_year* (YYYY).

        Caches the raw company facts JSON so subsequent calls for the same CIK
        (different fiscal years) hit the cache without another HTTP request.
        """
        with self._lf.start_as_current_observation(
            name="retriever.xbrl",
            as_type="span",
            trace_context=trace_context,
            input={"cik": cik, "fiscal_year": fiscal_year},
        ) as span:
            raw = await self._cache.get_xbrl_facts(cik, "raw")
            if raw is None:
                raw = await download_xbrl_facts(cik, self._data_headers)
                await self._cache.store_xbrl_facts(cik, "raw", raw)
                span.update(output={"source": "edgar"})
            else:
                span.update(output={"source": "cache"})

            metrics = fetch_xbrl_facts(cik, fiscal_year, raw=raw)
            span.update(output={"source": "cache" if raw else "edgar", "metrics_found": list(metrics.keys())})
        return metrics

    async def fetch_10k(
        self,
        ticker: str,
        trace_context: TraceContext,
        *,
        form_type: str = "10-K",
    ) -> FilingResult:
        """Return a FilingResult for *ticker*, using caches where possible."""
        with self._lf.start_as_current_observation(
            name="retriever",
            as_type="span",
            trace_context=trace_context,
            input={"ticker": ticker, "form_type": form_type},
        ) as span:
            try:
                result = await self._fetch(ticker, form_type, trace_context)
                fiscal_year = result.filed_date[:4]
                try:
                    result.xbrl_metrics = await self.fetch_xbrl(result.cik, None, trace_context)
                except Exception as xbrl_exc:
                    span.update(output={"xbrl_warning": str(xbrl_exc)})
            except Exception as exc:
                span.update(
                    output={"error": str(exc)},
                    level="ERROR",
                    status_message=str(exc),
                )
                raise
            span.update(
                output={
                    "cache_hit": result.cache_hit,
                    "cik": result.cik,
                    "filed_date": result.filed_date,
                    "document_path": str(result.document_path),
                    "xbrl_metrics": list(result.xbrl_metrics.keys()) if result.xbrl_metrics else [],
                }
            )
        return result

    async def fetch_10k_history(
        self,
        ticker: str,
        trace_context: TraceContext,
        *,
        years: int = 5,
        form_type: str = "10-K",
    ) -> list[FilingResult]:
        """Return up to *years* annual FilingResults, most-recent first."""
        with self._lf.start_as_current_observation(
            name="retriever.history",
            as_type="span",
            trace_context=trace_context,
            input={"ticker": ticker, "years": years},
        ) as span:
            cik = await self._cache.get_cik(ticker)
            async with httpx.AsyncClient(headers=self._headers, timeout=30.0, follow_redirects=True) as client:
                if not cik:
                    cik = await self._resolve_cik(ticker, client, trace_context)
                    await self._cache.store_cik(ticker, cik)

                await self._rl.acquire()
                url = f"{_EDGAR_BASE}/submissions/CIK{cik}.json"
                resp = await client.get(url)
                resp.raise_for_status()
                subs = resp.json()

                recent = subs.get("filings", {}).get("recent", {})
                forms = recent.get("form", [])
                accessions = recent.get("accessionNumber", [])
                dates = recent.get("filingDate", [])
                docs = recent.get("primaryDocument", [])

                # Collect all 10-K entries up to *years*
                entries = [
                    (accessions[i], dates[i], docs[i])
                    for i, form in enumerate(forms)
                    if form == form_type
                ][:years]

                results = []
                for accession, filed_date, doc_name in entries:
                    doc_url = _archive_url(cik, accession, doc_name)
                    cached_path = await self._cache.get_document_path(doc_url)
                    if cached_path and Path(cached_path).exists():
                        results.append(FilingResult(
                            ticker=ticker, cik=cik, accession_number=accession,
                            form_type=form_type, filed_date=filed_date,
                            document_url=doc_url, document_path=Path(cached_path),
                            cache_hit=True,
                        ))
                    else:
                        file_path = await self._download(doc_url, client, trace_context)
                        await self._cache.store_filing(
                            ticker, cik, accession, form_type, filed_date, doc_url, str(file_path)
                        )
                        results.append(FilingResult(
                            ticker=ticker, cik=cik, accession_number=accession,
                            form_type=form_type, filed_date=filed_date,
                            document_url=doc_url, document_path=file_path,
                            cache_hit=False,
                        ))

            span.update(output={"filings_found": len(results), "dates": [r.filed_date for r in results]})
        return results

    # ── Internal ──────────────────────────────────────────────────────────────

    async def _fetch(
        self, ticker: str, form_type: str, trace_context: TraceContext
    ) -> FilingResult:
        # 1 — Full cache hit: metadata + document on disk
        cached = await self._cache.get_filing(ticker, form_type)
        if cached and Path(cached.file_path).exists():
            with self._lf.start_as_current_observation(
                name="retriever.cache_hit",
                trace_context=trace_context,
                input={"ticker": ticker},
            ) as s:
                s.update(output={"path": cached.file_path})
            return _result_from_cached(cached, hit=True)

        async with httpx.AsyncClient(headers=self._headers, timeout=30.0, follow_redirects=True) as client:
            # 2 — Resolve CIK
            cik = await self._cache.get_cik(ticker)
            if not cik:
                cik = await self._resolve_cik(ticker, client, trace_context)
                await self._cache.store_cik(ticker, cik)

            # 3 — Find latest filing
            accession, filed_date, doc_name = await self._find_latest_filing(
                cik, form_type, client, trace_context
            )
            doc_url = _archive_url(cik, accession, doc_name)

            # 4 — Check doc-level cache
            cached_path = await self._cache.get_document_path(doc_url)
            if cached_path and Path(cached_path).exists():
                await self._cache.store_filing(
                    ticker, cik, accession, form_type, filed_date, doc_url, cached_path
                )
                return FilingResult(
                    ticker=ticker,
                    cik=cik,
                    accession_number=accession,
                    form_type=form_type,
                    filed_date=filed_date,
                    document_url=doc_url,
                    document_path=Path(cached_path),
                    cache_hit=True,
                )

            # 5 — Download document
            file_path = await self._download(doc_url, client, trace_context)
            await self._cache.store_filing(
                ticker, cik, accession, form_type, filed_date, doc_url, str(file_path)
            )
            return FilingResult(
                ticker=ticker,
                cik=cik,
                accession_number=accession,
                form_type=form_type,
                filed_date=filed_date,
                document_url=doc_url,
                document_path=file_path,
                cache_hit=False,
            )

    async def _resolve_cik(
        self,
        ticker: str,
        client: httpx.AsyncClient,
        trace_context: TraceContext,
    ) -> str:
        with self._lf.start_as_current_observation(
            name="retriever.resolve_cik",
            trace_context=trace_context,
            input={"ticker": ticker},
        ) as span:
            await self._rl.acquire()
            resp = await client.get(
                _TICKERS_URL,
                headers={**self._headers, "Host": "www.sec.gov"},
            )
            resp.raise_for_status()
            data: dict = resp.json()
            upper = ticker.upper()
            for entry in data.values():
                if entry.get("ticker", "").upper() == upper:
                    cik = str(entry["cik_str"]).zfill(10)
                    span.update(output={"cik": cik})
                    return cik
            span.update(output={"error": "not found"}, level="ERROR")
            raise ValueError(f"Ticker {ticker!r} not found in EDGAR company_tickers.json")

    async def _find_latest_filing(
        self,
        cik: str,
        form_type: str,
        client: httpx.AsyncClient,
        trace_context: TraceContext,
    ) -> tuple[str, str, str]:
        """Return (accession_number, filed_date, primary_document_name)."""
        with self._lf.start_as_current_observation(
            name="retriever.find_filing",
            trace_context=trace_context,
            input={"cik": cik, "form_type": form_type},
        ) as span:
            await self._rl.acquire()
            url = f"{_EDGAR_BASE}/submissions/CIK{cik}.json"
            resp = await client.get(url)
            resp.raise_for_status()
            subs = resp.json()

            recent = subs.get("filings", {}).get("recent", {})
            forms = recent.get("form", [])
            accessions = recent.get("accessionNumber", [])
            dates = recent.get("filingDate", [])
            docs = recent.get("primaryDocument", [])

            for i, form in enumerate(forms):
                if form == form_type:
                    accession = accessions[i]
                    filed_date = dates[i]
                    doc_name = docs[i]
                    span.update(output={"accession": accession, "filed_date": filed_date, "doc": doc_name})
                    return accession, filed_date, doc_name

            span.update(output={"error": f"no {form_type} found"}, level="ERROR")
            raise ValueError(f"No {form_type} filing found for CIK {cik}")

    async def _download(
        self,
        url: str,
        client: httpx.AsyncClient,
        trace_context: TraceContext,
    ) -> Path:
        with self._lf.start_as_current_observation(
            name="retriever.download",
            trace_context=trace_context,
            input={"url": url},
        ) as span:
            dest = self._cache.document_path_for(url)
            await self._rl.acquire()
            resp = await client.get(url, headers={**self._headers, "Host": "www.sec.gov"})
            resp.raise_for_status()
            dest.write_bytes(resp.content)
            span.update(output={"path": str(dest), "bytes": len(resp.content)})
        return dest


# ── Helpers ───────────────────────────────────────────────────────────────────

def _archive_url(cik: str, accession: str, doc_name: str) -> str:
    nodash = accession.replace("-", "")
    return f"{_EDGAR_ARCHIVES}/{int(cik)}/{nodash}/{doc_name}"


def _result_from_cached(c: CachedFiling, *, hit: bool) -> FilingResult:
    return FilingResult(
        ticker=c.ticker,
        cik=c.cik,
        accession_number=c.accession_number,
        form_type=c.form_type,
        filed_date=c.filed_date,
        document_url=c.document_url,
        document_path=Path(c.file_path),
        cache_hit=hit,
    )
