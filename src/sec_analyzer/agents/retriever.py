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

from sec_analyzer.cache.sqlite_cache import SQLiteCache
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
    xbrl_metrics: dict[str, float] | None = None

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
    """Fetches 10-K filings from SEC EDGAR with caching and rate limiting.

    Cache strategy:
      - ticker_cik   : ticker → CIK (stable; rarely changes)
      - filings      : accession metadata per ticker/form/accession
      - documents    : raw HTML keyed by URL (URL contains accession so it's
                       content-addressed — same document never downloaded twice)
      - xbrl_facts   : EDGAR company facts JSON keyed by (cik, accession_number)
                       A new filing = new accession = automatic cache miss = fresh fetch
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
        # Base headers for data.sec.gov; _www_headers overrides Host for www.sec.gov
        self._headers = {
            "User-Agent": user_agent,
            "Accept-Encoding": "gzip, deflate",
            "Host": "data.sec.gov",
        }
        self._www_headers = {**self._headers, "Host": "www.sec.gov"}

    def _client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(headers=self._headers, timeout=30.0, follow_redirects=True)

    # ── Public ────────────────────────────────────────────────────────────────

    async def fetch_xbrl(
        self,
        cik: str,
        accession_number: str,
        trace_context: TraceContext,
        *,
        fiscal_year: str | None = None,
    ) -> dict[str, float]:
        """Return XBRL-sourced metrics, cached by (cik, accession_number).

        A new 10-K filing gets a new accession number, so the cache is
        automatically invalidated without any TTL or manual intervention.
        """
        with self._lf.start_as_current_observation(
            name="retriever.xbrl",
            as_type="span",
            trace_context=trace_context,
            input={"cik": cik, "accession": accession_number},
        ) as span:
            raw = await self._cache.get_xbrl_facts(cik, accession_number)
            if raw is None:
                raw = await download_xbrl_facts(cik, self._headers)
                await self._cache.store_xbrl_facts(cik, accession_number, raw)
                span.update(output={"source": "edgar"})
            else:
                span.update(output={"source": "cache"})

            metrics = fetch_xbrl_facts(cik, fiscal_year, raw=raw, accession_number=accession_number)
            span.update(output={"metrics_found": list(metrics.keys())})
        return metrics

    async def fetch_10k(
        self,
        ticker: str,
        trace_context: TraceContext,
        *,
        form_type: str = "10-K",
    ) -> FilingResult:
        """Return the most recent FilingResult for *ticker*."""
        with self._lf.start_as_current_observation(
            name="retriever",
            as_type="span",
            trace_context=trace_context,
            input={"ticker": ticker, "form_type": form_type},
        ) as span:
            try:
                result = await self._fetch(ticker, form_type, trace_context)
                try:
                    result.xbrl_metrics = await self.fetch_xbrl(
                        result.cik, result.accession_number, trace_context
                    )
                except Exception as xbrl_exc:
                    span.update(output={"xbrl_warning": str(xbrl_exc)})
            except Exception as exc:
                span.update(output={"error": str(exc)}, level="ERROR", status_message=str(exc))
                raise
            span.update(output={
                "cache_hit": result.cache_hit,
                "cik": result.cik,
                "filed_date": result.filed_date,
                "document_path": str(result.document_path),
                "xbrl_metrics": list(result.xbrl_metrics.keys()) if result.xbrl_metrics else [],
            })
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
            async with self._client() as client:
                cik = await self._cache.get_cik(ticker)
                if not cik:
                    cik = await self._resolve_cik(ticker, client, trace_context)
                    await self._cache.store_cik(ticker, cik)

                await self._rl.acquire()
                resp = await client.get(f"{_EDGAR_BASE}/submissions/CIK{cik}.json")
                resp.raise_for_status()
                subs = resp.json()

                recent = subs.get("filings", {}).get("recent", {})
                forms = recent.get("form", [])
                accessions = recent.get("accessionNumber", [])
                dates = recent.get("filingDate", [])
                docs = recent.get("primaryDocument", [])

                entries = [
                    (accessions[i], dates[i], docs[i])
                    for i, form in enumerate(forms)
                    if form == form_type
                ][:years]

                results: list[FilingResult] = []
                for accession, filed_date, doc_name in entries:
                    doc_url = _archive_url(cik, accession, doc_name)
                    cached_path = await self._cache.get_document_path(doc_url)
                    if cached_path and Path(cached_path).exists():
                        filing = FilingResult(
                            ticker=ticker, cik=cik, accession_number=accession,
                            form_type=form_type, filed_date=filed_date,
                            document_url=doc_url, document_path=Path(cached_path),
                            cache_hit=True,
                        )
                        # Ensure filing metadata is recorded even on a doc cache hit
                        await self._cache.store_filing(
                            ticker, cik, accession, form_type, filed_date, doc_url, cached_path
                        )
                    else:
                        file_path = await self._download(doc_url, client, trace_context)
                        await self._cache.store_filing(
                            ticker, cik, accession, form_type, filed_date, doc_url, str(file_path)
                        )
                        filing = FilingResult(
                            ticker=ticker, cik=cik, accession_number=accession,
                            form_type=form_type, filed_date=filed_date,
                            document_url=doc_url, document_path=file_path,
                            cache_hit=False,
                        )
                    try:
                        filing.xbrl_metrics = await self.fetch_xbrl(
                            cik, accession, trace_context
                        )
                    except Exception:
                        pass
                    results.append(filing)

            span.update(output={"filings_found": len(results), "dates": [r.filed_date for r in results]})
        return results

    # ── Internal ──────────────────────────────────────────────────────────────

    async def _fetch(
        self, ticker: str, form_type: str, trace_context: TraceContext
    ) -> FilingResult:
        async with self._client() as client:
            # 1 — Resolve CIK
            cik = await self._cache.get_cik(ticker)
            if not cik:
                cik = await self._resolve_cik(ticker, client, trace_context)
                await self._cache.store_cik(ticker, cik)

            # 2 — Always fetch latest accession from EDGAR so we never serve
            #     a stale cached filing when a newer one has been published.
            accession, filed_date, doc_name = await self._find_latest_filing(
                cik, form_type, client, trace_context
            )
            doc_url = _archive_url(cik, accession, doc_name)

            # 3 — Document URL is content-addressed (contains accession number).
            #     Cache hit means we already have this exact filing on disk.
            cached_path = await self._cache.get_document_path(doc_url)
            if cached_path and Path(cached_path).exists():
                await self._cache.store_filing(
                    ticker, cik, accession, form_type, filed_date, doc_url, cached_path
                )
                return FilingResult(
                    ticker=ticker, cik=cik, accession_number=accession,
                    form_type=form_type, filed_date=filed_date,
                    document_url=doc_url, document_path=Path(cached_path),
                    cache_hit=True,
                )

            # 4 — Download document
            file_path = await self._download(doc_url, client, trace_context)
            await self._cache.store_filing(
                ticker, cik, accession, form_type, filed_date, doc_url, str(file_path)
            )
            return FilingResult(
                ticker=ticker, cik=cik, accession_number=accession,
                form_type=form_type, filed_date=filed_date,
                document_url=doc_url, document_path=file_path,
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
            resp = await client.get(_TICKERS_URL, headers=self._www_headers)
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
            resp = await client.get(f"{_EDGAR_BASE}/submissions/CIK{cik}.json")
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
                    span.update(output={"accession": accession, "filed_date": filed_date})
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
            resp = await client.get(url, headers=self._www_headers)
            resp.raise_for_status()
            dest.write_bytes(resp.content)
            span.update(output={"path": str(dest), "bytes": len(resp.content)})
        return dest


# ── Helpers ───────────────────────────────────────────────────────────────────

def _archive_url(cik: str, accession: str, doc_name: str) -> str:
    nodash = accession.replace("-", "")
    return f"{_EDGAR_ARCHIVES}/{int(cik)}/{nodash}/{doc_name}"
