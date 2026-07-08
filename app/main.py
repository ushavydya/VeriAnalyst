"""Streamlit chat interface for VeriAnalyst."""
from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

# Allow imports from src/
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

import streamlit as st

from sec_analyzer.cache.sqlite_cache import SQLiteCache
from sec_analyzer.comparison import compare
from sec_analyzer.gateway import get_gateway
from sec_analyzer.orchestration.graph import build_pipeline, initial_state
from sec_analyzer.trends import fetch_trends

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="VeriAnalyst",
    page_icon="📊",
    layout="wide",
)

st.title("📊 VeriAnalyst")
st.caption("SEC 10-K analysis powered by AI")


def _fmt_millions(value: float) -> str:
    """Format a value stored in millions as $XB or $XM."""
    if abs(value) >= 1_000:
        return f"${value / 1_000:.2f}B"
    return f"${value:,.0f}M"


def _extract_intelligence_section(report: str) -> str | None:
    """Extract the Investment Intelligence section from the report regardless of heading style.

    The LLM may render it as '## Investment Intelligence', '**Investment Intelligence**',
    '5. **Investment Intelligence**', etc.  We search case-insensitively for the keyword
    and grab everything up to the next same-level heading.
    """
    import re
    # Match any heading line that contains "investment intelligence" (case-insensitive)
    pattern = re.compile(r"((?:^|\n)(#{1,3}[^\n]*investment intelligence[^\n]*|[^\n]*\*\*investment intelligence\*\*[^\n]*))", re.IGNORECASE)
    m = pattern.search(report)
    if not m:
        return None
    start = m.start(1)
    # Find the next Markdown heading at the same or higher level after this section
    next_heading = re.search(r"\n#{1,3} ", report[start + 1:])
    # Also look for numbered bold headings like "\n5. **" or "\n**"
    next_bold = re.search(r"\n\d+\.\s+\*\*|\n\*\*[A-Z]", report[start + 1:])
    candidates = [c.start() + start + 1 for c in [next_heading, next_bold] if c]
    end = min(candidates) if candidates else len(report)
    return report[start:end].strip()

# ── Session state ─────────────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []          # chat history
if "report" not in st.session_state:
    st.session_state.report = None          # latest report text
if "metrics" not in st.session_state:
    st.session_state.metrics = {}
if "confidence" not in st.session_state:
    st.session_state.confidence = None
if "ticker" not in st.session_state:
    st.session_state.ticker = None
if "news_summary_json" not in st.session_state:
    st.session_state.news_summary_json = None
if "market_summary_json" not in st.session_state:
    st.session_state.market_summary_json = None

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Analyze a ticker")
    ticker_input = st.text_input("Ticker symbol", placeholder="e.g. AAPL, UBER, MSFT").upper().strip()
    run_btn = st.button("Run analysis", type="primary", use_container_width=True)

    if st.session_state.confidence is not None:
        st.divider()
        st.subheader("Last run")
        st.metric("Ticker", st.session_state.ticker)
        color = "green" if st.session_state.confidence >= 0.8 else "orange" if st.session_state.confidence >= 0.5 else "red"
        st.metric("Data confidence", f"{st.session_state.confidence:.0%}")

    if st.session_state.metrics:
        st.divider()
        st.subheader("Key metrics")
        m = st.session_state.metrics
        if "revenue" in m:
            st.metric("Revenue", _fmt_millions(m['revenue']))
        if "net_income" in m:
            st.metric("Net income", _fmt_millions(m['net_income']))
        if "eps_diluted" in m:
            st.metric("Diluted EPS", f"${m['eps_diluted']:.2f}")
        if "total_assets" in m:
            st.metric("Total assets", _fmt_millions(m['total_assets']))
        if "fiscal_year_end" in m:
            st.caption(f"Fiscal year end: {m['fiscal_year_end']}")

    if st.session_state.report and st.button("Clear / new analysis", use_container_width=True):
        st.session_state.messages = []
        st.session_state.report = None
        st.session_state.metrics = {}
        st.session_state.confidence = None
        st.session_state.ticker = None
        st.session_state.news_summary_json = None
        st.session_state.market_summary_json = None
        st.rerun()

# ── Pipeline runner ───────────────────────────────────────────────────────────

def _run_pipeline(ticker: str) -> dict:
    """Run the full pipeline in a fresh thread+event loop to avoid Streamlit conflicts."""
    import concurrent.futures
    from langfuse import Langfuse

    def _in_thread():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            async def _run():
                async with SQLiteCache() as cache:
                    langfuse = Langfuse(
                        public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
                        secret_key=os.environ["LANGFUSE_SECRET_KEY"],
                        host=os.environ.get("LANGFUSE_HOST", "http://localhost:3000"),
                    )
                    pipeline = build_pipeline(cache, langfuse)
                    state = initial_state(ticker)
                    return await pipeline.ainvoke(state)
            return loop.run_until_complete(_run())
        finally:
            loop.close()

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(_in_thread)
        return future.result()


# ── Trends runner ────────────────────────────────────────────────────────────

def _run_trends(ticker: str, years: int) -> list:
    """Fetch multi-year metrics in a fresh thread+event loop."""
    import concurrent.futures
    from langfuse import Langfuse

    def _in_thread():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            async def _run():
                async with SQLiteCache() as cache:
                    langfuse = Langfuse(
                        public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
                        secret_key=os.environ["LANGFUSE_SECRET_KEY"],
                        host=os.environ.get("LANGFUSE_HOST", "http://localhost:3000"),
                    )
                    return await fetch_trends(ticker, years=years, cache=cache, langfuse=langfuse)
            return loop.run_until_complete(_run())
        finally:
            loop.close()

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(_in_thread).result()


# ── Comparison runner ────────────────────────────────────────────────────────

def _run_comparison(tickers: list[str]):
    """Run parallel comparison pipeline in a fresh thread+event loop."""
    import concurrent.futures
    from langfuse import Langfuse

    def _in_thread():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            async def _run():
                async with SQLiteCache() as cache:
                    langfuse = Langfuse(
                        public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
                        secret_key=os.environ["LANGFUSE_SECRET_KEY"],
                        host=os.environ.get("LANGFUSE_HOST", "http://localhost:3000"),
                    )
                    return await compare(tickers, cache, langfuse)
            return loop.run_until_complete(_run())
        finally:
            loop.close()

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(_in_thread).result()


# ── Chat Q&A helper ───────────────────────────────────────────────────────────

def _answer_question(question: str, report: str) -> str:
    """Ask the LLM a follow-up question grounded in the report."""
    import concurrent.futures

    def _in_thread():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            gw = get_gateway()
            system = (
                "You are a financial analyst assistant. Answer questions based ONLY on the "
                "provided 10-K analysis report. Be concise and precise. If the answer is not "
                "in the report, say so — do not speculate."
            )
            messages = [{"role": "user", "content": f"Report:\n\n{report}\n\n---\nQuestion: {question}"}]
            return loop.run_until_complete(gw.complete(messages, system=system, max_tokens=512)).text.strip()
        finally:
            loop.close()

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(_in_thread).result()


# ── Main area: tabs ──────────────────────────────────────────────────────────

import json

tab_analysis, tab_trends, tab_compare, tab_intel = st.tabs(["📄 Analysis", "📈 Trends", "⚖️ Compare", "🧠 Intelligence"])

# ── Tab 1: Analysis + chat ────────────────────────────────────────────────────

with tab_analysis:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if run_btn and ticker_input:
        st.session_state.messages = []
        st.session_state.report = None

        with st.chat_message("assistant"):
            with st.spinner(f"Fetching and analysing {ticker_input} 10-K filing…"):
                try:
                    result = _run_pipeline(ticker_input)
                except Exception as e:
                    import traceback
                    msg = str(e) or type(e).__name__
                    st.error(f"Pipeline exception: {msg}")
                    with st.expander("Full traceback"):
                        st.code(traceback.format_exc())
                    st.stop()

            if result.get("error"):
                st.error(f"Pipeline error:\n\n```\n{result['error']}\n```")
                st.stop()

            report = result.get("report", "")
            st.markdown(report)

        st.session_state.report = report
        st.session_state.ticker = ticker_input
        st.session_state.messages.append({"role": "assistant", "content": report})

        if result.get("critique_json"):
            crit = json.loads(result["critique_json"])
            st.session_state.confidence = crit.get("confidence")
        if result.get("extracted_data_json"):
            data = json.loads(result["extracted_data_json"])
            st.session_state.metrics = data.get("metrics", {})
        st.session_state.news_summary_json = result.get("news_summary_json")
        st.session_state.market_summary_json = result.get("market_summary_json")

        st.rerun()

    if st.session_state.report:
        if question := st.chat_input("Ask a follow-up question about this filing…"):
            with st.chat_message("user"):
                st.markdown(question)
            st.session_state.messages.append({"role": "user", "content": question})
            with st.chat_message("assistant"):
                with st.spinner("Thinking…"):
                    answer = _answer_question(question, st.session_state.report)
                st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
    elif not run_btn:
        st.info("Enter a ticker symbol in the sidebar and click **Run analysis** to get started.")

# ── Tab 2: Multi-year trends ──────────────────────────────────────────────────

with tab_trends:
    st.subheader("Multi-year financial trends")

    col1, col2 = st.columns([2, 1])
    with col1:
        trend_ticker = st.text_input(
            "Ticker", value=st.session_state.ticker or "", key="trend_ticker",
            placeholder="e.g. AAPL"
        ).upper().strip()
    with col2:
        years = st.selectbox("Years", [3, 4, 5], index=2)

    if st.button("Fetch trends", type="primary"):
        if not trend_ticker:
            st.warning("Enter a ticker first.")
        else:
            with st.spinner(f"Fetching {years} years of {trend_ticker} 10-K filings…"):
                try:
                    trend_data = _run_trends(trend_ticker, years)
                except Exception as e:
                    st.error(f"Error: {e}")
                    st.stop()

            if not trend_data:
                st.warning("No filings found.")
                st.stop()

            # Build a dataframe of metrics over time
            import pandas as pd

            rows = []
            for ym in trend_data:
                label = ym.fiscal_year_end or ym.filed_date
                row = {"Period": label}
                row.update({k: v for k, v in ym.metrics.items() if isinstance(v, (int, float))})
                rows.append(row)

            df = pd.DataFrame(rows).set_index("Period").sort_index()

            # Charts
            _CHART_METRICS = {
                "revenue": "Revenue ($M)",
                "net_income": "Net Income ($M)",
                "gross_profit": "Gross Profit ($M)",
                "eps_diluted": "Diluted EPS ($)",
                "total_assets": "Total Assets ($M)",
                "cash_and_equivalents": "Cash & Equivalents ($M)",
            }

            available = [m for m in _CHART_METRICS if m in df.columns]
            if not available:
                st.warning("No numeric metrics extracted from these filings.")
            else:
                # Summary table
                st.dataframe(
                    df[[m for m in _CHART_METRICS if m in df.columns]]
                    .rename(columns=_CHART_METRICS)
                    .style.format("{:,.1f}"),
                    use_container_width=True,
                )

                # Individual charts
                chart_cols = st.columns(2)
                for i, metric in enumerate(available):
                    with chart_cols[i % 2]:
                        st.caption(_CHART_METRICS[metric])
                        st.line_chart(df[[metric]].rename(columns={metric: _CHART_METRICS[metric]}))

# ── Tab 3: Multi-company comparison ──────────────────────────────────────────

with tab_compare:
    st.subheader("Compare companies side-by-side")
    st.caption("Enter 2–4 ticker symbols to generate a comparative analysis.")

    compare_input = st.text_input(
        "Tickers (comma-separated)",
        placeholder="e.g. UBER, LYFT  or  AAPL, MSFT, GOOGL",
        key="compare_tickers",
    )
    compare_btn = st.button("Compare", type="primary", key="compare_btn")

    if compare_btn:
        raw_tickers = [t.strip().upper() for t in compare_input.split(",") if t.strip()]
        if len(raw_tickers) < 2:
            st.warning("Enter at least 2 ticker symbols separated by commas.")
        elif len(raw_tickers) > 4:
            st.warning("Maximum 4 tickers at once.")
        else:
            with st.spinner(f"Analysing {', '.join(raw_tickers)} in parallel…"):
                try:
                    result = _run_comparison(raw_tickers)
                except Exception as e:
                    st.error(f"Comparison failed: {e}")
                    st.stop()

            # ── Side-by-side metrics table ────────────────────────────────
            st.divider()
            st.subheader("Key metrics")

            import pandas as pd

            _CMP_METRICS = {
                "revenue":             "Revenue ($M)",
                "gross_profit":        "Gross Profit ($M)",
                "operating_income":    "Operating Income ($M)",
                "net_income":          "Net Income ($M)",
                "eps_diluted":         "Diluted EPS ($)",
                "total_assets":        "Total Assets ($M)",
                "total_equity":        "Total Equity ($M)",
                "cash_and_equivalents":"Cash & Equiv ($M)",
            }

            rows = {}
            for label in _CMP_METRICS.values():
                rows[label] = {}
            for extraction in result.extractions:
                fy = extraction.metrics.get("fiscal_year_end") or extraction.filed_date
                col_name = f"{extraction.ticker}\n(FY {fy})"
                for key, label in _CMP_METRICS.items():
                    val = extraction.metrics.get(key)
                    if val is None:
                        rows[label][col_name] = None
                    elif key == "eps_diluted":
                        rows[label][col_name] = round(val, 2)
                    else:
                        rows[label][col_name] = round(val, 0)

            df_cmp = pd.DataFrame(rows).T
            st.dataframe(
                df_cmp.style.format(
                    lambda v: f"{v:,.1f}" if isinstance(v, float) and abs(v) >= 10
                    else (f"{v:.2f}" if isinstance(v, float) else ("—" if v is None else str(v)))
                ),
                use_container_width=True,
            )

            # ── Bar charts ────────────────────────────────────────────────
            st.divider()
            st.subheader("Visual comparison")

            bar_metrics = [
                ("revenue", "Revenue ($M)"),
                ("net_income", "Net Income ($M)"),
                ("eps_diluted", "Diluted EPS ($)"),
                ("total_assets", "Total Assets ($M)"),
            ]
            bar_cols = st.columns(2)
            for i, (key, label) in enumerate(bar_metrics):
                chart_data = {
                    e.ticker: e.metrics.get(key)
                    for e in result.extractions
                    if e.metrics.get(key) is not None
                }
                if chart_data:
                    with bar_cols[i % 2]:
                        st.caption(label)
                        st.bar_chart(pd.DataFrame.from_dict(
                            {"Value": chart_data}
                        ))

            # ── Comparative report ────────────────────────────────────────
            st.divider()
            st.subheader("Comparative analysis")
            st.markdown(result.report)

# ── Tab 4: Investment Intelligence ───────────────────────────────────────────

with tab_intel:
    st.subheader("🧠 Investment Intelligence")

    news_json = st.session_state.news_summary_json
    market_json = st.session_state.market_summary_json
    intel_ticker = st.session_state.ticker

    if not intel_ticker:
        st.info("Run an analysis from the sidebar to see live market data and news here.")
    else:
        st.caption(f"Live data for **{intel_ticker}** fetched alongside the most recent analysis run.")

        # ── Market data ───────────────────────────────────────────────────
        st.markdown("### 📉 Market Data")
        if not market_json:
            st.warning("Market data unavailable — check that FINNHUB_API_KEY is set.")
        else:
            mkt = json.loads(market_json)
            quote = mkt.get("quote")
            ratios = mkt.get("ratios")

            if quote:
                price = quote.get("price") or 0
                change_pct = quote.get("change_pct") or 0
                volume = quote.get("volume") or 0

                col_p, col_c, col_v = st.columns(3)
                col_p.metric("Current Price", f"${price:,.2f}" if price else "—")
                col_c.metric("Intraday Change", f"{change_pct:+.2f}%" if change_pct else "—",
                             delta=f"{change_pct:+.2f}%" if change_pct else None)
                col_v.metric("Volume", f"{volume:,}" if volume else "—")

            if ratios:
                st.divider()
                col_pe, col_pb, col_b, col_hi, col_lo = st.columns(5)
                pe = ratios.get("pe_ratio")
                pb = ratios.get("pb_ratio")
                beta = ratios.get("beta")
                hi = ratios.get("week_52_high")
                lo = ratios.get("week_52_low")
                if pe:
                    pe_label = f"{pe:.1f}x"
                    pe_help = "Computed from current price ÷ EPS (diluted) from the most recent 10-K filing" if mkt.get("pe_computed") else None
                else:
                    xbrl_eps = st.session_state.metrics.get("eps_diluted")
                    if xbrl_eps is not None and xbrl_eps < 0:
                        pe_label = "N/M (loss)"
                        pe_help = "P/E not meaningful — company reported a net loss in the most recent filing"
                    else:
                        pe_label = "—"
                        pe_help = None
                col_pe.metric("P/E Ratio", pe_label, help=pe_help)
                col_pb.metric("P/B Ratio", f"{pb:.1f}x" if pb else "—")
                col_b.metric("Beta", f"{beta:.2f}" if beta else "—")
                col_hi.metric("52W High", f"${hi:,.2f}" if hi else "—")
                col_lo.metric("52W Low", f"${lo:,.2f}" if lo else "—")

                if quote and hi and quote.get("price"):
                    pct_vs_high = round((quote["price"] / hi - 1) * 100, 1)
                    st.caption(f"Current price is **{pct_vs_high:+.1f}%** vs 52-week high.")


            cache_hits = mkt.get("cache_hits", {})
            if cache_hits:
                cached = [k for k, v in cache_hits.items() if v]
                live = [k for k, v in cache_hits.items() if not v]
                parts = []
                if cached:
                    parts.append(f"cached: {', '.join(cached)}")
                if live:
                    parts.append(f"live: {', '.join(live)}")
                st.caption(f"Data sources — {'; '.join(parts)}")

        # ── News & sentiment ──────────────────────────────────────────────
        st.markdown("### 📰 Recent News & Sentiment")
        if not news_json:
            st.warning("News data unavailable — check that FINNHUB_API_KEY is set.")
        else:
            news = json.loads(news_json)
            sentiment_score = news.get("sentiment_score")
            articles = news.get("articles", [])
            date = news.get("date", "")

            # Sentiment gauge
            if sentiment_score is not None:
                label = "Bullish 📈" if sentiment_score >= 0.2 else ("Bearish 📉" if sentiment_score <= -0.2 else "Neutral ➡️")
                col_sl, col_sv = st.columns([3, 1])
                col_sl.progress(
                    int((sentiment_score + 1) / 2 * 100),
                    text=f"Sentiment: **{label}**",
                )
                col_sv.metric("Score", f"{sentiment_score:+.2f}", help="-1.0 = fully bearish, +1.0 = fully bullish")
            else:
                st.caption("Sentiment score not available.")

            # LLM-generated narrative
            narrative = news.get("narrative")
            if narrative:
                st.info(narrative)

            st.caption(f"Showing {len(articles)} article(s) fetched on {date}" + (" _(from cache)_" if news.get("cache_hit") else ""))

            if articles:
                for art in articles[:10]:
                    with st.expander(art.get("headline", "No headline"), expanded=False):
                        summary = art.get("summary", "")
                        if summary:
                            st.write(summary)
                        url = art.get("url", "")
                        source = art.get("source", "")
                        pub = art.get("published_at", "")
                        meta = " · ".join(filter(None, [source, pub[:10] if pub else ""]))
                        if meta:
                            st.caption(meta)
                        if url:
                            st.markdown(f"[Read full article →]({url})")
            else:
                st.info(f"No recent news articles found for {intel_ticker}.")

        # ── Intelligence synthesis ────────────────────────────────────────
        st.divider()
        st.markdown("### 🔍 Synthesis")
        st.caption("The Investment Intelligence section of the analysis report is reproduced below.")
        report = st.session_state.report
        if report:
            snippet = _extract_intelligence_section(report)
            if snippet:
                st.markdown(snippet)
            else:
                st.info("No Investment Intelligence section found in this report. Re-run the analysis with FINNHUB_API_KEY set to generate one.")
        else:
            st.info("Run an analysis to see the investment intelligence synthesis.")
