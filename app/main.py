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
            st.metric("Revenue", f"${m['revenue']:,.0f}M")
        if "net_income" in m:
            st.metric("Net income", f"${m['net_income']:,.0f}M")
        if "eps_diluted" in m:
            st.metric("Diluted EPS", f"${m['eps_diluted']:.2f}")
        if "total_assets" in m:
            st.metric("Total assets", f"${m['total_assets']:,.0f}M")
        if "fiscal_year_end" in m:
            st.caption(f"Fiscal year end: {m['fiscal_year_end']}")

    if st.session_state.report and st.button("Clear / new analysis", use_container_width=True):
        st.session_state.messages = []
        st.session_state.report = None
        st.session_state.metrics = {}
        st.session_state.confidence = None
        st.session_state.ticker = None
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

tab_analysis, tab_trends = st.tabs(["📄 Analysis", "📈 Trends"])

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
                    st.error(f"Pipeline exception: {e}")
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

