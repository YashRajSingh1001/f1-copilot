"""
F1 Copilot — Streamlit Chat Interface
Deployable to Streamlit Community Cloud.
Run locally: streamlit run ui/app.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import plotly.graph_objects as go
import json

# Load .env locally (no-op on Streamlit Cloud where st.secrets is used)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from src.agents.f1_agent import stream_query
from src.data.ingestion import ingest_race_session
from src.data.vectorstore import get_collection_stats


# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="F1 Copilot",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  .main-title { font-size: 2.2rem; font-weight: 800; color: #e8002d; letter-spacing: -1px; }
  .subtitle { color: #888; font-size: 0.95rem; margin-top: -12px; margin-bottom: 20px; }
  .tool-badge {
    display: inline-block; padding: 2px 10px; border-radius: 12px;
    font-size: 0.75rem; font-weight: 600; margin: 2px;
    background: #1a1a2e; color: #e8002d; border: 1px solid #e8002d;
  }
  div[data-testid="stChatMessage"] { border-radius: 8px; margin-bottom: 6px; }
</style>
""", unsafe_allow_html=True)


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏎️ F1 Copilot")
    st.caption("Agentic AI · FastF1 · LangGraph · RAG")
    st.divider()

    # ── RAG Ingestion ─────────────────────────────────────────────────────
    st.markdown("### Ingest Race into Knowledge Base")
    col1, col2 = st.columns(2)
    with col1:
        ingest_year = st.number_input("Year", min_value=2018, max_value=2025, value=2024, step=1)
    with col2:
        ingest_session_type = st.selectbox("Session", ["R", "Q", "FP1", "FP2", "FP3"])
    ingest_gp = st.text_input("Grand Prix", placeholder="e.g. Bahrain")

    if st.button("Pull & Ingest", type="primary", use_container_width=True):
        if not ingest_gp.strip():
            st.warning("Enter a Grand Prix name.")
        else:
            with st.spinner(f"Loading {ingest_year} {ingest_gp} {ingest_session_type}..."):
                try:
                    summary = ingest_race_session(
                        int(ingest_year), ingest_gp.strip(), ingest_session_type
                    )
                    st.success("Ingested into Qdrant!")
                    with st.expander("Generated summary"):
                        st.write(summary)
                except Exception as e:
                    st.error(f"Error: {e}")

    st.divider()

    # ── KB Stats ──────────────────────────────────────────────────────────
    if st.button("Knowledge base stats", use_container_width=True):
        try:
            stats = get_collection_stats()
            st.info(f"**{stats['total_vectors']}** chunks in Qdrant")
        except Exception as e:
            st.error(f"Cannot reach Qdrant: {e}")

    st.divider()

    # ── Example questions ─────────────────────────────────────────────────
    st.markdown("### Try these")
    examples = [
        "Why was Norris slower than Verstappen in Bahrain 2024?",
        "Compare Leclerc and Hamilton tire strategies in Monaco 2024",
        "What was the weather in Singapore 2023 and how did it affect the race?",
        "Who had the fastest Sector 2 in Silverstone 2024 qualifying?",
        "Analyse Verstappen's braking telemetry in Abu Dhabi 2023",
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True, key=f"ex_{ex[:20]}"):
            st.session_state["pending_question"] = ex

    st.divider()
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.tool_log = []
        st.rerun()

    st.divider()
    st.markdown("**Stack**")
    st.caption("GPT-4o · LangGraph · FastF1 · Qdrant · Streamlit")


# ── Session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "tool_log" not in st.session_state:
    st.session_state.tool_log = []


# ── Main layout ───────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">🏎️ F1 Copilot</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Ask anything about F1 — the agent pulls live telemetry, tire data, weather & race reports automatically</div>',
    unsafe_allow_html=True,
)

chat_col, activity_col = st.columns([3, 1])

with chat_col:
    # Render history
    for msg in st.session_state.messages:
        avatar = "🏎️" if msg["role"] == "assistant" else "👤"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])

    # Input — sidebar button or typed
    pending = st.session_state.pop("pending_question", None)
    user_input = st.chat_input("Ask about any F1 race, driver, or strategy...") or pending

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user", avatar="👤"):
            st.markdown(user_input)

        with st.chat_message("assistant", avatar="🏎️"):
            status_slot = st.empty()
            answer_slot = st.empty()

            active_tools: list[str] = []
            final_answer = ""

            for event in stream_query(user_input):
                if event["type"] == "tool_call":
                    active_tools.append(event["tool"])
                    st.session_state.tool_log.append(event)
                    badges = " ".join(
                        f'<span class="tool-badge">⚙ {t}</span>' for t in active_tools
                    )
                    status_slot.markdown(
                        f"**Fetching data...**<br>{badges}", unsafe_allow_html=True
                    )

                elif event["type"] == "answer":
                    status_slot.empty()
                    final_answer = event["content"]
                    answer_slot.markdown(final_answer)

                elif event["type"] == "done":
                    status_slot.empty()

            if final_answer:
                st.session_state.messages.append(
                    {"role": "assistant", "content": final_answer}
                )


with activity_col:
    st.markdown("#### Agent Activity")

    if not st.session_state.tool_log:
        st.caption("Tool calls appear here as the agent works through your question.")
    else:
        tool_icons = {
            "get_telemetry": "📡",
            "compare_telemetry": "⚖️",
            "get_sector_times": "⏱️",
            "get_tire_data": "🛞",
            "get_weather": "🌡️",
            "get_race_results": "🏆",
            "search_race_context": "🔍",
        }
        for tc in reversed(st.session_state.tool_log[-12:]):
            icon = tool_icons.get(tc.get("tool", ""), "⚙️")
            with st.expander(f"{icon} {tc.get('tool', '')}", expanded=False):
                st.json(tc.get("args", {}))

    st.divider()
    st.markdown("#### How it works")
    st.markdown("""
1. You ask a question
2. GPT-4o decides which tools to call
3. Agent pulls live FastF1 data
4. RAG retrieves race context
5. GPT-4o synthesises the answer
""")
