"""
F1 Copilot — Streamlit Chat Interface
Run: streamlit run ui/app.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import re
import json
import traceback
import streamlit as st
from typing import Optional, Tuple

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from src.agents.f1_agent import stream_query
from src.data.ingestion import ingest_race_session
from src.data.vectorstore import get_collection_stats
from src.agents.tools import get_cached_session
from src.data import fastf1_client as ff1
from ui.charts import (
    speed_trace_chart,
    sector_delta_chart,
    lap_time_chart,
    multi_driver_pace_chart,
    tire_strategy_chart,
    weather_chart,
)


# ── Constants ─────────────────────────────────────────────────────────────────
F1_GPS = [
    "Bahrain", "Saudi Arabia", "Australia", "Japan", "China", "Miami",
    "Imola", "Monaco", "Canada", "Spain", "Austria", "Great Britain",
    "Hungary", "Belgium", "Netherlands", "Italy", "Azerbaijan",
    "Singapore", "Austin", "Mexico", "Brazil", "Las Vegas", "Qatar", "Abu Dhabi",
    "Silverstone", "Monza",
]

SESSION_TYPES = {
    "Race": "R",
    "Qualifying": "Q",
    "Practice 1": "FP1",
    "Practice 2": "FP2",
    "Practice 3": "FP3",
    "Sprint": "S",
    "Sprint Qualifying": "SQ",
}

TOOL_META = {
    "get_telemetry":       ("📡", "Pulling lap telemetry"),
    "compare_telemetry":   ("⚖️", "Comparing driver telemetry"),
    "get_sector_times":    ("⏱️", "Analysing sector times"),
    "get_tire_data":       ("🛞", "Fetching tire strategy"),
    "get_lap_times_series":("📈", "Loading lap time series"),
    "compare_race_pace":   ("🏁", "Comparing race pace"),
    "get_weather":         ("🌡️", "Getting weather data"),
    "get_race_results":    ("🏆", "Fetching race results"),
    "search_race_context": ("🔍", "Searching knowledge base"),
}


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="F1 Copilot",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  [data-testid="stAppViewContainer"] { background: #0a0a0a; }
  [data-testid="stSidebar"] { background: #111111; border-right: 1px solid #222; }
  .f1-header {
    display: flex; align-items: center; gap: 16px;
    border-bottom: 2px solid #e8002d; padding-bottom: 12px; margin-bottom: 20px;
  }
  .f1-title { font-size: 2rem; font-weight: 900; color: #e8002d; letter-spacing: -1px; margin: 0; }
  .f1-sub { color: #666; font-size: 0.8rem; font-family: monospace; margin: 0; }
  .tool-pill {
    display: inline-flex; align-items: center; gap: 5px;
    padding: 3px 10px; border-radius: 20px; font-size: 0.72rem; font-weight: 600;
    background: #1a0a0a; color: #e8002d; border: 1px solid #3a0a0a; margin: 2px;
  }
  .chart-card {
    background: #111; border: 1px solid #222; border-radius: 8px;
    padding: 4px; margin-top: 10px;
  }
  .insight-box {
    background: #0f1a0f; border-left: 3px solid #39b54a;
    padding: 10px 14px; border-radius: 4px; margin-top: 8px;
    font-size: 0.85rem; color: #b0d0b0;
  }
  div[data-testid="stChatMessage"] { background: #111 !important; border-radius: 8px; margin-bottom: 6px; }
  .stButton > button { border-radius: 6px; font-size: 0.82rem; }
  .stButton > button:hover { border-color: #e8002d !important; color: #e8002d !important; }
  .metric-label { font-size: 0.7rem; color: #666; text-transform: uppercase; letter-spacing: 1px; }
  .metric-value { font-size: 1.4rem; font-weight: 700; color: #f0f0f0; }
</style>
""", unsafe_allow_html=True)


# ── Session state ─────────────────────────────────────────────────────────────
for key, default in [
    ("messages", []),
    ("tool_log", []),
    ("charts_per_turn", {}),
    ("ingested_races", set()),
    ("current_race", None),
    ("turn_count", 0),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ── Helpers ───────────────────────────────────────────────────────────────────
def detect_race(question: str) -> Optional[Tuple[int, str]]:
    year_match = re.search(r"\b(20\d{2})\b", question)
    if not year_match:
        return None
    year = int(year_match.group(1))
    q_lower = question.lower()
    for gp in F1_GPS:
        if gp.lower() in q_lower:
            return year, gp
    return None


def race_key(year: int, gp: str) -> str:
    return f"{year}_{gp}_R"


def maybe_auto_ingest(question: str) -> Optional[str]:
    """If the question mentions a race not yet in session cache, auto-ingest it."""
    detected = detect_race(question)
    if not detected:
        return None
    year, gp = detected
    key = race_key(year, gp)
    if key in st.session_state.ingested_races:
        return None
    return f"{year} {gp}"


def render_charts_for_turn(turn_id: int, tool_calls: list[dict]):
    """After agent responds, pull chart data from cached sessions and render."""
    charts_rendered = False

    for tc in tool_calls:
        tool = tc.get("tool")
        args = tc.get("args", {})

        year = args.get("year")
        gp = args.get("grand_prix")
        stype = args.get("session_type", "R")

        if not year or not gp:
            continue

        try:
            session = get_cached_session(year, gp, stype)
        except Exception:
            continue

        if tool == "compare_telemetry":
            da, db = args.get("driver_a"), args.get("driver_b")
            if da and db:
                try:
                    data = ff1.compare_drivers_telemetry(session, da, db)
                    fig = speed_trace_chart(data)
                    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
                    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
                    st.markdown("</div>", unsafe_allow_html=True)
                    charts_rendered = True
                except Exception:
                    pass

        if tool == "get_sector_times":
            da, db = args.get("driver_a"), args.get("driver_b")
            if da and db:
                try:
                    data = ff1.get_sector_analysis(session, da, db)
                    fig = sector_delta_chart(data)
                    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
                    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
                    st.markdown("</div>", unsafe_allow_html=True)
                    charts_rendered = True
                except Exception:
                    pass

        if tool == "get_tire_data":
            driver = args.get("driver")
            if driver:
                try:
                    strategy = ff1.get_tire_strategy(session, driver)
                    fig_s = tire_strategy_chart(strategy)
                    if fig_s:
                        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
                        st.plotly_chart(fig_s, use_container_width=True, config={"displayModeBar": False})
                        st.markdown("</div>", unsafe_allow_html=True)

                    lap_data = ff1.get_lap_times_series(session, driver)
                    fig_l = lap_time_chart(lap_data)
                    if fig_l:
                        st.plotly_chart(fig_l, use_container_width=True, config={"displayModeBar": False})
                    charts_rendered = True
                except Exception:
                    pass

        if tool == "get_lap_times_series":
            driver = args.get("driver")
            if driver:
                try:
                    lap_data = ff1.get_lap_times_series(session, driver)
                    fig = lap_time_chart(lap_data)
                    if fig:
                        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
                        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
                        st.markdown("</div>", unsafe_allow_html=True)
                        charts_rendered = True
                except Exception:
                    pass

        if tool == "compare_race_pace":
            drivers_str = args.get("drivers", "")
            if drivers_str:
                try:
                    driver_list = [d.strip() for d in drivers_str.split(",")]
                    pace_data = ff1.get_multi_driver_lap_times(session, driver_list)
                    fig = multi_driver_pace_chart(pace_data)
                    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
                    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
                    st.markdown("</div>", unsafe_allow_html=True)
                    charts_rendered = True
                except Exception:
                    pass

        if tool == "get_weather":
            try:
                w_data = ff1.get_session_weather(session)
                if "error" not in w_data:
                    fig = weather_chart(w_data)
                    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
                    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
                    st.markdown("</div>", unsafe_allow_html=True)
                    charts_rendered = True
            except Exception:
                pass

    return charts_rendered


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏎️ F1 Copilot")
    st.caption("GPT-4o · LangGraph · FastF1 · Pinecone")
    st.divider()

    st.markdown("### Ingest Race into RAG")
    ingest_year = st.number_input("Year", 2018, 2025, 2024, step=1, label_visibility="visible")
    ingest_stype_label = st.selectbox("Session", list(SESSION_TYPES.keys()))
    ingest_stype = SESSION_TYPES[ingest_stype_label]
    ingest_gp = st.selectbox("Grand Prix", F1_GPS, index=None, placeholder="Select a Grand Prix")

    if st.button("Pull & Ingest", type="primary", use_container_width=True):
        if not ingest_gp:
            st.warning("Select a Grand Prix.")
        else:
            with st.spinner(f"Loading {ingest_year} {ingest_gp}..."):
                try:
                    summary = ingest_race_session(int(ingest_year), ingest_gp, ingest_stype)
                    st.session_state.ingested_races.add(race_key(int(ingest_year), ingest_gp))
                    st.session_state.current_race = (int(ingest_year), ingest_gp)
                    st.success("Ingested!")
                    with st.expander("Summary"):
                        st.write(summary)
                except Exception as e:
                    st.error(str(e))
                    st.code(traceback.format_exc())

    st.divider()

    # Debug tool
    if st.button("🔧 Debug FastF1", use_container_width=True):
        import fastf1
        st.write(f"FastF1 version: `{fastf1.__version__}`")
        try:
            from pathlib import Path
            Path("/tmp/f1_cache").mkdir(parents=True, exist_ok=True)
            fastf1.Cache.enable_cache("/tmp/f1_cache")
            st.success("Step 1: Cache OK")
        except Exception as e:
            st.error(f"Step 1 FAIL: {e}")
        try:
            session = fastf1.get_session(2024, "Bahrain", "R")
            st.success("Step 2: Session object created")
        except Exception as e:
            st.error(f"Step 2 FAIL: {e}")
            st.stop()
        try:
            session.load(laps=True, telemetry=False, weather=True, messages=False)
            st.success("Step 3: session.load() complete")
        except Exception as e:
            st.error(f"Step 3 FAIL: {e}")
            st.code(traceback.format_exc())
            st.stop()
        try:
            r = len(session.results)
            st.success(f"Step 4: Results loaded — {r} drivers")
        except Exception as e:
            st.error(f"Step 4 FAIL: {e}")
        try:
            l = len(session.laps)
            st.success(f"Step 5: Laps loaded — {l} laps")
        except Exception as e:
            st.error(f"Step 5 FAIL: {e}")

    st.divider()

    # KB stats
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("KB Stats", use_container_width=True):
            try:
                s = get_collection_stats()
                st.info(f"{s['total_vectors']} chunks")
            except Exception as e:
                st.error(str(e))
    with col_b:
        if st.button("Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.tool_log = []
            st.session_state.charts_per_turn = {}
            st.session_state.turn_count = 0
            st.rerun()

    st.divider()
    st.markdown("### Try These")

    example_groups = {
        "Driver Comparison": [
            "Why was Norris slower than Verstappen in Bahrain 2024?",
            "Compare Hamilton and Russell race pace in Great Britain 2024",
        ],
        "Strategy": [
            "What tire strategy did Verstappen use in Monaco 2024?",
            "Compare Leclerc and Sainz tire strategies in Monaco 2024",
        ],
        "Race Overview": [
            "Who won the 2024 Singapore Grand Prix and how?",
            "What happened in the 2023 Abu Dhabi Grand Prix?",
        ],
        "Weather & Conditions": [
            "How did weather affect the 2023 Singapore Grand Prix?",
        ],
    }

    for group, examples in example_groups.items():
        with st.expander(group):
            for ex in examples:
                if st.button(ex, use_container_width=True, key=f"ex_{ex[:25]}"):
                    st.session_state["pending_question"] = ex

    st.divider()
    st.markdown(
        "<div style='font-size:0.7rem;color:#444;'>Memory: conversation history is preserved across turns</div>",
        unsafe_allow_html=True
    )


# ── Main header ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="f1-header">
  <div>
    <p class="f1-title">🏎️ F1 COPILOT</p>
    <p class="f1-sub">AGENTIC AI · FASTF1 TELEMETRY · RAG · LANGGRAPH · PINECONE</p>
  </div>
</div>
""", unsafe_allow_html=True)

chat_col, activity_col = st.columns([3, 1])

with chat_col:
    # Render history with charts
    for i, msg in enumerate(st.session_state.messages):
        avatar = "🏎️" if msg["role"] == "assistant" else "👤"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])
            # Re-render charts for this turn
            if msg["role"] == "assistant":
                turn_calls = st.session_state.charts_per_turn.get(i, [])
                if turn_calls:
                    render_charts_for_turn(i, turn_calls)

    # Input
    pending = st.session_state.pop("pending_question", None)
    user_input = st.chat_input("Ask anything about F1 — races, drivers, strategy, telemetry...") or pending

    if user_input:
        # Auto-ingest detection
        auto_race = maybe_auto_ingest(user_input)

        # Track/reuse race context: if the question names a race, remember it;
        # otherwise fall back to the last ingested/discussed race.
        detected = detect_race(user_input)
        if detected:
            st.session_state.current_race = detected

        query_for_agent = user_input
        if not detected and st.session_state.current_race:
            ctx_year, ctx_gp = st.session_state.current_race
            query_for_agent = (
                f"(Context: currently discussing the {ctx_year} {ctx_gp} Grand Prix.) {user_input}"
            )

        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user", avatar="👤"):
            st.markdown(user_input)

        with st.chat_message("assistant", avatar="🏎️"):

            # Auto-ingest banner
            if auto_race:
                ingest_slot = st.empty()
                ingest_slot.info(f"Detected **{auto_race}** — ingesting into knowledge base first...")
                try:
                    yr_str, gp_str = auto_race.split(" ", 1)
                    ingest_race_session(int(yr_str), gp_str, "R")
                    st.session_state.ingested_races.add(race_key(int(yr_str), gp_str))
                    st.session_state.current_race = (int(yr_str), gp_str)
                    ingest_slot.success(f"{auto_race} ingested into RAG ✓")
                except Exception:
                    ingest_slot.empty()

            status_slot = st.empty()
            answer_slot = st.empty()

            active_tools: list[str] = []
            turn_tool_calls: list[dict] = []
            final_answer = ""

            for event in stream_query(query_for_agent, history=st.session_state.messages[:-1]):
                if event["type"] == "tool_call":
                    tool_name = event["tool"]
                    active_tools.append(tool_name)
                    turn_tool_calls.append(event)
                    st.session_state.tool_log.append(event)

                    icon, label = TOOL_META.get(tool_name, ("⚙️", tool_name))
                    pills = " ".join(
                        f'<span class="tool-pill">{TOOL_META.get(t, ("⚙️",""))[0]} {t}</span>'
                        for t in active_tools
                    )
                    status_slot.markdown(
                        f"<div style='color:#666;font-size:0.8rem;margin-bottom:6px'>"
                        f"{icon} {label}...</div>{pills}",
                        unsafe_allow_html=True,
                    )

                elif event["type"] == "answer":
                    status_slot.empty()
                    final_answer = event["content"]
                    answer_slot.markdown(final_answer)

                elif event["type"] == "done":
                    status_slot.empty()

            # Store turn data and render charts
            turn_idx = len(st.session_state.messages)
            st.session_state.charts_per_turn[turn_idx] = turn_tool_calls
            if turn_tool_calls:
                render_charts_for_turn(turn_idx, turn_tool_calls)

            if final_answer:
                st.session_state.messages.append(
                    {"role": "assistant", "content": final_answer}
                )

        st.session_state.turn_count += 1


with activity_col:
    st.markdown("#### Agent Activity")

    if not st.session_state.tool_log:
        st.caption("Tool calls appear here as the agent works.")
    else:
        recent = list(reversed(st.session_state.tool_log[-15:]))
        for tc in recent:
            t = tc.get("tool", "")
            icon = TOOL_META.get(t, ("⚙️", ""))[0]
            with st.expander(f"{icon} {t}", expanded=False):
                st.json(tc.get("args", {}))

    st.divider()
    st.markdown("#### Session Stats")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f'<div class="metric-label">Questions</div><div class="metric-value">{st.session_state.turn_count}</div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-label">Tool Calls</div><div class="metric-value">{len(st.session_state.tool_log)}</div>', unsafe_allow_html=True)

    st.divider()
    st.markdown("#### How It Works")
    steps = [
        ("1", "You ask a question"),
        ("2", "Agent decides which tools to call"),
        ("3", "FastF1 returns live telemetry"),
        ("4", "Pinecone retrieves race context"),
        ("5", "GPT-4o synthesises answer + charts render"),
    ]
    for num, step in steps:
        st.markdown(
            f'<div style="font-size:0.78rem;color:#888;padding:2px 0">'
            f'<span style="color:#e8002d;font-weight:700">{num}.</span> {step}</div>',
            unsafe_allow_html=True
        )

    st.divider()
    st.markdown(
        '<div style="font-size:0.65rem;color:#333;text-align:center">Charts auto-render based on<br>which tools the agent calls</div>',
        unsafe_allow_html=True
    )
