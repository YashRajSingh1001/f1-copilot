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
from src.agents.tools import get_cached_session, peek_cached_session
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

TRACE_LABELS = {
    "get_telemetry": "Get telemetry",
    "compare_telemetry": "Compare telemetry",
    "get_sector_times": "Sector times",
    "get_tire_data": "Tire data",
    "get_lap_times_series": "Lap times",
    "compare_race_pace": "Race pace",
    "get_weather": "Weather",
    "get_race_results": "Race results",
    "search_race_context": "Knowledge search",
}

# Approximate 2024-era team colors, keyed by 3-letter driver code.
DRIVER_COLORS = {
    "VER": "#3671c6", "PER": "#3671c6",
    "NOR": "#ff8000", "PIA": "#ff8000",
    "LEC": "#e10600", "SAI": "#e10600",
    "HAM": "#27f4d2", "RUS": "#27f4d2",
    "ALO": "#00594f", "STR": "#00594f",
    "GAS": "#0090ff", "OCO": "#0090ff",
    "TSU": "#6692ff", "RIC": "#6692ff",
    "ALB": "#005aff", "SAR": "#005aff",
    "BOT": "#52e252", "ZHO": "#52e252",
    "HUL": "#b6babd", "MAG": "#b6babd",
}


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="F1 Copilot",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Titillium+Web:ital,wght@0,400;0,600;0,700;0,900;1,700;1,900&family=JetBrains+Mono:wght@400;500;700&display=swap">
<style>
  @keyframes pcpulse { 0%,100% { opacity: 1; } 50% { opacity: 0.35; } }
  html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
    background:
      repeating-linear-gradient(135deg, rgba(255,255,255,0.015) 0px, rgba(255,255,255,0.015) 1px, transparent 1px, transparent 5px),
      radial-gradient(ellipse at top left, rgba(255,30,45,0.06), transparent 55%),
      #0a0a0c !important;
    font-family: 'Titillium Web', sans-serif;
    color: #f4f5f6;
  }
  [data-testid="stHeader"] { background: transparent; }
  [data-testid="stToolbar"] { right: 1rem; }
  /* ── Sidebar shell ── */
  [data-testid="stSidebar"] {
    background: #0d0e10 !important;
    border-right: 1px solid #2a2c31;
  }
  [data-testid="stSidebar"] hr { border-color: #2a2c31 !important; margin: 14px 0 !important; }
  [data-testid="stSidebar"] label p { color: #6f7278 !important; font-size: 11px !important; }
  .f1-mono-label {
    font-family: 'JetBrains Mono', monospace; font-size: 11px; letter-spacing: 0.1em;
    text-transform: uppercase; color: #6f7278; margin-bottom: 6px;
  }
  .f1-logo-row { display:flex; align-items:center; gap:10px; margin-bottom: 2px; }
  .f1-logo-mark { width:22px; height:26px; background:#ff1e2d; transform: skewX(-10deg);
    box-shadow: 0 0 14px rgba(255,30,45,0.5); flex-shrink:0; }
  .f1-logo-word { font-weight:900; font-style:italic; font-size:18px; transform: skewX(-3deg); display:inline-block; }
  .f1-session-name { font-weight:900; font-style:italic; font-size:20px; margin: 2px 0 12px; transform: skewX(-2deg); }
  .f1-stat-grid { display:grid; grid-template-columns:1fr 1fr; gap:12px 16px; font-family:'JetBrains Mono',monospace; }
  .f1-stat-label { font-size:10px; color:#6f7278; text-transform:uppercase; letter-spacing:0.08em; }
  .f1-stat-value { font-size:14px; font-weight:700; color:#f4f5f6; }
  .f1-example-cat { font-size:11px; font-weight:700; color:#ff1e2d; text-transform:uppercase;
    letter-spacing:0.04em; margin: 12px 0 6px; }
  .f1-divider { border-top:1px solid #2a2c31; margin: 14px 0; }
  /* ── Buttons: hard edges everywhere ── */
  .stButton>button, .stDownloadButton>button {
    border-radius: 0 !important;
    font-family: 'Titillium Web', sans-serif;
  }
  .stButton>button[kind="primary"] {
    background: #ff1e2d !important; color: #0a0a0c !important; border: none !important;
    font-weight: 700; font-size: 12px; letter-spacing: 0.03em;
    clip-path: polygon(8px 0, 100% 0, calc(100% - 8px) 100%, 0 100%);
  }
  .stButton>button[kind="primary"]:hover { background: #ff444f !important; }
  .stButton>button:not([kind="primary"]) {
    background: #16181c; border: 1px solid #2a2c31 !important; color: #8d9096;
  }
  .stButton>button:not([kind="primary"]):hover {
    border-color: #ff1e2d !important; color: #ff1e2d !important;
  }
  .stButton>button:focus-visible {
    box-shadow: 0 0 0 2px rgba(255,30,45,0.4) !important;
  }
  div[class*="st-key-try-asking-block"] .stButton>button:not([kind="primary"]) {
    text-align: left; justify-content: flex-start; color: #c2c4c8;
  }
  div[class*="st-key-try-asking-block"] .stButton>button:not([kind="primary"]):hover {
    color: #f4f5f6 !important;
  }
  /* ── Inputs ── */
  [data-testid="stNumberInput"] input,
  [data-testid="stTextInput"] input,
  [data-baseweb="select"] > div,
  [data-baseweb="base-input"] {
    background: #16181c !important; border-color: #2a2c31 !important;
    color: #f4f5f6 !important; border-radius: 0 !important;
    font-family: 'JetBrains Mono', monospace; font-size: 13px;
  }
  [data-testid="stNumberInputContainer"] button { background: #16181c !important; border-color: #2a2c31 !important; }
  ul[data-testid="stSelectboxVirtualDropdown"] { background: #16181c !important; }
  [data-baseweb="popover"] { background: #16181c !important; }
  [data-baseweb="menu"] { background: #16181c !important; }
  /* ── Alerts (ingest banner) ── */
  [data-testid="stAlert"] { background: #101114 !important; border-radius: 0 !important; }
  [data-testid="stExpander"] { background: #101114; border: 1px solid #2a2c31 !important; border-radius: 0 !important; }
  /* ── Top bar ── */
  .f1-topbar {
    height: 56px; border-bottom: 1px solid #2a2c31; box-shadow: 0 1px 0 rgba(255,30,45,0.2);
    background: #101114; display:flex; align-items:center; justify-content:space-between;
    padding: 0 20px; margin: -1rem -1rem 24px -1rem; gap:16px;
  }
  .f1-topbar-left {
    font-family:'JetBrains Mono',monospace; font-size:12px; color:#8d9096; letter-spacing:0.04em;
    white-space:nowrap; overflow:hidden; text-overflow:ellipsis; min-width:0;
  }
  .f1-live-pill {
    font-family:'JetBrains Mono',monospace; font-size:11px; letter-spacing:0.06em;
    border:1px solid #ff1e2d; color:#ff1e2d; padding:5px 12px;
    display:flex; align-items:center; gap:7px; white-space:nowrap; flex-shrink:0;
  }
  .f1-live-dot { width:6px; height:6px; border-radius:50%; background:#ff1e2d;
    animation: pcpulse 1.6s ease-in-out infinite; flex-shrink:0; }
  /* ── Chat turn text ── */
  .f1-you-label {
    font-family:'JetBrains Mono',monospace; font-size:11px; text-transform:uppercase;
    color:#6f7278; letter-spacing:0.1em; margin-bottom:2px;
  }
  div[class*="st-key-turn-user-"] [data-testid="stMarkdownContainer"] p {
    font-size:15px; line-height:1.5; color:#eceeef; margin:0;
  }
  div[class*="st-key-turn-assistant-"] [data-testid="stMarkdownContainer"] p {
    font-size:15px; line-height:1.6; color:#eceeef;
  }
  div[class*="st-key-turn-assistant-"] [data-testid="stMarkdownContainer"] code {
    font-family:'JetBrains Mono',monospace; font-weight:700; color:#f4f5f6;
    background:transparent; padding:0;
  }
  /* ── Tool trace ── */
  .f1-trace { display:flex; align-items:center; flex-wrap:wrap; row-gap:8px; margin: 6px 0 14px; }
  .f1-trace-step { display:flex; align-items:center; gap:8px; }
  .f1-trace-num {
    width:24px; height:24px; border:2px solid #ff1e2d; display:flex; align-items:center;
    justify-content:center; font-family:'JetBrains Mono',monospace; font-size:11px;
    font-weight:800; color:#ff1e2d; flex-shrink:0;
  }
  .f1-trace-label { font-size:12px; font-weight:600; color:#c2c4c8; }
  .f1-trace-line { width:32px; height:1px; background:#2a2c31; margin:0 12px; }
  /* ── Chart card ── */
  .chart-card { background:#101114; border:1px solid #2a2c31; padding:18px; margin: 6px 0 16px; }
  .f1-chart-head { display:flex; justify-content:space-between; align-items:center; margin-bottom:12px; }
  .f1-chart-title { font-family:'JetBrains Mono',monospace; font-size:11px; letter-spacing:0.06em;
    text-transform:uppercase; color:#6f7278; }
  .f1-legend { display:flex; gap:14px; }
  .f1-legend-item { display:flex; align-items:center; gap:6px; font-family:'JetBrains Mono',monospace; font-size:11px; }
  .f1-legend-swatch { width:10px; height:10px; display:inline-block; }
  /* ── Chat input bar ── */
  [data-testid="stChatInput"] { background: #0d0e10 !important; border-top: 1px solid #2a2c31 !important; }
  [data-testid="stChatInput"] [data-baseweb="textarea"],
  [data-testid="stChatInput"] [data-baseweb="base-input"] {
    background: #16181c !important; border-color: #2a2c31 !important; border-radius: 0 !important;
  }
  [data-testid="stChatInputTextArea"] { color: #f4f5f6 !important; font-family: 'Titillium Web', sans-serif !important; }
  [data-testid="stChatInputSubmitButton"] {
    background: #ff1e2d !important; border-radius: 0 !important; color: #0a0a0c !important;
  }
  [data-testid="stChatInputSubmitButton"]:hover { background: #ff444f !important; }
  [data-testid="stChatInputSubmitButton"]:disabled { background: #2a2c31 !important; }
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
    ("kb_chunks_count", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default


def _refresh_kb_stats():
    try:
        st.session_state.kb_chunks_count = get_collection_stats()["total_vectors"]
    except Exception:
        st.session_state.kb_chunks_count = None


if st.session_state.kb_chunks_count is None:
    _refresh_kb_stats()


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


@st.cache_data(ttl=3600, show_spinner=False)
def _race_winner_snapshot(year: int, gp: str, session_type: str = "R") -> Optional[dict]:
    """Cheap winner lookup — uses the lightweight results-only loader (safe on Cloud)."""
    try:
        session = ff1.load_session_basic(year, gp, session_type)
        results = ff1.get_race_results(session)
        if "error" in results or not results.get("results"):
            return None
        winner = results["results"][0]
        surname = (winner.get("full_name") or winner["driver"]).split()[-1].upper()
        return {"driver": winner["driver"], "team": winner["team"], "surname": surname}
    except Exception:
        return None


def _fmt_laptime(td) -> Optional[str]:
    try:
        total = td.total_seconds()
    except Exception:
        return None
    if total != total:  # NaN
        return None
    m = int(total // 60)
    s = total - m * 60
    return f"{m}:{s:06.3f}"


def _session_extra_snapshot(year: int, gp: str, session_type: str = "R") -> dict:
    """Laps / fastest lap / conditions — only from an already-loaded session (no new network load)."""
    session = peek_cached_session(year, gp, session_type)
    out = {"laps": None, "fastest_lap": None, "conditions": None}
    if session is None:
        return out
    try:
        out["laps"] = int(session.laps["LapNumber"].max())
    except Exception:
        pass
    try:
        fl = session.laps.pick_fastest()
        out["fastest_lap"] = _fmt_laptime(fl["LapTime"])
    except Exception:
        pass
    try:
        w = ff1.get_session_weather(session)
        if "error" not in w:
            cond = "WET" if w.get("rainfall") else "DRY"
            out["conditions"] = f"{cond} · {w['track_temp_c']['avg']:.0f}°C"
    except Exception:
        pass
    return out


_NUM_PATTERN = re.compile(r"(\b\d+:\d{2}\.\d{3}\b|\b\d+\.\d+\s?(?:s|ms)\b|\b\d+\s?ms\b)")


def _highlight_numbers(text: str) -> str:
    return _NUM_PATTERN.sub(lambda m: f'<code>{m.group(0)}</code>', text)


def _trace_html(tool_calls: list) -> str:
    if not tool_calls:
        return ""
    parts = []
    n = len(tool_calls)
    for idx, tc in enumerate(tool_calls, start=1):
        label = TRACE_LABELS.get(tc.get("tool"), tc.get("tool", ""))
        parts.append(
            f'<div class="f1-trace-step"><span class="f1-trace-num">{idx}</span>'
            f'<span class="f1-trace-label">{label}</span></div>'
        )
        if idx != n:
            parts.append('<div class="f1-trace-line"></div>')
    return f'<div class="f1-trace">{"".join(parts)}</div>'


def _chart_card_open(title: str, legend: Optional[list] = None) -> str:
    legend_html = ""
    if legend:
        items = "".join(
            f'<span class="f1-legend-item" style="color:{color}">'
            f'<span class="f1-legend-swatch" style="background:{color}"></span>{code}</span>'
            for code, color in legend
        )
        legend_html = f'<div class="f1-legend">{items}</div>'
    return (
        '<div class="chart-card"><div class="f1-chart-head">'
        f'<span class="f1-chart-title">{title}</span>{legend_html}</div>'
    )


def render_charts_for_turn(turn_id: int, tool_calls: list) -> bool:
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
                    legend = [(da, DRIVER_COLORS.get(da, "#c2c4c8")), (db, DRIVER_COLORS.get(db, "#c2c4c8"))]
                    st.markdown(_chart_card_open(f"Speed trace — {da} vs {db}", legend), unsafe_allow_html=True)
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
                    st.markdown(_chart_card_open(f"Sector deltas — {da} vs {db}"), unsafe_allow_html=True)
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
                        st.markdown(_chart_card_open(f"Tire strategy — {driver}"), unsafe_allow_html=True)
                        st.plotly_chart(fig_s, use_container_width=True, config={"displayModeBar": False})
                        st.markdown("</div>", unsafe_allow_html=True)

                    lap_data = ff1.get_lap_times_series(session, driver)
                    fig_l = lap_time_chart(lap_data)
                    if fig_l:
                        st.markdown(_chart_card_open(f"Lap time progression — {driver}"), unsafe_allow_html=True)
                        st.plotly_chart(fig_l, use_container_width=True, config={"displayModeBar": False})
                        st.markdown("</div>", unsafe_allow_html=True)
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
                        st.markdown(_chart_card_open(f"Lap time progression — {driver}"), unsafe_allow_html=True)
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
                    legend = [(d, DRIVER_COLORS.get(d, "#c2c4c8")) for d in driver_list]
                    st.markdown(_chart_card_open("Race pace comparison", legend), unsafe_allow_html=True)
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
                    st.markdown(_chart_card_open("Weather conditions"), unsafe_allow_html=True)
                    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
                    st.markdown("</div>", unsafe_allow_html=True)
                    charts_rendered = True
            except Exception:
                pass

    return charts_rendered


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        '<div class="f1-logo-row"><div class="f1-logo-mark"></div>'
        '<span class="f1-logo-word">F1 COPILOT</span></div>',
        unsafe_allow_html=True,
    )

    # Current session card
    current_race = st.session_state.current_race
    if current_race:
        cs_year, cs_gp = current_race
        winner_snap = _race_winner_snapshot(cs_year, cs_gp, "R") or {}
        extra_snap = _session_extra_snapshot(cs_year, cs_gp, "R")
        winner_display = winner_snap.get("surname") or "—"
        winner_color = DRIVER_COLORS.get(winner_snap.get("driver", ""), "#f4f5f6")
        laps_val = extra_snap["laps"] if extra_snap["laps"] is not None else "—"
        fastest_val = extra_snap["fastest_lap"] or "—"
        conditions_val = extra_snap["conditions"] or "—"
        session_name = f"{cs_year} {cs_gp.upper()} GP"
    else:
        winner_display, winner_color = "—", "#f4f5f6"
        laps_val = fastest_val = conditions_val = "—"
        session_name = "NO ACTIVE SESSION"

    st.markdown(f"""
    <div class="f1-mono-label">Current session</div>
    <div class="f1-session-name">{session_name}</div>
    <div class="f1-stat-grid">
      <div><div class="f1-stat-label">Winner</div><div class="f1-stat-value" style="color:{winner_color}">{winner_display}</div></div>
      <div><div class="f1-stat-label">Laps</div><div class="f1-stat-value">{laps_val}</div></div>
      <div><div class="f1-stat-label">Fastest lap</div><div class="f1-stat-value" style="color:#b026ff">{fastest_val}</div></div>
      <div><div class="f1-stat-label">Conditions</div><div class="f1-stat-value">{conditions_val}</div></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="f1-divider"></div>', unsafe_allow_html=True)

    # Ingest form
    st.markdown('<div class="f1-mono-label">Ingest a race</div>', unsafe_allow_html=True)
    ingest_year = st.number_input("Year", 2018, 2025, 2024, step=1, label_visibility="collapsed")
    ingest_stype_label = st.selectbox("Session", list(SESSION_TYPES.keys()), label_visibility="collapsed")
    ingest_stype = SESSION_TYPES[ingest_stype_label]
    ingest_gp = st.selectbox("Grand Prix", F1_GPS, index=None, placeholder="Grand Prix", label_visibility="collapsed")

    if st.button("PULL & INGEST", type="primary", use_container_width=True):
        if not ingest_gp:
            st.warning("Select a Grand Prix.")
        else:
            with st.spinner(f"Loading {ingest_year} {ingest_gp}..."):
                try:
                    summary = ingest_race_session(int(ingest_year), ingest_gp, ingest_stype)
                    st.session_state.ingested_races.add(race_key(int(ingest_year), ingest_gp))
                    st.session_state.current_race = (int(ingest_year), ingest_gp)
                    _refresh_kb_stats()
                    st.success("Ingested!")
                    with st.expander("Summary"):
                        st.write(summary)
                except Exception as e:
                    st.error(str(e))
                    st.code(traceback.format_exc())

    st.markdown('<div class="f1-divider"></div>', unsafe_allow_html=True)

    # Try asking
    with st.container(key="try-asking-block"):
        st.markdown('<div class="f1-mono-label">Try asking</div>', unsafe_allow_html=True)

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
            st.markdown(f'<div class="f1-example-cat">{group}</div>', unsafe_allow_html=True)
            for ex in examples:
                if st.button(ex, use_container_width=True, key=f"ex_{ex[:25]}"):
                    st.session_state["pending_question"] = ex

    st.markdown('<div class="f1-divider"></div>', unsafe_allow_html=True)

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

    # Footer: stats + clear chat
    st.markdown('<div class="f1-divider"></div>', unsafe_allow_html=True)
    kb_display = st.session_state.kb_chunks_count if st.session_state.kb_chunks_count is not None else "—"
    st.markdown(f"""
    <div style="display:flex; gap:20px; font-family:'JetBrains Mono',monospace; margin-bottom:12px;">
      <div><div class="f1-stat-label">Questions</div><div style="font-size:19px;font-weight:700;">{st.session_state.turn_count}</div></div>
      <div><div class="f1-stat-label">Tool calls</div><div style="font-size:19px;font-weight:700;">{len(st.session_state.tool_log)}</div></div>
      <div><div class="f1-stat-label">KB chunks</div><div style="font-size:19px;font-weight:700;">{kb_display}</div></div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("CLEAR CHAT", use_container_width=True):
        st.session_state.messages = []
        st.session_state.tool_log = []
        st.session_state.charts_per_turn = {}
        st.session_state.turn_count = 0
        st.rerun()


# ── Top bar ───────────────────────────────────────────────────────────────────
if st.session_state.current_race:
    pill_year, pill_gp = st.session_state.current_race
    pill_text = f"{pill_year} {pill_gp.upper()} GP · RACE"
else:
    pill_text = "NO ACTIVE SESSION"

st.markdown(f"""
<div class="f1-topbar">
  <span class="f1-topbar-left">AGENTIC RAG · GPT-4O · LANGGRAPH · FASTF1</span>
  <span class="f1-live-pill"><span class="f1-live-dot"></span>{pill_text}</span>
</div>
""", unsafe_allow_html=True)


# ── Chat history ──────────────────────────────────────────────────────────────
for i, msg in enumerate(st.session_state.messages):
    if msg["role"] == "user":
        with st.container(key=f"turn-user-{i}"):
            st.markdown('<div class="f1-you-label">You</div>', unsafe_allow_html=True)
            st.markdown(msg["content"])
    else:
        turn_calls = st.session_state.charts_per_turn.get(i, [])
        with st.container(key=f"turn-assistant-{i}"):
            trace = _trace_html(turn_calls)
            if trace:
                st.markdown(trace, unsafe_allow_html=True)
            st.markdown(_highlight_numbers(msg["content"]), unsafe_allow_html=True)
            if turn_calls:
                render_charts_for_turn(i, turn_calls)
        st.markdown('<div class="f1-divider"></div>', unsafe_allow_html=True)

# ── Input ─────────────────────────────────────────────────────────────────────
pending = st.session_state.pop("pending_question", None)
user_input = st.chat_input("Ask anything about F1 — races, drivers, strategy, telemetry...") or pending

if user_input:
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
    with st.container(key=f"turn-user-{len(st.session_state.messages) - 1}"):
        st.markdown('<div class="f1-you-label">You</div>', unsafe_allow_html=True)
        st.markdown(user_input)

    turn_idx = len(st.session_state.messages)
    with st.container(key=f"turn-assistant-{turn_idx}"):

        if auto_race:
            ingest_slot = st.empty()
            ingest_slot.info(f"Detected **{auto_race}** — ingesting into knowledge base first...")
            try:
                yr_str, gp_str = auto_race.split(" ", 1)
                ingest_race_session(int(yr_str), gp_str, "R")
                st.session_state.ingested_races.add(race_key(int(yr_str), gp_str))
                st.session_state.current_race = (int(yr_str), gp_str)
                _refresh_kb_stats()
                ingest_slot.success(f"{auto_race} ingested into RAG ✓")
            except Exception:
                ingest_slot.empty()

        trace_slot = st.empty()
        answer_slot = st.empty()

        active_tools: list = []
        turn_tool_calls: list = []
        final_answer = ""

        for event in stream_query(query_for_agent, history=st.session_state.messages[:-1]):
            if event["type"] == "tool_call":
                turn_tool_calls.append(event)
                st.session_state.tool_log.append(event)
                trace_slot.markdown(_trace_html(turn_tool_calls), unsafe_allow_html=True)

            elif event["type"] == "answer":
                final_answer = event["content"]
                answer_slot.markdown(_highlight_numbers(final_answer), unsafe_allow_html=True)

            elif event["type"] == "done":
                pass

        st.session_state.charts_per_turn[turn_idx] = turn_tool_calls
        if turn_tool_calls:
            render_charts_for_turn(turn_idx, turn_tool_calls)

        if final_answer:
            st.session_state.messages.append({"role": "assistant", "content": final_answer})

    st.session_state.turn_count += 1
    # Rerun so the sidebar "Current session" card and header pill (rendered
    # earlier in script order) immediately reflect this turn's current_race.
    st.rerun()
