"""LangGraph-compatible tool definitions for the F1 agent."""

import json
from typing import Annotated, Optional

from langchain_core.tools import tool

from ..data import fastf1_client as ff1
from ..data.vectorstore import search


_session_cache: dict = {}


def _get_session(year: int, grand_prix: str, session_type: str = "R"):
    key = f"{year}_{grand_prix}_{session_type}"
    if key not in _session_cache:
        _session_cache[key] = ff1.load_session(year, grand_prix, session_type)
    return _session_cache[key]


def get_cached_session(year: int, grand_prix: str, session_type: str = "R"):
    """Public accessor so the UI can reuse already-loaded sessions for charts."""
    return _get_session(year, grand_prix, session_type)


@tool
def get_telemetry(
    year: Annotated[int, "Season year, e.g. 2024"],
    grand_prix: Annotated[str, "Grand Prix name, e.g. 'Bahrain'"],
    driver: Annotated[str, "3-letter driver code, e.g. 'NOR', 'VER'"],
    session_type: Annotated[str, "Session type: R=Race, Q=Qualifying, FP1/FP2/FP3"] = "R",
    lap_number: Annotated[Optional[int], "Specific lap number, or None for fastest lap"] = None,
) -> str:
    """Get detailed lap telemetry for a single driver: speed trace, throttle, braking, gears."""
    try:
        session = _get_session(year, grand_prix, session_type)
        result = ff1.get_driver_lap_telemetry(session, driver, lap_number)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def compare_telemetry(
    year: Annotated[int, "Season year"],
    grand_prix: Annotated[str, "Grand Prix name"],
    driver_a: Annotated[str, "First driver 3-letter code"],
    driver_b: Annotated[str, "Second driver 3-letter code"],
    session_type: Annotated[str, "Session type"] = "R",
) -> str:
    """Compare telemetry between two drivers — speed, throttle, braking on their fastest laps. Returns delta and who was faster."""
    try:
        session = _get_session(year, grand_prix, session_type)
        result = ff1.compare_drivers_telemetry(session, driver_a, driver_b)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def get_sector_times(
    year: Annotated[int, "Season year"],
    grand_prix: Annotated[str, "Grand Prix name"],
    driver_a: Annotated[str, "First driver code"],
    driver_b: Annotated[str, "Second driver code"],
    session_type: Annotated[str, "Session type"] = "R",
) -> str:
    """Compare sector-by-sector times between two drivers. Identifies where time is lost or gained."""
    try:
        session = _get_session(year, grand_prix, session_type)
        result = ff1.get_sector_analysis(session, driver_a, driver_b)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def get_tire_data(
    year: Annotated[int, "Season year"],
    grand_prix: Annotated[str, "Grand Prix name"],
    driver: Annotated[str, "Driver 3-letter code"],
    session_type: Annotated[str, "Session type"] = "R",
) -> str:
    """Get tire compound strategy for a driver: stint lengths, compounds used, degradation rate per lap."""
    try:
        session = _get_session(year, grand_prix, session_type)
        result = ff1.get_tire_strategy(session, driver)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def get_lap_times_series(
    year: Annotated[int, "Season year"],
    grand_prix: Annotated[str, "Grand Prix name"],
    driver: Annotated[str, "Driver 3-letter code"],
    session_type: Annotated[str, "Session type"] = "R",
) -> str:
    """Get lap-by-lap times for a driver across the full race, with tire compound per lap. Used for pace and degradation analysis."""
    try:
        session = _get_session(year, grand_prix, session_type)
        result = ff1.get_lap_times_series(session, driver)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def compare_race_pace(
    year: Annotated[int, "Season year"],
    grand_prix: Annotated[str, "Grand Prix name"],
    drivers: Annotated[str, "Comma-separated driver codes, e.g. 'NOR,VER,LEC'"],
    session_type: Annotated[str, "Session type"] = "R",
) -> str:
    """Compare race pace across multiple drivers over the full race distance."""
    try:
        session = _get_session(year, grand_prix, session_type)
        driver_list = [d.strip() for d in drivers.split(",")]
        result = ff1.get_multi_driver_lap_times(session, driver_list)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def get_weather(
    year: Annotated[int, "Season year"],
    grand_prix: Annotated[str, "Grand Prix name"],
    session_type: Annotated[str, "Session type"] = "R",
) -> str:
    """Get weather conditions during the session: air temp, track temp, humidity, wind, rainfall."""
    try:
        session = _get_session(year, grand_prix, session_type)
        result = ff1.get_session_weather(session)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def get_race_results(
    year: Annotated[int, "Season year"],
    grand_prix: Annotated[str, "Grand Prix name"],
    session_type: Annotated[str, "Session type"] = "R",
) -> str:
    """Get race classification: finishing positions, points, teams, driver status."""
    try:
        session = _get_session(year, grand_prix, session_type)
        result = ff1.get_race_results(session)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def search_race_context(
    query: Annotated[str, "Natural language query about F1 races, drivers, or strategies"],
    n_results: Annotated[int, "Number of context chunks to retrieve"] = 4,
) -> str:
    """
    RAG search over ingested race reports, summaries, and articles.
    Use this for narrative context, historical comparisons, and analyst commentary.
    Gracefully returns empty if knowledge base has no relevant data.
    """
    try:
        results = search(query, n_results=n_results)
        if not results:
            return json.dumps({"message": "No relevant context found in knowledge base — rely on live telemetry tools."})
        return json.dumps(
            [{"text": r["text"], "source": r["metadata"]} for r in results],
            indent=2,
        )
    except Exception:
        return json.dumps({"message": "Knowledge base unavailable — using live telemetry tools only."})


ALL_TOOLS = [
    get_telemetry,
    compare_telemetry,
    get_sector_times,
    get_tire_data,
    get_lap_times_series,
    compare_race_pace,
    get_weather,
    get_race_results,
    search_race_context,
]
