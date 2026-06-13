"""LangGraph-compatible tool definitions for the F1 agent."""

import json
import os
from typing import Annotated, Optional

from langchain_core.tools import tool

from ..data import fastf1_client as ff1
from ..data.vectorstore import search


# Session cache so repeated tool calls in one conversation don't re-download data
_session_cache: dict = {}


def _get_session(year: int, grand_prix: str, session_type: str = "R"):
    key = f"{year}_{grand_prix}_{session_type}"
    if key not in _session_cache:
        _session_cache[key] = ff1.load_session(year, grand_prix, session_type)
    return _session_cache[key]


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
    Use this to retrieve narrative context, historical comparisons, and analyst commentary.
    """
    try:
        results = search(query, n_results=n_results)
        if not results:
            return json.dumps({"message": "No relevant context found in knowledge base."})
        return json.dumps(
            [{"text": r["text"], "source": r["metadata"]} for r in results],
            indent=2,
        )
    except Exception as e:
        return json.dumps({"error": str(e)})


ALL_TOOLS = [
    get_telemetry,
    compare_telemetry,
    get_sector_times,
    get_tire_data,
    get_weather,
    get_race_results,
    search_race_context,
]
