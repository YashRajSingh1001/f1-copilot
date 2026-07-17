"""FastF1 wrapper for pulling session telemetry, weather, tire, and results data."""

import fastf1
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional
from ..config import get


def _setup_cache():
    cache_path = Path(get("FASTF1_CACHE_PATH", "/tmp/f1_cache"))
    cache_path.mkdir(exist_ok=True)
    fastf1.Cache.enable_cache(str(cache_path))


_setup_cache()


def load_session(year: int, grand_prix: str, session_type: str = "R") -> fastf1.core.Session:
    """Load an F1 session. session_type: R=Race, Q=Qualifying, FP1/FP2/FP3=Practice."""
    session = fastf1.get_session(year, grand_prix, session_type)
    session.load(laps=True, telemetry=True, weather=True, messages=True)
    return session


def get_driver_lap_telemetry(
    session: fastf1.core.Session,
    driver: str,
    lap_number: Optional[int] = None,
) -> dict:
    """Get per-lap telemetry for a driver. If lap_number is None, returns fastest lap."""
    driver_laps = session.laps.pick_drivers(driver)
    if driver_laps.empty:
        return {"error": f"No laps found for driver {driver}"}

    lap = (
        driver_laps[driver_laps["LapNumber"] == lap_number].iloc[0]
        if lap_number
        else driver_laps.pick_fastest()
    )

    tel = lap.get_car_data().add_distance()

    return {
        "driver": driver,
        "lap_number": int(lap["LapNumber"]),
        "lap_time": str(lap["LapTime"]),
        "compound": lap["Compound"],
        "telemetry_summary": {
            "max_speed_kmh": round(float(tel["Speed"].max()), 1),
            "avg_speed_kmh": round(float(tel["Speed"].mean()), 1),
            "full_throttle_pct": round(
                float((tel["Throttle"] > 95).mean() * 100), 1
            ),
            "braking_pct": round(float((tel["Brake"] == True).mean() * 100), 1),
            "distance_m": round(float(tel["Distance"].max()), 0),
        },
        "speed_trace": tel[["Distance", "Speed"]].round(2).to_dict(orient="records")[::10],
        "throttle_trace": tel[["Distance", "Throttle"]].round(2).to_dict(orient="records")[::10],
        "gear_trace": tel[["Distance", "nGear"]].round(0).to_dict(orient="records")[::10],
    }


def compare_drivers_telemetry(
    session: fastf1.core.Session,
    driver_a: str,
    driver_b: str,
    lap_number_a: Optional[int] = None,
    lap_number_b: Optional[int] = None,
) -> dict:
    """Compare telemetry between two drivers on their fastest (or specified) laps."""
    laps_a = session.laps.pick_drivers(driver_a)
    laps_b = session.laps.pick_drivers(driver_b)

    lap_a = (
        laps_a[laps_a["LapNumber"] == lap_number_a].iloc[0]
        if lap_number_a else laps_a.pick_fastest()
    )
    lap_b = (
        laps_b[laps_b["LapNumber"] == lap_number_b].iloc[0]
        if lap_number_b else laps_b.pick_fastest()
    )

    tel_a = lap_a.get_car_data().add_distance()
    tel_b = lap_b.get_car_data().add_distance()

    def _summary(tel, lap):
        return {
            "lap_time": str(lap["LapTime"]),
            "lap_number": int(lap["LapNumber"]),
            "compound": lap["Compound"],
            "max_speed_kmh": round(float(tel["Speed"].max()), 1),
            "avg_speed_kmh": round(float(tel["Speed"].mean()), 1),
            "full_throttle_pct": round(float((tel["Throttle"] > 95).mean() * 100), 1),
            "braking_pct": round(float((tel["Brake"] == True).mean() * 100), 1),
            "min_speed_kmh": round(float(tel["Speed"].min()), 1),
        }

    time_delta_ms = (
        lap_a["LapTime"].total_seconds() - lap_b["LapTime"].total_seconds()
    ) * 1000

    return {
        "driver_a": {"code": driver_a, **_summary(tel_a, lap_a)},
        "driver_b": {"code": driver_b, **_summary(tel_b, lap_b)},
        "delta_ms": round(time_delta_ms, 0),
        "faster_driver": driver_b if time_delta_ms > 0 else driver_a,
        "gap_seconds": round(abs(time_delta_ms) / 1000, 3),
        "speed_traces": {
            driver_a: tel_a[["Distance", "Speed"]].round(2).to_dict(orient="records")[::10],
            driver_b: tel_b[["Distance", "Speed"]].round(2).to_dict(orient="records")[::10],
        },
    }


def get_tire_strategy(session: fastf1.core.Session, driver: str) -> dict:
    """Get tire compound strategy and stint performance for a driver."""
    laps = session.laps.pick_drivers(driver).copy()
    if laps.empty:
        return {"error": f"No data for driver {driver}"}

    stints = []
    for (compound, stint_num), group in laps.groupby(["Compound", "Stint"]):
        valid_laps = group[group["LapTime"].notna() & group["PitOutTime"].isna()]
        if valid_laps.empty:
            continue

        lap_times_s = valid_laps["LapTime"].dt.total_seconds()
        stints.append({
            "stint": int(stint_num),
            "compound": compound,
            "laps": len(group),
            "start_lap": int(group["LapNumber"].min()),
            "end_lap": int(group["LapNumber"].max()),
            "avg_lap_time_s": round(float(lap_times_s.mean()), 3),
            "best_lap_time_s": round(float(lap_times_s.min()), 3),
            "degradation_s_per_lap": round(
                float(
                    np.polyfit(range(len(lap_times_s)), lap_times_s, 1)[0]
                    if len(lap_times_s) > 1 else 0
                ),
                4,
            ),
        })

    return {"driver": driver, "stints": stints, "total_stints": len(stints)}


def get_session_weather(session: fastf1.core.Session) -> dict:
    """Get weather summary for a session."""
    weather = session.weather_data
    if weather is None or weather.empty:
        return {"error": "No weather data available"}

    return {
        "air_temp_c": {
            "min": round(float(weather["AirTemp"].min()), 1),
            "max": round(float(weather["AirTemp"].max()), 1),
            "avg": round(float(weather["AirTemp"].mean()), 1),
        },
        "track_temp_c": {
            "min": round(float(weather["TrackTemp"].min()), 1),
            "max": round(float(weather["TrackTemp"].max()), 1),
            "avg": round(float(weather["TrackTemp"].mean()), 1),
        },
        "humidity_pct": round(float(weather["Humidity"].mean()), 1),
        "wind_speed_ms": round(float(weather["WindSpeed"].mean()), 2),
        "rainfall": bool(weather["Rainfall"].any()),
    }


def get_race_results(session: fastf1.core.Session) -> dict:
    """Get final race classification."""
    results = session.results
    if results is None or results.empty:
        return {"error": "No results available"}

    top_results = []
    for _, row in results.head(20).iterrows():
        top_results.append({
            "position": int(row["Position"]) if pd.notna(row["Position"]) else None,
            "driver": row["Abbreviation"],
            "full_name": f"{row['FirstName']} {row['LastName']}",
            "team": row["TeamName"],
            "points": float(row["Points"]) if pd.notna(row["Points"]) else 0,
            "status": row["Status"],
        })

    return {
        "event": str(session.event["EventName"]),
        "year": int(session.event["EventDate"].year),
        "results": top_results,
    }


def get_lap_times_series(session: fastf1.core.Session, driver: str) -> dict:
    """Get lap-by-lap times with compound and stint info — used for degradation charts."""
    laps = session.laps.pick_drivers(driver).copy()
    if laps.empty:
        return {"error": f"No data for driver {driver}"}

    records = []
    for _, lap in laps.iterrows():
        if pd.notna(lap["LapTime"]) and pd.notna(lap["LapNumber"]):
            records.append({
                "lap": int(lap["LapNumber"]),
                "time_s": round(float(lap["LapTime"].total_seconds()), 3),
                "compound": str(lap.get("Compound", "UNKNOWN")),
                "stint": int(lap.get("Stint", 0)),
                "is_pit_out": pd.notna(lap.get("PitOutTime")) and not pd.isna(lap.get("PitOutTime")),
            })

    return {"driver": driver, "laps": records}


def get_multi_driver_lap_times(
    session: fastf1.core.Session, drivers: list[str]
) -> dict:
    """Get lap times for multiple drivers — used for race pace comparison charts."""
    return {drv: get_lap_times_series(session, drv) for drv in drivers}


def get_sector_analysis(
    session: fastf1.core.Session, driver_a: str, driver_b: str
) -> dict:
    """Compare sector times between two drivers on their fastest laps."""
    laps_a = session.laps.pick_drivers(driver_a).pick_fastest()
    laps_b = session.laps.pick_drivers(driver_b).pick_fastest()

    def _sectors(lap):
        return {
            "s1": str(lap["Sector1Time"]),
            "s2": str(str(lap["Sector2Time"])),
            "s3": str(lap["Sector3Time"]),
            "s1_s": round(lap["Sector1Time"].total_seconds(), 3) if pd.notna(lap["Sector1Time"]) else None,
            "s2_s": round(lap["Sector2Time"].total_seconds(), 3) if pd.notna(lap["Sector2Time"]) else None,
            "s3_s": round(lap["Sector3Time"].total_seconds(), 3) if pd.notna(lap["Sector3Time"]) else None,
        }

    s_a = _sectors(laps_a)
    s_b = _sectors(laps_b)

    deltas = {}
    for s in ["s1_s", "s2_s", "s3_s"]:
        if s_a[s] and s_b[s]:
            deltas[s.replace("_s", "_delta_ms")] = round((s_a[s] - s_b[s]) * 1000, 1)

    return {
        "driver_a": {"code": driver_a, **s_a},
        "driver_b": {"code": driver_b, **s_b},
        "deltas_ms": deltas,
        "faster_by_sector": {
            s: (driver_b if v > 0 else driver_a)
            for s, v in deltas.items()
            if v != 0
        },
    }
