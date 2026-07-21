"""Plotly chart renderers for F1 Copilot."""

import plotly.graph_objects as go
from plotly.subplots import make_subplots

COMPOUND_COLORS = {
    "SOFT": "#ff1e2d",
    "MEDIUM": "#ffd400",
    "HARD": "#e8e8e8",
    "INTERMEDIATE": "#2ee6a6",
    "WET": "#0067ff",
    "UNKNOWN": "#8d9096",
}

DRIVER_PALETTE = ["#ff1e2d", "#3671c6", "#ff8000", "#2ee6a6", "#b026ff", "#ffd400"]

_BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="#101114",
    font=dict(color="#c2c4c8", family="JetBrains Mono, monospace", size=12),
    margin=dict(l=50, r=20, t=20, b=40),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#2a2c31"),
)

_AXIS = dict(gridcolor="#2a2c31", linecolor="#2a2c31", zerolinecolor="#2a2c31")


def speed_trace_chart(comparison_data: dict) -> go.Figure:
    """Overlay speed traces for two drivers vs track distance."""
    traces = comparison_data.get("speed_traces", {})
    driver_a = comparison_data.get("driver_a", {}).get("code", "A")
    driver_b = comparison_data.get("driver_b", {}).get("code", "B")
    gap = comparison_data.get("gap_seconds", 0)
    faster = comparison_data.get("faster_driver", "")

    fig = go.Figure()

    for i, (drv, color) in enumerate(zip([driver_a, driver_b], DRIVER_PALETTE)):
        pts = traces.get(drv, [])
        if not pts:
            continue
        distances = [p["Distance"] for p in pts]
        speeds = [p["Speed"] for p in pts]
        fig.add_trace(go.Scatter(
            x=distances, y=speeds,
            name=drv,
            line=dict(color=color, width=2),
            hovertemplate=f"<b>{drv}</b><br>Distance: %{{x:.0f}}m<br>Speed: %{{y:.0f}} km/h<extra></extra>",
        ))

    fig.update_layout(
        **_BASE,
        xaxis=dict(title="Distance (m)", **_AXIS),
        yaxis=dict(title="Speed (km/h)", **_AXIS),
        height=280,
    )
    return fig


def sector_delta_chart(sector_data: dict) -> go.Figure:
    """Horizontal bar chart showing sector time deltas (positive = driver_a slower)."""
    driver_a = sector_data.get("driver_a", {}).get("code", "A")
    driver_b = sector_data.get("driver_b", {}).get("code", "B")
    deltas = sector_data.get("deltas_ms", {})

    labels = ["Sector 1", "Sector 2", "Sector 3"]
    keys = ["s1_delta_ms", "s2_delta_ms", "s3_delta_ms"]
    values = [deltas.get(k, 0) for k in keys]
    colors = ["#ff1e2d" if v > 0 else "#2ee6a6" for v in values]

    fig = go.Figure(go.Bar(
        x=values,
        y=labels,
        orientation="h",
        marker_color=colors,
        text=[f"{abs(v):.0f}ms {'← ' + driver_b if v > 0 else driver_a + ' →'}" for v in values],
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>Delta: %{x:.0f} ms<extra></extra>",
    ))

    fig.update_layout(
        **_BASE,
        xaxis=dict(title="Delta (ms)", **_AXIS),
        yaxis=dict(**_AXIS),
        height=220,
        shapes=[dict(type="line", x0=0, x1=0, y0=-0.5, y1=2.5,
                     line=dict(color="#6f7278", width=1, dash="dash"))],
    )
    return fig


def lap_time_chart(lap_data: dict) -> go.Figure:
    """Lap time progression coloured by tire compound."""
    driver = lap_data.get("driver", "")
    laps = lap_data.get("laps", [])
    if not laps:
        return None

    fig = go.Figure()
    by_compound: dict[str, list] = {}
    for lap in laps:
        c = lap.get("compound", "UNKNOWN")
        by_compound.setdefault(c, []).append(lap)

    for compound, lap_list in by_compound.items():
        x = [l["lap"] for l in lap_list]
        y = [l["time_s"] for l in lap_list]
        fig.add_trace(go.Scatter(
            x=x, y=y,
            mode="lines+markers",
            name=compound,
            line=dict(color=COMPOUND_COLORS.get(compound, "#888"), width=2),
            marker=dict(size=5),
            hovertemplate=f"<b>{compound}</b><br>Lap %{{x}}<br>%{{y:.3f}}s<extra></extra>",
        ))

    fig.update_layout(
        **_BASE,
        xaxis=dict(title="Lap", **_AXIS),
        yaxis=dict(title="Lap Time (s)", **_AXIS, autorange="reversed"),
        height=260,
    )
    return fig


def multi_driver_pace_chart(pace_data: dict) -> go.Figure:
    """Overlay lap times for multiple drivers — race pace comparison."""
    fig = go.Figure()

    for i, (driver, data) in enumerate(pace_data.items()):
        laps = data.get("laps", [])
        if not laps:
            continue
        color = DRIVER_PALETTE[i % len(DRIVER_PALETTE)]
        x = [l["lap"] for l in laps]
        y = [l["time_s"] for l in laps]
        fig.add_trace(go.Scatter(
            x=x, y=y,
            mode="lines+markers",
            name=driver,
            line=dict(color=color, width=2),
            marker=dict(size=4),
            hovertemplate=f"<b>{driver}</b><br>Lap %{{x}}<br>%{{y:.3f}}s<extra></extra>",
        ))

    fig.update_layout(
        **_BASE,
        xaxis=dict(title="Lap", **_AXIS),
        yaxis=dict(title="Lap Time (s)", **_AXIS, autorange="reversed"),
        height=280,
    )
    return fig


def tire_strategy_chart(strategy_data: dict) -> go.Figure:
    """Gantt-style tire strategy visualisation."""
    driver = strategy_data.get("driver", "")
    stints = strategy_data.get("stints", [])
    if not stints:
        return None

    fig = go.Figure()
    for stint in stints:
        compound = stint.get("compound", "UNKNOWN")
        fig.add_trace(go.Bar(
            x=[stint["end_lap"] - stint["start_lap"] + 1],
            y=[driver],
            base=[stint["start_lap"] - 1],
            orientation="h",
            name=compound,
            marker_color=COMPOUND_COLORS.get(compound, "#8d9096"),
            text=f"{compound} ({stint['laps']} laps)",
            textposition="inside",
            hovertemplate=(
                f"<b>{compound}</b><br>"
                f"Laps {stint['start_lap']}–{stint['end_lap']}<br>"
                f"Avg: {stint['avg_lap_time_s']:.3f}s<br>"
                f"Deg: {stint['degradation_s_per_lap']:.4f}s/lap<extra></extra>"
            ),
            showlegend=True,
        ))

    fig.update_layout(
        **_BASE,
        xaxis=dict(title="Lap", **_AXIS),
        yaxis=dict(**_AXIS),
        barmode="stack",
        height=140,
    )
    return fig


def weather_chart(weather_data: dict) -> go.Figure:
    """Mini gauge-style cards for weather metrics."""
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=["Air Temp (°C)", "Track Temp (°C)", "Humidity (%)"],
    )

    def _add_gauge(row, col, value, min_v, max_v, color):
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=value,
            gauge=dict(
                axis=dict(range=[min_v, max_v], tickcolor="#8d9096"),
                bar=dict(color=color),
                bgcolor="#16181c",
                bordercolor="#2a2c31",
            ),
            number=dict(font=dict(color="#f4f5f6")),
        ), row=row, col=col)

    air = weather_data.get("air_temp_c", {})
    trk = weather_data.get("track_temp_c", {})

    _add_gauge(1, 1, air.get("avg", 0), 10, 50, "#ff8000")
    _add_gauge(1, 2, trk.get("avg", 0), 15, 65, "#ff1e2d")
    _add_gauge(1, 3, weather_data.get("humidity_pct", 0), 0, 100, "#3671c6")

    fig.update_layout(**_BASE, height=200, showlegend=False)
    return fig
