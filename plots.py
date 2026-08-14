# plots.py — Plotly figures for the driver-selection dashboard
#
# Colour and chrome are centralised here so every chart in the app reads as one
# system. Two rules are load-bearing:
#
#   * A driver's colour follows the DRIVER, not their rank or file slot. Driver
#     A is always blue and driver B always orange, so swapping which one is
#     quicker never repaints the chart under the reader.
#   * Every y-quantity gets its own subplot row. There are deliberately no
#     dual-axis plots: overlaying speed (km/h) and throttle (%) on one y-scale
#     would invent a correlation by choosing where the two scales line up.
#
# The two series colours were checked with the data-viz validator against both
# Streamlit surfaces (#ffffff light, #0e1117 dark) and pass the lightness band,
# chroma floor, CVD separation, normal-vision and 3:1 contrast gates
# (worst pair ΔE 24.7 protan light / 26.8 dark).

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from metrics import format_lap_time
from motec_parser import CHANNEL_LABELS

# Categorical slots 1 and 2, stepped per mode.
SERIES = {
    "light": ["#2a78d6", "#eb6834"],
    "dark": ["#3987e5", "#d95926"],
}

# Chart chrome. Grid and axis are solid hairlines one shade off the surface;
# labels sit in muted ink so the data is the loudest thing on screen.
CHROME = {
    "light": {
        "surface": "#ffffff",
        "text": "#0b0b0b",
        "secondary": "#52514e",
        "muted": "#898781",
        "grid": "#e1e0d9",
        "axis": "#c3c2b7",
    },
    "dark": {
        "surface": "#0e1117",
        "text": "#ffffff",
        "secondary": "#c3c2b7",
        "muted": "#898781",
        "grid": "#2c2c2a",
        "axis": "#383835",
    },
}

FONT_FAMILY = 'system-ui, -apple-system, "Segoe UI", sans-serif'


def series_color(index: int, mode: str = "light") -> str:
    """Colour for driver slot `index` (0 or 1) in the given mode."""
    return SERIES.get(mode, SERIES["light"])[index % 2]


def _apply_theme(fig: go.Figure, mode: str, height: int) -> go.Figure:
    """Apply the shared chrome to a figure."""
    c = CHROME.get(mode, CHROME["light"])
    fig.update_layout(
        height=height,
        paper_bgcolor=c["surface"],
        plot_bgcolor=c["surface"],
        font=dict(family=FONT_FAMILY, color=c["secondary"], size=13),
        margin=dict(l=64, r=24, t=56, b=56),
        hoverlabel=dict(font=dict(family=FONT_FAMILY, size=12)),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="left", x=0, bgcolor="rgba(0,0,0,0)"),
        title=dict(font=dict(color=c["text"], size=16)),
    )
    fig.update_xaxes(
        showgrid=True, gridcolor=c["grid"], gridwidth=1, griddash="solid",
        zeroline=False, showline=True, linecolor=c["axis"], linewidth=1,
        ticks="outside", tickcolor=c["axis"], tickfont=dict(color=c["muted"]),
        title_font=dict(color=c["secondary"]),
    )
    fig.update_yaxes(
        showgrid=True, gridcolor=c["grid"], gridwidth=1, griddash="solid",
        zeroline=False, showline=False,
        tickfont=dict(color=c["muted"]),
        title_font=dict(color=c["secondary"]),
    )
    return fig


def _lap_time_ticks(values: np.ndarray) -> tuple[list[float], list[str]]:
    """Build m:ss tick labels across the range of lap times shown."""
    vals = values[np.isfinite(values)]
    if vals.size == 0:
        return [], []
    lo, hi = float(np.min(vals)), float(np.max(vals))
    span = max(hi - lo, 1.0)
    # Pick a step from a set of round, human-readable intervals in seconds.
    for step in (1, 2, 5, 10, 15, 30, 60, 120, 300):
        if span / step <= 8:
            break
    start = np.floor(lo / step) * step
    ticks = np.arange(start, hi + step, step)
    return list(ticks), [format_lap_time(t) for t in ticks]


# --------------------------------------------------------------------------
# Lap-by-lap pace
# --------------------------------------------------------------------------

def lap_pace_chart(stints: list[dict], target_s: float = 210.0,
                   mode: str = "light") -> go.Figure:
    """Lap time across the stint, with the target lap as a reference line.

    `stints` is a list of dicts: {"name", "laps" (lap table), "keep" (bool mask)}.

    Laps the outlier filter rejected are drawn as hollow markers rather than
    hidden. The reader can then see *why* the median differs from the mean —
    a chart that quietly deletes the traffic laps is hiding its own workings.
    """
    c = CHROME.get(mode, CHROME["light"])
    fig = go.Figure()

    all_times: list[float] = []

    for i, stint in enumerate(stints):
        laps = stint.get("laps")
        if laps is None or len(laps) == 0:
            continue
        color = series_color(i, mode)
        name = stint.get("name", f"Driver {i + 1}")
        lap_no = laps["Lap"].to_numpy()
        lap_t = pd.to_numeric(laps["LapTime [s]"], errors="coerce").to_numpy()
        all_times.extend(lap_t[np.isfinite(lap_t)].tolist())

        keep = stint.get("keep")
        keep = (np.asarray(keep, dtype=bool) if keep is not None
                else np.ones(len(laps), dtype=bool))

        # The line joins every lap so the shape of the stint stays readable;
        # marker style carries kept-vs-excluded.
        fig.add_trace(go.Scatter(
            x=lap_no, y=lap_t, mode="lines",
            name=name, legendgroup=name,
            line=dict(color=color, width=2),
            hovertemplate=(f"<b>{name}</b><br>Lap %{{x}}<br>"
                           "%{customdata}<extra></extra>"),
            customdata=[format_lap_time(v) for v in lap_t],
        ))
        fig.add_trace(go.Scatter(
            x=lap_no[keep], y=lap_t[keep], mode="markers",
            name=name, legendgroup=name, showlegend=False,
            marker=dict(color=color, size=9,
                        line=dict(color=c["surface"], width=2)),
            hovertemplate=(f"<b>{name}</b><br>Lap %{{x}}<br>"
                           "%{customdata}<extra></extra>"),
            customdata=[format_lap_time(v) for v in lap_t[keep]],
        ))
        if (~keep).any():
            fig.add_trace(go.Scatter(
                x=lap_no[~keep], y=lap_t[~keep], mode="markers",
                name=f"{name} — excluded", legendgroup=name,
                marker=dict(color=c["surface"], size=9, symbol="circle",
                            line=dict(color=color, width=2)),
                hovertemplate=(f"<b>{name}</b><br>Lap %{{x}} (excluded)<br>"
                               "%{customdata}<extra></extra>"),
                customdata=[format_lap_time(v) for v in lap_t[~keep]],
            ))

    # The target lap. Dashed because it genuinely is a threshold, not a grid
    # line, and in muted ink so it never competes with a driver's colour.
    fig.add_hline(
        y=target_s, line=dict(color=c["muted"], width=1.5, dash="dash"),
        annotation_text=f"Target {format_lap_time(target_s)}",
        annotation_position="top right",
        annotation_font=dict(color=c["secondary"], size=12),
    )
    all_times.append(target_s)

    ticks, labels = _lap_time_ticks(np.asarray(all_times, dtype=float))
    fig.update_yaxes(title_text="Lap time", tickvals=ticks, ticktext=labels)
    fig.update_xaxes(title_text="Lap number", dtick=1)
    fig.update_layout(title="Lap-by-lap pace", hovermode="x unified")

    return _apply_theme(fig, mode, height=420)


# --------------------------------------------------------------------------
# Head-to-head telemetry overlay
# --------------------------------------------------------------------------

# The overlay rows: (canonical channel key, axis title). One row per quantity —
# never two of these on a shared y-axis.
OVERLAY_ROWS = [
    ("speed", "Speed [km/h]"),
    ("throttle", "Throttle [%]"),
    ("steering", "Steering [deg]"),
]


def overlay_chart(traces: list[dict], mode: str = "light",
                  x_col: str = "Lap Distance [m]") -> go.Figure:
    """Overlay one selected lap per driver against distance into the lap.

    `traces` is a list of dicts: {"name", "lap", "data" (per-lap DataFrame)}.

    Distance into the lap — not elapsed time — is the x-axis, because that is
    what makes the comparison physical: at 1,200 m both drivers are at the same
    corner, so a gap between the two speed traces is a difference in how they
    drove that corner rather than an artefact of one starting the lap earlier.
    """
    c = CHROME.get(mode, CHROME["light"])

    present = [(key, title) for key, title in OVERLAY_ROWS
               if any(CHANNEL_LABELS[key] in t["data"].columns for t in traces)]
    if not present:
        fig = go.Figure()
        fig.add_annotation(text="None of the overlay channels are in these logs.",
                           showarrow=False, font=dict(color=c["secondary"]))
        return _apply_theme(fig, mode, height=260)

    fig = make_subplots(
        rows=len(present), cols=1, shared_xaxes=True, vertical_spacing=0.06,
        subplot_titles=[title for _key, title in present],
    )

    for row, (key, _title) in enumerate(present, start=1):
        col = CHANNEL_LABELS[key]
        for i, t in enumerate(traces):
            data = t["data"]
            if col not in data.columns or x_col not in data.columns:
                continue
            label = f"{t['name']} — lap {t['lap']}"
            fig.add_trace(
                go.Scatter(
                    x=data[x_col], y=data[col],
                    mode="lines", name=label, legendgroup=label,
                    showlegend=(row == 1),          # one legend entry per driver
                    line=dict(color=series_color(i, mode), width=2),
                    hovertemplate="%{y:.1f}<extra>" + label + "</extra>",
                ),
                row=row, col=1,
            )

    # A shared crosshair across all three rows: hovering one corner shows what
    # both drivers were doing with speed, throttle and steering at that point.
    fig.update_layout(
        title="Head-to-head telemetry overlay",
        hovermode="x unified",
        hoversubplots="axis",
    )
    fig.update_xaxes(title_text=x_col, row=len(present), col=1)
    fig.update_xaxes(showspikes=True, spikemode="across",
                     spikecolor=c["muted"], spikethickness=1, spikedash="solid")

    fig = _apply_theme(fig, mode, height=240 * len(present))
    # Subplot titles are annotations, so they need colouring separately.
    for ann in fig.layout.annotations:
        ann.font.color = c["secondary"]
        ann.font.size = 13
    return fig
