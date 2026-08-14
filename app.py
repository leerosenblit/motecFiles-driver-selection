# app.py — Driver Selection Dashboard (MoTeC telemetry, Circuit Zolder)
#
# Compares driver stints from MoTeC .ld logs to answer one question: who do we
# put in the car for the endurance race?
#
# Run:
#     streamlit run app.py
#
# The pipeline for each uploaded log is:
#
#     read_ld            .ld binary            -> channels at native rates
#     to_dataframe       channels              -> one common time grid
#     build_lap_table    lap channel / .ldx    -> the logger's own lap division
#     add_lap_columns    lap table             -> per-row lap no. + lap distance
#     compute_driver_metrics                   -> the four decision metrics
#
# Note on laps: the telemetry system already sums laps geographically at the
# start/finish beacon. build_lap_table only READS that division (from a lap
# channel, or from .ldx beacon markers) — nothing here re-detects a lap trigger.

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import streamlit as st
from streamlit.runtime.scriptrunner import get_script_run_ctx

# Run with `python app.py` and Streamlit's widgets return None, which fails
# later with an unhelpful traceback. Say what to do instead.
if get_script_run_ctx() is None:
    sys.exit(
        # Plain ASCII: this goes to a terminal, and Windows consoles default to
        # cp1252, which mangles an em dash.
        "\nThis is a Streamlit app - start it with the Streamlit CLI:\n\n"
        "    streamlit run app.py\n"
    )

from metrics import (
    TARGET_LAP_TIME_S,
    compute_driver_metrics,
    flag_outlier_laps,
    format_delta,
    format_lap_time,
    metrics_table,
    rank_drivers,
)
from motec_parser import (
    CHANNEL_LABELS,
    add_lap_columns,
    apply_ldx_laps,
    build_lap_table,
    read_ld,
    read_ldx_markers,
    to_dataframe,
)
from plots import lap_pace_chart, overlay_chart, series_color

st.set_page_config(
    page_title="Driver Selection — MoTeC Telemetry",
    page_icon=":material/speed:",
    layout="wide",
)


def active_theme() -> str:
    """Which Streamlit theme is rendering, so the charts can match it."""
    try:
        base = st.context.theme.type            # Streamlit >= 1.44
        if base in ("light", "dark"):
            return base
    except Exception:
        pass
    return st.get_option("theme.base") or "light"


MODE = active_theme()


# --------------------------------------------------------------------------
# Loading (cached — parsing a 20 MB log should happen once per upload)
# --------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_stint(ld_bytes: bytes, ldx_bytes: bytes | None,
               max_hz: float, lap_source: str) -> dict:
    """Parse one driver's log into everything the dashboard needs.

    Returns a plain dict so the result pickles cleanly into Streamlit's cache.
    """
    log = read_ld(ld_bytes)
    df, found = to_dataframe(log, max_hz=max_hz)

    markers: list[float] = []
    ldx_error = None
    if ldx_bytes:
        try:
            markers = read_ldx_markers(ldx_bytes, log_duration_s=log.duration_s)
        except ValueError as exc:
            ldx_error = str(exc)

    # ".ldx beacons" is offered because an engineer who added or corrected a
    # beacon in i2 saved that edit to the .ldx, not back into the .ld.
    if lap_source == "ldx" and markers:
        laps = apply_ldx_laps(df, markers)
    else:
        laps = build_lap_table(df)
        if len(laps) == 0 and markers:
            laps = apply_ldx_laps(df, markers)

    df = add_lap_columns(df, laps)
    # Read attrs before reset_index — pandas does not reliably propagate attrs
    # through frame operations.
    lap_source = laps.attrs.get("lap_source", "unknown")

    return {
        "driver": log.driver,
        "venue": log.venue,
        "event": log.event,
        "session": log.session,
        "datetime": log.datetime,
        "df": df,
        "laps": laps.reset_index(drop=True),
        "lap_source": lap_source,
        "channel_map": {k: (c.name, c.unit, c.freq) for k, c in found.items()},
        "all_channels": [(c.name, c.unit, c.freq, len(c.data)) for c in log.channels],
        "sample_hz": df.attrs.get("sample_hz", max_hz),
        "n_markers": len(markers),
        "warnings": list(log.warnings),
        "ldx_error": ldx_error,
    }


@st.cache_data(show_spinner=False)
def load_demo(max_hz: float) -> dict[str, dict]:
    """Build the two demo stints through the same pipeline as a real upload."""
    from sample_data import demo_stints

    out = {}
    for label, (frame, _laps) in demo_stints().items():
        laps = build_lap_table(frame)
        lap_source = laps.attrs.get("lap_source", "unknown")
        out[label] = {
            "driver": label,
            "venue": "Zolder",
            "event": "Driver Selection (demo)",
            "session": "Synthetic",
            "datetime": None,
            "df": add_lap_columns(frame, laps),
            "laps": laps.reset_index(drop=True),
            "lap_source": lap_source,
            "channel_map": {k: (CHANNEL_LABELS[k], "", int(frame.attrs.get("sample_hz", 20)))
                            for k in ("speed", "throttle", "steering", "g_lat",
                                      "distance", "lap_time", "lap_number")},
            "all_channels": [(c, "", int(frame.attrs.get("sample_hz", 20)), len(frame))
                             for c in frame.columns if c != "Time [s]"],
            "sample_hz": frame.attrs.get("sample_hz", 20),
            "n_markers": 0,
            "warnings": [],
            "ldx_error": None,
        }
    return out


# --------------------------------------------------------------------------
# Sidebar — one control panel scoping every chart below
# --------------------------------------------------------------------------

st.sidebar.title("Telemetry source")

source = st.sidebar.radio(
    "Data", ["Upload MoTeC logs", "Demo data"], key="source",
    help="Demo data generates two synthetic Zolder stints so the dashboard can "
         "be reviewed before a real export is available.",
)

uploads: list[dict] = []
if source == "Upload MoTeC logs":
    st.sidebar.caption(
        "Upload one log for a single-driver report, or two for a head-to-head. "
        "The `.ldx` is optional — add it if beacons were edited in i2."
    )
    for slot in (1, 2):
        with st.sidebar.expander(f"Driver {slot}" + ("" if slot == 1 else "  (optional)"),
                                 expanded=(slot == 1)):
            ld = st.file_uploader(f"`.ld` log — driver {slot}", type=["ld"],
                                  key=f"ld_{slot}")
            ldx = st.file_uploader(f"`.ldx` laps/beacons — driver {slot} (optional)",
                                   type=["ldx"], key=f"ldx_{slot}")
            label = st.text_input("Driver name (optional)", key=f"name_{slot}",
                                  placeholder="taken from the log if blank")
            uploads.append({"ld": ld, "ldx": ldx, "label": label})

st.sidebar.title("Analysis settings")

target_s = st.sidebar.number_input(
    "Target lap time [s]", min_value=30.0, max_value=1200.0,
    value=float(TARGET_LAP_TIME_S), step=0.5, key="target_s",
    help="The energy-budget lap. Default 210 s = 3:30.",
)
st.sidebar.caption(f"Target = **{format_lap_time(target_s)}**")

lap_source_choice = st.sidebar.selectbox(
    "Lap division", ["Auto (from the log)", "Prefer .ldx beacons"], key="lap_div",
    help="Laps come from the logger, which already sums them at the start/finish "
         "beacon. 'Auto' reads the lap channel; the .ldx option uses the beacon "
         "markers instead, including any you edited in i2.",
)

with st.sidebar.expander("Lap filtering", expanded=False):
    drop_first = st.checkbox("Exclude the first lap (out-lap)", value=True,
                             key="drop_first")
    drop_last = st.checkbox("Exclude the last lap (in-lap)", value=True,
                            key="drop_last")
    mad_k = st.slider(
        "Outlier threshold (× robust σ)", min_value=1.0, max_value=6.0,
        value=3.0, step=0.5, key="mad_k",
        help="Laps further than this many robust standard deviations from the "
             "median are treated as traffic or yellow-flag laps. Lower = stricter.",
    )

with st.sidebar.expander("Sampling", expanded=False):
    max_hz = st.slider(
        "Resample cap [Hz]", min_value=5, max_value=50, value=25, step=5,
        key="max_hz",
        help="Channels are logged at different rates and get resampled onto one "
             "grid. None of the metrics resolve anything above ~25 Hz.",
    )

# --------------------------------------------------------------------------
# Load
# --------------------------------------------------------------------------

st.title("Driver Selection — Endurance Stint Analysis")
st.caption("MoTeC telemetry comparison · Circuit Zolder")

stints: list[dict] = []
lap_pref = "ldx" if lap_source_choice.startswith("Prefer") else "auto"

if source == "Demo data":
    for label, stint in load_demo(float(max_hz)).items():
        stints.append(stint)
else:
    for slot, up in enumerate(uploads, start=1):
        if up["ld"] is None:
            continue
        try:
            with st.spinner(f"Parsing driver {slot}…"):
                stint = load_stint(
                    up["ld"].getvalue(),
                    up["ldx"].getvalue() if up["ldx"] is not None else None,
                    float(max_hz), lap_pref,
                )
            stint = dict(stint)
            stint["driver"] = (up["label"].strip() or stint["driver"]
                               or up["ld"].name.rsplit(".", 1)[0])
            stints.append(stint)
        except ValueError as exc:
            st.error(f"**Driver {slot} — could not read `{up['ld'].name}`**\n\n{exc}")
        except Exception as exc:                                   # noqa: BLE001
            st.error(
                f"**Driver {slot} — unexpected error reading `{up['ld'].name}`**\n\n"
                f"`{type(exc).__name__}: {exc}`\n\nThis usually means the file is "
                "from a device layout this parser does not recognise."
            )

if not stints:
    st.info(
        "**Upload a MoTeC `.ld` log to begin** — one for a single-driver report, "
        "two for a head-to-head comparison. Or pick **Demo data** in the sidebar "
        "to see the dashboard populated."
    )
    with st.expander("What the dashboard measures"):
        st.markdown(
            f"""
| Metric | Definition | Why it decides the seat |
|---|---|---|
| **Median lap time** | Median of the valid laps | Robust to one lap lost in traffic, unlike the mean |
| **Pace adherence** | Mean of \\|lap − {format_lap_time(target_s)}\\| | Both directions are failures: under target burns energy we don't have, over it loses distance |
| **Consistency** | Standard deviation of lap times | A metronomic driver is a predictable energy budget |
| **Smoothness** | Variance of d(Throttle)/dt, in (%/s)² | Rewards a driver who eases the pedal; pedal pumping wastes energy |

Required channels: `LapTime`, `Throttle Pos [%]`, `Steering Angle [deg]`,
`G Force Lat [G]`, `Corr Speed [km/h]`, `Distance [m]`.
"""
        )
    st.stop()

# Warnings from parsing, surfaced rather than swallowed.
for i, s in enumerate(stints):
    for w in s["warnings"]:
        st.warning(f"**{s['driver']}** — {w}")
    if s["ldx_error"]:
        st.warning(f"**{s['driver']}** — `.ldx` ignored: {s['ldx_error']}")
    if len(s["laps"]) == 0:
        st.error(
            f"**{s['driver']}** — no lap division was found in this log. The "
            "dashboard needs a lap channel (`Lap Number` or `Lap Time`) or an "
            "`.ldx` with beacon markers."
        )

# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------

all_metrics = []
keep_masks = []
for s in stints:
    keep = flag_outlier_laps(s["laps"]["LapTime [s]"], mad_k=mad_k,
                             drop_first=drop_first, drop_last=drop_last) \
        if len(s["laps"]) else pd.Series(dtype=bool)
    keep_masks.append(keep)
    all_metrics.append(compute_driver_metrics(
        s["driver"], s["df"], s["laps"], target_s=target_s,
        mad_k=mad_k, drop_first=drop_first, drop_last=drop_last,
    ))


def driver_heading(name: str, index: int, subtitle: str = "") -> None:
    """Driver name with its chart colour as a swatch.

    The swatch is the secondary identity cue: the name, not the hue alone, is
    what ties a row of numbers to a line on the chart.
    """
    color = series_color(index, MODE)
    st.markdown(
        f"<div style='display:flex;align-items:center;gap:.5rem;margin:.25rem 0'>"
        f"<span style='width:12px;height:12px;border-radius:3px;"
        f"background:{color};display:inline-block;flex:0 0 auto'></span>"
        f"<span style='font-weight:600;font-size:1.05rem'>{name}</span>"
        f"<span style='color:#898781;font-size:.85rem'>{subtitle}</span></div>",
        unsafe_allow_html=True,
    )


st.subheader("Driver metrics")

baseline = all_metrics[0]
for i, (s, m) in enumerate(zip(stints, all_metrics)):
    bits = [f"{m.n_laps_used}/{m.n_laps_total} laps used"]
    if s["venue"]:
        bits.append(s["venue"])
    if s["session"]:
        bits.append(s["session"])
    driver_heading(m.name, i, " · ".join(bits))

    c1, c2, c3, c4 = st.columns(4)

    # For the second driver every metric also carries its delta vs the first, so
    # the comparison is readable without arithmetic.
    def delta(value: float, base: float, digits: int = 2) -> str | None:
        if i == 0 or not (np.isfinite(value) and np.isfinite(base)):
            return None
        return f"{value - base:+.{digits}f} vs {baseline.name}"

    c1.metric(
        "Median lap time", format_lap_time(m.median_lap_s),
        delta=delta(m.median_lap_s, baseline.median_lap_s),
        delta_color="inverse",           # quicker (negative) is better
        help="Median of the valid laps — robust to traffic and yellows.",
    )
    c2.metric(
        f"Pace adherence vs {format_lap_time(target_s)}",
        f"{m.pace_adherence_s:.2f} s" if np.isfinite(m.pace_adherence_s) else "—",
        delta=delta(m.pace_adherence_s, baseline.pace_adherence_s),
        delta_color="inverse",           # smaller deviation is better
        help=f"Mean absolute deviation from the target lap. Median lap sits "
             f"{format_delta(m.median_delta_s)} from target.",
    )
    c3.metric(
        "Consistency (σ)",
        f"{m.consistency_s:.2f} s" if np.isfinite(m.consistency_s) else "—",
        delta=delta(m.consistency_s, baseline.consistency_s),
        delta_color="inverse",           # tighter spread is better
        help="Standard deviation of the valid lap times.",
    )
    c4.metric(
        "Smoothness score",
        f"{m.smoothness_score:.0f} / 100" if np.isfinite(m.smoothness_score) else "—",
        delta=delta(m.smoothness_score, baseline.smoothness_score, digits=0),
        delta_color="normal",            # higher score is better
        help=(f"Throttle rate RMS {m.smoothness_rms:.1f} %/s "
              f"(variance {m.smoothness_var:,.0f} (%/s)²). Higher score = "
              "smoother pedal = less wasted energy.")
        if np.isfinite(m.smoothness_rms) else "No throttle channel found.",
    )

    if m.excluded_laps:
        st.caption(
            f"Excluded from the statistics: lap "
            f"{', '.join(str(v) for v in m.excluded_laps)} "
            "(out/in-lap or outside the outlier threshold)."
        )
    for note in m.notes:
        st.caption(f":warning: {note}")

# --------------------------------------------------------------------------
# Comparison table + ranking
# --------------------------------------------------------------------------

if len(all_metrics) >= 2:
    st.subheader("Side-by-side")
    st.dataframe(metrics_table(all_metrics, target_s=target_s),
                 hide_index=True, width="stretch")

    ranking = rank_drivers(all_metrics)
    if len(ranking):
        winner = ranking.iloc[0]
        st.markdown(
            f"On equally-weighted ranks across pace adherence, consistency and "
            f"throttle smoothness, **{winner['Driver']}** comes out ahead "
            f"(total rank {winner['Total rank']:.0f} vs "
            f"{ranking.iloc[1]['Total rank']:.0f})."
        )
        with st.expander("How that ranking was computed"):
            st.dataframe(ranking, hide_index=True, width="stretch")
            st.caption(
                "Each driver is ranked 1..N on each metric (lower is better in "
                "all three) and the ranks are summed with equal weight. It is "
                "deliberately transparent rather than a tuned score — the point "
                "is to expose the trade-off, not to hide it behind a weighting."
            )

# --------------------------------------------------------------------------
# Lap-by-lap pace
# --------------------------------------------------------------------------

st.subheader("Lap-by-lap pace")

pace_stints = [
    {"name": s["driver"], "laps": s["laps"], "keep": keep_masks[i]}
    for i, s in enumerate(stints) if len(s["laps"])
]
if pace_stints:
    st.plotly_chart(
        lap_pace_chart(pace_stints, target_s=target_s, mode=MODE),
        width="stretch",
        config={"displaylogo": False},
    )
    st.caption(
        "Hollow markers are laps excluded from the statistics. The dashed line "
        f"is the {format_lap_time(target_s)} target."
    )

    with st.expander("Table view — lap times"):
        for i, s in enumerate(stints):
            if not len(s["laps"]):
                continue
            table = s["laps"][["Lap", "LapTime [s]", "Distance [m]"]].copy()
            table["Lap time"] = table["LapTime [s]"].map(format_lap_time)
            table["Δ vs target [s]"] = (table["LapTime [s]"] - target_s).round(2)
            table["Used"] = keep_masks[i].to_numpy()
            driver_heading(s["driver"], i, f"laps from: {s['lap_source']}")
            st.dataframe(
                table[["Lap", "Lap time", "LapTime [s]", "Δ vs target [s]",
                       "Distance [m]", "Used"]].round(2),
                hide_index=True, width="stretch",
            )

# --------------------------------------------------------------------------
# Head-to-head telemetry overlay
# --------------------------------------------------------------------------

st.subheader("Telemetry overlay")

usable = [(i, s) for i, s in enumerate(stints) if len(s["laps"])]
if not usable:
    st.info("No laps available to overlay.")
else:
    if len(usable) == 1:
        st.caption(
            "Upload a second `.ld` log to overlay two drivers. Showing the "
            "selected lap for one driver."
        )

    sel_cols = st.columns(len(usable))
    traces = []
    for col, (i, s) in zip(sel_cols, usable):
        laps = s["laps"]
        options = [int(v) for v in laps["Lap"].tolist()]
        keep_i = keep_masks[i].to_numpy()
        labels = {}
        for idx in range(len(laps)):
            lap_no = int(laps["Lap"].iat[idx])
            suffix = "" if keep_i[idx] else "  (excluded)"
            labels[lap_no] = (f"Lap {lap_no} — "
                              f"{format_lap_time(laps['LapTime [s]'].iat[idx])}{suffix}")

        # Default to the driver's quickest kept lap — their best representative
        # effort, rather than lap 1 which is an out-lap.
        kept = laps[keep_masks[i].to_numpy()] if keep_masks[i].any() else laps
        best_lap = int(kept.loc[kept["LapTime [s]"].idxmin(), "Lap"])

        with col:
            driver_heading(s["driver"], i)
            chosen = st.selectbox(
                "Lap to overlay", options,
                index=options.index(best_lap) if best_lap in options else 0,
                format_func=lambda v: labels.get(int(v), f"Lap {int(v)}"),
                key=f"lap_pick_{i}",
            )

        row = laps[laps["Lap"] == chosen].iloc[0]
        lo, hi = int(row.start_idx), int(row.end_idx)
        traces.append({
            "name": s["driver"],
            "lap": int(chosen),
            "data": s["df"].iloc[lo:hi + 1],
        })

    x_col = ("Lap Distance [m]" if all("Lap Distance [m]" in t["data"].columns
                                       for t in traces) else "Time [s]")
    st.plotly_chart(
        overlay_chart(traces, mode=MODE, x_col=x_col),
        width="stretch",
        config={"displaylogo": False},
    )
    if x_col == "Lap Distance [m]":
        st.caption(
            "X-axis is distance into the lap, so both drivers line up corner for "
            "corner. Hover to read all three channels at one point on the track."
        )
    else:
        st.caption(
            ":warning: No `Distance` channel was found, so the overlay falls back "
            "to elapsed time — the two traces are **not** aligned by track "
            "position."
        )

# --------------------------------------------------------------------------
# Diagnostics
# --------------------------------------------------------------------------

with st.expander("Channel mapping & log details"):
    st.caption(
        "Channel names vary between MoTeC configurations, so the parser matches "
        "them against a list of aliases. This is exactly which channel fed each "
        "metric — check it before trusting the numbers."
    )
    for i, s in enumerate(stints):
        driver_heading(s["driver"], i,
                       f"{s['sample_hz']:.0f} Hz common grid · laps from: {s['lap_source']}")
        meta = []
        if s["event"]:
            meta.append(f"event: {s['event']}")
        if s["session"]:
            meta.append(f"session: {s['session']}")
        if s["datetime"]:
            meta.append(f"logged: {s['datetime']:%Y-%m-%d %H:%M}")
        if s["n_markers"]:
            meta.append(f"{s['n_markers']} .ldx markers")
        if meta:
            st.caption(" · ".join(meta))

        mapping = pd.DataFrame(
            [{"Dashboard channel": CHANNEL_LABELS[k],
              "Log channel": v[0],
              "Log unit": v[1] or "—",
              "Log rate [Hz]": v[2]}
             for k, v in s["channel_map"].items()]
        )
        st.dataframe(mapping, hide_index=True, width="stretch")

        missing = [CHANNEL_LABELS[k] for k in
                   ("speed", "throttle", "steering", "g_lat", "distance")
                   if k not in s["channel_map"]]
        if missing:
            st.caption(f":warning: Not found in this log: {', '.join(missing)}")

        # A checkbox rather than a nested expander/popover, which Streamlit
        # does not allow inside an expander.
        if st.checkbox(f"List all {len(s['all_channels'])} channels in this log",
                       key=f"all_ch_{i}"):
            st.dataframe(
                pd.DataFrame(s["all_channels"],
                             columns=["Channel", "Unit", "Rate [Hz]", "Samples"]),
                hide_index=True, width="stretch",
            )
