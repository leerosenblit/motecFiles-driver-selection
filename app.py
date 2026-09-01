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


def _running_under_streamlit() -> bool:
    """Whether a Streamlit script-run context is active.

    Deliberately defensive. The context lives on a semi-internal Streamlit path
    that has moved between releases, and a hosted deployment installs whatever
    version is current — so this check must never be the thing that takes the
    app down. Any failure to determine the answer is treated as "yes, we are
    running normally", because refusing to start is far worse than skipping a
    convenience message.
    """
    for module, attr in (
        ("streamlit.runtime.scriptrunner", "get_script_run_ctx"),
        ("streamlit.runtime.scriptrunner_utils.script_run_context",
         "get_script_run_ctx"),
    ):
        try:
            mod = __import__(module, fromlist=[attr])
            return getattr(mod, attr)() is not None
        except Exception:
            continue
    return True


# Run with `python app.py` and Streamlit's widgets return None, which fails
# later with an unhelpful traceback. Say what to do instead.
if not _running_under_streamlit():
    sys.exit(
        # Plain ASCII: this goes to a terminal, and Windows consoles default to
        # cp1252, which mangles an em dash.
        "\nThis is a Streamlit app - start it with the Streamlit CLI:\n\n"
        "    streamlit run app.py\n"
    )

from metrics import (
    DEFAULT_SCORE_WEIGHTS,
    ENERGY_BUDGET_WH_PER_LAP,
    STRATEGY_REFERENCE,
    TARGET_LAP_TIME_S,
    compute_driver_metrics,
    energy_per_lap,
    expected_energy_wh,
    flag_outlier_laps,
    format_delta,
    format_lap_time,
    leaderboard,
    metrics_table,
    rank_drivers,
)
from motec_parser import (
    CHANNEL_LABELS,
    add_lap_columns,
    apply_ldx_laps,
    average_lap_trace,
    build_lap_table,
    read_ld,
    read_ldx_markers,
    to_dataframe,
)
from plots import (
    MAX_DRIVERS,
    OVERLAY_ROWS,
    energy_chart,
    lap_pace_chart,
    overlay_chart,
    series_color,
)

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
def load_stint(ld_file_id: str, ldx_file_id: str | None,
               _ld_bytes: bytes, _ldx_bytes: bytes | None,
               max_hz: float, lap_source: str) -> dict:
    """Parse one driver's log into everything the dashboard needs.

    Returns a plain dict so the result pickles cleanly into Streamlit's cache.

    The cache key is built from `ld_file_id`/`ldx_file_id` — Streamlit's own
    per-upload identifiers, a few bytes each — NOT from the actual file bytes.
    An underscore-prefixed parameter is Streamlit's documented signal to skip
    hashing that argument for the cache key, and `_ld_bytes` badly needs it: a
    30-45 MB log takes ~150-250 ms to hash, and without this, `st.cache_data`
    would redo that hash on *every* rerun the script makes for *any* reason —
    dragging an unrelated sidebar slider, ticking a checkbox, typing a driver
    name — because Streamlit reruns the whole script top to bottom on every
    widget interaction and must re-check the cache key each time. With two or
    three real logs loaded that tax stacked into a very noticeable multi-second
    stall on every click, unrelated to parsing (which is well under a second)
    or to the actual one-time network upload.
    """
    log = read_ld(_ld_bytes)
    df, found = to_dataframe(log, max_hz=max_hz)

    markers: list[float] = []
    ldx_error = None
    if _ldx_bytes:
        try:
            markers = read_ldx_markers(_ldx_bytes, log_duration_s=log.duration_s)
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
def load_demo(max_hz: float, n_drivers: int = 2) -> dict[str, dict]:
    """Build the demo stints through the same pipeline as a real upload."""
    from sample_data import demo_stints

    out = {}
    for label, (frame, _laps) in demo_stints(n_drivers).items():
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
                                      "g_lon", "distance", "lap_time", "lap_number")},
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
n_demo = 2
if source == "Demo data":
    n_demo = st.sidebar.number_input(
        "Demo drivers", min_value=1, max_value=MAX_DRIVERS, value=2, step=1,
        key="n_demo",
        help="Two or more brings up the leaderboard alongside the head-to-head.",
    )
elif source == "Upload MoTeC logs":
    # Default 2 (the head-to-head case), openable up to a full squad of 10.
    n_drivers = st.sidebar.number_input(
        "Drivers to compare", min_value=1, max_value=MAX_DRIVERS, value=2, step=1,
        key="n_drivers",
        help=f"How many upload slots to show, 1 to {MAX_DRIVERS}. Slots you "
             "leave empty are ignored, so you can raise this and fill them in "
             "as logs arrive.",
    )
    st.sidebar.caption(
        "One log gives a single-driver report; two or more compare them. "
        "The `.ldx` is optional — add it if beacons were edited in i2."
    )
    for slot in range(1, int(n_drivers) + 1):
        with st.sidebar.expander(f"Driver {slot}" + ("" if slot <= 2 else "  (optional)"),
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

budget_wh = st.sidebar.number_input(
    "Energy budget [Wh/lap]", min_value=1.0, max_value=2000.0,
    value=float(ENERGY_BUDGET_WH_PER_LAP), step=1.0, key="budget_wh",
    help="Watt-hours per lap the strategy allows. Default 80 Wh, matching the "
         "team's Base (210 s) strategy.",
)

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

with st.sidebar.expander("Driver score weights", expanded=False):
    st.caption(
        "How much each metric counts toward the Driver Score on the leaderboard "
        "(shown from two drivers up). Each metric is scored against the best "
        "(lowest) driver on file, as a percentage of that best. Weights are "
        "renormalised, so only their relative size matters."
    )
    w_cons = st.slider("Consistency", 0.0, 1.0,
                       DEFAULT_SCORE_WEIGHTS["consistency"], 0.05, key="w_cons")
    w_energy = st.slider("Energy (per lap)", 0.0, 1.0,
                         DEFAULT_SCORE_WEIGHTS["energy"], 0.05, key="w_energy")
    w_accel = st.slider("Avg acceleration", 0.0, 1.0,
                        DEFAULT_SCORE_WEIGHTS["avg_accel"], 0.05, key="w_accel")
    w_decel = st.slider("Avg deceleration", 0.0, 1.0,
                        DEFAULT_SCORE_WEIGHTS["avg_decel"], 0.05, key="w_decel")
    w_lat = st.slider("Avg lateral acceleration", 0.0, 1.0,
                      DEFAULT_SCORE_WEIGHTS["avg_lat_g"], 0.05, key="w_lat")
    w_maxlat = st.slider("Max lateral acceleration", 0.0, 1.0,
                         DEFAULT_SCORE_WEIGHTS["max_lat_g"], 0.05, key="w_maxlat")

    raw_weights = {"consistency": w_cons, "energy": w_energy,
                   "avg_accel": w_accel, "avg_decel": w_decel,
                   "avg_lat_g": w_lat, "max_lat_g": w_maxlat}
    if sum(raw_weights.values()) <= 0:
        st.caption(":warning: All weights are zero — falling back to the defaults.")
        score_weights = dict(DEFAULT_SCORE_WEIGHTS)
    else:
        score_weights = raw_weights

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
    with st.spinner(f"Generating {int(n_demo)} synthetic stint(s)…"):
        for label, stint in load_demo(float(max_hz), int(n_demo)).items():
            stints.append(stint)
else:
    for slot, up in enumerate(uploads, start=1):
        if up["ld"] is None:
            continue
        try:
            with st.spinner(f"Parsing driver {slot}…"):
                stint = load_stint(
                    up["ld"].file_id,
                    up["ldx"].file_id if up["ldx"] is not None else None,
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
| **Smoothness** | Variance of d(Throttle)/dt, in (%/s)² | A proxy for efficiency where no energy channel exists |
| **Energy** | Wh per lap, and Wh above what that pace should cost | This is a solar car: energy is the binding constraint |
| **Acceleration** | Avg forward/deceleration and avg/max lateral G | Descriptive — how hard the car is driven, not scored good or bad |

With two or more drivers, a **leaderboard** combines consistency, energy per
lap, and average/max acceleration, deceleration and lateral G into a single
0-100 Driver Score, each metric scored as a percentage of whoever posts the
best (lowest) value.

Required channels: `LapTime`, `Throttle Pos [%]`, `Steering Angle [deg]`,
`G Force Lat [G]`, `Corr Speed [km/h]`, `Distance [m]`.

For energy, any one of: a power channel (`mms_power_W`, `Power`), pack
**voltage + current** (`bms_voltage_V` + `bms_current_A`), or a cumulative
energy counter (`total_race_energy`).
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
lap_energy = []
for s in stints:
    keep = flag_outlier_laps(s["laps"]["LapTime [s]"], mad_k=mad_k,
                             drop_first=drop_first, drop_last=drop_last) \
        if len(s["laps"]) else pd.Series(dtype=bool)
    keep_masks.append(keep)
    all_metrics.append(compute_driver_metrics(
        s["driver"], s["df"], s["laps"], target_s=target_s,
        mad_k=mad_k, drop_first=drop_first, drop_last=drop_last,
        budget_wh=budget_wh,
    ))
    lap_energy.append(energy_per_lap(s["df"], s["laps"])[0])

any_energy = any(np.isfinite(np.asarray(e, dtype=float)).any() for e in lap_energy)


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

# Five metric cards per driver reads well up to about four drivers; past that the
# page becomes a wall of tiles, so the comparison table leads instead and the
# cards move into an expander.
CARD_LIMIT = 4
cards_inline = len(all_metrics) <= CARD_LIMIT
card_host = (st.container() if cards_inline
             else st.expander(f"Per-driver metric cards ({len(all_metrics)} drivers)",
                              expanded=False))
if not cards_inline:
    st.caption(
        f"With more than {CARD_LIMIT} drivers the side-by-side table below is the "
        "clearer read; the individual metric cards are in the expander."
    )

baseline = all_metrics[0]
for i, (s, m) in enumerate(zip(stints, all_metrics)):
  with card_host:
    bits = [f"{m.n_laps_used}/{m.n_laps_total} laps used"]
    if s["venue"]:
        bits.append(s["venue"])
    if s["session"]:
        bits.append(s["session"])
    driver_heading(m.name, i, " · ".join(bits))

    c1, c2, c3, c4, c5 = st.columns(5)

    # Every driver after the first also carries its delta vs the first, so the
    # comparison is readable without arithmetic.
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
              "smoother pedal. A proxy for efficiency — where energy channels "
              "exist, the energy figure measures it directly.")
        if np.isfinite(m.smoothness_rms) else "No throttle channel found.",
    )
    c5.metric(
        "Energy per lap",
        f"{m.median_energy_wh:.1f} Wh" if np.isfinite(m.median_energy_wh) else "—",
        delta=delta(m.median_energy_wh, baseline.median_energy_wh, digits=1),
        delta_color="inverse",           # fewer watt-hours is better
        help=(f"Median over the valid laps. Budget {budget_wh:.0f} Wh "
              f"({format_delta(m.energy_delta_wh).replace(' s', ' Wh')}). "
              f"A lap at this driver's own pace should cost "
              f"{expected_energy_wh(m.median_lap_s, budget_wh):.1f} Wh, so "
              f"{m.energy_excess_wh:+.1f} Wh is down to how they drove. "
              f"Efficiency {m.wh_per_km:.1f} Wh/km. Source: {m.energy_source}.")
        if np.isfinite(m.median_energy_wh)
        else "No power, voltage+current, or energy channel found in this log.",
    )

    # Acceleration: descriptive, not scored good/bad — delta_color="off" so a
    # bigger number isn't implied to be either better or worse, unlike the row
    # above where every delta has a direction that means something for the seat.
    c6, c7, c8, c9 = st.columns(4)
    c6.metric(
        "Avg forward accel", f"{m.avg_accel_g:.2f} G" if np.isfinite(m.avg_accel_g) else "—",
        delta=delta(m.avg_accel_g, baseline.avg_accel_g), delta_color="off",
        help="Mean of the positive longitudinal-G samples (accelerating) over "
             "the valid laps.",
    )
    c7.metric(
        "Avg deceleration", f"{m.avg_decel_g:.2f} G" if np.isfinite(m.avg_decel_g) else "—",
        delta=delta(m.avg_decel_g, baseline.avg_decel_g), delta_color="off",
        help="Mean magnitude of the negative longitudinal-G samples (braking) "
             "over the valid laps.",
    )
    c8.metric(
        "Avg lateral accel", f"{m.avg_lat_g:.2f} G" if np.isfinite(m.avg_lat_g) else "—",
        delta=delta(m.avg_lat_g, baseline.avg_lat_g), delta_color="off",
        help="Mean of |lateral G| over the valid laps. Uses the absolute value "
             "because a signed mean would cancel a left-hander against a "
             "right-hander and read near zero regardless of cornering load.",
    )
    c9.metric(
        "Max lateral accel", f"{m.max_lat_g:.2f} G" if np.isfinite(m.max_lat_g) else "—",
        delta=delta(m.max_lat_g, baseline.max_lat_g), delta_color="off",
        help="Peak |lateral G| reached during the valid laps — the hardest "
             "corner taken, in either direction.",
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

# The leaderboard shows from two drivers up: a head-to-head is a leaderboard of
# two, and showing the same Driver Score there — not just once a field exists —
# means the number is already familiar by the time a third driver joins.
if len(all_metrics) >= 2:
    st.subheader("Leaderboard")

    board = leaderboard(all_metrics, weights=score_weights)
    if len(board):
        leader = board.iloc[0]
        st.markdown(
            f"**{leader['Driver']}** leads on Driver Score with "
            f"**{leader['Driver score']:.1f}/100**"
            + (f", ahead of {board.iloc[1]['Driver']} on "
               f"{board.iloc[1]['Driver score']:.1f}." if len(board) > 1 else ".")
        )

        show = board[["Pos", "Driver", "Driver score", "Consistency pts",
                      "Energy pts", "Avg accel pts", "Avg decel pts",
                      "Avg lateral G pts", "Max lateral G pts",
                      "Median lap", "Laps used"]]
        st.dataframe(
            show, hide_index=True, width="stretch",
            column_config={
                "Driver score": st.column_config.ProgressColumn(
                    "Driver score", min_value=0, max_value=100,
                    format="%.1f", help="Weighted 0-100 composite of all six metrics.",
                ),
                **{c: st.column_config.NumberColumn(c, format="%.0f")
                   for c in ("Consistency pts", "Energy pts", "Avg accel pts",
                             "Avg decel pts", "Avg lateral G pts",
                             "Max lateral G pts")},
            },
        )

        with st.expander("How the Driver Score is calculated"):
            st.markdown(
                f"""
Each metric is "lower is better". Whoever posts the best (lowest) value among
the drivers on file scores 100 on it; everyone else scores a straight
percentage of that best:

`points = 100 x best / value`

Those points are then combined with the weights in the sidebar:

| Metric | Weight |
|---|---|
| Consistency (σ) | {score_weights['consistency']:.0%} |
| Energy per lap | {score_weights['energy']:.0%} |
| Avg acceleration | {score_weights['avg_accel']:.0%} |
| Avg deceleration | {score_weights['avg_decel']:.0%} |
| Avg lateral acceleration | {score_weights['avg_lat_g']:.0%} |
| Max lateral acceleration | {score_weights['max_lat_g']:.0%} |

**Peer-relative, not absolute.** Every number above is scored against the best
driver currently on file, not a fixed engineering target — adding or removing
a driver can move everyone else's score, because the bar they're measured
against just moved.

Missing metrics are dropped and the remaining weights renormalised, so an
incomplete log gives a less informed score rather than a punished one.
"""
            )
            st.dataframe(board, hide_index=True, width="stretch")

if len(all_metrics) >= 2:
    st.subheader("Side-by-side")
    st.dataframe(metrics_table(all_metrics, target_s=target_s),
                 hide_index=True, width="stretch")

    ranking, eff_col = rank_drivers(all_metrics)
    if len(ranking):
        winner = ranking.iloc[0]
        st.markdown(
            f"On equally-weighted ranks across pace adherence, consistency and "
            f"{eff_col.split(' [')[0].lower()}, **{winner['Driver']}** comes out "
            f"ahead (total rank {winner['Total rank']:.0f} vs "
            f"{ranking.iloc[1]['Total rank']:.0f})."
        )
        with st.expander("How that ranking was computed"):
            st.dataframe(ranking, hide_index=True, width="stretch")
            st.caption(
                f"Each driver is ranked 1..N on each metric (lower is better in "
                f"all three) and the ranks are summed with equal weight. The "
                f"efficiency slot used **{eff_col}** — energy when the car logged "
                f"it, throttle smoothness as a proxy when it did not. Only one of "
                f"the two is used, since ranking on both would weight efficiency "
                f"twice. This is a deliberately different method from the Driver "
                f"Score above, so agreement between them is a useful check."
            )

# --------------------------------------------------------------------------
# Lap-by-lap pace
# --------------------------------------------------------------------------

st.subheader("Lap-by-lap pace")

pace_stints = [
    {"name": s["driver"], "laps": s["laps"], "keep": keep_masks[i], "slot": i}
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
# Energy consumption
# --------------------------------------------------------------------------

st.subheader("Energy consumption")

if not any_energy:
    st.info(
        "**No energy data in these logs.** To measure consumption the log needs "
        "one of: a power channel (`Drivetrain Power`, `mms_power_W`, `Power`), "
        "pack **voltage and current** (`bms_voltage_V` + `bms_current_A`) from "
        "which power is derived, or a cumulative energy counter "
        "(`total_race_energy`). Without any of them, throttle smoothness is the "
        "only efficiency signal — and it is a proxy, not a measurement."
    )

    # Channel names vary a lot between logger configurations, so rather than
    # leaving the reader to guess, list what each log actually contains that
    # looks energy-related. If the right channel is sitting here under a name the
    # alias list does not know, this is where you spot it.
    import re as _re

    candidate = _re.compile(
        r"volt|amp|curr|power|energ|batt|soc|charge|kers|ers|watt|joule|"
        r"consum|fuel|torque|hp\b",
        _re.I,
    )
    for i, s in enumerate(stints):
        hits = [c for c in s["all_channels"]
                if candidate.search(c[0]) or candidate.search(str(c[1]))]
        driver_heading(s["driver"], i,
                       f"{len(hits)} of {len(s['all_channels'])} channels look "
                       "energy-related")
        if hits:
            st.dataframe(
                pd.DataFrame(hits, columns=["Channel", "Unit", "Rate [Hz]", "Samples"]),
                hide_index=True, width="stretch",
            )
            st.caption(
                "If one of these is the car's power or energy channel, it only "
                "needs adding to `CHANNEL_ALIASES` in motec_parser.py."
            )
        else:
            st.caption(
                "Nothing energy-related in this log at all — the logger was not "
                "recording the battery. That is a MoTeC channel-configuration "
                "change, not something the dashboard can recover."
            )
else:
    energy_stints = [
        {"name": s["driver"], "laps": s["laps"], "keep": keep_masks[i],
         "energy": lap_energy[i], "slot": i}
        for i, s in enumerate(stints) if len(s["laps"])
    ]
    st.plotly_chart(
        energy_chart(energy_stints, budget_wh=budget_wh, mode=MODE),
        width="stretch",
        config={"displaylogo": False},
    )
    st.caption(
        f"The dashed line is the {budget_wh:.0f} Wh/lap budget. Read this "
        "together with the pace chart above: a driver under the budget but off "
        "the pace target is not saving energy, only losing distance."
    )

    with st.expander("Pace vs energy — the team's strategy table"):
        st.caption(
            "Raw watt-hours alone would reward whoever drove slowest, so the "
            "dashboard also reports **energy excess**: consumption minus what a "
            "lap at that driver's own median pace should cost, interpolated from "
            "this table. Positive excess is energy spent on how they drove "
            "rather than on how fast they went."
        )
        strat = pd.DataFrame(STRATEGY_REFERENCE,
                             columns=["Strategy", "Lap time [s]", "Energy [Wh/lap]"])
        strat["Lap time"] = strat["Lap time [s]"].map(format_lap_time)
        st.dataframe(strat[["Strategy", "Lap time", "Energy [Wh/lap]"]],
                     hide_index=True, width="stretch")

        rows = []
        for m in all_metrics:
            if not np.isfinite(m.median_energy_wh):
                continue
            rows.append({
                "Driver": m.name,
                "Median lap": format_lap_time(m.median_lap_s),
                "Actual [Wh/lap]": round(m.median_energy_wh, 1),
                "Expected at that pace [Wh]": round(
                    expected_energy_wh(m.median_lap_s, budget_wh), 1),
                "Excess [Wh/lap]": round(m.energy_excess_wh, 1),
                "Efficiency [Wh/km]": round(m.wh_per_km, 1),
                "Stint total [Wh]": round(m.total_energy_wh, 0),
                "Source": m.energy_source,
            })
        if rows:
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

    with st.expander("Table view — energy per lap"):
        for i, s in enumerate(stints):
            e = np.asarray(lap_energy[i], dtype=float)
            if not len(s["laps"]) or not np.isfinite(e).any():
                continue
            table = s["laps"][["Lap"]].copy()
            table["Lap time"] = s["laps"]["LapTime [s]"].map(format_lap_time)
            table["Energy [Wh]"] = np.round(e, 1)
            table["Δ vs budget [Wh]"] = np.round(e - budget_wh, 1)
            table["Used"] = keep_masks[i].to_numpy()
            driver_heading(s["driver"], i)
            st.dataframe(table, hide_index=True, width="stretch")

# --------------------------------------------------------------------------
# Head-to-head telemetry overlay
# --------------------------------------------------------------------------

st.subheader("Telemetry overlay")
st.caption(
    "Each line is the average of the driver's valid laps, resampled onto a "
    "common distance grid and averaged point by point — not one raw recorded "
    "lap. A single lap can be a lucky (or unlucky) sample of a driver's "
    "technique; the average across the stint is the more honest comparison."
)

available = [(i, s) for i, s in enumerate(stints) if len(s["laps"])]
if not available:
    st.info("No laps available to overlay.")
else:
    if len(available) == 1:
        st.caption("Upload a second `.ld` log to overlay two drivers.")
        usable = available
    else:
        # Overlaying a whole field of ten traces is unreadable, so the drivers
        # to compare are chosen explicitly — defaulting to the first two.
        names = {i: s["driver"] for i, s in available}
        picked = st.multiselect(
            "Drivers to overlay", options=[i for i, _s in available],
            default=[i for i, _s in available][:2],
            format_func=lambda i: names[i], key="overlay_pick",
            help="Two or three traces per chart stay readable; more gets busy.",
        )
        usable = [(i, s) for i, s in available if i in set(picked)]
        if not usable:
            st.info("Pick at least one driver to overlay.")

    channel_cols = [CHANNEL_LABELS[key] for key, _title in OVERLAY_ROWS]
    traces = []
    for i, s in usable:
        keep_i = keep_masks[i]
        n_used = int(keep_i.sum()) if len(keep_i) and keep_i.any() else len(s["laps"])
        label = f"avg of {n_used} lap{'s' if n_used != 1 else ''}"

        avg_df = average_lap_trace(s["df"], s["laps"], keep_i, channel_cols)
        driver_heading(s["driver"], i, label)
        if avg_df.empty or len(avg_df) < 2:
            st.caption(
                ":warning: Could not build an averaged trace for this driver — "
                "no `Distance` channel, or no lap has usable distance data."
            )
            continue
        traces.append({"name": s["driver"], "label": label, "data": avg_df, "slot": i})

    if traces:
        st.plotly_chart(
            overlay_chart(traces, mode=MODE, x_col="Lap Distance [m]"),
            width="stretch",
            config={"displaylogo": False},
        )
        st.caption(
            "X-axis is distance into the lap, so the drivers line up corner for "
            "corner. Hover to read every channel at one point on the track — "
            "the power row shows which corner exit cost the energy. The grid "
            "only extends to each driver's shortest kept lap, so no line is "
            "extrapolated past real data."
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
