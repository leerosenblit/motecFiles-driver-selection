# metrics.py — driver-selection metrics computed from a parsed stint
#
# Four numbers decide which driver we put in the car for the endurance race at
# Zolder. Each one answers a different question:
#
#   Median lap time  — how quick is this driver, ignoring traffic and yellows?
#   Pace adherence   — can they hit the energy-budget target lap of 3:30?
#   Consistency      — how repeatable are they lap after lap?
#   Smoothness       — how gently do they use the throttle? (energy efficiency)
#
# The first three are lap-level statistics; the last is a signal-level one.

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from motec_parser import CHANNEL_LABELS

# Energy-budget target lap: 3:30. The whole strategy is built around holding
# this pace, so "fast" alone is not the goal — hitting 210 s is.
TARGET_LAP_TIME_S = 210.0

# Energy budget for a lap at the target pace, in watt-hours.
#
# Note this is 100 Wh, while Pit_Dashboard/constants.py still lists the
# "Base (210s)" strategy at 80 Wh — the two want reconciling, and the number
# here is the one the team currently races to. It is a sidebar input, so it can
# be changed per session without editing code.
ENERGY_BUDGET_WH_PER_LAP = 100.0

# The team's pace/energy trade-off from Pit_Dashboard/constants.py, kept as
# RELATIVE factors against the base strategy rather than absolute watt-hours.
#
# Storing ratios is what lets the budget above be changed freely: the expected
# cost of a lap is always `budget x factor(lap_time)`, so the whole curve moves
# with the budget and can never contradict it. Hard-coding the absolute table
# would leave the pace correction anchored to a stale 80 Wh while the dashboard
# reported against 100 Wh.
#
# (label, lap time [s], energy relative to the base lap)
STRATEGY_REFERENCE = [
    ("Fast (-10%)", 189.0, 88.0 / 80.0),
    ("Med-Fast (-5%)", 199.5, 84.0 / 80.0),
    ("Base (210s)", 210.0, 1.0),
    ("Med-Slow (+5%)", 220.5, 76.0 / 80.0),
    ("Slow (+10%)", 231.0, 72.0 / 80.0),
]

# Reference throttle-derivative variance used to put the smoothness score on a
# 0-100 scale. A driver whose throttle rate-of-change has this variance scores
# 50; smoother scores higher. It is a presentation constant only — every
# ranking decision uses the raw variance, which is unit-bearing and honest.
SMOOTHNESS_REFERENCE_VAR = 400.0        # (%/s)^2


@dataclass
class DriverMetrics:
    """The metric set for one driver's stint."""
    name: str
    n_laps_total: int = 0
    n_laps_used: int = 0
    median_lap_s: float = float("nan")
    mean_lap_s: float = float("nan")
    best_lap_s: float = float("nan")
    pace_adherence_s: float = float("nan")     # mean |lap - target|
    median_delta_s: float = float("nan")       # median lap - target (signed)
    consistency_s: float = float("nan")        # std dev of lap times
    smoothness_var: float = float("nan")       # var of dThrottle/dt, (%/s)^2
    smoothness_rms: float = float("nan")       # sqrt of the above, %/s
    smoothness_score: float = float("nan")     # 0-100, higher = smoother
    # Energy — the binding constraint on a solar car.
    median_energy_wh: float = float("nan")     # Wh per lap
    energy_delta_wh: float = float("nan")      # median Wh/lap - budget
    wh_per_km: float = float("nan")            # energy per distance
    total_energy_wh: float = float("nan")      # across the whole stint
    energy_excess_wh: float = float("nan")     # Wh/lap above the pace-matched expectation
    energy_source: str | None = None           # how energy was obtained
    excluded_laps: list[int] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def format_lap_time(seconds: float) -> str:
    """Render a lap time as m:ss.mmm, the way the timing screens show it."""
    if seconds is None or not np.isfinite(seconds):
        return "—"
    sign = "-" if seconds < 0 else ""
    seconds = abs(float(seconds))
    minutes, rem = divmod(seconds, 60.0)
    return f"{sign}{int(minutes)}:{rem:06.3f}"


def format_delta(seconds: float) -> str:
    """Render a signed delta in seconds, e.g. '+1.42 s'."""
    if seconds is None or not np.isfinite(seconds):
        return "—"
    return f"{seconds:+.2f} s"


# --------------------------------------------------------------------------
# Lap filtering
# --------------------------------------------------------------------------

def flag_outlier_laps(lap_times: pd.Series, mad_k: float = 3.0,
                      drop_first: bool = True,
                      drop_last: bool = True) -> pd.Series:
    """Return a boolean mask of laps to KEEP.

    Traffic, a yellow flag or a driver change turn a lap into an outlier that
    would drag the mean and inflate the standard deviation. We reject them with
    a median-absolute-deviation test rather than a mean/sigma test, because MAD
    is itself robust — a single 5-minute lap barely moves it, whereas it would
    inflate a standard deviation enough to hide itself.

        robust_sigma = 1.4826 * MAD        (1.4826 makes MAD comparable to the
                                            standard deviation of a normal
                                            distribution)
        keep if |lap - median| <= mad_k * robust_sigma

    The out-lap and in-lap are dropped by default: they include pit entry/exit
    and are not representative racing laps.
    """
    lt = pd.to_numeric(lap_times, errors="coerce")
    keep = lt.notna() & (lt > 0)

    if drop_first and len(keep) > 0:
        keep.iloc[0] = False
    if drop_last and len(keep) > 1:
        keep.iloc[-1] = False

    valid = lt[keep]
    if len(valid) >= 3:
        median = float(valid.median())
        mad = float((valid - median).abs().median())
        robust_sigma = 1.4826 * mad
        if robust_sigma > 0:
            keep &= (lt - median).abs() <= mad_k * robust_sigma
        else:
            # Every kept lap is identical to the median — nothing to reject.
            pass

    return keep


# --------------------------------------------------------------------------
# Smoothness
# --------------------------------------------------------------------------

def throttle_derivative(df: pd.DataFrame, laps: pd.DataFrame | None = None,
                        keep_mask: pd.Series | None = None) -> np.ndarray:
    """Compute dThrottle/dt over the laps we are scoring.

    The smoothness metric is the variance of the throttle position's first
    derivative:

        throttle[%] sampled at times t[s]
        rate[i] = d(throttle)/dt  ->  units of % per second
        smoothness = Var(rate)

    Why the derivative and not the throttle itself: a driver holding 60%
    throttle steadily and a driver oscillating 40-80% can have the *same* mean
    and the same distribution of throttle positions. What separates them is how
    fast the pedal moves. Differentiating turns "pedal pumping" into large
    positive and negative rate values, and the variance of that rate is then a
    single number that grows with pumping.

    Why the variance and not the mean rate: over a closed lap the pedal returns
    to where it started, so the mean rate is ~0 for everyone and carries no
    information. The spread around that zero is the signal. Variance also
    weights big stabs quadratically, which is what we want — one violent stab
    costs more energy than several gentle corrections.

    Two implementation details that matter:

      * np.gradient is used with the actual time vector, so the result is a
        true rate in %/s. A plain np.diff would give "% per sample", whose
        magnitude changes with the logging rate and means nothing physical.
      * The derivative is taken per lap and the results concatenated, never
        across a lap boundary. Consecutive laps are contiguous in the log, but
        an excluded lap (or a pit stop) would otherwise contribute one enormous
        spurious rate value at the seam.

    One honest caveat. Correct units do NOT make the variance itself independent
    of the sample rate: differentiating amplifies high frequencies, so a log
    sampled faster resolves more throttle-sensor noise and reports a larger
    variance for the same driving. The metric is therefore only comparable
    between drivers measured on the SAME time grid — which the dashboard
    guarantees, because to_dataframe resamples every log onto one common grid
    before any of this runs. What it means in practice: changing the resample
    cap shifts every driver's smoothness number together, so the ranking holds
    while the absolute value should not be quoted across sessions logged at
    different rates.
    """
    thr_col = CHANNEL_LABELS["throttle"]
    if thr_col not in df:
        return np.array([])

    t_all = df["Time [s]"].to_numpy(dtype=np.float64)
    thr_all = pd.to_numeric(df[thr_col], errors="coerce").to_numpy(dtype=np.float64)

    # Which contiguous row ranges to differentiate over.
    if laps is not None and len(laps):
        if keep_mask is not None:
            used = laps[keep_mask.to_numpy()]
        else:
            used = laps
        ranges = [(int(r.start_idx), int(r.end_idx)) for r in used.itertuples(index=False)]
    else:
        ranges = [(0, len(df) - 1)]

    pieces = []
    for lo, hi in ranges:
        t = t_all[lo:hi + 1]
        thr = thr_all[lo:hi + 1]
        good = np.isfinite(t) & np.isfinite(thr)
        t, thr = t[good], thr[good]
        if len(t) < 3:
            continue
        # Guard against a duplicated timestamp, which would divide by zero.
        keep = np.concatenate(([True], np.diff(t) > 0))
        t, thr = t[keep], thr[keep]
        if len(t) < 3:
            continue
        pieces.append(np.gradient(thr, t))

    return np.concatenate(pieces) if pieces else np.array([])


def smoothness_from_rate(rate: np.ndarray) -> tuple[float, float, float]:
    """Turn the derivative samples into (variance, RMS, 0-100 score)."""
    rate = rate[np.isfinite(rate)]
    if rate.size < 2:
        return float("nan"), float("nan"), float("nan")

    var = float(np.var(rate, ddof=1))
    rms = float(np.sqrt(var))
    # Map variance onto 0-100 with a rational curve: score = 100 * R / (R + var).
    # Monotonic, bounded, and equals 50 at var == R. Presentation only.
    score = 100.0 * SMOOTHNESS_REFERENCE_VAR / (SMOOTHNESS_REFERENCE_VAR + var)
    return var, rms, float(score)


# --------------------------------------------------------------------------
# Energy
# --------------------------------------------------------------------------

def expected_energy_wh(lap_time_s: float,
                       budget_wh: float = ENERGY_BUDGET_WH_PER_LAP) -> float:
    """Energy a lap at this pace *should* cost, given the budget at target pace.

    Needed because raw Wh/lap is not a fair way to compare drivers: going slower
    always uses less energy, so ranking on it alone would crown whoever was
    least committed. Scaling the team's pace/energy curve by the budget gives the
    expected cost at the pace actually driven, and the difference between actual
    and expected is the part attributable to the driver rather than their speed.

    Outside the tabulated 189-231 s range the endpoints are held flat, which is
    the conservative reading — we don't invent a slope we have no data for.
    """
    if not np.isfinite(lap_time_s):
        return float("nan")
    times = [t for _label, t, _factor in STRATEGY_REFERENCE]
    factors = [f for _label, _t, f in STRATEGY_REFERENCE]
    return float(budget_wh * np.interp(lap_time_s, times, factors))


def energy_per_lap(df: pd.DataFrame,
                   laps: pd.DataFrame) -> tuple[np.ndarray, str | None]:
    """Energy consumed on each lap, in watt-hours.

    Returns (array aligned with `laps`, description of the source used).

    Three sources are tried, most trustworthy first:

      1. A cumulative energy counter (`total_race_energy`). The difference
         between its value at the end and the start of a lap is that lap's
         consumption. This is the car's own coulomb-counted accounting, already
         net of regen, so it beats anything we recompute.
      2. Integrating the power channel over the lap. Trapezoidal, because power
         is a continuous signal sampled at intervals — a plain rectangular sum
         would systematically over-read every acceleration ramp:

             E[J] = sum( (P[i] + P[i+1]) / 2 * dt[i] )
             E[Wh] = E[J] / 3600                (1 Wh = 3600 J)

         Regen shows up as negative power and is therefore subtracted, giving
         net consumption on the same basis as source 1.
      3. A per-lap energy channel (`last_lap_energy`). Used last because it
         needs an assumption: a channel reporting the LAST lap's energy holds
         lap N-1's figure while lap N is being driven, so it is read one lap
         behind and the final lap comes out unknown.
    """
    if laps is None or len(laps) == 0:
        return np.array([]), None

    e_col = CHANNEL_LABELS["energy"]
    p_col = CHANNEL_LABELS["power"]
    le_col = CHANNEL_LABELS["lap_energy"]
    n = len(laps)
    spans = [(int(r.start_idx), int(r.end_idx))
             for r in laps.itertuples(index=False)]

    # --- 1. Cumulative counter -------------------------------------------
    if e_col in df:
        e = pd.to_numeric(df[e_col], errors="coerce").to_numpy()
        if np.isfinite(e).sum() >= 2:
            out = np.full(n, np.nan)
            for i, (lo, hi) in enumerate(spans):
                seg = e[lo:hi + 1]
                seg = seg[np.isfinite(seg)]
                if seg.size >= 2:
                    out[i] = float(seg[-1] - seg[0])
            if np.isfinite(out).any():
                return out, f"{e_col} counter delta"

    # --- 2. Integrate power ------------------------------------------------
    if p_col in df:
        p = pd.to_numeric(df[p_col], errors="coerce").to_numpy()
        t = df["Time [s]"].to_numpy(dtype=np.float64)
        if np.isfinite(p).sum() >= 2:
            out = np.full(n, np.nan)
            for i, (lo, hi) in enumerate(spans):
                pt, tt = p[lo:hi + 1], t[lo:hi + 1]
                good = np.isfinite(pt) & np.isfinite(tt)
                pt, tt = pt[good], tt[good]
                if pt.size < 2:
                    continue
                joules = float(np.sum((pt[:-1] + pt[1:]) / 2.0 * np.diff(tt)))
                out[i] = joules / 3600.0
            if np.isfinite(out).any():
                return out, f"integrated {p_col}"

    # --- 3. Per-lap channel ------------------------------------------------
    if le_col in df:
        le = pd.to_numeric(df[le_col], errors="coerce").to_numpy()
        if np.isfinite(le).any():
            out = np.full(n, np.nan)
            for i in range(n):
                # Lap i's energy is the value held during lap i+1.
                if i + 1 >= n:
                    continue
                lo, hi = spans[i + 1]
                seg = le[lo:hi + 1]
                seg = seg[np.isfinite(seg)]
                if seg.size:
                    out[i] = float(np.median(seg))
            if np.isfinite(out).any():
                return out, f"{le_col} channel (read one lap behind)"

    return np.full(n, np.nan), None


# --------------------------------------------------------------------------
# Top-level metric computation
# --------------------------------------------------------------------------

def compute_driver_metrics(name: str, df: pd.DataFrame, laps: pd.DataFrame,
                           target_s: float = TARGET_LAP_TIME_S,
                           mad_k: float = 3.0,
                           drop_first: bool = True,
                           drop_last: bool = True,
                           budget_wh: float = ENERGY_BUDGET_WH_PER_LAP) -> DriverMetrics:
    """Compute the full metric set for one driver.

    All lap statistics are computed on the KEPT laps only, and the laps that
    were rejected are recorded on the result so the UI can show them — a metric
    that silently drops data is not one an engineer should trust.
    """
    m = DriverMetrics(name=name)

    if laps is None or len(laps) == 0:
        m.notes.append("No laps were found in this log, so no lap statistics "
                       "could be computed.")
        rate = throttle_derivative(df, None, None)
        m.smoothness_var, m.smoothness_rms, m.smoothness_score = smoothness_from_rate(rate)
        if np.isfinite(m.smoothness_var):
            m.notes.append("Smoothness was computed over the whole log instead "
                           "of per lap.")
        return m

    lt = pd.to_numeric(laps["LapTime [s]"], errors="coerce")
    m.n_laps_total = int(len(laps))

    keep = flag_outlier_laps(lt, mad_k=mad_k, drop_first=drop_first, drop_last=drop_last)
    used = lt[keep]
    m.n_laps_used = int(len(used))
    m.excluded_laps = [int(v) for v in laps.loc[~keep.to_numpy(), "Lap"].tolist()]

    if m.n_laps_used == 0:
        m.notes.append("Every lap was filtered out — loosen the outlier filter "
                       "or keep the first/last lap.")
    else:
        # Median, not mean: one lap spent behind a slower car should not move
        # our estimate of the driver's true pace.
        m.median_lap_s = float(used.median())
        m.mean_lap_s = float(used.mean())
        m.best_lap_s = float(used.min())

        # Pace adherence: mean absolute deviation from the 210 s target. Absolute
        # because both directions are failures — a lap under target burns energy
        # we do not have, a lap over it loses distance.
        m.pace_adherence_s = float((used - target_s).abs().mean())
        m.median_delta_s = float(m.median_lap_s - target_s)

        # Consistency: sample standard deviation (ddof=1, since these laps are a
        # sample of the driver's ability, not the whole population).
        if m.n_laps_used >= 2:
            m.consistency_s = float(used.std(ddof=1))
        else:
            m.notes.append("Consistency needs at least two valid laps.")

    rate = throttle_derivative(df, laps, keep if m.n_laps_used else None)
    m.smoothness_var, m.smoothness_rms, m.smoothness_score = smoothness_from_rate(rate)
    if not np.isfinite(m.smoothness_var):
        m.notes.append("No throttle channel was found, so smoothness could not "
                       "be scored.")

    # --- Energy ----------------------------------------------------------
    lap_wh, m.energy_source = energy_per_lap(df, laps)
    if m.energy_source is None:
        m.notes.append(
            "No energy channels were found (power, or voltage + current, or an "
            "energy counter), so energy consumption could not be measured — "
            "throttle smoothness is the only efficiency signal available."
        )
    elif len(lap_wh):
        # Energy stats use the same kept laps as the pace stats, so a lap
        # dropped for traffic does not distort the energy figure either.
        kept_wh = pd.Series(lap_wh)[keep.to_numpy()] if m.n_laps_used else pd.Series(lap_wh)
        kept_wh = kept_wh[np.isfinite(kept_wh)]
        finite_all = lap_wh[np.isfinite(lap_wh)]

        if finite_all.size:
            m.total_energy_wh = float(np.sum(finite_all))
        if len(kept_wh):
            m.median_energy_wh = float(kept_wh.median())
            m.energy_delta_wh = m.median_energy_wh - budget_wh
            # Pace-corrected: how much more energy than a lap at THIS pace
            # should have cost. Positive = energy wasted by how they drove.
            if np.isfinite(m.median_lap_s):
                m.energy_excess_wh = (m.median_energy_wh
                                      - expected_energy_wh(m.median_lap_s, budget_wh))

            # Wh/km normalises away any difference in lap length, so a lap cut
            # short by a pit entry cannot look artificially efficient.
            if "Distance [m]" in laps:
                dist_km = (pd.to_numeric(laps["Distance [m]"], errors="coerce")
                           .to_numpy() / 1000.0)
                with np.errstate(divide="ignore", invalid="ignore"):
                    per_km = np.where(dist_km > 0.1, lap_wh / dist_km, np.nan)
                per_km = pd.Series(per_km)[keep.to_numpy()] if m.n_laps_used else pd.Series(per_km)
                per_km = per_km[np.isfinite(per_km)]
                if len(per_km):
                    m.wh_per_km = float(per_km.median())
        else:
            m.notes.append("Energy channels were found but no valid lap had a "
                           "usable energy figure.")

    return m


def metrics_table(metrics: list[DriverMetrics],
                  target_s: float = TARGET_LAP_TIME_S) -> pd.DataFrame:
    """Assemble a side-by-side comparison table for the drivers given."""
    def num(value: float, fmt: str = "{:.2f}") -> str:
        return fmt.format(value) if np.isfinite(value) else "—"

    rows = []
    for m in metrics:
        rows.append({
            "Driver": m.name,
            "Laps used": f"{m.n_laps_used} / {m.n_laps_total}",
            "Median lap": format_lap_time(m.median_lap_s),
            "Best lap": format_lap_time(m.best_lap_s),
            f"Median Δ vs {format_lap_time(target_s)}": format_delta(m.median_delta_s),
            "Pace adherence [s]": num(m.pace_adherence_s),
            "Consistency σ [s]": num(m.consistency_s),
            "Throttle rate RMS [%/s]": num(m.smoothness_rms, "{:.1f}"),
            "Smoothness score": num(m.smoothness_score, "{:.1f}"),
            "Energy [Wh/lap]": num(m.median_energy_wh, "{:.1f}"),
            "Energy excess [Wh/lap]": num(m.energy_excess_wh, "{:+.1f}"),
            "Efficiency [Wh/km]": num(m.wh_per_km, "{:.1f}"),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Driver Score — one number per driver, for a leaderboard
# --------------------------------------------------------------------------

# Half-credit references. Each is the value at which that metric scores 50/100:
# a driver exactly this far off gets half marks, better scores higher, worse
# lower. These are engineering judgements about what "good" looks like for a
# solar endurance stint at Zolder, and they are the numbers to argue with if the
# leaderboard ever looks wrong.
SCORE_REFERENCES = {
    "pace": 2.5,          # s of mean absolute deviation from target
    "consistency": 2.0,   # s of lap-time standard deviation
    "energy": 4.0,        # Wh/lap above the pace-matched expectation
}

# What each metric is worth. Pace and energy lead because they are what the race
# is actually limited by; consistency is the enabler; smoothness is only a proxy.
DEFAULT_SCORE_WEIGHTS = {
    "pace": 0.35,
    "consistency": 0.25,
    "energy": 0.30,
    "smoothness": 0.10,
}


def _diminishing(value: float, ref: float) -> float:
    """Map a "lower is better" metric onto 0-100 points.

        points = 100 * ref / (ref + value)

    Chosen over a linear scale because it is bounded, always monotonic, and has
    diminishing returns in both directions: it never goes negative for a very
    poor value (which would let one bad metric wipe out a whole score), and it
    cannot run away above 100 for an implausibly good one. It equals 50 exactly
    at `value == ref`, which is what makes the references above readable.

    Values below zero are clamped to zero, so a driver who beats the expectation
    (negative energy excess) simply takes full marks rather than over-scoring.
    """
    if not np.isfinite(value):
        return float("nan")
    return 100.0 * ref / (ref + max(float(value), 0.0))


def score_components(m: DriverMetrics) -> dict[str, float]:
    """The four 0-100 sub-scores behind a driver's overall score."""
    return {
        "pace": _diminishing(m.pace_adherence_s, SCORE_REFERENCES["pace"]),
        "consistency": _diminishing(m.consistency_s, SCORE_REFERENCES["consistency"]),
        "energy": _diminishing(m.energy_excess_wh, SCORE_REFERENCES["energy"]),
        "smoothness": m.smoothness_score,
    }


def driver_score(m: DriverMetrics,
                 weights: dict[str, float] | None = None) -> tuple[float, dict]:
    """Combine the four metrics into a single 0-100 Driver Score.

    Returns (score, per-component sub-scores).

    Three decisions worth knowing about:

      * **Absolute, not relative.** Each metric is scored against a fixed
        reference rather than against the other drivers in the comparison. A
        driver's score therefore does not change when someone else is added to
        or removed from the upload list, and scores are comparable across
        sessions. Peer-relative scoring (z-scores, min-max) would make the
        leaderboard shift under you for reasons that have nothing to do with
        driving.

      * **Smoothness is deliberately the smallest weight**, and when energy data
        is missing it inherits energy's weight instead. Smoothness exists to
        estimate energy waste, so with real watt-hours available it is close to
        redundant — and the synthetic sweep shows the two can disagree entirely
        (fast pedal oscillation is filtered out by vehicle inertia and costs
        almost nothing, while slow surging costs plenty and barely registers as
        pedal activity).

      * **Missing metrics are dropped, not zeroed.** The weights of whatever is
        available are renormalised to sum to 1, so a log with no throttle
        channel yields a slightly less informed score rather than a punished one.
    """
    weights = dict(weights or DEFAULT_SCORE_WEIGHTS)
    parts = score_components(m)

    # Without measured energy, the proxy carries the efficiency weight.
    if not np.isfinite(parts["energy"]):
        weights["smoothness"] = weights.get("smoothness", 0.0) + weights.get("energy", 0.0)
        weights["energy"] = 0.0

    total_w = sum(w for k, w in weights.items()
                  if w > 0 and np.isfinite(parts.get(k, float("nan"))))
    if total_w <= 0:
        return float("nan"), parts

    score = sum(weights[k] * parts[k] for k in weights
                if weights[k] > 0 and np.isfinite(parts.get(k, float("nan"))))
    # Clamp: every component is already within [0, 100] and the weights are
    # renormalised, so this only absorbs floating-point drift at the ends — but
    # the score is documented as 0-100 and should not print 100.00000000000001.
    return float(min(100.0, max(0.0, score / total_w))), parts


def leaderboard(metrics: list[DriverMetrics],
                weights: dict[str, float] | None = None) -> pd.DataFrame:
    """Drivers ordered by Driver Score, best first."""
    rows = []
    for m in metrics:
        if m.n_laps_used == 0:
            continue
        score, parts = driver_score(m, weights)
        rows.append({
            "Driver": m.name,
            "Driver score": score,
            "Pace pts": parts["pace"],
            "Consistency pts": parts["consistency"],
            "Energy pts": parts["energy"],
            "Smoothness pts": parts["smoothness"],
            "Median lap": format_lap_time(m.median_lap_s),
            "Pace adherence [s]": m.pace_adherence_s,
            "Consistency σ [s]": m.consistency_s,
            "Energy [Wh/lap]": m.median_energy_wh,
            "Energy excess [Wh/lap]": m.energy_excess_wh,
            "Throttle rate RMS [%/s]": m.smoothness_rms,
            "Laps used": f"{m.n_laps_used} / {m.n_laps_total}",
        })

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows).sort_values(
        "Driver score", ascending=False, na_position="last"
    ).reset_index(drop=True)
    out.insert(0, "Pos", range(1, len(out) + 1))
    return out


def rank_drivers(metrics: list[DriverMetrics]) -> tuple[pd.DataFrame, str]:
    """Rank drivers on the metrics, lower rank number being better.

    Returns (table, efficiency column used).

    Deliberately simple and transparent: each driver is ranked on each metric
    and the ranks are summed with equal weight. The point is to make the
    trade-offs visible (fast but erratic vs. slower but metronomic), not to
    collapse the decision into one authoritative score.

    Three slots are ranked — pace, consistency, and efficiency — and the
    efficiency slot takes whichever measure is available:

      * **Energy excess** when the car logged energy. This is the real thing:
        watt-hours above what a lap at that driver's own pace should cost.
      * **Throttle rate RMS** otherwise, which is only a proxy for the same
        quantity.

    Those two are NOT both used. Smoothness exists to estimate energy waste, so
    ranking on both would weight efficiency twice and quietly outvote pace and
    consistency together.
    """
    usable = [m for m in metrics if m.n_laps_used > 0]
    if len(usable) < 2:
        return pd.DataFrame(), ""

    # Energy is only usable as a ranking column if every driver has it —
    # otherwise the drivers with data would be ranked against blanks.
    have_energy = all(np.isfinite(m.energy_excess_wh) for m in usable)
    eff_col = "Energy excess [Wh/lap]" if have_energy else "Throttle rate RMS [%/s]"
    eff_vals = [m.energy_excess_wh if have_energy else m.smoothness_rms
                for m in usable]

    df = pd.DataFrame({
        "Driver": [m.name for m in usable],
        "Pace adherence [s]": [m.pace_adherence_s for m in usable],
        "Consistency σ [s]": [m.consistency_s for m in usable],
        eff_col: eff_vals,
    })

    # Every column here is "lower is better".
    rank_cols = []
    for col in ("Pace adherence [s]", "Consistency σ [s]", eff_col):
        r = f"rank({col})"
        df[r] = df[col].rank(method="min")
        rank_cols.append(r)

    df["Total rank"] = df[rank_cols].sum(axis=1)
    return df.sort_values("Total rank").reset_index(drop=True), eff_col
