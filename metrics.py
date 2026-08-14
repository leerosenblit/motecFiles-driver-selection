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
# Top-level metric computation
# --------------------------------------------------------------------------

def compute_driver_metrics(name: str, df: pd.DataFrame, laps: pd.DataFrame,
                           target_s: float = TARGET_LAP_TIME_S,
                           mad_k: float = 3.0,
                           drop_first: bool = True,
                           drop_last: bool = True) -> DriverMetrics:
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

    return m


def metrics_table(metrics: list[DriverMetrics],
                  target_s: float = TARGET_LAP_TIME_S) -> pd.DataFrame:
    """Assemble a side-by-side comparison table for the drivers given."""
    rows = []
    for m in metrics:
        rows.append({
            "Driver": m.name,
            "Laps used": f"{m.n_laps_used} / {m.n_laps_total}",
            "Median lap": format_lap_time(m.median_lap_s),
            "Best lap": format_lap_time(m.best_lap_s),
            f"Median Δ vs {format_lap_time(target_s)}": format_delta(m.median_delta_s),
            "Pace adherence [s]": (f"{m.pace_adherence_s:.2f}"
                                   if np.isfinite(m.pace_adherence_s) else "—"),
            "Consistency σ [s]": (f"{m.consistency_s:.2f}"
                                  if np.isfinite(m.consistency_s) else "—"),
            "Throttle rate RMS [%/s]": (f"{m.smoothness_rms:.1f}"
                                        if np.isfinite(m.smoothness_rms) else "—"),
            "Smoothness score": (f"{m.smoothness_score:.1f}"
                                 if np.isfinite(m.smoothness_score) else "—"),
        })
    return pd.DataFrame(rows)


def rank_drivers(metrics: list[DriverMetrics]) -> pd.DataFrame:
    """Rank drivers on the metrics, lower rank number being better.

    Deliberately simple and transparent: each driver is ranked on each metric
    and the ranks are summed with equal weight. The point is to make the
    trade-offs visible (fast but erratic vs. slower but metronomic), not to
    collapse the decision into one authoritative score.
    """
    usable = [m for m in metrics if m.n_laps_used > 0]
    if len(usable) < 2:
        return pd.DataFrame()

    df = pd.DataFrame({
        "Driver": [m.name for m in usable],
        "Pace adherence [s]": [m.pace_adherence_s for m in usable],
        "Consistency σ [s]": [m.consistency_s for m in usable],
        "Throttle rate RMS [%/s]": [m.smoothness_rms for m in usable],
    })

    # Every column here is "lower is better".
    rank_cols = []
    for col in ("Pace adherence [s]", "Consistency σ [s]", "Throttle rate RMS [%/s]"):
        r = f"rank({col})"
        df[r] = df[col].rank(method="min")
        rank_cols.append(r)

    df["Total rank"] = df[rank_cols].sum(axis=1)
    return df.sort_values("Total rank").reset_index(drop=True)
