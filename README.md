# Driver Selection Dashboard

Compares driver stints from MoTeC telemetry to decide who drives the endurance
race at Circuit Zolder. Upload **1 to 10** `.ld` logs — one gives a single-driver
report, two a head-to-head, three or more a scored leaderboard.

```bash
pip install -r requirements.txt
streamlit run app.py
```

No real log to hand? Pick **Demo data** in the sidebar and set the driver count
(2 for the head-to-head, 3+ for the leaderboard). Or generate sample files and
upload them:

```bash
python sample_data.py        # -> samples/driver_a.{ld,ldx} + driver_b
python sample_data.py 10     # -> a full ten-driver field
```

## Modules

| File | Responsibility |
|---|---|
| [app.py](app.py) | Streamlit UI: upload, controls, KPIs, leaderboard, charts |
| [motec_parser.py](motec_parser.py) | `.ld` binary + `.ldx` XML reading, channel matching, resampling, lap table |
| [metrics.py](metrics.py) | The decision metrics, lap filtering, energy, Driver Score |
| [plots.py](plots.py) | Plotly figures and the shared colour/chrome system |
| [sample_data.py](sample_data.py) | Synthetic stints + a `.ld`/`.ldx` writer |
| [test_analysis.py](test_analysis.py) | Parser and pace metrics — 34 tests |
| [test_energy.py](test_energy.py) | Energy, Driver Score, palette — 30 tests |

```bash
python -m pytest test_analysis.py test_energy.py -q     # 64 tests
```

Pipeline per log:

```
read_ld           .ld binary          -> channels at their native rates
to_dataframe      channels            -> one common time grid (+ P = V*I)
build_lap_table   lap channel / .ldx  -> the logger's own lap division
add_lap_columns   lap table           -> per-row lap number + lap distance
compute_driver_metrics                -> the five metrics
```

## The metrics

| Metric | Definition |
|---|---|
| **Median lap time** | Median of the valid laps — robust to one lap lost in traffic, unlike the mean |
| **Pace adherence** | Mean of \|lap − 210 s\|. Absolute, because both directions are failures: under target burns energy we don't have, over it loses distance |
| **Consistency** | Sample standard deviation (ddof=1) of the valid lap times |
| **Smoothness** | Variance of d(Throttle)/dt in (%/s)², plus a 0–100 presentation score |
| **Energy** | Wh per lap, Wh/km, and **energy excess** — Wh above what that driver's own pace should cost |

**Why the throttle derivative.** A driver holding 60% throttle and one
oscillating 40–80% can share the same mean and the same distribution of throttle
positions. What separates them is how fast the pedal moves, so the signal is
differentiated: pumping becomes large ± rate values. The *variance* is then taken
rather than the mean, because over a closed lap the pedal returns to where it
started — the mean rate is ≈ 0 for everyone. Variance also weights big stabs
quadratically, which matches the energy cost.

`np.gradient` is used against the real time vector, so the result is a true
rate in %/s rather than "% per sample". Two caveats, both handled:

- **Rate sensitivity.** Differentiating amplifies high frequencies, so a
  faster-sampled log resolves more sensor noise and reports a larger variance.
  The metric is therefore only comparable on a shared time grid — which
  `to_dataframe` guarantees, since every log is resampled onto one. Changing the
  resample cap moves all drivers together, so the ranking holds; don't quote the
  absolute value across sessions logged at different rates.
- **Lap seams.** The derivative is taken per lap and concatenated, never across
  a boundary, so an excluded lap or a pit stop can't inject one enormous
  spurious rate value at the seam.

## Energy — the binding constraint

This is a solar car, so energy is measured, not just proxied. Consumption comes
from whichever source the log offers, most trustworthy first:

1. a cumulative counter (`total_race_energy`) differenced per lap — the car's own
   coulomb-counted accounting, already net of regen;
2. **trapezoidal integration of power** over the lap, `E[Wh] = Σ (P₁+P₂)/2·Δt / 3600`,
   where power is `mms_power_W` or derived as `V × I` from `bms_voltage_V` and
   `bms_current_A`;
3. a per-lap channel (`last_lap_energy`), read one lap behind, since a "last lap"
   figure holds lap *N−1* while lap *N* is being driven.

Power is derived **after** resampling, because voltage and current are usually
logged at different rates and multiplying them on separate time bases would pair
samples taken at different moments. If the pack current logs discharge as
negative, the sign is detected from the median and flipped, with a warning.

**Energy excess is the metric that matters for driver selection.** Raw Wh/lap
would crown whoever drove slowest — going slower always uses less energy. So
consumption is compared against what a lap at that driver's *own* median pace
should cost, interpolated from the team's strategy table in
`Pit_Dashboard/constants.py` (189 s → 88 Wh … 231 s → 72 Wh). The difference is
the part attributable to the driver rather than to their speed.

**Smoothness is only a proxy for this, and the two can disagree outright.** The
car is a low-pass filter with a time constant of order *m/(ρ·CdA·v)* ≈ 90 s, so a
driver sawing at the pedal at 2 Hz wrecks their smoothness score while spending
essentially no extra energy; a driver who surges slowly spends real energy
because drag is convex in speed. `test_energy.py` pins this as a rank inversion.
Consequently the efficiency slot uses **energy when the car logged it, smoothness
only as a fallback** — never both, since that would weight efficiency twice.

## Driver Score and the leaderboard

From three drivers up, a leaderboard ranks the field on a single 0–100 score.
Each metric is mapped onto points against a fixed reference — the value worth
exactly half marks — then combined with the sidebar weights (default: pace 35%,
energy 30%, consistency 25%, smoothness 10%):

```
points = 100 · ref / (ref + value)
```

Bounded, always monotonic, equal to 50 at the reference. It cannot go negative,
so one catastrophic metric can't cancel out an otherwise strong driver, and it
can't run away above 100.

References: 2.5 s pace adherence, 2.0 s consistency σ, 4.0 Wh/lap energy excess.
These are engineering judgements — they are the numbers to argue with if the
board ever looks wrong.

Three deliberate properties:

- **Absolute, not peer-relative.** Adding or removing a driver never changes
  anyone else's score, and scores compare across sessions. Z-score or min-max
  normalisation would make the board shift for reasons unrelated to driving.
- **Missing metrics are dropped, not zeroed** — the remaining weights are
  renormalised, so an incomplete log gives a less informed score, not a punished
  one. A log with no energy data hands energy's weight to smoothness.
- **A second, independent method is shown alongside** (equal-weight rank-summing).
  Agreement between two different methods is a useful check; disagreement is
  itself informative.

**Lap filtering.** Outliers are rejected with a median-absolute-deviation test,
not mean/σ: MAD is itself robust, so a single five-minute lap barely moves it,
whereas it would inflate a standard deviation enough to hide itself. The out-lap
and in-lap are dropped by default. Rejected laps are always listed in the UI and
drawn as hollow markers on the pace chart rather than silently deleted.

## Laps come from the logger

The telemetry system already sums laps geographically at the start/finish
beacon, so nothing here re-detects a lap trigger. `build_lap_table` only
*reads* that existing division, trying in order of trustworthiness:

1. a lap-number channel (an explicit integer per lap);
2. a lap-time channel — either a running timer that resets at the line
   (sawtooth) or a hold of the last completed lap time (staircase), told apart
   by how much the signal moves within a segment;
3. a lap-distance channel that resets each lap.

`.ldx` beacon markers are used when selected in the sidebar, or automatically if
the `.ld` carries no lap channel. Prefer them when beacons were added or
corrected in i2 — those edits are saved to the `.ldx`, not back into the `.ld`.

## Notes on the file format

There is no maintained MoTeC parser on PyPI (neither `ldparser` nor
`motec-log-parser` publishes a distribution), so `motec_parser.py` implements the
binary layout directly. It follows the reverse-engineered `ld` format used by
MoTeC ADL/i2, matching the reference implementation at
[gotzl/ldparser](https://github.com/gotzl/ldparser); raw counts become physical
values via `(raw / scale · 10^-dec + shift) · mul`. `.ldx` is XML, read with the
standard library — MoTeC has never published that schema and it varies between
i2 versions, so the reader walks the tree for marker-like tags instead of binding
to a fixed element path, and infers the marker time unit from the log duration.

Because channel names vary between car configurations, channels are matched
against an alias list (`Corr Speed`, `Ground Speed`, `Speed`, …) and normalised
to km/h, %, deg, G and m regardless of how the logger was set up. The
**Channel mapping** expander shows exactly which log channel fed each metric —
worth a glance before trusting the numbers on an unfamiliar log.

Round-trip tested: `sample_data.write_ld` writes the same layout the parser
reads, which is what verifies the struct offsets, the unit conversion and the
per-channel resampling without needing a confidential team log.

## Charts

**Eight** validated categorical hues, stepped per theme, checked against both
Streamlit surfaces (`#ffffff` / `#0e1117`) for the lightness band, chroma floor,
colour-vision-deficiency separation, normal-vision floor and 3:1 contrast — worst
adjacent pair ΔE 9.1 protan light, 8.4 dark. Marker symbols give identity a
second channel beyond hue. On the light surface three hues sit below 3:1
contrast, which obliges the relief rule — hence the table view beside every
chart.

Drivers 9 and 10 reuse the first two hues with a **dashed** line rather than an
invented ninth colour, which would be indistinguishable from an existing slot
under CVD. Identity is then carried by hue *and* line style together.

Colour follows the **driver**, not their position in any list: a driver whose log
had no laps doesn't shift everyone else's colour, and filtering the overlay down
to two drivers never repaints the survivors.

Each quantity gets its own subplot row; there are no dual-axis plots, since
overlaying km/h and W on one y-scale would invent a correlation by choosing where
the scales align.

The overlay's x-axis is **distance into the lap**, not elapsed time: at 1,200 m
both drivers are at the same corner, so a gap between their speed traces is a
difference in driving rather than an artefact of one starting the lap earlier.
This needs a `Distance` channel; without one the chart falls back to elapsed time
and says so. A **power** row is added when the log has it — that's the one that
shows which corner exit actually cost the watt-hours.

## Synthetic data

`sample_data.py` models the car so the demo is physically honest rather than
decorative: tractive force from inertia + rolling resistance + aerodynamic drag,
asymmetric drivetrain (90% out) and regen (55% back), pack voltage sagging as
`V_oc − I·R`, and a 35 W auxiliary load. Crr and CdA are calibrated so a steady
210 s lap lands on the team's 80 Wh budget (it comes out at 80.2 Wh).

Driver style is two separate knobs because it is two separate physical
mechanisms: `pump_amplitude`/`pump_hz` drive the *throttle* trace (and so the
smoothness metric), while `surge_pct` drives slow *speed* variation (and so the
energy metric). Coupling fast pedal ripple into speed — the first thing I
tried — is wrong, and produced 22 kW peaks and 625 A before the vehicle time
constant was accounted for.
