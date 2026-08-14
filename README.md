# Driver Selection Dashboard

Compares driver stints from MoTeC telemetry to decide who drives the endurance
race at Circuit Zolder. Upload one `.ld` log for a single-driver report, or two
for a head-to-head.

```bash
pip install -r requirements.txt
streamlit run app.py
```

No real log to hand? Pick **Demo data** in the sidebar, or generate a pair of
sample files and upload them:

```bash
python sample_data.py     # -> samples/driver_a.ld + .ldx, driver_b.ld + .ldx
```

## Modules

| File | Responsibility |
|---|---|
| [app.py](app.py) | Streamlit UI: upload, controls, KPIs, charts |
| [motec_parser.py](motec_parser.py) | `.ld` binary + `.ldx` XML reading, channel matching, resampling, lap table |
| [metrics.py](metrics.py) | The four decision metrics and lap filtering |
| [plots.py](plots.py) | Plotly figures and the shared colour/chrome system |
| [sample_data.py](sample_data.py) | Synthetic stints + a `.ld`/`.ldx` writer |
| [test_analysis.py](test_analysis.py) | `python -m pytest test_analysis.py -q` — 34 tests |

Pipeline per log:

```
read_ld           .ld binary          -> channels at their native rates
to_dataframe      channels            -> one common time grid
build_lap_table   lap channel / .ldx  -> the logger's own lap division
add_lap_columns   lap table           -> per-row lap number + lap distance
compute_driver_metrics                -> the four metrics
```

## The metrics

| Metric | Definition |
|---|---|
| **Median lap time** | Median of the valid laps — robust to one lap lost in traffic, unlike the mean |
| **Pace adherence** | Mean of \|lap − 210 s\|. Absolute, because both directions are failures: under target burns energy we don't have, over it loses distance |
| **Consistency** | Sample standard deviation (ddof=1) of the valid lap times |
| **Smoothness** | Variance of d(Throttle)/dt in (%/s)², plus a 0–100 presentation score |

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

Two series colours (blue `#2a78d6` / orange `#eb6834` light, `#3987e5` /
`#d95926` dark), validated against both Streamlit surfaces for the lightness
band, chroma floor, colour-vision-deficiency separation, normal-vision floor and
3:1 contrast — worst pair ΔE 24.7 protan light, 26.8 dark. Colour follows the
*driver*, not their rank, so a change in who is quicker never repaints the chart.
Each quantity gets its own subplot row; there are no dual-axis plots, since
overlaying km/h and % on one y-scale would invent a correlation by choosing
where the scales align.

The overlay's x-axis is **distance into the lap**, not elapsed time: at 1,200 m
both drivers are at the same corner, so a gap between their speed traces is a
difference in driving rather than an artefact of one starting the lap earlier.
This needs a `Distance` channel; without one the chart falls back to elapsed time
and says so.
