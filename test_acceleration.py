# test_acceleration.py — acceleration metrics, the score-weight swap, and the
# average-lap overlay trace
#
#     python -m pytest test_acceleration.py -q

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import metrics as M
import motec_parser as P
import plots
from sample_data import synth_stint


# --------------------------------------------------------------------------
# g_lon channel identification and unit handling
# --------------------------------------------------------------------------

def test_glon_channel_recognised_by_alias():
    log = P.LDLog(channels=[
        P.Channel("CG Accel Longitudinal", "glon", "G", 50, np.zeros(10)),
    ])
    found = P.identify_channels(log)
    assert found["g_lon"].name == "CG Accel Longitudinal"


def test_glat_and_glon_metres_per_second_squared_converted_to_g():
    """A pre-existing gap: CHANNEL_UNIT_HINTS accepted m/s^2 for g_lat/g_lon,
    but nothing converted it, so such a channel silently stayed ~9.8x too big."""
    lat = P.Channel("G Force Lat", "gl", "m/s2", 10, np.array([9.80665, -4.903325]))
    assert np.allclose(P._convert_units("g_lat", lat), [1.0, -0.5])

    lon = P.Channel("G Force Lon", "gn", "m/s^2", 10, np.array([19.6133]))
    assert P._convert_units("g_lon", lon)[0] == pytest.approx(2.0, rel=1e-3)

    # A channel already in G must pass through unchanged.
    already_g = P.Channel("G Force Lat", "gl", "G", 10, np.array([0.42]))
    assert P._convert_units("g_lat", already_g)[0] == pytest.approx(0.42)


def test_glon_alias_does_not_collide_with_glat():
    """Two distinct channels must resolve to two distinct canonical keys."""
    log = P.LDLog(channels=[
        P.Channel("G Force Lat", "gl", "G", 50, np.full(10, 0.3)),
        P.Channel("G Force Lon", "gn", "G", 50, np.full(10, 0.1)),
    ])
    found = P.identify_channels(log)
    assert found["g_lat"].name == "G Force Lat"
    assert found["g_lon"].name == "G Force Lon"


# --------------------------------------------------------------------------
# acceleration_stats()
# --------------------------------------------------------------------------

def _df_with_g(lon=None, lat=None, n=None):
    n = n if n is not None else (len(lon) if lon is not None else len(lat))
    data = {"Time [s]": np.arange(n) / 10.0}
    if lon is not None:
        data[P.CHANNEL_LABELS["g_lon"]] = lon
    if lat is not None:
        data[P.CHANNEL_LABELS["g_lat"]] = lat
    return pd.DataFrame(data)


def test_avg_accel_and_decel_split_on_sign():
    # +0.2, +0.4 accelerating; -0.1, -0.3 braking.
    lon = np.array([0.2, 0.4, -0.1, -0.3])
    df = _df_with_g(lon=lon)
    stats = M.acceleration_stats(df)
    assert stats["avg_accel_g"] == pytest.approx(0.3)     # mean(0.2, 0.4)
    assert stats["avg_decel_g"] == pytest.approx(0.2)     # mean(|-0.1|, |-0.3|)


def test_avg_and_max_lateral_use_absolute_value():
    """A signed mean of left/right corners would read near zero and carry no
    information; the average must use |g_lat|, same as the specified max."""
    lat = np.array([0.6, -0.6, 0.3, -0.9])
    df = _df_with_g(lat=lat)
    stats = M.acceleration_stats(df)
    assert stats["avg_lat_g"] == pytest.approx(np.mean([0.6, 0.6, 0.3, 0.9]))
    assert stats["max_lat_g"] == pytest.approx(0.9)
    # A plain signed mean would have been (0.6-0.6+0.3-0.9)/4 = -0.15 — nowhere
    # near the correct answer, which is the point of this test.
    assert stats["avg_lat_g"] != pytest.approx(np.mean(lat))


def test_missing_g_channels_reported_as_nan_not_crashed():
    df = pd.DataFrame({"Time [s]": np.arange(5) / 10.0})
    stats = M.acceleration_stats(df)
    assert all(not np.isfinite(v) for v in stats.values())


def test_acceleration_stats_restricted_to_kept_laps():
    """A lap flagged as an outlier (traffic, out-lap) must not leak into the
    acceleration figures, matching the convention used by every other metric."""
    lon = np.concatenate([np.full(50, 0.5), np.full(50, 5.0)])   # lap 2 = absurd
    df = _df_with_g(lon=lon)
    laps = pd.DataFrame({
        "Lap": [1, 2], "LapTime [s]": [210.0, 210.0],
        "start_idx": [0, 50], "end_idx": [49, 99],
    })
    keep_all = pd.Series([True, True])
    keep_lap1_only = pd.Series([True, False])

    both = M.acceleration_stats(df, laps, keep_all)
    filtered = M.acceleration_stats(df, laps, keep_lap1_only)
    assert both["avg_accel_g"] == pytest.approx(2.75)     # mean(0.5, 5.0)
    assert filtered["avg_accel_g"] == pytest.approx(0.5)  # lap 2 excluded


def test_acceleration_stats_falls_back_to_whole_log_with_no_lap_table():
    lon = np.array([1.0, -1.0, 1.0, -1.0])
    df = _df_with_g(lon=lon)
    stats = M.acceleration_stats(df, None, None)
    assert stats["avg_accel_g"] == pytest.approx(1.0)
    assert stats["avg_decel_g"] == pytest.approx(1.0)


def test_compute_driver_metrics_wires_up_acceleration_fields():
    df, _ = synth_stint("d", n_laps=4, seed=9)
    laps = P.build_lap_table(df)
    m = M.compute_driver_metrics("d", P.add_lap_columns(df, laps), laps)
    # synth_stint's G Force Lon channel exists, so all four should be finite.
    assert np.isfinite(m.avg_accel_g)
    assert np.isfinite(m.avg_decel_g)
    assert np.isfinite(m.avg_lat_g)
    assert np.isfinite(m.max_lat_g)
    assert m.max_lat_g >= m.avg_lat_g >= 0


def test_missing_longitudinal_channel_reported_in_notes():
    df, _ = synth_stint("d", n_laps=3, seed=2)
    laps = P.build_lap_table(df)
    stripped = P.add_lap_columns(df, laps).drop(columns=["G Force Lon [G]"])
    m = M.compute_driver_metrics("d", stripped, laps)
    assert not np.isfinite(m.avg_accel_g)
    assert not np.isfinite(m.avg_decel_g)
    assert np.isfinite(m.avg_lat_g)          # lateral channel untouched
    assert any("longitudinal" in n.lower() for n in m.notes)


def test_metrics_table_includes_acceleration_columns():
    df, _ = synth_stint("d", n_laps=3, seed=1)
    laps = P.build_lap_table(df)
    m = M.compute_driver_metrics("d", P.add_lap_columns(df, laps), laps)
    table = M.metrics_table([m])
    for col in ("Avg accel [G]", "Avg decel [G]", "Avg lateral G", "Max lateral G"):
        assert col in table.columns


# --------------------------------------------------------------------------
# Driver Score weight swap
# --------------------------------------------------------------------------

def test_pace_and_consistency_weights_are_swapped():
    assert M.DEFAULT_SCORE_WEIGHTS["pace"] == pytest.approx(0.25)
    assert M.DEFAULT_SCORE_WEIGHTS["consistency"] == pytest.approx(0.35)
    # Energy and smoothness untouched by the swap.
    assert M.DEFAULT_SCORE_WEIGHTS["energy"] == pytest.approx(0.30)
    assert M.DEFAULT_SCORE_WEIGHTS["smoothness"] == pytest.approx(0.10)
    assert sum(M.DEFAULT_SCORE_WEIGHTS.values()) == pytest.approx(1.0)


def _fake(name, pace, cons, excess=1.0, smooth_score=60.0, laps=10):
    m = M.DriverMetrics(name=name, n_laps_used=laps, n_laps_total=laps)
    m.median_lap_s = 210.0
    m.pace_adherence_s = pace
    m.consistency_s = cons
    m.energy_excess_wh = excess
    m.median_energy_wh = 80.0 + excess
    m.smoothness_score = smooth_score
    m.smoothness_rms = 20.0
    return m


def test_weight_swap_changes_who_the_leaderboard_prefers():
    """The swap is only meaningful if it can flip a decision. Construct a driver
    who is markedly better on pace but worse on consistency than another (equal
    on everything else), and confirm the DEFAULT weighting (consistency-heavy)
    now favours the consistent one — the opposite of the pre-swap 0.35/0.25
    split. The gap is made large and asymmetric on purpose: with only 10
    percentage points moving between the two weights, a mild scenario (checked
    by hand first) does not flip — near-saturating both diminishing-returns
    curves is what makes a 10-point weight shift decisive rather than marginal.
    """
    sharp_but_erratic = _fake("sharp", pace=0.05, cons=20.0)
    steady_but_off = _fake("steady", pace=20.0, cons=0.05)

    board = M.leaderboard([sharp_but_erratic, steady_but_off])
    assert board.iloc[0]["Driver"] == "steady"

    pre_swap_weights = {"pace": 0.35, "consistency": 0.25,
                        "energy": 0.30, "smoothness": 0.10}
    board_pre_swap = M.leaderboard([sharp_but_erratic, steady_but_off],
                                   weights=pre_swap_weights)
    assert board_pre_swap.iloc[0]["Driver"] == "sharp"


def test_score_still_bounded_and_sums_to_one_after_swap():
    score, parts = M.driver_score(_fake("x", 1.0, 1.0, 1.0))
    assert 0.0 <= score <= 100.0
    assert set(parts) == {"pace", "consistency", "energy", "smoothness"}


# --------------------------------------------------------------------------
# average_lap_trace()
# --------------------------------------------------------------------------

def _distance_log(per_lap_speeds: list[float], track_m: float = 1000.0,
                  freq: float = 10.0, n_per_lap: int = 100) -> tuple[pd.DataFrame, pd.DataFrame]:
    """A log with `len(per_lap_speeds)` laps, each held at a constant speed
    (km/h) so the analytically expected average is trivial to compute by hand."""
    dfs, rows = [], []
    t_cursor = 0.0
    for lap, speed_kmh in enumerate(per_lap_speeds, start=1):
        speed_ms = speed_kmh / 3.6
        lap_time = track_m / speed_ms
        t = t_cursor + np.arange(n_per_lap) / freq * (lap_time / (n_per_lap / freq))
        # Simpler: just space samples evenly across [0, lap_time).
        t = t_cursor + np.linspace(0.0, lap_time, n_per_lap, endpoint=False)
        dist = np.linspace(0.0, track_m, n_per_lap, endpoint=False)
        dfs.append(pd.DataFrame({
            "Time [s]": t,
            "Distance [m]": dist,
            "Corr Speed [km/h]": np.full(n_per_lap, speed_kmh),
            "Lap Number": np.full(n_per_lap, lap, dtype=float),
        }))
        rows.append({"Lap": lap, "LapTime [s]": lap_time})
        t_cursor += lap_time
    df = pd.concat(dfs, ignore_index=True)
    return df, pd.DataFrame(rows)


def test_average_trace_matches_hand_computed_mean_of_constant_laps():
    """Three laps, each at a different CONSTANT speed. The average trace at
    every distance point must equal the plain mean of the three constants —
    the simplest possible case to verify by hand."""
    df, laps = _distance_log([60.0, 80.0, 100.0])
    lap_table = P.build_lap_table(df)
    annotated = P.add_lap_columns(df, lap_table)
    keep = pd.Series([True, True, True])

    avg = P.average_lap_trace(annotated, lap_table, keep, ["Corr Speed [km/h]"])
    assert not avg.empty
    assert np.allclose(avg["Corr Speed [km/h]"], 80.0, atol=1e-6)   # mean(60,80,100)


def test_average_trace_grid_spans_only_the_shortest_lap():
    """A longer lap must not have its tail extrapolated into the average —
    the grid should stop at the shortest kept lap's distance."""
    df, laps = _distance_log([60.0, 60.0], track_m=1000.0)
    # Shorten lap 2's recorded distance so it is the "shortest" kept lap.
    lap_table = P.build_lap_table(df)
    short_end = int(lap_table.iloc[1].start_idx) + 40     # cut lap 2 short
    df.loc[short_end:, "Distance [m]"] = np.nan
    # Rebuild with the truncated data.
    annotated = P.add_lap_columns(df, lap_table)
    keep = pd.Series([True, True])

    avg = P.average_lap_trace(annotated, lap_table, keep, ["Corr Speed [km/h]"])
    full_lap_len = df["Distance [m]"].to_numpy()[
        int(lap_table.iloc[0].start_idx):int(lap_table.iloc[0].end_idx) + 1
    ].max()
    assert avg["Lap Distance [m]"].max() < full_lap_len - 1.0


def test_average_trace_with_one_kept_lap_equals_that_lap():
    df, laps = _distance_log([90.0, 90.0])
    lap_table = P.build_lap_table(df)
    annotated = P.add_lap_columns(df, lap_table)
    keep = pd.Series([True, False])       # only lap 1 kept

    avg = P.average_lap_trace(annotated, lap_table, keep, ["Corr Speed [km/h]"])
    assert np.allclose(avg["Corr Speed [km/h]"], 90.0, atol=1e-6)


def test_average_trace_falls_back_to_all_laps_when_none_kept():
    df, laps = _distance_log([70.0, 70.0])
    lap_table = P.build_lap_table(df)
    annotated = P.add_lap_columns(df, lap_table)
    keep = pd.Series([False, False])      # every lap excluded

    avg = P.average_lap_trace(annotated, lap_table, keep, ["Corr Speed [km/h]"])
    assert not avg.empty                  # still produces something, not empty


def test_average_trace_empty_without_distance_channel():
    df, laps = _distance_log([70.0])
    lap_table = P.build_lap_table(df)
    bare = P.add_lap_columns(df, lap_table).drop(columns=["Lap Distance [m]"])
    avg = P.average_lap_trace(bare, lap_table, pd.Series([True]), ["Corr Speed [km/h]"])
    assert avg.empty


def test_average_trace_on_real_shaped_synthetic_stint():
    """End-to-end on the actual demo-data generator, not a hand-built fixture."""
    df, _ = synth_stint("d", n_laps=6, lap_sigma_s=0.0, seed=3)
    laps = P.build_lap_table(df)
    annotated = P.add_lap_columns(df, laps)
    keep = M.flag_outlier_laps(laps["LapTime [s]"])

    cols = [P.CHANNEL_LABELS[k] for k, _ in plots.OVERLAY_ROWS]
    avg = P.average_lap_trace(annotated, laps, keep, cols)
    assert len(avg) == 200                                   # default n_points
    assert avg["Lap Distance [m]"].iloc[0] == pytest.approx(0.0)
    assert avg["Lap Distance [m]"].is_monotonic_increasing
    for col in cols:
        assert col in avg.columns
        assert avg[col].notna().all()


# --------------------------------------------------------------------------
# overlay_chart() with the generalised "label" field
# --------------------------------------------------------------------------

def test_overlay_chart_uses_label_field_not_lap_number():
    df = pd.DataFrame({
        "Lap Distance [m]": np.linspace(0, 100, 10),
        "Corr Speed [km/h]": np.linspace(50, 90, 10),
    })
    fig = plots.overlay_chart(
        [{"name": "Driver X", "label": "avg of 7 laps", "data": df, "slot": 0}],
        mode="light",
    )
    names = {t.name for t in fig.data}
    assert "Driver X — avg of 7 laps" in names


def test_overlay_chart_tolerates_a_missing_label():
    df = pd.DataFrame({
        "Lap Distance [m]": np.linspace(0, 100, 10),
        "Corr Speed [km/h]": np.linspace(50, 90, 10),
    })
    # No "label" key at all — must not KeyError.
    fig = plots.overlay_chart([{"name": "Driver X", "data": df, "slot": 0}])
    assert len(fig.data) >= 1
