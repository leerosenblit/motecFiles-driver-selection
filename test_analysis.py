# test_analysis.py — checks on the parser, the lap division and the metrics
#
#     python -m pytest test_analysis.py -q
#
# The .ld tests round-trip through sample_data.write_ld, which writes the same
# binary layout motec_parser reads. That verifies the struct offsets, the
# raw->physical conversion and the per-channel resampling without needing a
# confidential team log.

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import metrics as M
import motec_parser as P
from sample_data import TARGET_LAP_S, TRACK_LENGTH_M, synth_stint, write_ld


@pytest.fixture(scope="module")
def stint():
    return synth_stint("Test Driver", n_laps=6, lap_sigma_s=2.0, seed=5)


@pytest.fixture(scope="module")
def ld_path(tmp_path_factory, stint):
    df, _laps = stint
    dest = tmp_path_factory.mktemp("logs") / "test.ld"
    return write_ld(str(dest), df, driver="Test Driver", venue="Zolder")


# --------------------------------------------------------------------------
# Binary parsing
# --------------------------------------------------------------------------

def test_header_round_trips(ld_path):
    log = P.read_ld(ld_path)
    assert log.driver == "Test Driver"
    assert log.venue == "Zolder"
    assert log.warnings == []
    assert len(log.channels) == 7


def test_channel_values_and_rates_round_trip(ld_path, stint):
    df_src, _ = stint
    log = P.read_ld(ld_path)

    speed = log.channel("Corr Speed")
    assert speed is not None
    assert speed.unit == "km/h"
    assert speed.freq == 25
    # Written as float32 at 25 Hz from a 20 Hz source, so compare ranges rather
    # than sample-for-sample.
    assert speed.data.min() == pytest.approx(df_src["Corr Speed [km/h]"].min(), abs=1.0)
    assert speed.data.max() == pytest.approx(df_src["Corr Speed [km/h]"].max(), abs=1.0)

    # Channels really are stored at different rates.
    assert {c.name: c.freq for c in log.channels}["Distance"] == 5
    assert {c.name: c.freq for c in log.channels}["Steering Angle"] == 10


def test_read_ld_accepts_bytes_and_file_objects(ld_path):
    import io

    with open(ld_path, "rb") as fh:
        raw = fh.read()
    assert P.read_ld(raw).driver == "Test Driver"
    assert P.read_ld(io.BytesIO(raw)).driver == "Test Driver"


def test_read_ld_rejects_garbage():
    with pytest.raises(ValueError):
        P.read_ld(b"not an ld file")


def test_resampling_puts_all_channels_on_one_grid(ld_path):
    log = P.read_ld(ld_path)
    df, found = P.to_dataframe(log, max_hz=25.0)

    assert set(found) == {"speed", "throttle", "steering", "g_lat",
                          "distance", "lap_time", "lap_number"}
    assert df["Time [s]"].is_monotonic_increasing
    # One row count for every channel, and no gaps left by the resample.
    for key in found:
        assert df[P.CHANNEL_LABELS[key]].notna().all()
    dt = np.diff(df["Time [s]"].to_numpy())
    assert np.allclose(dt, 1 / 25.0)


def test_lap_number_resample_stays_integral(ld_path):
    """A step channel must not be linearly interpolated into fractional laps."""
    log = P.read_ld(ld_path)
    df, _ = P.to_dataframe(log)
    lap = df[P.CHANNEL_LABELS["lap_number"]].to_numpy()
    assert np.allclose(lap, np.round(lap))


# --------------------------------------------------------------------------
# Lap division — every source path
# --------------------------------------------------------------------------

def test_laps_from_lap_number_channel(stint):
    df, src_laps = stint
    laps = P.build_lap_table(df)
    assert laps.attrs["lap_source"].startswith("Lap Number")
    assert len(laps) == len(src_laps)
    # The lap times the logger recorded come back, not our grid-quantised guess.
    assert np.allclose(laps["LapTime [s]"], src_laps["LapTime [s]"], atol=0.15)
    # Each lap covers a lap of the circuit.
    assert np.allclose(laps["Distance [m]"], TRACK_LENGTH_M, atol=25.0)


def test_laps_from_running_timer_sawtooth(stint):
    """With no lap counter, the sawtooth lap timer must still divide the stint."""
    df, src_laps = stint
    df2 = df.drop(columns=["Lap Number"])
    laps = P.build_lap_table(df2)
    assert "resets" in laps.attrs["lap_source"]
    assert len(laps) == len(src_laps)
    assert np.allclose(laps["LapTime [s]"][:-1], src_laps["LapTime [s]"][:-1], atol=0.25)


def test_laps_from_staircase_lap_time(stint):
    """A 'last completed lap time' channel holds a value and steps at the line."""
    df, src_laps = stint
    df2 = df.drop(columns=["Lap Number"]).copy()
    staircase = np.zeros(len(df2))
    lap_col = df["Lap Number"].to_numpy()
    for idx, row in src_laps.iterrows():
        staircase[lap_col == row["Lap"]] = row["LapTime [s]"]
    df2["LapTime [s]"] = staircase

    laps = P.build_lap_table(df2)
    assert "steps" in laps.attrs["lap_source"]
    # A staircase only reveals a boundary when the held value changes, so we
    # recover one segment per step.
    assert len(laps) >= len(src_laps) - 2


def test_laps_from_distance_resets(stint):
    """Fall back to a lap-distance channel that returns to zero each lap."""
    df, src_laps = stint
    df2 = df.drop(columns=["Lap Number", "LapTime [s]"]).copy()
    lap_col = df["Lap Number"].to_numpy()
    d = df2["Distance [m]"].to_numpy().copy()
    for lap in np.unique(lap_col):                  # make it per-lap distance
        mask = lap_col == lap
        d[mask] -= d[mask][0]
    df2["Distance [m]"] = d

    laps = P.build_lap_table(df2)
    assert "resets" in laps.attrs["lap_source"]
    assert len(laps) == len(src_laps)


def test_laps_from_ldx_markers(stint):
    """Beacon markers in the .ldx define the laps when no channel does."""
    df, src_laps = stint
    starts = [0.0]
    for t in src_laps["LapTime [s]"]:
        starts.append(starts[-1] + float(t))

    ldx = "<LDXFile><Layers><Layer><MarkerBlock><MarkerGroup>" + "".join(
        f'<Marker Time="{int(round(t * 1e6))}" Name=""/>' for t in starts
    ) + "</MarkerGroup></MarkerBlock></Layer></Layers></LDXFile>"

    markers = P.read_ldx_markers(ldx.encode(), log_duration_s=float(df["Time [s]"].iloc[-1]))
    assert len(markers) == len(starts)
    assert markers[-1] == pytest.approx(starts[-1], abs=0.01)

    laps = P.apply_ldx_laps(df.drop(columns=["Lap Number", "LapTime [s]"]), markers)
    assert laps.attrs["lap_source"] == ".ldx beacon markers"
    assert len(laps) == len(src_laps)
    assert np.allclose(laps["LapTime [s]"], src_laps["LapTime [s]"], atol=0.02)


def test_ldx_unit_autodetection(stint):
    """Marker times in milliseconds must not be read as microseconds."""
    df, _ = stint
    duration = float(df["Time [s]"].iloc[-1])
    ldx = ('<LDXFile><MarkerBlock>'
           '<Marker Time="0"/><Marker Time="210000"/><Marker Time="420000"/>'
           '</MarkerBlock></LDXFile>')
    markers = P.read_ldx_markers(ldx.encode(), log_duration_s=duration)
    # 420000 as microseconds would be 0.42 s — nonsense for a 3-lap gap; as
    # milliseconds it is 420 s, which fits inside the log.
    assert markers[-1] == pytest.approx(420.0)


def test_no_lap_source_returns_empty_table(stint):
    df, _ = stint
    bare = df[["Time [s]", "Corr Speed [km/h]", "Throttle Pos [%]"]]
    laps = P.build_lap_table(bare)
    assert len(laps) == 0
    assert laps.attrs["lap_source"] == "none"


def test_lap_distance_resets_each_lap(stint):
    df, _ = stint
    laps = P.build_lap_table(df)
    annotated = P.add_lap_columns(df, laps)
    for lap in laps["Lap"]:
        seg = annotated.loc[annotated["_Lap"] == lap, "Lap Distance [m]"]
        assert seg.iloc[0] == pytest.approx(0.0, abs=1e-6)
        assert seg.max() == pytest.approx(TRACK_LENGTH_M, abs=25.0)


# --------------------------------------------------------------------------
# Unit normalisation
# --------------------------------------------------------------------------

def test_speed_converted_from_metres_per_second():
    ch = P.Channel(name="Corr Speed", short_name="spd", unit="m/s",
                   freq=10, data=np.array([10.0, 20.0]))
    assert np.allclose(P._convert_units("speed", ch), [36.0, 72.0])


def test_throttle_ratio_scaled_to_percent():
    ch = P.Channel(name="Throttle Pos", short_name="thr", unit="",
                   freq=10, data=np.array([0.0, 0.5, 1.0]))
    assert np.allclose(P._convert_units("throttle", ch), [0.0, 50.0, 100.0])
    # An already-percent channel must be left alone.
    ch2 = P.Channel(name="Throttle Pos", short_name="thr", unit="%",
                    freq=10, data=np.array([0.0, 50.0, 100.0]))
    assert np.allclose(P._convert_units("throttle", ch2), [0.0, 50.0, 100.0])


def test_distance_converted_from_km():
    ch = P.Channel(name="Distance", short_name="d", unit="km",
                   freq=1, data=np.array([0.0, 1.5]))
    assert np.allclose(P._convert_units("distance", ch), [0.0, 1500.0])


def test_lap_time_converted_from_milliseconds():
    ch = P.Channel(name="Lap Time", short_name="lt", unit="ms",
                   freq=1, data=np.array([0.0, 210_000.0]))
    assert np.allclose(P._convert_units("lap_time", ch), [0.0, 210.0])


def test_channel_alias_matching_prefers_corr_speed():
    log = P.LDLog(channels=[
        P.Channel("Speed", "spd", "km/h", 10, np.zeros(10)),
        P.Channel("Corr Speed", "cspd", "km/h", 10, np.zeros(10)),
    ])
    assert P.identify_channels(log)["speed"].name == "Corr Speed"


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------

def test_median_ignores_a_traffic_lap():
    laps = pd.Series([210.0, 211.0, 209.0, 300.0, 210.5, 209.5, 210.0])
    keep = M.flag_outlier_laps(laps, drop_first=False, drop_last=False)
    assert not keep.iloc[3]                       # the 300 s lap is rejected
    assert keep.sum() == 6


def test_out_and_in_laps_dropped_by_default():
    laps = pd.Series([222.0, 210.0, 210.5, 209.5, 219.0])
    keep = M.flag_outlier_laps(laps)
    assert not keep.iloc[0] and not keep.iloc[-1]
    assert keep.sum() == 3


def test_outlier_filter_survives_identical_laps():
    """MAD is zero here; the filter must not reject everything."""
    laps = pd.Series([210.0] * 6)
    keep = M.flag_outlier_laps(laps, drop_first=False, drop_last=False)
    assert keep.all()


def test_pace_adherence_is_absolute():
    """A lap 5 s under target is as much a miss as 5 s over."""
    fast = pd.DataFrame({"Lap": [1, 2, 3], "LapTime [s]": [205.0, 205.0, 205.0],
                         "start_idx": [0, 1, 2], "end_idx": [0, 1, 2]})
    slow = fast.copy()
    slow["LapTime [s]"] = [215.0, 215.0, 215.0]
    df = pd.DataFrame({"Time [s]": [0.0, 1.0, 2.0]})

    a = M.compute_driver_metrics("fast", df, fast, drop_first=False, drop_last=False)
    b = M.compute_driver_metrics("slow", df, slow, drop_first=False, drop_last=False)
    assert a.pace_adherence_s == pytest.approx(5.0)
    assert b.pace_adherence_s == pytest.approx(5.0)
    # The signed median delta still distinguishes them.
    assert a.median_delta_s == pytest.approx(-5.0)
    assert b.median_delta_s == pytest.approx(+5.0)


def test_consistency_is_sample_std():
    laps = pd.DataFrame({"Lap": [1, 2, 3, 4], "LapTime [s]": [208.0, 210.0, 212.0, 210.0],
                         "start_idx": [0, 1, 2, 3], "end_idx": [0, 1, 2, 3]})
    df = pd.DataFrame({"Time [s]": [0.0, 1.0, 2.0, 3.0]})
    m = M.compute_driver_metrics("x", df, laps, drop_first=False, drop_last=False)
    expected = pd.Series([208.0, 210.0, 212.0, 210.0]).std(ddof=1)
    assert m.consistency_s == pytest.approx(expected)


def test_smoothness_separates_a_pumper_from_a_smooth_driver():
    smooth, _ = synth_stint("smooth", n_laps=3, pump_amplitude=1.0, seed=1)
    pumper, _ = synth_stint("pumper", n_laps=3, pump_amplitude=12.0, seed=1)

    s_var = M.smoothness_from_rate(M.throttle_derivative(smooth))[0]
    p_var = M.smoothness_from_rate(M.throttle_derivative(pumper))[0]
    assert p_var > s_var * 3
    # The 0-100 score must move the other way.
    assert (M.smoothness_from_rate(M.throttle_derivative(pumper))[2]
            < M.smoothness_from_rate(M.throttle_derivative(smooth))[2])


@pytest.mark.parametrize("freq", [10.0, 25.0, 50.0])
def test_throttle_rate_is_a_true_rate_in_percent_per_second(freq):
    """On a band-limited signal the derivative matches calculus at any rate.

    throttle = A sin(2*pi*f*t)  ->  d/dt = A*2*pi*f cos(2*pi*f*t)
    so RMS(rate) = A*2*pi*f / sqrt(2).

    This is the property that makes the metric physical: np.gradient against the
    real time vector recovers %/s, whereas np.diff would return "% per sample"
    and scale with `freq`.
    """
    amp, sig_f, duration = 10.0, 0.5, 60.0
    t = np.arange(int(duration * freq)) / freq
    df = pd.DataFrame({
        "Time [s]": t,
        "Throttle Pos [%]": amp * np.sin(2 * np.pi * sig_f * t),
    })
    rms = M.smoothness_from_rate(M.throttle_derivative(df))[1]
    expected = amp * 2 * np.pi * sig_f / np.sqrt(2.0)
    # 5% covers np.gradient's O(h^2) central-difference error at 10 Hz.
    assert rms == pytest.approx(expected, rel=0.05)


def test_pipeline_puts_two_logs_on_a_common_grid(tmp_path, stint):
    """Smoothness is only comparable on a shared grid, so the pipeline must
    produce one even when the two logs were recorded at different rates."""
    df_slow, _ = synth_stint("slow", n_laps=3, freq=10.0, seed=8)
    df_fast, _ = synth_stint("fast", n_laps=3, freq=50.0, seed=8)

    a = P.to_dataframe(P.read_ld(write_ld(str(tmp_path / "a.ld"), df_slow, "slow")),
                       max_hz=25.0)[0]
    b = P.to_dataframe(P.read_ld(write_ld(str(tmp_path / "b.ld"), df_fast, "fast")),
                       max_hz=25.0)[0]
    assert a.attrs["sample_hz"] == b.attrs["sample_hz"]
    assert np.diff(a["Time [s]"])[0] == pytest.approx(np.diff(b["Time [s]"])[0])


def test_smoothness_skips_excluded_lap_boundaries():
    """Differentiating across a dropped lap would inject a huge false rate."""
    df, _src = synth_stint("d", n_laps=4, pump_amplitude=2.0, seed=7)
    laps = P.build_lap_table(df)
    annotated = P.add_lap_columns(df, laps)

    keep_all = pd.Series([True] * len(laps))
    keep_gap = pd.Series([True, False, True, True])          # lap 2 removed

    rate_all = M.throttle_derivative(annotated, laps, keep_all)
    rate_gap = M.throttle_derivative(annotated, laps, keep_gap)
    assert len(rate_gap) < len(rate_all)
    # No seam artefact: the gapped variance stays the same order of magnitude.
    v_all = M.smoothness_from_rate(rate_all)[0]
    v_gap = M.smoothness_from_rate(rate_gap)[0]
    assert v_gap == pytest.approx(v_all, rel=0.5)


def test_metrics_handle_a_log_with_no_laps():
    df, _ = synth_stint("d", n_laps=2, seed=2)
    empty = P.build_lap_table(df[["Time [s]", "Throttle Pos [%]"]])
    m = M.compute_driver_metrics("d", df, empty)
    assert not np.isfinite(m.median_lap_s)
    assert m.notes                                  # the reason is reported
    assert np.isfinite(m.smoothness_var)            # smoothness still computable


def test_missing_throttle_channel_is_reported_not_crashed():
    df, _ = synth_stint("d", n_laps=3, seed=4)
    laps = P.build_lap_table(df)
    m = M.compute_driver_metrics("d", df.drop(columns=["Throttle Pos [%]"]), laps)
    assert np.isfinite(m.median_lap_s)
    assert not np.isfinite(m.smoothness_var)
    assert any("throttle" in n.lower() for n in m.notes)


def test_ranking_orders_the_better_driver_first():
    good, _ = synth_stint("good", n_laps=10, median_offset_s=0.2, lap_sigma_s=0.8,
                          pump_amplitude=1.5, seed=21)
    poor, _ = synth_stint("poor", n_laps=10, median_offset_s=-4.0, lap_sigma_s=4.5,
                          pump_amplitude=10.0, seed=22)

    ms = []
    for name, frame in (("good", good), ("poor", poor)):
        laps = P.build_lap_table(frame)
        ms.append(M.compute_driver_metrics(name, P.add_lap_columns(frame, laps), laps))

    ranking = M.rank_drivers(ms)
    assert ranking.iloc[0]["Driver"] == "good"


def test_lap_time_formatting():
    assert M.format_lap_time(210.0) == "3:30.000"
    assert M.format_lap_time(209.456) == "3:29.456"
    assert M.format_lap_time(float("nan")) == "—"
    assert M.format_delta(1.4249) == "+1.42 s"
    assert M.format_delta(-0.5) == "-0.50 s"
