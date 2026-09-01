# test_energy.py — checks on the energy channels, the Driver Score and the palette
#
#     python -m pytest test_energy.py -q
#
# Split from test_analysis.py (parser + pace metrics) so the energy and scoring
# work has its own home.

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import metrics as M
import motec_parser as P
import plots
from plots import overlay_chart
from sample_data import synth_stint, write_ld


@pytest.fixture(scope="module")
def stint():
    return synth_stint("Test Driver", n_laps=6, lap_sigma_s=2.0, seed=5)


@pytest.fixture(scope="module")
def ld_path(tmp_path_factory, stint):
    df, _laps = stint
    dest = tmp_path_factory.mktemp("logs") / "test.ld"
    return write_ld(str(dest), df, driver="Test Driver", venue="Zolder")


# --------------------------------------------------------------------------
# Channel identification and power derivation
# --------------------------------------------------------------------------

def test_power_derived_from_voltage_times_current(ld_path):
    """The sample .ld logs volts and amps but no watts, so power is computed."""
    log = P.read_ld(ld_path)
    assert log.channel("Power") is None            # genuinely absent from the file
    df, found = P.to_dataframe(log)

    assert "power" in found
    assert "derived" in found["power"].name
    v = df[P.CHANNEL_LABELS["voltage"]].to_numpy()
    i = df[P.CHANNEL_LABELS["current"]].to_numpy()
    assert np.allclose(df[P.CHANNEL_LABELS["power"]].to_numpy(), v * i)


def test_inverted_current_sign_is_corrected():
    """Some installations log discharge as negative; energy must stay positive."""
    n = 200
    log = P.LDLog(channels=[
        P.Channel("Lap Number", "lap", "", 10, np.ones(n)),
        P.Channel("Battery Voltage", "v", "V", 10, np.full(n, 100.0)),
        P.Channel("Battery Current", "a", "A", 10, np.full(n, -12.0)),
    ])
    df, _found = P.to_dataframe(log)
    assert (df[P.CHANNEL_LABELS["power"]] > 0).all()
    assert any("inverted" in w for w in log.warnings)


def test_power_alias_does_not_capture_a_power_map_channel():
    """The team logs 'Power Map', a controller setting with no unit.

    A bare prefix match on the alias "power" would grab it and feed a map number
    into the energy calculation.
    """
    log = P.LDLog(channels=[
        P.Channel("Power Map", "map", "", 10, np.full(10, 3.0)),
        P.Channel("Corr Speed", "spd", "km/h", 10, np.zeros(10)),
    ])
    assert "power" not in P.identify_channels(log)

    # A real power channel, in watts, still matches.
    log2 = P.LDLog(channels=[
        P.Channel("Power Map", "map", "", 10, np.full(10, 3.0)),
        P.Channel("Power", "pwr", "W", 10, np.full(10, 900.0)),
    ])
    assert P.identify_channels(log2)["power"].name == "Power"


def test_team_channel_names_are_recognised():
    """The names the car's own telemetry uses must map straight through."""
    log = P.LDLog(channels=[
        P.Channel("bms_voltage_V", "v", "V", 10, np.full(10, 100.0)),
        P.Channel("bms_current_A", "a", "A", 10, np.full(10, 15.0)),
        P.Channel("bms_soc_percent", "soc", "%", 1, np.full(10, 90.0)),
        P.Channel("mms_power_W", "p", "W", 10, np.full(10, 1500.0)),
        P.Channel("total_race_energy", "e", "Wh", 1, np.linspace(0, 50, 10)),
    ])
    found = P.identify_channels(log)
    assert found["voltage"].name == "bms_voltage_V"
    assert found["current"].name == "bms_current_A"
    assert found["soc"].name == "bms_soc_percent"
    assert found["power"].name == "mms_power_W"
    assert found["energy"].name == "total_race_energy"


def test_energy_units_normalised():
    for unit, raw, expected in (("kWh", 0.085, 85.0), ("J", 306000.0, 85.0),
                                ("kJ", 306.0, 85.0)):
        ch = P.Channel("Energy Used", "e", unit, 1, np.array([0.0, raw]))
        assert P._convert_units("energy", ch)[1] == pytest.approx(expected, rel=1e-3)
    kw = P.Channel("Power", "p", "kW", 1, np.array([1.5]))
    assert P._convert_units("power", kw)[0] == pytest.approx(1500.0)
    mv = P.Channel("Battery Voltage", "v", "mV", 1, np.array([98000.0]))
    assert P._convert_units("voltage", mv)[0] == pytest.approx(98.0)


# --------------------------------------------------------------------------
# Energy per lap — every source path
# --------------------------------------------------------------------------

def test_energy_integrates_power_over_each_lap():
    """A constant 1800 W for 200 s is exactly 100 Wh."""
    freq, lap_s, watts = 10.0, 200.0, 1800.0
    n = int(lap_s * freq)
    df = pd.DataFrame({
        "Time [s]": np.arange(2 * n) / freq,
        "Power [W]": np.full(2 * n, watts),
        "Lap Number": np.repeat([1.0, 2.0], n),
    })
    laps = P.build_lap_table(df)
    wh, source = M.energy_per_lap(df, laps)

    assert "integrated" in source
    # W * s / 3600 = Wh. The lap's final sample belongs to the next segment, so
    # allow one sample interval of tolerance.
    assert wh[0] == pytest.approx(watts * lap_s / 3600.0, rel=0.01)


def test_energy_counts_regen_as_a_credit():
    """Negative power is recovery, so it must reduce net consumption."""
    n = 400                                        # two laps of 200 samples
    lap_col = np.repeat([1.0, 2.0], n // 2)
    steady = pd.DataFrame({"Time [s]": np.arange(n) / 10.0,
                           "Power [W]": np.full(n, 1000.0),
                           "Lap Number": lap_col})
    with_regen = steady.copy()
    # Lap 1 pushes for 10 s then recovers for 10 s.
    with_regen.loc[100:199, "Power [W]"] = -1000.0

    a = M.energy_per_lap(steady, P.build_lap_table(steady))[0][0]
    b = M.energy_per_lap(with_regen, P.build_lap_table(with_regen))[0][0]
    assert b < a
    assert b == pytest.approx(0.0, abs=0.5)        # equal out and back = net zero


def test_energy_prefers_the_cumulative_counter_over_integration():
    """The car's own coulomb-counted total outranks anything we recompute."""
    n = 400
    df = pd.DataFrame({
        "Time [s]": np.arange(n) / 10.0,
        "Power [W]": np.full(n, 1800.0),                  # would integrate to ~10 Wh
        "Energy Used [Wh]": np.linspace(0.0, 60.0, n),    # counter says 30 Wh/lap
        "Lap Number": np.repeat([1.0, 2.0], n // 2),
    })
    laps = P.build_lap_table(df)
    wh, source = M.energy_per_lap(df, laps)
    assert "counter" in source
    assert wh[0] == pytest.approx(30.0, abs=0.5)


def test_energy_from_per_lap_channel_is_read_one_lap_behind():
    """A 'last lap energy' channel reports the PREVIOUS lap while driving."""
    n_per = 100
    # During lap 1 nothing is known; during lap 2 it holds lap 1's 81 Wh;
    # during lap 3 it holds lap 2's 86 Wh.
    df = pd.DataFrame({
        "Time [s]": np.arange(n_per * 3) / 10.0,
        "Lap Energy [Wh]": np.repeat([0.0, 81.0, 86.0], n_per),
        "Lap Number": np.repeat([1.0, 2.0, 3.0], n_per),
    })
    laps = P.build_lap_table(df)
    wh, source = M.energy_per_lap(df, laps)
    assert "one lap behind" in source
    assert wh[0] == pytest.approx(81.0)
    assert wh[1] == pytest.approx(86.0)
    assert not np.isfinite(wh[2])                  # the last lap is unknowable


def test_missing_energy_channels_reported_not_crashed():
    df, _ = synth_stint("d", n_laps=3, seed=4)
    bare = df.drop(columns=["Power [W]", "Battery Voltage [V]",
                            "Battery Current [A]"])
    laps = P.build_lap_table(bare)
    m = M.compute_driver_metrics("d", bare, laps)
    assert m.energy_source is None
    assert not np.isfinite(m.median_energy_wh)
    assert any("energy" in n.lower() for n in m.notes)


def test_energy_stats_use_the_same_kept_laps_as_pace():
    """A traffic lap must not distort the energy figure either.

    Note which way this cuts: a lap stuck behind traffic is SLOWER, and a slower
    lap uses LESS energy (drag falls with speed — the whole premise of the
    strategy table). So leaving traffic laps in would flatter a driver's energy
    figure, not penalise it. Applying the same lap filter to both keeps pace and
    energy talking about the same set of laps.
    """
    df, _ = synth_stint("d", n_laps=8, traffic_laps={4: 45.0}, seed=6)
    laps = P.build_lap_table(df)
    annotated = P.add_lap_columns(df, laps)
    m = M.compute_driver_metrics("d", annotated, laps)

    wh, _src = M.energy_per_lap(annotated, laps)
    assert 4 in m.excluded_laps
    # The slow lap used less than the median, and was left out of it.
    assert wh[3] < m.median_energy_wh


# --------------------------------------------------------------------------
# Pace correction
# --------------------------------------------------------------------------

def test_expected_energy_interpolates_and_clamps():
    # The curve is anchored to the budget at target pace, so a lap at target
    # costs exactly the budget and the rest scale with it.
    assert M.expected_energy_wh(210.0, 100.0) == pytest.approx(100.0)
    assert M.expected_energy_wh(189.0, 100.0) == pytest.approx(110.0)
    assert 100.0 < M.expected_energy_wh(204.75, 100.0) < 105.0
    # Outside the table the endpoints hold flat rather than extrapolating.
    assert M.expected_energy_wh(120.0, 100.0) == pytest.approx(110.0)
    assert M.expected_energy_wh(400.0, 100.0) == pytest.approx(90.0)
    # Changing the budget rescales the whole curve, never contradicts it.
    assert M.expected_energy_wh(210.0, 80.0) == pytest.approx(80.0)
    assert M.expected_energy_wh(189.0, 80.0) == pytest.approx(88.0)


def test_slower_laps_are_expected_to_use_less_energy():
    """The premise of the correction: raw Wh/lap would reward going slowly."""
    assert (M.expected_energy_wh(231.0) < M.expected_energy_wh(210.0)
            < M.expected_energy_wh(189.0))


def test_energy_excess_is_measured_against_the_pace_driven():
    df = pd.DataFrame({"Time [s]": [0.0, 1.0, 2.0]})
    results = []
    for lap_s in (225.0, 195.0):
        expected = M.expected_energy_wh(lap_s)
        n = 3
        laps = pd.DataFrame({
            "Lap": range(1, n + 1), "LapTime [s]": [lap_s] * n,
            "start_idx": range(n), "end_idx": range(n),
            "Distance [m]": [4011.0] * n,
        })
        m = M.compute_driver_metrics("d", df, laps, drop_first=False, drop_last=False)
        # Both drivers sit exactly 8 Wh above their own pace's expectation.
        m.median_energy_wh = expected + 8.0
        m.energy_excess_wh = m.median_energy_wh - M.expected_energy_wh(m.median_lap_s)
        results.append(m)

    slow, quick = results
    assert slow.median_energy_wh < quick.median_energy_wh        # slow used less
    assert slow.energy_excess_wh == pytest.approx(quick.energy_excess_wh)


def test_smoothness_and_energy_can_rank_two_drivers_the_opposite_way():
    """Why smoothness is only a proxy, and must not be double-counted.

    Vehicle inertia low-passes fast pedal oscillation away, so a driver sawing at
    the pedal at 2 Hz destroys their smoothness score while spending essentially
    no extra energy. A driver who instead surges slowly spends real energy. Rank
    those two on smoothness and you get the opposite answer to ranking them on
    watt-hours — so where energy is measured, it is the one that counts.
    """
    def measure(**kw):
        df, _ = synth_stint("d", n_laps=3, lap_sigma_s=0.0, seed=1, **kw)
        laps = P.build_lap_table(df)
        return M.compute_driver_metrics("d", P.add_lap_columns(df, laps), laps)

    calm = measure(pump_amplitude=1.0, pump_hz=0.3, surge_pct=1.0)
    pumping = measure(pump_amplitude=12.0, pump_hz=2.0, surge_pct=1.0)
    surging = measure(pump_amplitude=1.0, pump_hz=0.3, surge_pct=6.0)

    # Fast pumping wrecks smoothness while leaving energy alone.
    assert pumping.smoothness_score < calm.smoothness_score / 3
    assert pumping.median_energy_wh == pytest.approx(calm.median_energy_wh, rel=0.02)
    # Slow surging costs real energy.
    assert surging.median_energy_wh > calm.median_energy_wh * 1.05

    # The inversion: the pumper looks far worse on the proxy, yet is the one
    # actually using less energy.
    assert pumping.smoothness_score < surging.smoothness_score
    assert pumping.median_energy_wh < surging.median_energy_wh


def test_ranking_uses_energy_when_available_and_the_proxy_otherwise():
    def build(frame):
        laps = P.build_lap_table(frame)
        return M.compute_driver_metrics("x", P.add_lap_columns(frame, laps), laps)

    a, _ = synth_stint("a", n_laps=8, surge_pct=1.0, seed=21)
    b, _ = synth_stint("b", n_laps=8, surge_pct=5.0, seed=22)

    with_energy = [build(a), build(b)]
    with_energy[0].name, with_energy[1].name = "a", "b"
    assert M.rank_drivers(with_energy)[1] == "Energy excess [Wh/lap]"

    stripped = [build(f.drop(columns=["Power [W]", "Battery Voltage [V]",
                                      "Battery Current [A]"]))
                for f in (a, b)]
    stripped[0].name, stripped[1].name = "a", "b"
    assert M.rank_drivers(stripped)[1] == "Throttle rate RMS [%/s]"


# --------------------------------------------------------------------------
# Driver Score / leaderboard
# --------------------------------------------------------------------------

def _fake(name, cons, energy, accel=1.0, decel=1.0, lat=1.0, maxlat=1.0, laps=10):
    """A DriverMetrics with the six scored inputs set directly."""
    m = M.DriverMetrics(name=name, n_laps_used=laps, n_laps_total=laps)
    m.median_lap_s = 210.0
    m.consistency_s = cons
    m.median_energy_wh = energy
    m.avg_accel_g = accel
    m.avg_decel_g = decel
    m.avg_lat_g = lat
    m.max_lat_g = maxlat
    return m


def test_score_is_bounded_and_monotonic():
    a = _fake("a", 0.1, 40.0, 0.1, 0.1, 0.1, 0.1)
    b = _fake("b", 5.0, 400.0, 5.0, 5.0, 5.0, 5.0)
    scores = M.driver_score([a, b])
    assert 0.0 <= scores["b"][0] < scores["a"][0] <= 100.0

    # Monotonic in each metric with the others held fixed and a frozen peer.
    peer = _fake("peer", 2.0, 100.0, 2.0, 2.0, 2.0, 2.0)
    for key, values in (("cons", (0.5, 1.0, 2.0, 4.0)),
                        ("energy", (50.0, 75.0, 100.0, 150.0)),
                        ("accel", (0.5, 1.0, 2.0, 4.0)),
                        ("decel", (0.5, 1.0, 2.0, 4.0)),
                        ("lat", (0.5, 1.0, 2.0, 4.0)),
                        ("maxlat", (0.5, 1.0, 2.0, 4.0))):
        pts = []
        for v in values:
            args = {"cons": 0.5, "energy": 50.0, "accel": 0.5,
                    "decel": 0.5, "lat": 0.5, "maxlat": 0.5}
            args[key] = v
            x = _fake("x", args["cons"], args["energy"], args["accel"],
                      args["decel"], args["lat"], args["maxlat"])
            pts.append(M.driver_score([x, peer])["x"][0])
        assert pts == sorted(pts, reverse=True), key


def test_double_the_best_value_scores_half_marks():
    """points = 100 * best / value, so twice the best value is worth 50."""
    best = _fake("best", 1.0, 50.0, 1.0, 1.0, 1.0, 1.0)
    double = _fake("double", 2.0, 100.0, 2.0, 2.0, 2.0, 2.0)
    parts = M.score_components([best, double])["double"]
    for key in ("consistency", "energy", "avg_accel", "avg_decel",
               "avg_lat_g", "max_lat_g"):
        assert parts[key] == pytest.approx(50.0)


def test_one_bad_metric_cannot_wipe_out_the_score():
    """The bounded mapping never goes negative, so nothing gets cancelled."""
    peer = _fake("peer", 1.0, 50.0, 1.0, 1.0, 1.0, 1.0)
    # excellent on five metrics, catastrophic on max lateral G
    m = _fake("x", 0.5, 25.0, 0.5, 0.5, 0.5, 500.0)
    score, parts = M.driver_score([peer, m])["x"]
    assert parts["max_lat_g"] < 1.0
    assert score > 50.0                        # the good metrics still count


def test_zero_metric_value_scores_full_marks():
    """A value of zero (e.g. a hypothetical zero-std-dev driver) can't be
    divided into, so it is treated as already best-possible."""
    zero = _fake("zero", 0.0, 50.0)
    other = _fake("other", 1.0, 50.0)
    assert M.score_components([zero, other])["zero"]["consistency"] == pytest.approx(100.0)


def test_score_is_peer_relative_and_can_shift_when_the_field_changes():
    """Unlike an absolute-reference score, adding a driver CAN change everyone
    else's score: each metric is graded against whoever is best among the
    drivers being compared."""
    a = _fake("a", 1.0, 80.0)
    pair = M.leaderboard([a, _fake("b", 3.0, 120.0)])
    field = M.leaderboard([a, _fake("b", 3.0, 120.0),
                           _fake("c", 0.2, 40.0),
                           _fake("d", 9.0, 300.0)])
    a_pair = pair.loc[pair["Driver"] == "a", "Driver score"].iloc[0]
    a_field = field.loc[field["Driver"] == "a", "Driver score"].iloc[0]
    assert a_pair != pytest.approx(a_field)


def test_leaderboard_is_sorted_best_first_and_positioned():
    board = M.leaderboard([
        _fake("mid", 2.0, 120.0),
        _fake("best", 0.3, 40.0),
        _fake("worst", 6.0, 300.0),
    ])
    assert board["Driver"].tolist() == ["best", "mid", "worst"]
    assert board["Pos"].tolist() == [1, 2, 3]
    assert board["Driver score"].is_monotonic_decreasing


def test_missing_metric_is_dropped_not_zeroed():
    """A log with no energy channel must not be scored as if energy were
    zero — the weight is dropped and the rest renormalised."""
    a = _fake("a", 1.0, float("nan"))
    b = _fake("b", 3.0, float("nan"))
    score_a, parts_a = M.driver_score([a, b])["a"]

    assert not np.isfinite(parts_a["energy"])
    assert np.isfinite(score_a)
    assert parts_a["consistency"] == pytest.approx(100.0)


def test_score_survives_a_driver_with_no_metrics_at_all():
    blank = M.DriverMetrics(name="blank", n_laps_used=3, n_laps_total=3)
    score, _parts = M.driver_score([blank])["blank"]
    assert not np.isfinite(score)


def test_leaderboard_skips_drivers_with_no_usable_laps():
    board = M.leaderboard([_fake("ok", 1.0, 80.0),
                           _fake("empty", 1.0, 80.0, laps=0)])
    assert board["Driver"].tolist() == ["ok"]


def test_custom_weights_change_the_order():
    """Consistency-first and energy-first weightings should disagree about
    these two drivers."""
    quick_thirsty = _fake("quick", cons=0.4, energy=200.0)
    steady_frugal = _fake("frugal", cons=2.6, energy=20.0)
    zero = {"avg_accel": 0.0, "avg_decel": 0.0, "avg_lat_g": 0.0, "max_lat_g": 0.0}

    consistency_first = {"consistency": 0.9, "energy": 0.05, **zero}
    energy_first = {"consistency": 0.05, "energy": 0.9, **zero}

    assert M.leaderboard([quick_thirsty, steady_frugal],
                         consistency_first).iloc[0]["Driver"] == "quick"
    assert M.leaderboard([quick_thirsty, steady_frugal],
                         energy_first).iloc[0]["Driver"] == "frugal"


# --------------------------------------------------------------------------
# Palette / plotting for a field of drivers
# --------------------------------------------------------------------------

def test_palette_never_invents_a_ninth_hue():
    """Slots 9 and 10 reuse hues with a dashed line, not a generated colour."""
    for mode in ("light", "dark"):
        colours = [plots.series_style(i, mode)[0] for i in range(plots.MAX_DRIVERS)]
        assert len(set(colours)) == 8              # eight validated hues, no more
        assert colours[8] == colours[0] and colours[9] == colours[1]

        dashes = [plots.series_style(i, mode)[1] for i in range(plots.MAX_DRIVERS)]
        assert dashes[:8] == ["solid"] * 8
        assert dashes[8:] == ["dash", "dash"]


def test_colour_follows_the_driver_not_their_position(stint):
    """Overlaying a subset must not repaint the drivers that remain."""
    df, _ = stint
    laps = P.build_lap_table(df)
    seg = P.add_lap_columns(df, laps).iloc[:400]

    # The driver in colour slot 4 keeps slot 4 even when drawn first.
    fig = overlay_chart([{"name": "e", "lap": 1, "data": seg, "slot": 4}],
                        mode="light")
    assert fig.data[0].line.color == plots.series_style(4, "light")[0]


def test_overlay_includes_a_power_row_when_power_exists(stint):
    df, _ = stint
    laps = P.build_lap_table(df)
    seg = P.add_lap_columns(df, laps).iloc[:400]

    fig = overlay_chart([{"name": "a", "lap": 1, "data": seg, "slot": 0}],
                        mode="light")
    titles = [a.text for a in fig.layout.annotations]
    assert "Power [W]" in titles

    # And drops the row when the channel is absent, rather than plotting a blank.
    fig2 = overlay_chart(
        [{"name": "a", "lap": 1, "data": seg.drop(columns=["Power [W]"]), "slot": 0}],
        mode="light")
    assert "Power [W]" not in [a.text for a in fig2.layout.annotations]


def test_energy_chart_handles_missing_energy_gracefully(stint):
    df, _ = stint
    laps = P.build_lap_table(df)
    fig = plots.energy_chart(
        [{"name": "a", "laps": laps, "keep": None,
          "energy": np.full(len(laps), np.nan), "slot": 0}], mode="light")
    assert any("No energy data" in (a.text or "") for a in fig.layout.annotations)


# --------------------------------------------------------------------------
# Real-log channel naming (found against actual MoTeC exports)
# --------------------------------------------------------------------------

def test_sim_and_drivetrain_channel_names_are_recognised():
    """Names taken from real MoTeC exports the team actually uses.

    These logs carry no bms_*/mms_* channels at all: power arrives as
    "Drivetrain Power" in horsepower, lateral G as "CG Accel Lateral", and state
    of charge as "KERS Charge". Before these aliases the energy section was
    simply empty.
    """
    log = P.LDLog(channels=[
        P.Channel("Ground Speed", "spd", "kph", 30, np.full(10, 90.0)),
        P.Channel("CG Accel Lateral", "gl", "G", 50, np.zeros(10)),
        P.Channel("Drivetrain Power", "pwr", "hp", 10, np.full(10, 20.0)),
        P.Channel("KERS Charge", "soc", "%", 10, np.full(10, 80.0)),
    ])
    found = P.identify_channels(log)
    assert found["g_lat"].name == "CG Accel Lateral"
    assert found["power"].name == "Drivetrain Power"
    assert found["soc"].name == "KERS Charge"


def test_horsepower_converted_to_watts():
    hp = P.Channel("Drivetrain Power", "p", "hp", 10, np.array([1.0]))
    assert P._convert_units("power", hp)[0] == pytest.approx(745.7, rel=1e-3)
    ps = P.Channel("Drivetrain Power", "p", "PS", 10, np.array([1.0]))
    assert P._convert_units("power", ps)[0] == pytest.approx(735.5, rel=1e-3)


def test_kers_deployed_energy_is_not_treated_as_total_consumption():
    """A subsystem counter must not masquerade as the car's energy total.

    "KERS Deployed Energy" counts only the hybrid boost released from the store.
    Taking it as the energy counter would outrank integrating the power channel
    and under-report consumption by an order of magnitude.
    """
    log = P.LDLog(channels=[
        P.Channel("Lap Number", "lap", "", 10, np.repeat([1.0, 2.0], 100)),
        P.Channel("KERS Deployed Energy", "ke", "kJ", 10,
                  np.linspace(0.0, 1790.0, 200)),
        P.Channel("Drivetrain Power", "pwr", "hp", 10, np.full(200, 20.0)),
    ])
    found = P.identify_channels(log)
    assert "energy" not in found            # excluded as a subsystem channel
    assert found["power"].name == "Drivetrain Power"

    df, _ = P.to_dataframe(log)
    laps = P.build_lap_table(df)
    _wh, source = M.energy_per_lap(df, laps)
    assert "integrated" in source           # power, not the KERS counter


# --------------------------------------------------------------------------
# Distance channels that reset each lap
# --------------------------------------------------------------------------

def _resetting_distance_log(track_m=3960.0, start_offset_m=3537.0, n_laps=3):
    """A log whose distance channel resets at the line, starting mid-lap.

    This is what the real exports look like: "Lap Distance" runs 0 -> track
    length and wraps, and the recording begins partway around.
    """
    per_lap, freq = 200, 10.0
    n = per_lap * n_laps
    frac = (np.arange(n) / per_lap)                      # laps completed
    dist = ((start_offset_m + frac * track_m) % track_m)
    lap = np.floor((start_offset_m + frac * track_m) / track_m) + 1.0
    return pd.DataFrame({
        "Time [s]": np.arange(n) / freq,
        "Distance [m]": dist,
        "Lap Number": lap,
        "Corr Speed [km/h]": np.full(n, track_m / (per_lap / freq) * 3.6),
    })


def test_resetting_distance_channel_gives_real_lap_length():
    """End-minus-start reads a full lap of a resetting channel as ~zero."""
    track = 3960.0
    df = _resetting_distance_log(track_m=track)
    laps = P.build_lap_table(df)

    # Ignore the partial first and last laps; the middle ones are complete.
    complete = laps["Distance [m]"].to_numpy()[1:-1]
    assert len(complete) >= 1
    assert np.allclose(complete, track, rtol=0.02), complete
    # The naive calculation would have produced roughly nothing.
    assert complete.min() > track * 0.9


def test_overlay_axis_is_monotonic_across_a_distance_reset():
    """A reset must not send the overlay x-axis sharply negative."""
    df = _resetting_distance_log()
    laps = P.build_lap_table(df)
    annotated = P.add_lap_columns(df, laps)

    for lap in laps["Lap"]:
        seg = annotated.loc[annotated["_Lap"] == lap, "Lap Distance [m]"].to_numpy()
        assert seg[0] == pytest.approx(0.0)
        assert np.all(np.diff(seg) >= 0), f"lap {lap} axis goes backwards"
        assert seg.min() >= 0.0


def test_cumulative_odometer_still_works():
    """The same code path must keep handling a non-resetting odometer."""
    n, freq, track = 600, 10.0, 3960.0
    df = pd.DataFrame({
        "Time [s]": np.arange(n) / freq,
        "Distance [m]": np.linspace(0.0, track * 3, n),      # never resets
        "Lap Number": np.repeat([1.0, 2.0, 3.0], n // 3),
    })
    laps = P.build_lap_table(df)
    assert np.allclose(laps["Distance [m]"].to_numpy(), track, rtol=0.02)

    annotated = P.add_lap_columns(df, laps)
    seg = annotated.loc[annotated["_Lap"] == 2, "Lap Distance [m]"].to_numpy()
    assert seg[0] == pytest.approx(0.0)
    assert np.all(np.diff(seg) >= 0)
