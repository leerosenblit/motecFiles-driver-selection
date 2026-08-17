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
    assert M.expected_energy_wh(210.0) == pytest.approx(80.0)
    assert M.expected_energy_wh(189.0) == pytest.approx(88.0)
    assert 80.0 < M.expected_energy_wh(204.75) < 84.0     # halfway 199.5 -> 210
    # Outside the table the endpoints hold flat rather than extrapolating.
    assert M.expected_energy_wh(120.0) == pytest.approx(88.0)
    assert M.expected_energy_wh(400.0) == pytest.approx(72.0)


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

def _fake(name, pace, cons, excess, smooth_score=60.0, laps=10):
    """A DriverMetrics with the four inputs set directly."""
    m = M.DriverMetrics(name=name, n_laps_used=laps, n_laps_total=laps)
    m.median_lap_s = 210.0
    m.pace_adherence_s = pace
    m.consistency_s = cons
    m.energy_excess_wh = excess
    m.median_energy_wh = 80.0 + (excess if np.isfinite(excess) else 0.0)
    m.smoothness_score = smooth_score
    m.smoothness_rms = 20.0
    return m


def test_score_is_bounded_and_monotonic():
    best = M.driver_score(_fake("a", 0.0, 0.0, 0.0, 100.0))[0]
    worst = M.driver_score(_fake("b", 50.0, 50.0, 200.0, 0.0))[0]
    assert 0.0 <= worst < best <= 100.0

    # Monotonic in each metric with the others held fixed.
    for key, values in (("pace", (0.5, 1.5, 3.0, 6.0)),
                        ("cons", (0.5, 1.5, 3.0, 6.0)),
                        ("energy", (0.0, 2.0, 6.0, 15.0))):
        scores = []
        for v in values:
            args = {"pace": 1.5, "cons": 1.5, "energy": 2.0}
            args[key] = v
            scores.append(M.driver_score(
                _fake("x", args["pace"], args["cons"], args["energy"]))[0])
        assert scores == sorted(scores, reverse=True), key


def test_score_reference_gives_half_marks():
    parts = M.score_components(
        _fake("x", M.SCORE_REFERENCES["pace"], M.SCORE_REFERENCES["consistency"],
              M.SCORE_REFERENCES["energy"])
    )
    for key in ("pace", "consistency", "energy"):
        assert parts[key] == pytest.approx(50.0)


def test_one_bad_metric_cannot_wipe_out_the_score():
    """The bounded mapping never goes negative, so nothing gets cancelled."""
    m = _fake("x", 0.3, 0.3, 500.0, 95.0)      # excellent except catastrophic energy
    score, parts = M.driver_score(m)
    assert parts["energy"] < 5.0
    assert score > 50.0                        # the good metrics still count


def test_beating_the_energy_expectation_takes_full_marks():
    """Negative excess is clamped, not rewarded without limit."""
    assert M.score_components(_fake("x", 1.0, 1.0, -25.0))["energy"] == pytest.approx(100.0)


def test_score_does_not_depend_on_the_peer_group():
    """Adding a driver must not change anyone else's score."""
    a = _fake("a", 1.0, 1.0, 1.0)
    pair = M.leaderboard([a, _fake("b", 3.0, 3.0, 6.0)])
    field = M.leaderboard([a, _fake("b", 3.0, 3.0, 6.0),
                           _fake("c", 0.2, 0.2, 0.0),
                           _fake("d", 9.0, 9.0, 30.0)])
    assert (pair.loc[pair["Driver"] == "a", "Driver score"].iloc[0]
            == pytest.approx(field.loc[field["Driver"] == "a", "Driver score"].iloc[0]))


def test_leaderboard_is_sorted_best_first_and_positioned():
    board = M.leaderboard([
        _fake("mid", 2.0, 2.0, 4.0),
        _fake("best", 0.3, 0.4, 0.2),
        _fake("worst", 6.0, 5.0, 14.0),
    ])
    assert board["Driver"].tolist() == ["best", "mid", "worst"]
    assert board["Pos"].tolist() == [1, 2, 3]
    assert board["Driver score"].is_monotonic_decreasing


def test_smoothness_inherits_the_energy_weight_when_energy_is_missing():
    """A log with no energy data must not be scored as if energy were zero."""
    high = _fake("x", 1.5, 1.5, float("nan"), smooth_score=90.0)
    low = _fake("y", 1.5, 1.5, float("nan"), smooth_score=10.0)
    score, parts = M.driver_score(high)

    assert not np.isfinite(parts["energy"])
    assert np.isfinite(score)
    # Smoothness now carries 0.10 + 0.30 of the weight, so the gap between a 90
    # and a 10 must exceed what a bare 10% weight could produce.
    assert score - M.driver_score(low)[0] > 20.0


def test_score_survives_a_driver_with_no_metrics_at_all():
    blank = M.DriverMetrics(name="blank", n_laps_used=3, n_laps_total=3)
    score, _parts = M.driver_score(blank)
    assert not np.isfinite(score)


def test_leaderboard_skips_drivers_with_no_usable_laps():
    board = M.leaderboard([_fake("ok", 1.0, 1.0, 1.0),
                           _fake("empty", 1.0, 1.0, 1.0, laps=0)])
    assert board["Driver"].tolist() == ["ok"]


def test_custom_weights_change_the_order():
    """Pace-first and energy-first weightings should disagree about these two."""
    quick_thirsty = _fake("quick", 0.4, 1.2, 12.0)
    steady_frugal = _fake("frugal", 2.6, 1.2, 0.2)

    pace_first = {"pace": 0.9, "consistency": 0.05, "energy": 0.05, "smoothness": 0.0}
    energy_first = {"pace": 0.05, "consistency": 0.05, "energy": 0.9, "smoothness": 0.0}

    assert M.leaderboard([quick_thirsty, steady_frugal],
                         pace_first).iloc[0]["Driver"] == "quick"
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
