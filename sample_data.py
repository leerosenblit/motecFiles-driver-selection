# sample_data.py — synthetic Zolder stints, and a minimal .ld writer
#
# Two jobs:
#
#   1. Demo mode. The dashboard is usable (and reviewable) before anyone has
#      exported a real log, by generating two plausible stints in memory.
#   2. Round-trip testing. write_ld() emits a real MoTeC .ld file using the same
#      binary layout motec_parser reads, so the parser can be exercised
#      end-to-end without a confidential team log.
#
# Run directly to write two sample logs you can then upload in the app:
#
#     python sample_data.py            # -> samples/driver_a.ld, samples/driver_b.ld

from __future__ import annotations

import os
import struct
import datetime

import numpy as np
import pandas as pd

from motec_parser import _CHAN_FMT, _EVENT_FMT, _HEAD_FMT, _HEAD_SIZE

# Circuit Zolder, and the pace the energy budget is built around.
TRACK_LENGTH_M = 4011.0
TARGET_LAP_S = 210.0

# Corner layout as (position along the lap 0-1, severity 0-1, direction).
# Not a survey of Zolder — a plausible seven-corner rhythm that gives the
# charts realistic shapes to compare.
_CORNERS = [
    (0.08, 0.75, +1), (0.19, 0.45, -1), (0.31, 0.85, +1), (0.46, 0.55, -1),
    (0.61, 0.70, +1), (0.74, 0.35, -1), (0.88, 0.80, +1),
]


def _lap_profiles(s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Relative speed and steering angle as a function of lap fraction `s`.

    Each corner is a Gaussian dip in speed and a matching Gaussian bump in
    steering, so the two channels are physically consistent — the driver is
    slowest where the wheel is most turned.
    """
    speed = np.ones_like(s)
    steer = np.zeros_like(s)
    for pos, severity, direction in _CORNERS:
        width = 0.020 + 0.030 * severity
        # Wrap the distance so a corner near the line still shapes both ends.
        d = np.abs(((s - pos + 0.5) % 1.0) - 0.5)
        bell = np.exp(-0.5 * (d / width) ** 2)
        speed -= 0.55 * severity * bell
        steer += direction * 105.0 * severity * bell
    return np.clip(speed, 0.25, None), steer


def synth_stint(name: str, n_laps: int = 14, freq: float = 20.0,
                median_offset_s: float = 0.0, lap_sigma_s: float = 2.0,
                pump_amplitude: float = 3.0, pump_hz: float = 0.8,
                traffic_laps: dict[int, float] | None = None,
                seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate one driver's stint as (dataframe, lap table).

    The knobs map onto the metrics the dashboard measures:
      median_offset_s  shifts the driver's median lap off the 210 s target
      lap_sigma_s      lap-to-lap scatter, i.e. their consistency
      pump_amplitude   extra throttle oscillation, i.e. their smoothness
      traffic_laps     {lap number: seconds lost} to plant outlier laps
    """
    rng = np.random.default_rng(seed)
    traffic_laps = traffic_laps or {}

    t_parts, speed_parts, thr_parts = [], [], []
    steer_parts, glat_parts, dist_parts = [], [], []
    lapno_parts, laptime_parts = [], []

    t_cursor = 0.0
    dist_cursor = 0.0
    rows = []

    for lap in range(1, n_laps + 1):
        lap_time = TARGET_LAP_S + median_offset_s + rng.normal(0.0, lap_sigma_s)
        if lap == 1:
            lap_time += 12.0                      # out-lap
        if lap == n_laps:
            lap_time += 9.0                       # in-lap
        lap_time += traffic_laps.get(lap, 0.0)
        lap_time = max(lap_time, 60.0)

        n = max(int(round(lap_time * freq)), 8)
        t_lap = np.arange(n) / freq
        s = t_lap / lap_time                      # fraction of the lap covered

        rel_speed, steer = _lap_profiles(s)
        # Scale the profile so this lap really covers the track length in the
        # lap time: mean(speed) = length / time.
        speed = rel_speed * (TRACK_LENGTH_M / lap_time) / float(np.mean(rel_speed))
        speed_kmh = speed * 3.6

        # Throttle: broadly the demand needed to accelerate, plus the driver's
        # own pedal oscillation. `pump_amplitude` is what the smoothness metric
        # is designed to catch.
        accel = np.gradient(speed, t_lap)
        base = 55.0 + 40.0 * np.tanh(accel / 0.6)
        # `pump_hz` is the real oscillation frequency of the pedal — around
        # 1-2 Hz for a driver who saws at it, well under 0.5 Hz for a smooth one.
        pump = pump_amplitude * np.sin(2 * np.pi * pump_hz * t_lap)
        # A fixed 0.3% sensor noise floor. Deliberately independent of the
        # driver knobs: sensor noise is a property of the car, not the driver.
        thr = np.clip(base + pump + rng.normal(0.0, 0.3, n), 0.0, 100.0)

        steer = steer + rng.normal(0.0, 1.2, n)
        # Lateral acceleration from the steering geometry: a_lat = v^2 / R, and
        # 1/R rises with steering angle. The divisor folds in wheelbase and
        # steering ratio; it is tuned so a solar car peaks near 0.5 G, not the
        # 2 G a formula car would pull.
        curvature = np.deg2rad(np.abs(steer)) / 40.0
        glat = np.sign(steer) * (speed ** 2) * curvature / 9.81

        dist = dist_cursor + np.concatenate(([0.0], np.cumsum(np.diff(t_lap) * speed[:-1])))

        t_parts.append(t_cursor + t_lap)
        speed_parts.append(speed_kmh)
        thr_parts.append(thr)
        steer_parts.append(steer)
        glat_parts.append(glat)
        dist_parts.append(dist)
        lapno_parts.append(np.full(n, lap, dtype=float))
        laptime_parts.append(t_lap)               # running timer, resets at the line

        rows.append({"Lap": lap, "LapTime [s]": lap_time})
        t_cursor += n / freq
        dist_cursor = dist[-1]

    df = pd.DataFrame({
        "Time [s]": np.concatenate(t_parts),
        "Corr Speed [km/h]": np.concatenate(speed_parts),
        "Throttle Pos [%]": np.concatenate(thr_parts),
        "Steering Angle [deg]": np.concatenate(steer_parts),
        "G Force Lat [G]": np.concatenate(glat_parts),
        "Distance [m]": np.concatenate(dist_parts),
        "LapTime [s]": np.concatenate(laptime_parts),
        "Lap Number": np.concatenate(lapno_parts),
    })
    df.attrs["sample_hz"] = freq
    df.attrs["driver"] = name
    return df, pd.DataFrame(rows)


def demo_stints() -> dict[str, pd.DataFrame]:
    """Two contrasting drivers, so the comparison has something to say.

    Driver A is quicker but scruffier on the pedal; driver B sits almost exactly
    on the 210 s target and is smoother. Which one you want depends on whether
    the strategy is chasing lap time or energy — which is the decision the
    dashboard exists to inform.
    """
    a_df, a_laps = synth_stint(
        "Driver A", n_laps=14, median_offset_s=-2.6, lap_sigma_s=3.1,
        pump_amplitude=9.0, pump_hz=1.5, traffic_laps={6: 41.0}, seed=11,
    )
    b_df, b_laps = synth_stint(
        "Driver B", n_laps=14, median_offset_s=+0.4, lap_sigma_s=1.2,
        pump_amplitude=2.5, pump_hz=0.35, traffic_laps={9: 28.0}, seed=29,
    )
    return {"Driver A": (a_df, a_laps), "Driver B": (b_df, b_laps)}


# --------------------------------------------------------------------------
# Minimal .ld writer (for generating test/demo logs)
# --------------------------------------------------------------------------

# Channels written to the sample files, as (dataframe column, MoTeC name, short
# name, unit, sample rate). The rates deliberately differ between channels —
# that is how real logs are configured, and it exercises the parser's
# resampling onto a common time grid.
_SAMPLE_CHANNELS = [
    ("Corr Speed [km/h]", "Corr Speed", "CorrSpd", "km/h", 25),
    ("Throttle Pos [%]", "Throttle Pos", "Thr", "%", 25),
    ("Steering Angle [deg]", "Steering Angle", "Steer", "deg", 10),
    ("G Force Lat [G]", "G Force Lat", "GLat", "G", 10),
    ("Distance [m]", "Distance", "Dist", "m", 5),
    ("LapTime [s]", "Lap Time", "LapT", "s", 5),
    ("Lap Number", "Lap Number", "LapNo", "", 5),
]


def _downsample(t_src: np.ndarray, v_src: np.ndarray, freq: int) -> np.ndarray:
    """Resample a column onto `freq` Hz for writing."""
    n = int(np.floor((t_src[-1] - t_src[0]) * freq)) + 1
    t_dst = t_src[0] + np.arange(n) / float(freq)
    return np.interp(t_dst, t_src, v_src).astype(np.float32)


def write_ld(path: str, df: pd.DataFrame, driver: str,
             venue: str = "Zolder", vehicle_id: str = "Solar Car",
             event: str = "Driver Selection", session: str = "Practice") -> str:
    """Write a MoTeC .ld file holding the given stint.

    Everything is written as float32 with scale=mul=1 and shift=dec=0, so the
    reader's raw->physical conversion is the identity and the values round-trip
    exactly (to float32 precision).
    """
    chan_size = struct.calcsize(_CHAN_FMT)
    event_size = struct.calcsize(_EVENT_FMT)

    cols = [c for c in _SAMPLE_CHANNELS if c[0] in df.columns]
    if not cols:
        raise ValueError("None of the sample channels are present in this frame.")

    t = df["Time [s]"].to_numpy(dtype=np.float64)
    arrays = [_downsample(t, df[col].to_numpy(dtype=np.float64), freq)
              for col, _n, _s, _u, freq in cols]

    event_ptr = _HEAD_SIZE
    meta_ptr = _HEAD_SIZE + event_size
    data_ptr = meta_ptr + chan_size * len(cols)

    now = datetime.datetime.now()
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)

    with open(path, "wb") as f:
        f.write(struct.pack(
            _HEAD_FMT,
            0x40,                              # ldmarker
            meta_ptr, data_ptr, event_ptr,
            1, 0x4240, 0x000F,                 # static words
            0x1F44,                            # device serial
            b"ADL",                            # device type
            420, 0xADB0,                       # device version, static
            len(cols),                         # num_channs
            now.strftime("%d/%m/%Y").encode(),
            now.strftime("%H:%M:%S").encode(),
            driver.encode()[:63], vehicle_id.encode()[:63], venue.encode()[:63],
            0x000C81A4,                        # "pro logging"
            b"Synthetic log from sample_data.py",
        ))

        f.write(struct.pack(_EVENT_FMT, event.encode()[:63], session.encode()[:63],
                            b"Generated for dashboard testing", 0))

        # Channel meta blocks: a linked list, each pointing at the next.
        offset = data_ptr
        for i, (_col, name, short, unit, freq) in enumerate(cols):
            prev_ptr = meta_ptr + chan_size * (i - 1) if i > 0 else 0
            next_ptr = meta_ptr + chan_size * (i + 1) if i < len(cols) - 1 else 0
            f.write(struct.pack(
                _CHAN_FMT,
                prev_ptr, next_ptr, offset, len(arrays[i]),
                0x2EE1 + i,                    # counter
                0x07, 4,                       # dtype_a=float, 4 bytes
                freq,
                0, 1, 1, 0,                    # shift, mul, scale, dec
                name.encode()[:31], short.encode()[:7], unit.encode()[:11],
            ))
            offset += arrays[i].nbytes

        for arr in arrays:
            f.write(arr.tobytes())

    return path


def write_ldx(path: str, laps: pd.DataFrame) -> str:
    """Write a MoTeC .ldx holding one beacon marker per lap boundary.

    Marker times are microseconds, which is what i2 writes. The element nesting
    mirrors a real .ldx closely enough to exercise the reader, which walks the
    tree for marker-ish tags rather than binding to a fixed path.
    """
    edges = [0.0]
    for lap_time in laps["LapTime [s]"]:
        edges.append(edges[-1] + float(lap_time))

    markers = "\n".join(
        f'          <Marker Name="" Time="{int(round(t * 1e6))}" />' for t in edges
    )
    xml = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<LDXFile Version="1.6">\n'
        '  <Layers>\n'
        '    <Layer Index="0">\n'
        '      <MarkerBlock>\n'
        '        <MarkerGroup Type="Beacon">\n'
        f"{markers}\n"
        '        </MarkerGroup>\n'
        '      </MarkerBlock>\n'
        '    </Layer>\n'
        '  </Layers>\n'
        '</LDXFile>\n'
    )
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(xml)
    return path


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(here, "samples")
    for label, (frame, laps) in demo_stints().items():
        slug = label.lower().replace(" ", "_")
        ld = write_ld(os.path.join(out_dir, f"{slug}.ld"), frame, driver=label)
        ldx = write_ldx(os.path.join(out_dir, f"{slug}.ldx"), laps)
        print(f"wrote {ld}  ({os.path.getsize(ld) / 1e6:.2f} MB)")
        print(f"wrote {ldx}  ({len(laps) + 1} beacon markers)")
