# motec_parser.py — MoTeC .ld / .ldx reader for the driver-selection dashboard
#
# There is no maintained MoTeC parser on PyPI, so this module implements the
# binary layout directly. The layout below is the reverse-engineered ld format
# used by MoTeC ADL/i2 (and matches the reference implementation at
# github.com/gotzl/ldparser). Everything is little-endian.
#
# File layout:   HEADER -> EVENT -> VENUE -> VEHICLE -> CHANNEL META (linked
#                list) -> CHANNEL DATA (one contiguous block per channel)
#
# Two things about this format drive the design of this module:
#
#   1. Channels are logged at DIFFERENT frequencies (e.g. speed at 50 Hz,
#      steering at 20 Hz). Each channel therefore has its own implicit time
#      axis t[i] = i / freq. To get one tidy DataFrame we resample every
#      channel onto a single common time grid (see to_dataframe).
#
#   2. Laps are NOT something we detect here. The logger already divides the
#      stint into laps geographically (start/finish beacon), and that division
#      reaches us as either a lap-number channel, a lap-time channel, or beacon
#      markers in the sibling .ldx file. build_lap_table just READS whichever
#      of those is present — it never infers a lap trigger from GPS/position.

from __future__ import annotations

import io
import re
import struct
import datetime
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Binary layout
# --------------------------------------------------------------------------

# The ld header. '4x'/'20x'/... are padding runs whose meaning is still
# unknown; they must be kept because they set the offsets of the real fields.
_HEAD_FMT = "<" + (
    "I4x"      # ldmarker (0x40)
    "II"       # chann_meta_ptr, chann_data_ptr
    "20x"      # ??
    "I"        # event_ptr
    "24x"      # ??
    "HHH"      # static numbers
    "I"        # device serial
    "8s"       # device type
    "H"        # device version
    "H"        # static number
    "I"        # num_channs
    "4x"       # ??
    "16s"      # date
    "16x"      # ??
    "16s"      # time
    "16x"      # ??
    "64s"      # driver
    "64s"      # vehicle id
    "64x"      # ??
    "64s"      # venue
    "64x"      # ??
    "1024x"    # ??
    "I"        # "pro logging" flag
    "66x"      # ??
    "64s"      # short comment
    "126x"     # ??
)

_EVENT_FMT = "<64s64s1024sH"      # name, session, comment, venue_ptr
_VENUE_FMT = "<64s1034xH"         # name, vehicle_ptr

# Channel meta block. Note the trailing padding is 40 bytes in files written by
# ACC and 32 in some MoTeC devices. That difference is harmless: we always seek
# to an explicit meta pointer, so an over-long read is simply ignored.
_CHAN_FMT = "<" + (
    "IIII"     # prev_addr, next_addr, data_ptr, n_data
    "H"        # counter
    "HHH"      # datatype_a, datatype, rec_freq
    "hhhh"     # shift, mul, scale, dec_places
    "32s"      # name
    "8s"       # short name
    "12s"      # unit
    "40x"      # ??
)

_HEAD_SIZE = struct.calcsize(_HEAD_FMT)
_CHAN_SIZE = struct.calcsize(_CHAN_FMT)

_LD_MARKER = 0x40


def _decode(raw: bytes) -> str:
    """Decode a fixed-width, NUL-padded ASCII field from the file."""
    try:
        return raw.decode("ascii", errors="replace").strip().rstrip("\0").strip()
    except Exception:
        return ""


def _resolve_dtype(dtype_a: int, dtype: int):
    """Map the ld type-code pair onto a numpy dtype.

    The format stores the type as two words: `dtype_a` selects the family
    (float / int) and `dtype` the width in bytes.
    """
    if dtype_a == 0x07:                                  # floating point
        return {2: np.float16, 4: np.float32}.get(dtype)
    if dtype_a in (0x00, 0x03, 0x05):                    # integer
        return {2: np.int16, 4: np.int32}.get(dtype)
    if dtype_a == 0x08 and dtype == 0x08:                # float64
        return np.dtype("<d")
    return None


# --------------------------------------------------------------------------
# Parsed representation
# --------------------------------------------------------------------------

@dataclass(eq=False)          # eq=False keeps Channel hashable (identity-based),
class Channel:                # so it can be used as a dict key when matching names
    """One logged channel, already converted to physical units."""
    name: str
    short_name: str
    unit: str
    freq: int                 # sample rate in Hz
    data: np.ndarray          # physical values

    @property
    def duration_s(self) -> float:
        return len(self.data) / self.freq if self.freq else 0.0

    @property
    def time_s(self) -> np.ndarray:
        """Implicit time axis of this channel: sample i was taken at i / freq."""
        return np.arange(len(self.data), dtype=np.float64) / float(self.freq or 1)


@dataclass
class LDLog:
    """A parsed .ld file plus any lap markers found in its .ldx sibling."""
    driver: str = ""
    vehicle_id: str = ""
    venue: str = ""
    event: str = ""
    session: str = ""
    datetime: datetime.datetime | None = None
    channels: list[Channel] = field(default_factory=list)
    ldx_marker_times_s: list[float] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def channel(self, name: str) -> Channel | None:
        """Look up a channel by exact (case-insensitive) name."""
        target = name.strip().lower()
        for ch in self.channels:
            if ch.name.strip().lower() == target:
                return ch
        return None

    @property
    def duration_s(self) -> float:
        return max((c.duration_s for c in self.channels), default=0.0)


# --------------------------------------------------------------------------
# .ld reading
# --------------------------------------------------------------------------

def read_ld(source) -> LDLog:
    """Parse a MoTeC .ld file.

    `source` may be a path, a bytes object, or any file-like object (which is
    what Streamlit's uploader hands us). We read the whole thing into memory
    once — endurance logs are tens of MB at most, and random access into a
    buffer is far simpler than juggling file seeks.
    """
    if isinstance(source, (bytes, bytearray)):
        buf = bytes(source)
    elif hasattr(source, "read"):
        try:
            source.seek(0)
        except Exception:
            pass
        buf = source.read()
    else:
        with open(source, "rb") as fh:
            buf = fh.read()

    if len(buf) < _HEAD_SIZE:
        raise ValueError(
            f"File is only {len(buf)} bytes — too short to be a MoTeC .ld file."
        )

    log = LDLog()

    (marker, meta_ptr, _data_ptr, event_ptr,
     _a, _b, _c,
     _serial, _dev_type, _dev_ver, _d, n_channels,
     date_s, time_s,
     driver, vehicle_id, venue,
     _pro, short_comment) = struct.unpack(_HEAD_FMT, buf[:_HEAD_SIZE])

    if marker != _LD_MARKER:
        # Not fatal: some devices/versions differ here. Flag it and continue,
        # because a wrong guess shows up immediately as garbage channel names.
        log.warnings.append(
            f"Unexpected file marker 0x{marker:x} (expected 0x{_LD_MARKER:x}); "
            "this may not be a MoTeC .ld file."
        )

    log.driver = _decode(driver)
    log.vehicle_id = _decode(vehicle_id)
    log.venue = _decode(venue)

    date_s, time_s = _decode(date_s), _decode(time_s)
    for fmt in ("%d/%m/%Y %H:%M:%S", "%d/%m/%Y %H:%M"):
        try:
            log.datetime = datetime.datetime.strptime(f"{date_s} {time_s}", fmt)
            break
        except ValueError:
            continue

    # Event / venue / vehicle blocks are optional and only used for labelling.
    if 0 < event_ptr < len(buf):
        try:
            size = struct.calcsize(_EVENT_FMT)
            ev_name, ev_session, _ev_comment, venue_ptr = struct.unpack(
                _EVENT_FMT, buf[event_ptr:event_ptr + size]
            )
            log.event, log.session = _decode(ev_name), _decode(ev_session)
            if 0 < venue_ptr < len(buf) and not log.venue:
                v_size = struct.calcsize(_VENUE_FMT)
                v_name, _veh_ptr = struct.unpack(
                    _VENUE_FMT, buf[venue_ptr:venue_ptr + v_size]
                )
                log.venue = _decode(v_name)
        except struct.error:
            log.warnings.append("Could not read the event block; labels may be blank.")

    # Channel meta blocks form a singly linked list starting at meta_ptr.
    seen: set[int] = set()
    ptr = meta_ptr
    while 0 < ptr < len(buf) and ptr not in seen:
        seen.add(ptr)
        chunk = buf[ptr:ptr + _CHAN_SIZE]
        if len(chunk) < _CHAN_SIZE:
            break
        (_prev, nxt, ch_data_ptr, n_data, _cnt,
         dtype_a, dtype_w, freq,
         shift, mul, scale, dec,
         name, short_name, unit) = struct.unpack(_CHAN_FMT, chunk)

        np_dtype = _resolve_dtype(dtype_a, dtype_w)
        ch_name = _decode(name)

        if np_dtype is None:
            log.warnings.append(
                f"Skipped channel '{ch_name}': unsupported data type "
                f"({dtype_a:#x}/{dtype_w:#x})."
            )
        else:
            width = np.dtype(np_dtype).itemsize
            end = ch_data_ptr + n_data * width
            if ch_data_ptr <= 0 or end > len(buf):
                log.warnings.append(
                    f"Skipped channel '{ch_name}': data block runs past the end "
                    "of the file (truncated log?)."
                )
            else:
                raw = np.frombuffer(buf, dtype=np_dtype,
                                    count=n_data, offset=ch_data_ptr)
                # Raw counts -> physical units. `scale` and `10^-dec` undo the
                # logger's fixed-point packing; `shift` is an offset applied in
                # scaled space; `mul` is the final unit multiplier.
                vals = raw.astype(np.float64)
                if scale:
                    vals = vals / float(scale)
                vals = (vals * (10.0 ** -dec) + shift) * (mul if mul else 1)
                log.channels.append(
                    Channel(
                        name=ch_name,
                        short_name=_decode(short_name),
                        unit=_decode(unit),
                        freq=int(freq) if freq else 1,
                        data=vals,
                    )
                )
        ptr = nxt

    if not log.channels:
        raise ValueError(
            "No channels could be read from this file. It may not be a MoTeC "
            ".ld log, or it may use a device layout this parser does not know."
        )

    if n_channels and len(log.channels) != n_channels:
        log.warnings.append(
            f"Header declares {n_channels} channels but {len(log.channels)} were "
            "read."
        )

    return log


# --------------------------------------------------------------------------
# .ldx reading (lap / beacon markers)
# --------------------------------------------------------------------------

def read_ldx_markers(source, log_duration_s: float | None = None) -> list[float]:
    """Extract beacon/lap marker times (in seconds) from a MoTeC .ldx file.

    .ldx is XML holding maths channels, display layouts and — the part we care
    about — the beacon markers that define the laps. MoTeC has never published
    the schema and it varies between i2 versions, so rather than binding to one
    element path we walk the whole tree and collect any element whose tag looks
    like a marker, reading whichever time-ish attribute it carries.

    Marker times are stored as integers in an unknown unit (microseconds in
    every sample seen). If the caller passes the log duration we use it to pick
    the unit that makes the markers fall inside the log — trying the COARSEST
    unit first, because several units can technically "fit". A last marker
    reading 420000 fits a 20-minute log both as 0.42 s (µs) and as 420 s (ms),
    and 420 s is the right reading: lap beacons span most of a stint, so the
    interpretation that covers the most of the log is the correct one.
    """
    if isinstance(source, (bytes, bytearray)):
        data = bytes(source)
    elif hasattr(source, "read"):
        try:
            source.seek(0)
        except Exception:
            pass
        data = source.read()
    else:
        with open(source, "rb") as fh:
            data = fh.read()

    try:
        root = ET.fromstring(data)
    except ET.ParseError as exc:
        raise ValueError(f"Could not parse .ldx as XML: {exc}") from exc

    raw: list[float] = []
    for el in root.iter():
        tag = el.tag.split("}")[-1].lower()          # drop any XML namespace
        if "marker" not in tag or tag.endswith("block") or tag.endswith("group"):
            continue
        for attr in ("Time", "time", "TimeStamp", "Pos", "Position", "Value"):
            if attr in el.attrib:
                try:
                    raw.append(float(el.attrib[attr]))
                except ValueError:
                    pass
                break

    if not raw:
        return []

    raw = sorted(set(raw))

    # Coarsest-first: seconds, then milliseconds, then microseconds.
    if log_duration_s and log_duration_s > 0:
        for factor in (1.0, 1e-3, 1e-6):
            if raw[-1] * factor <= log_duration_s * 1.05:
                return [t * factor for t in raw]
    return [t * 1e-6 for t in raw]          # no duration to check against


# --------------------------------------------------------------------------
# Channel identification
# --------------------------------------------------------------------------

# Canonical channel keys used everywhere downstream, mapped to the aliases seen
# across MoTeC devices and configurations. Matching is done on a squashed form
# of the name (lower-case, alphanumerics only), so "Throttle Pos",
# "throttle_pos" and "ThrottlePos" all collapse to the same key.
CHANNEL_ALIASES: dict[str, list[str]] = {
    "speed": [
        "corr speed", "corrected speed", "speed", "ground speed",
        "gps speed", "vehicle speed", "wheel speed", "drive speed",
    ],
    "throttle": [
        "throttle pos", "throttle position", "throttle", "tps",
        "throttle pedal", "apps", "accelerator pos",
    ],
    "steering": [
        "steering angle", "steered angle", "steering pos", "steering",
        "steering wheel angle",
    ],
    "g_lat": [
        "g force lat", "g force lateral", "lateral g", "lat g",
        "g lat", "acceleration lateral", "accel lat",
    ],
    "distance": [
        "distance", "corr distance", "corrected distance", "lap distance",
        "dist", "odometer",
    ],
    "lap_time": ["lap time", "laptime", "running lap time", "lap t"],
    "lap_number": [
        "lap number", "lap num", "lap no", "lap", "lap count", "laps",
    ],
}

# How each canonical channel should be resampled onto the common time grid.
# Step-like channels (a lap counter, a lap timer that resets at the line) must
# use nearest-neighbour, otherwise interpolation smears the step into a ramp.
_RESAMPLE_KIND = {
    "lap_number": "nearest",
    "lap_time": "nearest",
}

# Display labels, matching the channel names the team uses in i2.
CHANNEL_LABELS = {
    "speed": "Corr Speed [km/h]",
    "throttle": "Throttle Pos [%]",
    "steering": "Steering Angle [deg]",
    "g_lat": "G Force Lat [G]",
    "distance": "Distance [m]",
    "lap_time": "LapTime [s]",
    "lap_number": "Lap Number",
}


def _squash(name: str) -> str:
    """Normalise a channel name for alias matching."""
    return re.sub(r"[^a-z0-9]+", "", name.lower())


def identify_channels(log: LDLog) -> dict[str, Channel]:
    """Match the log's channels onto our canonical keys.

    Exact alias matches win; a channel whose squashed name merely *starts with*
    an alias is accepted as a fallback (e.g. "Throttle Pos Sensor"). Earlier
    aliases in each list have priority, so "Corr Speed" beats a plain "Speed"
    when a log carries both.
    """
    squashed = {ch: _squash(ch.name) for ch in log.channels}
    found: dict[str, Channel] = {}

    for key, aliases in CHANNEL_ALIASES.items():
        for alias in aliases:
            a = _squash(alias)
            hit = next((ch for ch, s in squashed.items() if s == a), None)
            if hit is None:
                hit = next((ch for ch, s in squashed.items() if s.startswith(a)), None)
            if hit is not None and hit.name not in {c.name for c in found.values()}:
                found[key] = hit
                break

    return found


def _convert_units(key: str, ch: Channel) -> np.ndarray:
    """Bring a channel into the unit the dashboard assumes.

    The dashboard works in km/h, %, deg, G and m. Loggers are configured
    differently between cars, so normalise here rather than trusting the label.
    """
    vals = ch.data.astype(np.float64)
    unit = _squash(ch.unit)

    if key == "speed":
        if unit in ("ms", "msec", "ms1", "mpers"):
            vals = vals * 3.6                       # m/s -> km/h
        elif unit in ("mph",):
            vals = vals * 1.609344                  # mph -> km/h
    elif key == "throttle":
        # Some loggers record throttle as a 0-1 ratio or as a raw voltage.
        finite = vals[np.isfinite(vals)]
        if finite.size and np.nanmax(np.abs(finite)) <= 1.5:
            vals = vals * 100.0
    elif key == "distance":
        if unit in ("km",):
            vals = vals * 1000.0
        elif unit in ("mi", "mile", "miles"):
            vals = vals * 1609.344
        elif unit in ("ft", "feet"):
            vals = vals * 0.3048
    elif key == "lap_time":
        # A lap timer logged in ms would read ~210000 for a target lap.
        finite = vals[np.isfinite(vals)]
        if finite.size and np.nanmax(finite) > 10_000:
            vals = vals / 1000.0

    return vals


# --------------------------------------------------------------------------
# Resampling onto a common time grid
# --------------------------------------------------------------------------

def _resample(t_src: np.ndarray, v_src: np.ndarray,
              t_dst: np.ndarray, kind: str = "linear") -> np.ndarray:
    """Put one channel onto the shared time grid.

    'linear' interpolates (right for continuous signals like speed);
    'nearest' snaps to the closest source sample (right for step signals like a
    lap counter, where interpolating would invent fractional lap numbers).
    """
    if len(t_src) == 0:
        return np.full(len(t_dst), np.nan)
    if len(t_src) == 1:
        return np.full(len(t_dst), v_src[0])

    if kind == "nearest":
        idx = np.searchsorted(t_src, t_dst, side="left").clip(1, len(t_src) - 1)
        left, right = t_src[idx - 1], t_src[idx]
        pick = np.where(np.abs(t_dst - left) <= np.abs(right - t_dst), idx - 1, idx)
        return v_src[pick]

    return np.interp(t_dst, t_src, v_src, left=v_src[0], right=v_src[-1])


def to_dataframe(log: LDLog, target_hz: float | None = None,
                 max_hz: float = 25.0) -> tuple[pd.DataFrame, dict[str, Channel]]:
    """Build a tidy DataFrame of the canonical channels on one time grid.

    Returns the frame (indexed by a `Time [s]` column) and the mapping of
    canonical key -> source Channel so the UI can report exactly which log
    channel it used for each metric.

    `max_hz` caps the grid: an endurance stint sampled at 200 Hz would be
    millions of rows for no analytical gain, since none of our metrics resolve
    anything faster than ~25 Hz.
    """
    found = identify_channels(log)
    if not found:
        raise ValueError(
            "None of the required channels were found in this log. Channels "
            f"present: {', '.join(c.name for c in log.channels[:25])}"
        )

    if target_hz is None:
        target_hz = min(max((ch.freq for ch in found.values()), default=10), max_hz)
    target_hz = float(max(target_hz, 0.1))

    # Only span the range every channel covers, so no metric is computed off
    # the end of a short channel's extrapolated tail.
    duration = min(ch.duration_s for ch in found.values())
    n = int(np.floor(duration * target_hz))
    if n < 2:
        raise ValueError(
            f"Log is too short to analyse ({duration:.1f} s of overlapping data)."
        )
    t = np.arange(n, dtype=np.float64) / target_hz

    df = pd.DataFrame({"Time [s]": t})
    for key, ch in found.items():
        vals = _convert_units(key, ch)
        df[CHANNEL_LABELS[key]] = _resample(
            ch.time_s, vals, t, _RESAMPLE_KIND.get(key, "linear")
        )

    df.attrs["sample_hz"] = target_hz
    return df, found


# --------------------------------------------------------------------------
# Lap division — read, never inferred
# --------------------------------------------------------------------------

def _segments_to_table(df: pd.DataFrame, bounds: list[int],
                       source: str,
                       lap_times: list[float] | None = None) -> pd.DataFrame:
    """Turn a list of boundary row indices into a lap table."""
    t = df["Time [s]"].to_numpy()
    dist_col = CHANNEL_LABELS["distance"]
    dist = df[dist_col].to_numpy() if dist_col in df else None

    rows = []
    for i in range(len(bounds) - 1):
        lo, hi = bounds[i], bounds[i + 1] - 1
        if hi <= lo:
            continue
        measured = float(t[hi] - t[lo])
        # Prefer the logger's own lap time when we have it: it is timed at the
        # beacon crossing, not quantised to our resample grid.
        native = lap_times[i] if lap_times and i < len(lap_times) else None
        lap_time = float(native) if native and np.isfinite(native) and native > 0 else measured

        rows.append({
            "Lap": len(rows) + 1,
            "LapTime [s]": lap_time,
            "Measured [s]": measured,
            "Start [s]": float(t[lo]),
            "End [s]": float(t[hi]),
            "start_idx": int(lo),
            "end_idx": int(hi),
            "Distance [m]": (float(dist[hi] - dist[lo])
                             if dist is not None and np.isfinite(dist[hi]) else np.nan),
        })

    out = pd.DataFrame(rows)
    out.attrs["lap_source"] = source
    return out


def build_lap_table(df: pd.DataFrame) -> pd.DataFrame:
    """Read the lap division the logger already produced.

    The telemetry system sums laps itself from the start/finish beacon, so this
    function only has to recover that existing division. It tries, in order of
    trustworthiness:

      1. A lap-number channel — an explicit integer per lap.
      2. A lap-time channel — either a running timer that resets at the line
         (sawtooth) or a hold of the last completed lap time (staircase).
      3. A lap-distance channel that resets each lap.

    No geographic lap-trigger detection is attempted anywhere.
    """
    lapnum_col = CHANNEL_LABELS["lap_number"]
    laptime_col = CHANNEL_LABELS["lap_time"]
    dist_col = CHANNEL_LABELS["distance"]

    # --- 1. Explicit lap counter -----------------------------------------
    if lapnum_col in df:
        lap = pd.to_numeric(df[lapnum_col], errors="coerce").to_numpy()
        if np.isfinite(lap).any():
            lap = np.round(lap)
            change = np.flatnonzero(np.diff(lap) != 0) + 1
            if change.size:
                bounds = [0, *change.tolist(), len(df)]
                native = _native_lap_times(df, bounds, laptime_col)
                return _segments_to_table(df, bounds, f"{lapnum_col} channel", native)

    # --- 2. Lap timer ------------------------------------------------------
    if laptime_col in df:
        lt = pd.to_numeric(df[laptime_col], errors="coerce").to_numpy()
        if np.isfinite(lt).any():
            table = _laps_from_lap_timer(df, lt, laptime_col)
            if table is not None and len(table):
                return table

    # --- 3. Lap distance that resets --------------------------------------
    if dist_col in df:
        d = pd.to_numeric(df[dist_col], errors="coerce").to_numpy()
        if np.isfinite(d).any():
            drops = np.flatnonzero(np.diff(d) < -50.0) + 1     # metres
            if drops.size:
                bounds = [0, *drops.tolist(), len(df)]
                return _segments_to_table(df, bounds, f"{dist_col} resets")

    empty = pd.DataFrame(columns=["Lap", "LapTime [s]", "Measured [s]", "Start [s]",
                                  "End [s]", "start_idx", "end_idx", "Distance [m]"])
    empty.attrs["lap_source"] = "none"
    return empty


def _native_lap_times(df: pd.DataFrame, bounds: list[int],
                      laptime_col: str) -> list[float] | None:
    """Read the logger's lap time for each segment, if a lap-time channel exists."""
    if laptime_col not in df:
        return None
    lt = pd.to_numeric(df[laptime_col], errors="coerce").to_numpy()
    if not np.isfinite(lt).any():
        return None

    out = []
    for i in range(len(bounds) - 1):
        lo, hi = bounds[i], bounds[i + 1] - 1
        seg = lt[lo:hi + 1]
        seg = seg[np.isfinite(seg)]
        if seg.size == 0:
            out.append(np.nan)
            continue
        # A running timer peaks just before the line; a staircase holds the
        # completed time. Taking the segment maximum reads both correctly.
        out.append(float(np.nanmax(seg)))
    return out


def _laps_from_lap_timer(df: pd.DataFrame, lt: np.ndarray,
                         laptime_col: str) -> pd.DataFrame | None:
    """Recover laps from a lap-time channel.

    Two shapes occur in the wild:

      running timer  0 -> 210 -> 0 -> 210 ...   (sawtooth; resets at the line)
      last lap time  209 209 209 | 211 211 ...  (staircase; steps at the line)

    We tell them apart by how much the signal moves within a segment: a running
    timer climbs continuously, a staircase is flat.
    """
    finite = np.isfinite(lt)
    if finite.sum() < 3:
        return None

    filled = pd.Series(lt).ffill().bfill().to_numpy()
    d = np.diff(filled)

    # Sawtooth: a large negative jump is a reset at the start/finish line.
    resets = np.flatnonzero(d < -1.0) + 1
    # Staircase: the held value changes (in either direction) at the line.
    steps = np.flatnonzero(np.abs(d) > 1e-6) + 1

    rising = float(np.mean(d > 0)) if d.size else 0.0

    if resets.size and rising > 0.5:
        # Running timer. Each reset opens a new lap; the peak before the reset
        # is that lap's time as the logger measured it.
        bounds = [0, *resets.tolist(), len(df)]
        native = [float(filled[b - 1]) for b in resets] + [np.nan]
        return _segments_to_table(df, bounds, f"{laptime_col} resets", native)

    if steps.size:
        # Staircase. The value adopted at a step is the time of the lap that
        # just finished, so segment i's time is the value held from step i on.
        bounds = [0, *steps.tolist(), len(df)]
        native = [float(filled[b]) for b in steps] + [np.nan]
        return _segments_to_table(df, bounds, f"{laptime_col} steps", native)

    return None


def apply_ldx_laps(df: pd.DataFrame, marker_times_s: list[float]) -> pd.DataFrame:
    """Build the lap table from .ldx beacon markers instead of a channel.

    Used when the .ld carries no lap channel, or when the engineer has added or
    corrected beacons in i2 — those edits live in the .ldx, not the .ld.
    """
    t = df["Time [s]"].to_numpy()
    if len(marker_times_s) < 2 or len(t) == 0:
        empty = pd.DataFrame(columns=["Lap", "LapTime [s]", "Measured [s]", "Start [s]",
                                      "End [s]", "start_idx", "end_idx", "Distance [m]"])
        empty.attrs["lap_source"] = "none"
        return empty

    # Accept a marker one sample period outside the grid at each end. The
    # closing beacon of a stint lands on the lap boundary, which is a fraction
    # of a sample past the final row — a strict bound would silently throw away
    # the last lap.
    dt = float(t[1] - t[0]) if len(t) > 1 else 0.0
    marks = [m for m in sorted(marker_times_s) if t[0] - dt <= m <= t[-1] + dt]
    if len(marks) < 2:
        empty = pd.DataFrame(columns=["Lap", "LapTime [s]", "Measured [s]", "Start [s]",
                                      "End [s]", "start_idx", "end_idx", "Distance [m]"])
        empty.attrs["lap_source"] = "none"
        return empty

    bounds = sorted({int(np.clip(np.searchsorted(t, m), 0, len(t))) for m in marks})
    # Beacon times are exact; use their differences as the lap times rather
    # than the row indices they snap to.
    native = [marks[i + 1] - marks[i] for i in range(len(marks) - 1)]
    return _segments_to_table(df, bounds, ".ldx beacon markers", native)


def add_lap_columns(df: pd.DataFrame, laps: pd.DataFrame) -> pd.DataFrame:
    """Annotate each row with its lap number and distance-into-lap.

    `Lap Distance [m]` is what makes a head-to-head overlay meaningful: the raw
    Distance channel is a cumulative odometer, so two drivers on the same lap
    would otherwise sit kilometres apart on the x-axis.
    """
    out = df.copy()
    out["_Lap"] = np.nan
    dist_col = CHANNEL_LABELS["distance"]
    if dist_col in out:
        out["Lap Distance [m]"] = np.nan

    for row in laps.itertuples(index=False):
        lo, hi = int(row.start_idx), int(row.end_idx)
        out.iloc[lo:hi + 1, out.columns.get_loc("_Lap")] = row.Lap
        if dist_col in out:
            seg = out[dist_col].to_numpy()[lo:hi + 1]
            base = seg[0] if len(seg) and np.isfinite(seg[0]) else np.nan
            out.iloc[lo:hi + 1, out.columns.get_loc("Lap Distance [m]")] = seg - base

    return out
