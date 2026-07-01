"""Parse an SCR server datagram into a typed :class:`SensorFrame`.

This module touches no socket. It turns one raw sensor string —
``(angle 0.0021)(speedX 1.94)(track 4.0 ... 4.0)(trackPos -0.001)...`` — into a
frozen, typed record. Off-track sentinels (``-1`` in ``track``/``focus``) are
passed through unchanged; interpreting them is RL semantics and belongs to
T-004/T-005, not here. See docs/design/scr-client.md §4.
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass

from .config import (
    FOCUS_SENSOR_COUNT,
    OPPONENTS_SENSOR_COUNT,
    TRACK_SENSOR_COUNT,
    WHEEL_COUNT,
)
from .errors import ScrProtocolError

logger = logging.getLogger(__name__)

# Wire camelCase -> Python snake_case. Single map driving both parsing and any
# future re-emission (design §4.1).
WIRE_TO_FIELD: dict[str, str] = {
    "angle": "angle",
    "curLapTime": "cur_lap_time",
    "damage": "damage",
    "distFromStart": "dist_from_start",
    "distRaced": "dist_raced",
    "focus": "focus",
    "fuel": "fuel",
    "gear": "gear",
    "lastLapTime": "last_lap_time",
    "opponents": "opponents",
    "racePos": "race_pos",
    "rpm": "rpm",
    "speedX": "speed_x",
    "speedY": "speed_y",
    "speedZ": "speed_z",
    "track": "track",
    "trackPos": "track_pos",
    "wheelSpinVel": "wheel_spin_vel",
    "z": "z",
}

# Fields exposed as vectors (tuples) and as ints, respectively.
_VECTOR_FIELDS = frozenset({"focus", "opponents", "track", "wheel_spin_vel"})
_INT_FIELDS = frozenset({"gear", "race_pos"})

# Expected vector cardinalities; a mismatch is logged as a protocol anomaly but
# is not fatal (a single garbled datagram must not kill a training run).
_EXPECTED_CARDINALITY: dict[str, int] = {
    "track": TRACK_SENSOR_COUNT,
    "focus": FOCUS_SENSOR_COUNT,
    "opponents": OPPONENTS_SENSOR_COUNT,
    "wheel_spin_vel": WHEEL_COUNT,
}

# One ``(name v0 v1 ...)`` group. ``[^()]*`` keeps the groups flat (the protocol
# has no nesting), which is more robust than split-on-``)(``.
_GROUP_RE = re.compile(r"\(([^()]*)\)")


@dataclass(frozen=True)
class SensorFrame:
    """One server tic of non-visual sensors (manual §6.1).

    Vectors are tuples (immutable, so the frame is fully frozen). Off-track
    ``-1`` markers in ``track``/``focus`` are preserved verbatim. ``focus`` and
    ``opponents`` are parsed but unused in V0.
    """

    angle: float = 0.0               # rad, car vs track axis, [-pi, +pi]
    cur_lap_time: float = 0.0        # s
    damage: float = 0.0
    dist_from_start: float = 0.0     # m
    dist_raced: float = 0.0          # m
    focus: tuple[float, ...] = ()    # 5 beams, [0,200] m, -1 unusable
    fuel: float = 0.0                # l
    gear: int = 0                    # {-1,0,1..6}
    last_lap_time: float = 0.0       # s
    opponents: tuple[float, ...] = ()  # 36 beams; parsed, unused in V0
    race_pos: int = 0
    rpm: float = 0.0
    speed_x: float = 0.0             # km/h longitudinal
    speed_y: float = 0.0             # km/h transverse
    speed_z: float = 0.0             # km/h vertical
    track: tuple[float, ...] = ()    # 19 beams, [0,200] m, -1 off-track
    track_pos: float = 0.0           # 0 axis, +/-1 edges, |.|>1 off-track
    wheel_spin_vel: tuple[float, ...] = ()  # 4 wheels, rad/s
    z: float = 0.0                   # m, mass-center height


def _coerce_floats(tokens: list[str], wire_key: str) -> list[float]:
    """Coerce wire tokens to floats, dropping (and logging) garbled ones."""
    values: list[float] = []
    for token in tokens:
        try:
            values.append(float(token))
        except ValueError:
            logger.warning("non-numeric token %r in group %r; skipping", token, wire_key)
    return values


def parse_sensors(datagram: str) -> SensorFrame:
    """Parse one server datagram into a :class:`SensorFrame`.

    Raises :class:`ScrProtocolError` only when *no* sensor group can be found
    (an empty or wholly unparseable datagram). Per-field problems — unknown
    keys, garbled numbers, wrong cardinalities — are logged and tolerated so the
    control loop survives a bad packet.
    """
    groups = _GROUP_RE.findall(datagram)
    if not groups:
        raise ScrProtocolError(f"no sensor groups in datagram: {datagram[:80]!r}")

    parsed: dict[str, object] = {}
    for group in groups:
        tokens = group.split()
        if not tokens:
            continue
        wire_key, raw_values = tokens[0], tokens[1:]
        field = WIRE_TO_FIELD.get(wire_key)
        if field is None:
            logger.debug("ignoring unknown sensor group %r", wire_key)
            continue

        values = _coerce_floats(raw_values, wire_key)
        if not values:
            logger.warning("empty/garbled values for sensor %r; using default", wire_key)
            continue

        if field in _VECTOR_FIELDS:
            expected = _EXPECTED_CARDINALITY.get(field)
            if expected is not None and len(values) != expected:
                logger.warning(
                    "protocol anomaly: %r has %d values, expected %d",
                    wire_key, len(values), expected,
                )
            parsed[field] = tuple(values)
        elif field in _INT_FIELDS:
            # float() accepts "inf"/"nan"/"1e400" without raising, but the
            # int() conversion below would then throw (OverflowError/ValueError)
            # and kill the loop — guard non-finite values like garbled tokens.
            if not math.isfinite(values[0]):
                logger.warning(
                    "non-finite value %r for int sensor %r; using default",
                    values[0], wire_key,
                )
                continue
            parsed[field] = int(round(values[0]))
        else:
            parsed[field] = values[0]

    return SensorFrame(**parsed)  # type: ignore[arg-type]
