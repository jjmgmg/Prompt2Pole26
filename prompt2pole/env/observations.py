"""Map a typed :class:`SensorFrame` to the normalized observation vector (T-004).

The canonical gym_torcs 29-dim layout (design §6), in order:
``[angle, track(n), trackPos, speedX, speedY, speedZ, wheelSpinVel(4), rpm]``, then the
D-0018 extension ``[gear, z, distFromStart, gridFlag, racePos, opponents(36)]``, the
D-0020 per-wheel slip block ``[slip(4)]``, and the D-0025 previous-action block
``[prev_steer, prev_long]`` (appended last) — 76 dims total.
Off-track ``-1`` ``track`` beams are mapped to a distinct marker (``TRACK_OFFTRACK_MARK``,
not clipped to 0) so the policy sees a clear "off the track" signal rather than a phantom
wall — recovery itself rides on ``track_pos`` (P5/P6). This is the one place the T-003
pass-through ``-1`` gets interpreted (RL semantics). The vector is clipped into the space
bounds so every emitted observation is in-space.
"""

from __future__ import annotations

from functools import cache

import numpy as np

from prompt2pole.scr.config import OPPONENTS_SENSOR_COUNT, WHEEL_COUNT
from prompt2pole.scr.sensors import SensorFrame

from .config import (
    ANGLE_SCALE,
    DIST_SCALE,
    GEAR_SCALE,
    OPPONENT_RANGE_M,
    RACEPOS_SCALE,
    RPM_SCALE,
    SLIP_OBS_SCALE,
    SPEED_SCALE,
    TRACK_OFFTRACK_MARK,
    TRACK_RANGE_M,
    WHEEL_RADIUS_M,
    WHEEL_SPIN_SCALE,
    Z_SCALE,
)
from .spaces import build_observation_space


@cache
def _bounds(n_track: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel (low, high) for clipping, from the space (single source)."""
    space = build_observation_space(n_track)
    return space.low, space.high


def _fixed(values: tuple[float, ...], n: int) -> list[float]:
    """Coerce a sensor vector to exactly ``n`` entries (zero-fill / truncate).

    The T-003 parser tolerates wrong-cardinality datagrams; the observation must
    still be a fixed-length vector, so pad short and truncate long.
    """
    out = list(values[:n])
    if len(out) < n:
        out += [0.0] * (n - len(out))
    return out


def _normalize(
    frame: SensorFrame,
    n_track: int,
    crossed_start: bool,
    prev_action: tuple[float, float],
) -> np.ndarray:
    """Build the raw (pre-clip) normalized vector in the canonical+extension order.

    Order matches ``spaces.build_observation_space``: the gym_torcs 29-dim set, then the
    D-0018 extension (gear, z, distFromStart, gridFlag, racePos, opponents), the D-0020
    per-wheel slip block, then the D-0025 previous action (last). ``crossed_start`` is episode
    state (the car has passed the start line since reset); ``prev_action`` is the last commanded
    ``(steer, long)`` (both already in [-1,1]) — neither is in the frame.
    """
    wheels_raw = _fixed(frame.wheel_spin_vel, WHEEL_COUNT)
    track = [
        b / TRACK_RANGE_M if b >= 0.0 else TRACK_OFFTRACK_MARK
        for b in _fixed(frame.track, n_track)
    ]
    wheels = [w / WHEEL_SPIN_SCALE for w in wheels_raw]
    # Opponents: same off-track/unusable handling as track (a <0 beam is undefined, not a
    # real distance). All 200 (-> 1.0) in single-car hotlap; carried for forward-compat.
    opponents = [
        b / OPPONENT_RANGE_M if b >= 0.0 else TRACK_OFFTRACK_MARK
        for b in _fixed(frame.opponents, OPPONENTS_SENSOR_COUNT)
    ]
    # Per-wheel slip (D-0020): (ω·r − v) normalized. The exact signal the reward penalizes,
    # handed to the policy explicitly (easier credit assignment; per-wheel distinguishes
    # front/rear and inside/outside). A single WHEEL_RADIUS_M -> order-agnostic. speed_x is
    # km/h -> /3.6 to m/s. Clipped into [-2, 2] by observation_from_frame like the rest.
    car_ms = frame.speed_x / 3.6
    slip = [(w * WHEEL_RADIUS_M - car_ms) / SLIP_OBS_SCALE for w in wheels_raw]
    return np.array(
        [
            frame.angle / ANGLE_SCALE,
            *track,
            frame.track_pos,
            frame.speed_x / SPEED_SCALE,
            frame.speed_y / SPEED_SCALE,
            frame.speed_z / SPEED_SCALE,
            *wheels,
            frame.rpm / RPM_SCALE,
            # -- D-0018 extension --
            frame.gear / GEAR_SCALE,
            frame.z / Z_SCALE,
            frame.dist_from_start / DIST_SCALE,
            1.0 if crossed_start else 0.0,
            frame.race_pos / RACEPOS_SCALE,
            *opponents,
            # -- D-0020 per-wheel slip --
            *slip,
            # -- D-0025 previous action a_{t-1} (last); already in [-1,1], no scaling --
            prev_action[0],
            prev_action[1],
        ],
        dtype=np.float32,
    )


def observation_from_frame(
    frame: SensorFrame,
    n_track: int,
    *,
    crossed_start: bool = False,
    prev_action: tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    """Normalize ``frame`` and clip into the observation-space bounds.

    ``crossed_start`` is the grid flag (False on the grid / before the start line, True after
    the lap-wrap crossing — see torcs_env). ``prev_action`` is the last commanded ``(steer, long)``
    fed back as the a_{t-1} obs channel (D-0025); defaults to neutral ``(0,0)`` for reset. A garbled
    sensor value can be non-finite (the parser coerces ``"inf"``/``"nan"`` via ``float()``);
    ``np.clip`` would pass ``NaN`` through, poisoning the policy. Sanitize to finite first (NaN→0,
    ±inf→large, then clipped to the bounds) so the observation is always finite and in-space.
    """
    low, high = _bounds(n_track)
    vec = np.nan_to_num(
        _normalize(frame, n_track, crossed_start, prev_action),
        nan=0.0, posinf=1e6, neginf=-1e6,
    )
    return np.clip(vec, low, high).astype(np.float32)
