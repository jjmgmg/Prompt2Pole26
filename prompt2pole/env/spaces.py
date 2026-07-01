"""Observation / action space construction (T-004).

The observation is the canonical gym_torcs 29-dim normalized vector (design §6);
its ``track`` block width is read from ``DEFAULT_TRACK_ANGLES`` (one source of
truth with T-003), so the total length follows the angle set rather than a
hard-coded 29. The action is the 2-D ``[steer, long]`` box (design §4). Bounds are
finite and generous; ``observations.py`` clips into them so every emitted vector
is in-space. See docs/design/torcs-env.md.
"""

from __future__ import annotations

import numpy as np
from gymnasium import spaces

from prompt2pole.scr.config import (
    DEFAULT_TRACK_ANGLES,
    OPPONENTS_SENSOR_COUNT,
    WHEEL_COUNT,
)

# Observation blocks, in order, as (name, dim, low, high). ``track`` and ``opponents``
# dims are filled from the sensor counts at build time. These bounds are the
# post-normalization ranges the observation is clipped to (design §6 + D-0018).
#
# Layout: the canonical gym_torcs 29-dim set first (angle, track, trackPos, speeds,
# wheelSpinVel, rpm) for continuity, then the D-0018 extension block (gear, z,
# distFromStart, gridFlag, racePos, opponents), then the D-0020 per-wheel slip block
# (4 dims), then the D-0025 previous-action block (2 dims, appended last) — see
# env/config.py for the rationale.
_SCALAR_BLOCKS_BEFORE_TRACK = (("angle", 1, -1.0, 1.0),)
_SCALAR_BLOCKS_AFTER_TRACK = (
    ("trackPos", 1, -2.0, 2.0),
    ("speedX", 1, -2.0, 2.0),
    ("speedY", 1, -2.0, 2.0),
    ("speedZ", 1, -2.0, 2.0),
    ("wheelSpinVel", 4, 0.0, 5.0),
    ("rpm", 1, 0.0, 1.1),
    # -- D-0018 extension (forward-compat across hotlap / competition) --
    ("gear", 1, -1.0, 1.1),        # gear / 6 -> {-1/6 .. 1.0}
    ("z", 1, -5.0, 5.0),           # mass-center height (m), generous bound
    ("distFromStart", 1, 0.0, 1.1),  # dist_from_start / L -> [0, 1]
    ("gridFlag", 1, 0.0, 1.0),     # 0 on grid / pre-start-line, 1 after crossing
    ("racePos", 1, 0.0, 1.1),      # race_pos / 40 (max grid); constant in hotlap
)
# track + opponents beams: [0,1] in range; the off-track/unusable marker (-1) sets the low.
_TRACK_LOW, _TRACK_HIGH = -1.0, 1.0
_OPP_LOW, _OPP_HIGH = -1.0, 1.0
# Per-wheel slip (D-0020): appended after opponents, 4 dims. slip = (ω·r − v)/scale,
# ~[-1,1] for gross wheelspin; the Box bound [-2,2] clips. See env/config.py / observations.py.
_SLIP_LOW, _SLIP_HIGH = -2.0, 2.0
# Previous action a_{t-1} (D-0025, reward v4): the last commanded [steer, long], appended LAST, 2
# dims in [-1, 1] (the action space). Lets a temporally-smooth policy emerge — the obs-side of CAPS
# (Mysore 2021) within stock SAC; pairs with the raised K_SMOOTH. (0,0) on reset / at handover.
_PREVACT_DIM = 2
_PREVACT_LOW, _PREVACT_HIGH = -1.0, 1.0


def observation_dim(n_track: int = len(DEFAULT_TRACK_ANGLES)) -> int:
    """Total length for a ``track`` beam count (+ opponents + slip + prev-action blocks)."""
    before = sum(d for _, d, *_ in _SCALAR_BLOCKS_BEFORE_TRACK)
    after = sum(d for _, d, *_ in _SCALAR_BLOCKS_AFTER_TRACK)
    return before + n_track + after + OPPONENTS_SENSOR_COUNT + WHEEL_COUNT + _PREVACT_DIM


def build_observation_space(
    n_track: int = len(DEFAULT_TRACK_ANGLES),
) -> spaces.Box:
    """Build the fixed ``Box`` observation space (design §6 + D-0018)."""
    lows: list[float] = []
    highs: list[float] = []
    for _, dim, lo, hi in _SCALAR_BLOCKS_BEFORE_TRACK:
        lows += [lo] * dim
        highs += [hi] * dim
    lows += [_TRACK_LOW] * n_track
    highs += [_TRACK_HIGH] * n_track
    for _, dim, lo, hi in _SCALAR_BLOCKS_AFTER_TRACK:
        lows += [lo] * dim
        highs += [hi] * dim
    lows += [_OPP_LOW] * OPPONENTS_SENSOR_COUNT
    highs += [_OPP_HIGH] * OPPONENTS_SENSOR_COUNT
    lows += [_SLIP_LOW] * WHEEL_COUNT          # per-wheel slip (D-0020)
    highs += [_SLIP_HIGH] * WHEEL_COUNT
    lows += [_PREVACT_LOW] * _PREVACT_DIM      # previous action a_{t-1} (D-0025), appended last
    highs += [_PREVACT_HIGH] * _PREVACT_DIM
    return spaces.Box(
        low=np.array(lows, dtype=np.float32),
        high=np.array(highs, dtype=np.float32),
        dtype=np.float32,
    )


def build_action_space() -> spaces.Box:
    """Build the 2-D ``[steer, long]`` action space, both in [-1, 1] (design §4)."""
    return spaces.Box(
        low=np.array([-1.0, -1.0], dtype=np.float32),
        high=np.array([1.0, 1.0], dtype=np.float32),
        dtype=np.float32,
    )
