"""Constants for the SCR telemetry client (T-003).

Single source of truth for protocol-level constants: default connection
endpoint, the track range-finder angle set, sensor-vector cardinalities, and
the socket timeout / retry budget. Nothing here knows about RL — see
``docs/design/scr-client.md`` and DECISIONS.md D-0011.
"""

from __future__ import annotations

# == Connection defaults (Windows debug_gui profile, D-0001/D-0010) ==
DEFAULT_HOST = "localhost"
DEFAULT_PORT = 3001
DEFAULT_SID = "SCR"

# == Track range-finder angles (the one real design call — D-0011 §3) ==
# Front-dense ±45° set, the de-facto deep-RL-on-TORCS standard inherited from
# yanpanlau/DDPG-Keras-Torcs → gym_torcs (reference/snakeoil3_gym.py:158).
# Decided from the domain literature, not as a library default; see
# docs/design/scr-client.md §5 and BIBLIOGRAPHY_NOTES.md. This constant is the
# single source of truth for the ``track`` dimension count that T-004 will read
# (do not hard-code the angles downstream).
DEFAULT_TRACK_ANGLES: tuple[float, ...] = (
    -45.0, -19.0, -12.0, -7.0, -4.0, -2.5, -1.7, -1.0, -0.5,
    0.0,
    0.5, 1.0, 1.7, 2.5, 4.0, 7.0, 12.0, 19.0, 45.0,
)

# == Sensor-vector cardinalities (manual §6.1) ==
TRACK_SENSOR_COUNT = 19
FOCUS_SENSOR_COUNT = 5
OPPONENTS_SENSOR_COUNT = 36
WHEEL_COUNT = 4

# == Socket I/O budget (tune against the live debug_gui profile — design §10) ==
RECV_TIMEOUT_S = 1.0          # per-recv socket timeout; generous for GUI startup
HANDSHAKE_RETRIES = 5         # (legacy) min init datagrams; handshake is now time-bound
RECV_RETRIES = 3              # bounded recv re-tries on a steady-state miss
RECV_BUFFER_SIZE = 2 ** 17    # mirrors the reference clients' data_size
# Wall-clock patience for the handshake. On Windows, sending init to a port with
# no listener yields an ICMP unreachable (WinError 10054), so a count-based retry
# budget burns out fast; bound by time + backoff so the race has time to come up
# while the client keeps re-sending init.
HANDSHAKE_GRACE_S = 30.0
HANDSHAKE_BACKOFF_S = 0.5

# == Control sentinels (datagram body, not a sensor group — manual §2/§6) ==
SENTINEL_IDENTIFIED = "***identified***"
SENTINEL_SHUTDOWN = "***shutdown***"
SENTINEL_RESTART = "***restart***"
