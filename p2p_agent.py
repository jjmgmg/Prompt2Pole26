"""Prompt2Pole agent driver for the IBM AI Racing League deliverable.

Plugs the trained **v9c** SAC policy (`sac_975000`) into the gym_torcs /
`torcs_jm_par.py` SCR client. The data path is exactly the training/eval one,
reused verbatim from the `prompt2pole` package so the deployed behaviour matches
the trained agent:

    snakeoil ServerState dict  ->  prompt2pole SensorFrame
                               ->  observation_from_frame  (76-D, normalized)
                               ->  SAC.predict(deterministic=True)  ->  [steer, long]
                               ->  accel/brake split  +  automatic gearbox (by speed)
                               ->  (steer, accel, brake, gear, clutch=0) for TORCS

Two subtleties MUST be reproduced or the car loses control (both come from the
training env stack, not the policy):

* **Action repeat (frame-skip 2).** Training/eval wrap the env in
  ``ActionRepeat(repeat=2)``: the agent decides at ~25 Hz (one action held for 2
  SCR tics), NOT every 50 Hz frame. Predicting every frame doubles the control
  frequency and the learned steering/throttle timing diverges. Override with
  ``P2P_ACTION_REPEAT``.
* **Green-light start.** The env starts the episode at the green light; the agent
  never saw the standing-start countdown. We hold on the grid until the countdown
  ends (``curLapTime >= 0``).

Set ``P2P_DEBUG=1`` to dump per-decision sensors + observation stats + action.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from stable_baselines3 import SAC

# Importing the custom policy module registers `LayerNormSACPolicy` so SB3 can
# resolve the class stored inside the checkpoint (v9c was trained with LayerNorm).
from prompt2pole.training import layernorm_policy  # noqa: F401
from prompt2pole.env.config import TRACK_LENGTH_M
from prompt2pole.env.gearbox import AutomaticGearbox
from prompt2pole.env.observations import observation_from_frame
from prompt2pole.scr.config import DEFAULT_TRACK_ANGLES
from prompt2pole.scr.sensors import SensorFrame

# Number of `track` range-finder beams the observation expects (19). The
# competition `torcs_jm_par.py` init string requests these exact angles.
N_TRACK = len(DEFAULT_TRACK_ANGLES)

# Frame-skip. MUST match the training env (ActionRepeat, D-0017); default 2.
ACTION_REPEAT = int(os.environ.get("P2P_ACTION_REPEAT", "2"))

# Deployment checkpoint (frozen v9c). Overridable with P2P_MODEL for testing.
_HERE = Path(__file__).resolve().parent
DEFAULT_MODEL = _HERE / "models" / "sac_975000.zip"


def _as_tuple(value: object) -> tuple[float, ...]:
    """Coerce a ServerState entry (scalar or list) to a float tuple."""
    if value is None:
        return ()
    if isinstance(value, (list, tuple)):
        return tuple(float(v) for v in value)
    return (float(value),)


def frame_from_serverstate(s: dict) -> SensorFrame:
    """Map a snakeoil `ServerState.d` dict to a typed `SensorFrame`.

    The wire keys and units are identical to what the `prompt2pole` SCR client
    parses (raw SCR values: speeds in km/h, beams in metres with -1 off-track),
    so no rescaling happens here — normalization is done downstream by
    `observation_from_frame`, exactly as in training.
    """
    return SensorFrame(
        angle=float(s.get("angle", 0.0)),
        cur_lap_time=float(s.get("curLapTime", 0.0)),
        damage=float(s.get("damage", 0.0)),
        dist_from_start=float(s.get("distFromStart", 0.0)),
        dist_raced=float(s.get("distRaced", 0.0)),
        focus=_as_tuple(s.get("focus")),
        fuel=float(s.get("fuel", 0.0)),
        gear=int(s.get("gear", 0)),
        last_lap_time=float(s.get("lastLapTime", 0.0)),
        opponents=_as_tuple(s.get("opponents")),
        race_pos=int(s.get("racePos", 0)),
        rpm=float(s.get("rpm", 0.0)),
        speed_x=float(s.get("speedX", 0.0)),
        speed_y=float(s.get("speedY", 0.0)),
        speed_z=float(s.get("speedZ", 0.0)),
        track=_as_tuple(s.get("track")),
        track_pos=float(s.get("trackPos", 0.0)),
        wheel_spin_vel=_as_tuple(s.get("wheelSpinVel")),
        z=float(s.get("z", 0.0)),
    )


class Prompt2PoleDriver:
    """Stateful deployment driver: ServerState dict -> TORCS control dict.

    Reproduces the training-time control loop: action-repeat frame-skip, the
    green-light start, the previous-action / grid-flag observation state, and the
    deterministic automatic gearbox.
    """

    def __init__(self, model_path: str | os.PathLike[str] | None = None) -> None:
        path = Path(model_path or os.environ.get("P2P_MODEL", DEFAULT_MODEL))
        if not path.exists():
            raise FileNotFoundError(f"deployment checkpoint not found: {path}")
        # Deterministic CPU inference (no GPU, matches how it was trained/evaluated).
        # The checkpoint stored TRAINING-only references (e.g. the stratified replay-buffer
        # class) that inference does not need; null them out so loading never imports training
        # modules absent from this minimal inference package (and skips a big buffer alloc).
        self.model = SAC.load(
            str(path),
            device="cpu",
            custom_objects={
                "replay_buffer_class": None,
                "replay_buffer_kwargs": {},
                "buffer_size": 1,
            },
        )
        self.gearbox = AutomaticGearbox()
        self._repeat = max(1, ACTION_REPEAT)
        self._debug = os.environ.get("P2P_DEBUG") == "1"
        self.reset()

    def reset(self) -> None:
        """Reset per-episode state (call once per race/episode)."""
        self.gearbox.reset(gear=1)
        self._prev_action: tuple[float, float] = (0.0, 0.0)
        self._crossed_start = False
        self._prev_dist: float | None = None
        self._phase = 0                    # 0 -> predict a fresh action; else hold
        self._steer = 0.0
        self._long = 0.0
        self._green = False
        self._agent_steps = 0

    def act(self, server_state: dict) -> dict:
        """Compute the TORCS control dict for one server tic.

        Returns keys understood by `DriverAction.d`: steer, accel, brake, gear,
        clutch. `focus`/`meta` are left to the transport's defaults.
        """
        frame = frame_from_serverstate(server_state)

        # Green-light start: hold on the grid through the countdown (curLapTime < 0);
        # the agent was only ever trained from the green light. Latches once green.
        if not self._green and frame.cur_lap_time < 0.0:
            return {"steer": 0.0, "accel": 0.0, "brake": 0.0, "gear": 1, "clutch": 0.0}
        self._green = True

        # Grid flag latch (D-0018): start line crossed when distFromStart wraps L -> 0
        # (single drop larger than half the track). Evaluated every SCR frame, as TorcsEnv.
        if (
            self._prev_dist is not None
            and not self._crossed_start
            and frame.dist_from_start - self._prev_dist < -TRACK_LENGTH_M / 2.0
        ):
            self._crossed_start = True
        self._prev_dist = frame.dist_from_start

        # Action-repeat (frame-skip): predict once per `repeat` SCR frames, hold in
        # between (the agent decided at ~25 Hz in training). The prediction uses the
        # frame at the start of each repeat block, matching ActionRepeat's last-obs.
        if self._phase == 0:
            obs = observation_from_frame(
                frame,
                N_TRACK,
                crossed_start=self._crossed_start,
                prev_action=self._prev_action,
            )
            action, _ = self.model.predict(obs, deterministic=True)
            self._steer = float(np.clip(action[0], -1.0, 1.0))
            self._long = float(np.clip(action[1], -1.0, 1.0))
            self._prev_action = (self._steer, self._long)
            self._agent_steps += 1
            if self._debug:
                self._dump(frame, obs)
        self._phase = (self._phase + 1) % self._repeat

        # Single longitudinal pedal: positive -> throttle, negative -> brake (design §4).
        accel, brake = (self._long, 0.0) if self._long >= 0.0 else (0.0, -self._long)
        # Gearbox shifts on road speed, not raw rpm (auto-clutch inflates rpm; D-0014).
        # Advanced every SCR frame, like the inner env step.
        gear = self.gearbox.update(frame.speed_x)

        return {
            "steer": self._steer,
            "accel": accel,
            "brake": brake,
            "gear": gear,
            "clutch": 0.0,
        }

    def _dump(self, frame: SensorFrame, obs: np.ndarray) -> None:
        """Print one line of diagnostics per agent decision (P2P_DEBUG=1)."""
        tr = frame.track
        print(
            f"[p2p] n={self._agent_steps:04d} green={int(self._green)} "
            f"spd={frame.speed_x:6.1f} ang={frame.angle:+.3f} tp={frame.track_pos:+.3f} "
            f"gearS={frame.gear} rpm={frame.rpm:5.0f} "
            f"trk[0/9/18]={tr[0] if tr else 0:.0f}/{tr[9] if len(tr) > 9 else 0:.0f}/"
            f"{tr[-1] if tr else 0:.0f} "
            f"| obs[min/max]={obs.min():+.2f}/{obs.max():+.2f} "
            f"-> steer={self._steer:+.3f} long={self._long:+.3f} gearBox={self.gearbox.gear}",
            flush=True,
        )
