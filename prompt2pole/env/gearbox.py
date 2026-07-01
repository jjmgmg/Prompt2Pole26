"""Automatic gearbox for the env (T-004) — berniw's shift law, calibrated to car1-ow1.

The policy never shifts; the env runs this fixed controller. The shift law is
**berniw's** — the canonical TORCS AI for car1-ow1 (``driver.cpp``): upshift when
road speed exceeds ``SHIFT_FRACTION`` (0.95) of the gear's redline speed; downshift
when speed falls ``SHIFT_MARGIN_KMH`` (= berniw's 4 m/s) below the gear-below's
shift speed (hysteresis). It triggers on **road speed**, not raw rpm, because
TORCS' automatic clutch inflates the reported rpm by thousands while slipping
(launch, hard acceleration), which makes raw-rpm shifting hunt (EXP-0002 / D-0014).

Redline speed per gear = ``REVS_LIMITER_RPM / rpm_per_kmh[gear]``, where
``rpm_per_kmh`` is an empirically-measured calibration (1st gear) scaled by the XML
gear ratios — no wheel-radius assumption. SCR commands only gears 1..6, so 6th is
top. Note car1-ow1 is geared for ~310 km/h: 2nd engages ~117, 3rd ~157, 4th ~198,
5th ~244, 6th ~271 km/h — so a slow track only exercises the low gears, by design.
See docs/design/torcs-env.md §5 and DECISIONS.md D-0012 / D-0014.
"""

from __future__ import annotations

from .config import (
    GEAR1_RPM_PER_KMH,
    GEAR_RATIOS,
    MAX_GEAR,
    REVS_LIMITER_RPM,
    SHIFT_COOLDOWN_STEPS,
    SHIFT_FRACTION,
    SHIFT_MARGIN_KMH,
)


class AutomaticGearbox:
    """Deterministic gear state machine: ``(current gear, speed) -> next gear``.

    Stateful (holds the current gear and a post-shift cooldown). Shift points are
    road speeds (berniw's law); a per-gear hysteresis margin keeps a gear engaged
    through normal speed swings.
    """

    def __init__(
        self,
        *,
        ratios: tuple[float, ...] = GEAR_RATIOS,
        max_gear: int = MAX_GEAR,
        redline_rpm: float = REVS_LIMITER_RPM,
        gear1_rpm_per_kmh: float = GEAR1_RPM_PER_KMH,
        shift_fraction: float = SHIFT_FRACTION,
        shift_margin_kmh: float = SHIFT_MARGIN_KMH,
        shift_cooldown_steps: int = SHIFT_COOLDOWN_STEPS,
    ) -> None:
        self._max_gear = max_gear
        self._cooldown = shift_cooldown_steps
        # Redline speed (km/h) for each gear: engaged rpm-per-km/h from the 1st-gear
        # calibration, scaled by the XML ratios; redline / that = top speed in gear.
        redline_speed = {
            g: redline_rpm / (gear1_rpm_per_kmh * ratios[g - 1] / ratios[0])
            for g in range(1, max_gear + 1)
        }
        # Upshift speed for gears 1..max-1 = SHIFT_FRACTION of that gear's redline
        # speed. Downshift speed for gears 2..max = the gear-below's upshift speed
        # minus the hysteresis margin (berniw's law).
        self._up_speed = {
            g: shift_fraction * redline_speed[g] for g in range(1, max_gear)
        }
        self._down_speed = {
            g: shift_fraction * redline_speed[g - 1] - shift_margin_kmh
            for g in range(2, max_gear + 1)
        }
        self.reset()

    def reset(self, gear: int = 1) -> None:
        """Reset to the standing-start gear (1) and arm shifting immediately."""
        self._gear = gear
        self._since_shift = self._cooldown  # ready to shift on the first update

    @property
    def gear(self) -> int:
        return self._gear

    def update(self, speed_kmh: float) -> int:
        """Advance the gearbox one control step and return the gear to command."""
        self._since_shift += 1
        if self._since_shift < self._cooldown:
            return self._gear

        speed = max(0.0, speed_kmh)
        if self._gear < self._max_gear and speed >= self._up_speed[self._gear]:
            self._gear += 1
            self._since_shift = 0
        elif self._gear > 1 and speed <= self._down_speed[self._gear]:
            self._gear -= 1
            self._since_shift = 0
        return self._gear
