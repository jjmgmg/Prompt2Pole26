"""Constants for the Gymnasium environment wrapper (T-004).

Single source of truth for the RL-side configuration: the `car1-ow1` physical
spec (engine/gearbox, used by the automatic gearbox), observation normalization
scales, the gearbox shift schedule, and episode limits. Nothing here touches the
socket — that is the `scr` package (T-003). See docs/design/torcs-env.md and
DECISIONS.md D-0012/D-0013.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# == Car: car1-ow1 (V0 competition car, D-0012) ==
# Read verbatim from the live C:\torcs\torcs\cars\car1-ow1\car1-ow1.xml. The
# automatic gearbox (gearbox.py) is derived from these numbers, not a heuristic.
CAR_NAME = "car1-ow1"

# Forward gear ratios 1..7 from the XML. The SCR `gear` actuator only commands
# {-1,0,..,6}, so the 7th gear (1.46) is UNREACHABLE and is kept here only for
# provenance; MAX_GEAR caps the gearbox at 6 (design §5.1).
GEAR_RATIOS: tuple[float, ...] = (3.9, 2.9, 2.3, 1.87, 1.68, 1.54, 1.46)
MAX_GEAR = 6                  # highest SCR-commandable forward gear (6th = top)

REVS_LIMITER_RPM = 18700.0    # `revs limiter` from the XML
REVS_MAXI_RPM = 20000.0       # `revs maxi`; used as the rpm normalization scale
TICKOVER_RPM = 5000.0         # idle rpm at the standing start

# == Automatic gearbox shift schedule (design §5; D-0014, validated vs berniw) ==
# The gearbox triggers on **road speed**, not raw rpm. The live runs (EXP-0002)
# showed TORCS' automatic clutch inflates the reported rpm by several thousand
# while slipping (standing-start launch, hard acceleration), which made raw-rpm
# shifting hunt badly. Speed is smooth and clutch-slip-immune.
#
# Shift law = berniw's (the canonical TORCS AI for car1-ow1, driver.cpp): upshift
# when speed exceeds SHIFT_FRACTION of the gear's redline speed; downshift when
# speed falls SHIFT_MARGIN_KMH below the gear-below's shift speed (hysteresis).
# Redline speed per gear = REVS_LIMITER_RPM / (engaged rpm-per-km/h of that gear).
# 1st gear's rpm/(km/h) is measured from clean (non-slipping) telemetry
# (67.0 km/h -> 10201 rpm, 90.2 -> 13614 => ~152); other gears scale by the XML
# ratios. No wheel-radius assumption needed.
GEAR1_RPM_PER_KMH = 152.0
SHIFT_FRACTION = 0.95         # berniw SHIFT: upshift at 95% of redline speed
SHIFT_MARGIN_KMH = 14.4       # berniw SHIFT_MARGIN = 4.0 m/s = 14.4 km/h hysteresis
# Steps to hold after any shift before another is allowed (debounce; berniw relies
# on the margin alone, this is a small extra guard).
SHIFT_COOLDOWN_STEPS = 5

# == Observation normalization (design §6 — gym_torcs 29-dim lineage) ==
ANGLE_SCALE = math.pi         # angle / pi  -> ~[-1, 1]
TRACK_RANGE_M = 200.0         # track / 200 -> [0, 1] on-track
# Off-track, the SCR track rangefinders return -1 (the beam is undefined outside the track,
# not a real distance — manual §6.1; confirmed in the deep-RL literature: a learned policy
# cannot navigate back from the beams, recovery is via track_pos + termination). So off-track
# beams are mapped to a DISTINCT marker instead of clipped to 0 — clipping to 0 would
# masquerade as "wall right ahead" everywhere, whereas a clear negative flags "off the track"
# (complements track_pos, which is the real recovery signal). See P5/P6 and D-0017.
TRACK_OFFTRACK_MARK = -1.0
SPEED_SCALE = 300.0           # speed_{x,y,z} / 300 (km/h)
WHEEL_SPIN_SCALE = 100.0      # wheel_spin_vel / 100 (rad/s)
RPM_SCALE = REVS_MAXI_RPM     # rpm / 20000

# == Per-wheel slip observation (D-0020, obs 70 -> 74) ==
# The raw wheelSpinVel are already in the obs, so slip is *inferable*; exposing it
# explicitly is cheap feature engineering that hands the policy the exact signal it is also
# penalized for (easier credit assignment -> faster learning of throttle modulation), and
# per-wheel it distinguishes front vs rear (RWD launch wheelspin) and inside vs outside
# (corner spin). slip_i = (wheel_spin_vel[i] * WHEEL_RADIUS_M - speed_x/3.6) / SLIP_OBS_SCALE.
# A SINGLE wheel radius for all four -> order-agnostic (front 0.302 m vs rear 0.315 m differ
# ~4 %, negligible for a feature). SLIP_OBS_SCALE ~ gross wheelspin (20-30 m/s) -> ~[-1, 1];
# Box bound [-2, 2] clips. See docs/design/torcs-learn-to-lap.md §B.0.
WHEEL_RADIUS_M = 0.31         # single radius for all wheels (m); slip feature is order-agnostic
SLIP_OBS_SCALE = 30.0         # m/s slip normalization (gross wheelspin -> ~[-1, 1])

# Corkscrew length (m): the finish-line wrap point for dist_from_start. A single drop larger
# than half the track in one ~20 ms tic can only be a lap wrap (design §2). Measured in
# EXP-0003: max dist_from_start before the wrap was 3608.2 m across all three smokes (grid sits
# at 3598.4, ~10 m behind the line). Used by both the reward wrap correction and the obs
# (dist_from_start normalization + the grid-flag wrap detection).
TRACK_LENGTH_M = 3608.2

# Physical per-step ceiling on track progress (m), used to reject the RSI-handover **resync
# teleport** (EXP-0010). The SCR server can stream a stale frame across the green-light/handover
# for ~tens of steps, then resync dist_from_start in one tic by tens-to-thousands of metres.
# The old wrap correction only unwrapped jumps > L/2, so a +657 m resync passed through RAW as
# phantom progress and a +3088 m one became a spurious -520 m — together ~50 % of EXP-0010's
# ENTIRE reward mass (verify: scratch/verify_crux.py). At ~25 Hz control (action_repeat=2,
# ~0.04 s/step) even 310 km/h advances ≲3.4 m/step; the largest *legitimate* per-step progress
# observed was 5.98 m. So any |Δ| above this ceiling is a resync, not motion → it contributes
# **0 progress** (not a raw jump, not an L-subtraction). Generous margin over the physical bound.
PROGRESS_CLAMP_M = 6.0

# == Extended observation (D-0018 — one obs across training modes, forward-compatible) ==
# Beyond the canonical gym_torcs 29-dim set, the obs carries six more channels so the SAME
# observation (and reward) serve hotlap V0 *and* the later multi-car competition mode without
# re-architecting (some channels are constant/inert in single-car hotlap):
#   gear           : drivetrain state (redundant-ish with speed, but cheap; D-0012 knob).
#   z              : mass-center height above the surface (m) — Corkscrew has elevation.
#   dist_from_start: ABSOLUTE curvilinear track position. This admits position-indexed
#                    ("geometric") learning, which DEFINITION §4 excluded; D-0018 brings it
#                    into V0 by explicit decision (it references the track; the user owns the
#                    trade-off vs generalization). Paired with grid_flag to disambiguate the
#                    grid (~L) from the finish (~L) — see torcs_env._crossed_start.
#   grid_flag      : 0 on the grid / before the start line, 1 after crossing it (wrap-detected,
#                    NOT cur_lap_time which is already >0 on the grid). Resolves the ambiguity
#                    that dist_from_start ≈ 3598 at the grid AND ≈ 3608 when finishing.
#   race_pos       : position in the field; constant 1 in single-car hotlap (no signal), varies
#                    with opponents in competition. In obs and reward (inert here) for one set.
#   opponents x36  : range finders to nearby cars; all 200 (-> 1.0) in single-car hotlap. Carried
#                    now so a hotlap policy can be fine-tuned for racing without an obs change.
GEAR_SCALE = float(MAX_GEAR)  # gear / 6 -> {-1/6 .. 1.0}
Z_SCALE = 1.0                 # z (m) passthrough; ~0.3-0.5 m typical, bounded generously
DIST_SCALE = TRACK_LENGTH_M   # dist_from_start / L -> [0, 1] (Corkscrew; single-track V0)
RACEPOS_SCALE = 40.0          # race_pos / max grid size (training XML "maximum number" = 40)
OPPONENT_RANGE_M = 200.0      # opponents / 200 -> [0, 1]; an unusable (<0) beam -> off-track mark

# == Reward (T-005, D-0015 — dense progress, racing-line-agnostic) ==
# Main signal = wrap-corrected Δ dist_from_start (curvilinear track progress), NOT the
# odometer dist_raced (which a doughnut/spin would inflate). No centering term: any
# line is free. See docs/design/torcs-reward.md §3. Coefficients are tuned against
# debug_gui like the gearbox (D-0014): the design fixes the shape, validation the scale.
K_PROGRESS = 1.0           # reward weight per metre of track progress
K_DAMAGE = 0.01            # penalty weight per unit of new damage this step (collisions)
K_OFFCOURSE = 0.1          # penalty weight for (|track_pos| - 1) while off the surface
# K_SMOOTH: reverted 0.10 -> 0.05 (D-0027). The D-0025 bump to 0.10 BACKFIRED (EXP-0015): a stronger
# |Δsteer| penalty rewards a CONSTANT steering angle, so the agent learned to HOLD a large steer
# (|steer|>0.5 rose 49->58% while oscillation fell) -> at speed a held angle = the car turns in a
# circle = heading runaway WORSE (|angle| p90 50->90 deg). So it fed the very failure it targeted.
# Back to 0.05 (the validated EXP-0008c value). The real steering/oscillation fix is the a_{t-1} obs
# channel + the LayerNorm-stabilized critic (critic_layernorm, D-0027), not a heavier smooth term.
K_SMOOTH = 0.05            # penalty weight for the per-step action change |Δsteer|+|Δlong|
# Speed gate for the THROTTLE part of the smoothness penalty (EXP-0008/0008c). Penalizing throttle
# rate at the standing start is counterproductive: it taxes the throttle exploration the launch
# needs and rewards sitting still, so SAC settled into a stuck local optimum. So |Δlong| is gated
# to apply only ABOVE this speed. The STEERING part (|Δsteer|) is NOT gated — it applies at all
# speeds, because removing it let the car pirouette off the track at low speed (the trompo,
# EXP-0008c). This is action-smoothness, not centering: the racing line still emerges.
SMOOTH_MIN_SPEED_KMH = 30.0
# Anti-OSCILLATION term (D-0048 Option 2). The magnitude smooth above taxes ALL action change, incl.
# the sharp/bang-bang inputs a fast lap needs (no proven racing reward uses |Daction|; Remonda
# reduces oscillation ARCHITECTURALLY via delta-actions). The reward-side alternative: penalize a
# SIGN REVERSAL of an action channel only when BOTH the previous and current command are
# meaningfully large (|·| > OSC_THRESHOLD) — i.e. chatter (+0.5,−0.5,…) and violent one-step slams
# (0.8→−0.8). Same-sign ramps (1→0.2) and a transition through ~0 (1→0.2→−0.8, "lose one step") are
# FREE. Steer + long counted independently. Default 0 = inert (set K_SMOOTH=0 + K_OSC>0 to switch).
K_OSC = 0.0                # penalty per sign-reversal (steer, long) when both |·| > OSC_THRESHOLD
OSC_THRESHOLD = 0.3        # magnitude above which an opposite-sign step counts as a reversal/slam
# P_FAIL REVERTED 8->1 (D-0029, reward v6): the cross-run A/B (D-0028) proved P_FAIL=8 made the SAC
# critic diverge (reward-scale sensitivity: a -8..-15 terminal spike vs ~0.3/step progress makes the
# Q-target unfittable). v2 (P_FAIL=1 + kinetic) was STABLE (critic_loss 2-5, +25 return). Back to 1;
# discouraging off-track is now the DENSE slide term's job (below), not a high-variance spike.
P_FAIL = 1.0               # one-shot penalty on a failure terminal (off-course/stuck/damage)
# Position-gain reward (D-0018): + per position advanced (race_pos decreasing). INERT in V0
# hotlap — race_pos is constant 1, so the term is identically 0 (the validated EXP-0003 reward
# is unchanged). It activates only in the multi-car competition mode, keeping ONE reward across
# modes (the overtaking/anti-gaming channel of D-0017 P9). Tuned when competition lands.
K_RACEPOS = 1.0            # reward weight per race position gained this step

# == Reward v2 (D-0020 — research-grounded, same GT-Sophy structure) ==
# Two principled terms added to the T-005 reward to break the EXP-0007 plateau
# (wheelspin-at-launch + lurch-into-the-first-corner-without-braking). See
# docs/design/torcs-learn-to-lap.md §B.1 and §10 (GT Sophy / GTS / driver-modeling).
#
# 1. Wheel-slip penalty (configurable threshold). Penalize only slip ABOVE a threshold
#    (gross wheelspin), via the fastest wheel's surface speed vs car speed — penalizing ALL
#    slip makes the policy over-conservative (literature). slip_ms = max(wheel_spin_vel) *
#    WHEEL_RADIUS_M - speed_x/3.6; penalty = -K_SLIP * max(0, slip_ms - SLIP_THRESHOLD_MS).
#    SLIP_THRESHOLD_MS is the user-requested init hyperparameter (m/s tolerated before
#    penalizing); K_SLIP keeps a hard launch's slip term O(0.1-0.5)/step (~the progress term).
SLIP_THRESHOLD_MS = 2.5    # m/s of wheel slip tolerated before penalizing (gross wheelspin)
K_SLIP = 0.02             # penalty weight per m/s of slip above the threshold
# 2. Kinetic-energy-scaled failure penalty (anti lurch-and-crash). fail = -(P_FAIL +
#    K_KINETIC * (speed_x / V_REF_KMH)^2): a 100 km/h off-course hurts far more than a 30 km/h
#    one, so the policy learns to brake/corner instead of flooring into the wall; a `stuck`
#    fail (speed ~ 0) keeps ~ P_FAIL. Counters the discount's anti-braking bias (GTS KE wall).
K_KINETIC = 5.0           # extra fail penalty at V_REF_KMH (100 km/h crash ~ -6 vs a clean lap)
V_REF_KMH = 100.0         # reference speed for the kinetic-energy fail scaling

# == Reward v3 (EXP-0013 — the reward PAID for crashing; fix it, line-agnostic) ==
# EXP-0013 (12h) found leaving the track was REWARD-RATIONAL: drifting to the edge was net-POSITIVE
# (progress dominates; off_course is a cliff at |tp|>1, ~5% of progress) and the crash step scored
# +0.27. Spinning ON the track was FREE (progress~0, off_course=0). So the agent drifts/spins off
# everywhere (straights too), heading to ~50-180 deg. Two line-agnostic fixes (NO centering):
#  1. P_FAIL raised (above) -> leaving the track is clearly net-NEGATIVE (crash > progress stolen).
#  2. Anti-spin below = a DENSE per-step penalty on heading beyond a threshold. The only steering
#     feedback before was the ~19-step-delayed off_course; this gives an immediate "you are rotating
#     out of control" signal. Threshold validated > clean driving (|angle| p90~30 deg; failure tail
#     50-180) so a good line (out-in-out, apex) stays UNDER it = 0 penalty.
# K_SPIN LOWERED 2.0->0.5 (D-0025, reward v4): at 2.0 the anti-spin DOMINATED progress (EXP-0014:
# spin -0.22..-0.83/step vs progress +0.31, NET always negative) -> critic_loss blew up 15-20x and
# the agent turned timid (54% of grid starts never moved). SAC is sensitive to reward scale
# (Haarnoja 2018) and GT Sophy warns "too high a penalty -> drivers become timid" (Wurman 2022).
# At 0.5 the worst case (|angle|~1.2) is ~progress, not 2.4x -> NET positive driving well, negative
# only when spinning. Still a corrective dense signal, no longer swamping. Threshold unchanged.
K_SPIN = 0.0                    # SUPERSEDED by the dense slide term (D-0029); kept as an inert knob
ANGLE_SPIN_THRESHOLD_RAD = 0.7  # ~40deg (unused while K_SPIN=0)

# == Reward v6 (D-0029 — the missing dense heading signal is SLIDE, not a cliff) ==
# Root-cause study: the missing recipe element is the gym_torcs dense heading term. KEY FINDING
# (448k steps): progress = Δdist_from_start is curvilinear, so it ALREADY ~= Vx*cos(angle) (ratio
# tracks cos: 15-30deg->0.94, 30-50->0.75) -> the cos(angle) REWARD is already in `progress`.
# What is MISSING is the gym_torcs LATERAL-SLIDE penalty -Vx*|sin(angle)|: at 51deg the agent still
# banks cos(51)=63% of progress, so going sideways is only mildly less-rewarded, never costly.
# The slide term makes slip actively expensive with a smooth gradient (vs the old 40deg cliff). It
# penalizes going SIDEWAYS (slip), NOT lateral POSITION -> line-agnostic: out-in-out keeps |angle|
# moderate (heading follows the corner), so only spins/slides (the failure mode, esp. Corkscrew
# hairpins + esses) are hit. Speed-normalized + clamped (|Vx|/V_REF<=1) so it can't blow the scale:
# slide = -K_LAT*min(|Vx|/V_REF,1)*max(0,|sin(slip)|-SLIP_FREE), slip = atan2(speed_y, speed_x).
# D-0046: the angle is the TRUE side-slip (velocity vs heading, from speed_y), NOT frame.angle
# (heading vs track) — counter-steer kept the old term ~0 while the car slid wide (EXP-0034). The
# FREE band leaves the normal racing slip unpenalized. Calibration (clean-lap data EXP-0034: slip
# p90 ~6.4deg -> |sin|~0.11; failure slides 25-40deg -> 0.42-0.64): SLIP_FREE 0.12 (~7deg) frees the
# racing line; K_LAT 0.3 makes a 25-40deg slide cost ~0.09-0.16/step (vs progress ~1.5) -> bites the
# slide-off without dominating or going timid (GT-Sophy). Tune via the run.
K_LAT = 0.3                # side-slip weight (0 in v11/forward; ON for D-0046 vs the 2490m slide)
SLIP_FREE = 0.12           # |sin(slip)| below this is free (normal racing slip ~<7deg)

# == Reward v7 (D-0030 — dense anti-EDGE signal: the last missing piece of the proven reward) ==
# EXP-0018 (reward v6): still plateaus ~26 m, off_course-dominant (~60-76/decile); the agent drifts
# to the edge and falls off, from RSI starts too (not just launch). The proven gym_torcs reward
# that LAPS is Vx*cos - Vx*|sin| - Vx*|trackPos|; we have cos (progress) + |sin| (slide) but NOT the
# lateral-position term -> no dense gradient to avoid the edge (off_course only fires at |tp|>1).
# This adds it LINE-AGNOSTIC: penalize only the OUTER band (|trackPos| > edge_inner, def 0.6) -> the
# central 60% is free (the line still emerges; out-in-out only grazes the edge at the apex), but
# drifting to the brink costs continuously. NOT centering (no pull to tp=0). Speed-normalized +
# clamped like slide: edge = -K_EDGE*min(|Vx|/V_REF,1)*max(0,|trackPos|-edge_inner).
K_EDGE = 1.0               # anti-edge weight (outer-band only; line-agnostic, not centering)
EDGE_INNER = 0.6           # |track_pos| beyond which the dense anti-edge penalty starts to ramp
# D-0044 fence shape: width (in |track_pos|) of the linear ramp past edge_inner before the penalty
# PLATEAUS at 1.0 (so the cost is time-based, not depth-proportional -> Q-stable; D-0040 §A.1). With
# edge_inner=1.28 the fence reaches full strength by |tp|=1.35, then flat to the |tp|>2 termination.
EDGE_RAMP_W = 0.07

# Minimum-time pressure (D-0036): a CONSTANT per-step penalty. The RSI episode ends at the finish
# (fixed distance), so the dense progress reward is speed-indifferent once completed -> the agent
# settles at a moderate, reliable speed (the slow-coast). -K_TIME/step makes finishing FASTER worth
# more, so the fast racing line + speed EMERGE from the lap-time objective (the racing-RL consensus:
# progress is the lap-time proxy, the line emerges; Fuchs 2021, GT Sophy/Wurman 2022). NOT a raw
# speed term (which imposes "go fast" + causes off-track rushing). Default 0 = inert (V0 unchanged).
K_TIME = 0.0

# == Reward — modo carrera con tráfico (D-0051; inertes en hotlap, default 0) ==
# Términos de oponentes que SOLO muerden con un rival cerca (gateados por los haces `opponents`).
# En pista libre (todos los haces ~200 m -> sin objetivo) el reward es IDÉNTICO al del forward, así
# que el hotlap no se degrada. Diseño: docs/design/torcs-race-finetune.md §8.
#
# (1) Proximidad sigmoide frontal (F1TENTH, Steiner 2025, arXiv:2510.26040): evasión temprana del
#     coche de delante. -K_PROX*sigmoid(STEEPNESS*(1 - d_front/TRIGGER_M)); satura
#     a ~K_PROX cuando d->0 y ~0 cuando d >> trigger. d_front_min = haz frontal más cercano (m).
K_PROX = 0.0               # peso de la penalización de proximidad frontal (0 = inerte)
PROX_TRIGGER_M = 20.0      # distancia (m) a la que la penalización está a ~la mitad
PROX_STEEPNESS = 6.0       # pendiente de la sigmoide (mayor = transición más brusca)
# Índices de los 36 haces `opponents` que miran HACIA DELANTE / HACIA ATRÁS. Layout del manual SCR
# (36 sectores de 10°, horario de -180° a +180°; el frente 0° cae ~índices 17-18). El frontal son
# ~±50° y el trasero ~±50° respecto a atrás. CONVENCIÓN A VALIDAR EN VIVO (design §15) — si el build
# real ordena distinto, solo cambian estas tuplas (los términos no).
OPPONENT_FRONT_SECTORS: tuple[int, ...] = (13, 14, 15, 16, 17, 18, 19, 20, 21, 22)
OPPONENT_REAR_SECTORS: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 31, 32, 33, 34, 35)
#
# (2) Contacto escalado por velocidad de impacto (GT Sophy / GT Sport, ∝Δv²): embestir rápido duele,
#     el roce lento casi no. -K_CONTACT * Δdamage⁺ * (speed_x/V_REF_KMH)². NO se sube K_DAMAGE plano
#     (la literatura: subir la penalización de colisión -> agente tímido; Wurman 2022).
K_CONTACT = 0.0            # peso del contacto escalado por Δv² (0 = inerte; K_DAMAGE sigue aparte)
#
# (3) Bonus de pase por haces (denso por EVENTO, agnóstico a N; cubre el DOBLAJE). +K_PASS cuando un
#     rival CERCANO migra de un sector frontal a uno trasero entre frames (pase completado); -K_PASS
#     al revés (te pasan = falta de defensa). Evento discreto -> robusto al ruido y SIN premiar
#     "acercarse" (evita el tailgating-hack que F1TENTH quitó). Solo cuenta rivales a < PASS_NEAR_M.
K_PASS = 0.0              # peso del evento de pase/ser-pasado (0 = inerte)
PASS_NEAR_M = 30.0        # un haz por debajo de esto = "rival presente" en ese sector

# == Episode termination (T-005, D-0015; patiences tuned in EXP-0003) ==
# SCR cadence ≈ 20 ms/tic (~50 Hz), so step patiences are real-time budgets: a brief,
# penalized excursion is survivable (the agent learns recovery); only a sustained one
# ends the episode. Two off-course tiers (EXP-0003): the RECOVERABLE band (1 < |tp| < 2,
# a wheel or two off) gets the full patience so the buffer sees useful edge/recovery
# states (V0 has no diverse-start mechanism — that is a T-007 lever); a HARD off-course
# (|tp| > 2, a full track-width into the run-off/scenery) terminates at once, so the car
# can't spend ~2 s ploughing the scenery generating low-value transitions (EXP-0003 saw
# trackPos reach -7 under the reckless policy). Literature: terminate on course-out
# (CarRacing/DeepRacer/TORCS lane-keeping); get extreme-state exposure from diverse start
# states, not long off-track tails (Reference State Init / Contrastive Initial State Buffer).
OFFCOURSE_LIMIT = 1.0         # |track_pos| beyond which the car is off the track surface
OFFCOURSE_HARD_LIMIT = 2.0    # |track_pos| beyond which it terminates at once (deep off)
OFFCOURSE_PATIENCE = 100      # steps in the recoverable band before terminating (~2.0 s)
STUCK_SPEED_KMH = 5.0         # speed below which the car counts as not moving
STUCK_PATIENCE = 100          # steps stuck before terminating (~2.0 s)
DAMAGE_LIMIT = 8000.0         # damage beyond which the car is wrecked (TORCS removes ~10000)

# Truncation safety net so a hung episode can't run forever. Also the time budget that
# makes "maximize Σ progress" ≈ "maximize average speed" ≈ fastest lap (design §3).
# Generous enough to fit a full slow Corkscrew lap (the T-003 smoke lap was ~12k frames);
# T-007 will pick a training-appropriate value.
MAX_EPISODE_STEPS = 20_000

# Max control-sentinel acks to drain on reset while waiting for the first sensor
# frame (past the ***identified***/***restart*** acks after a meta=1 restart).
RESET_DRAIN_LIMIT = 100
# Wall-clock patience (s) for the first frame after reset. The GUI race can take
# several seconds to load + count down before the first packet; tolerate recv
# timeouts up to this budget (mirrors the T-003 smoke's startup grace).
RESET_STARTUP_GRACE_S = 30.0


# == Configurable reward parameters (D-0020 / P4) ==
# A frozen bundle of the reward coefficients/thresholds, so the reward is configurable
# per run (TrainingConfig -> RewardParams -> TorcsEnv -> compute_reward_terms) without
# touching globals. Defaults reproduce the module constants above, so DEFAULT_REWARD_PARAMS
# is the single default source and the standalone constants remain the documentation of
# record. Only the slip/kinetic knobs are exposed on the CLI today (the rest are stable).
@dataclass(frozen=True)
class RewardParams:
    """Reward coefficients/thresholds for one run (defaults = the module constants)."""

    k_progress: float = K_PROGRESS
    k_damage: float = K_DAMAGE
    k_offcourse: float = K_OFFCOURSE
    k_smooth: float = K_SMOOTH
    smooth_min_speed_kmh: float = SMOOTH_MIN_SPEED_KMH
    # Anti-oscillation (D-0048 Option 2): sign-reversal penalty; alternative to the smooth term.
    k_osc: float = K_OSC
    osc_threshold: float = OSC_THRESHOLD
    k_racepos: float = K_RACEPOS
    p_fail: float = P_FAIL
    offcourse_limit: float = OFFCOURSE_LIMIT
    # Reward v2 (D-0020):
    slip_threshold_ms: float = SLIP_THRESHOLD_MS
    k_slip: float = K_SLIP
    k_kinetic: float = K_KINETIC
    v_ref_kmh: float = V_REF_KMH
    wheel_radius_m: float = WHEEL_RADIUS_M
    # Integrity (EXP-0010): reject the RSI-handover resync teleport (see PROGRESS_CLAMP_M).
    progress_clamp_m: float = PROGRESS_CLAMP_M
    # Reward v3 (EXP-0013): dense anti-spin cliff — SUPERSEDED by k_lat (D-0029); default 0 (inert).
    k_spin: float = K_SPIN
    angle_spin_threshold_rad: float = ANGLE_SPIN_THRESHOLD_RAD
    # Reward (D-0046, fixes D-0029): dense side-slip penalty on the TRUE slip (velocity vs heading,
    # from speed_y) with a free band -> -K_LAT*min(|Vx|/V_REF,1)*max(0,|sin(slip)|-slip_free).
    k_lat: float = K_LAT
    slip_free: float = SLIP_FREE
    # Reward v7 (D-0030) -> D-0044 fence: speed-scaled off-course penalty -K_EDGE*min(|Vx|/V_REF,1)*
    # min(max(0,|tp|-edge_inner)/edge_ramp_w, 1) -- free below edge_inner, short ramp, then PLATEAUS
    # (time-based, not depth-proportional). Forward hotlap sets edge_inner=1.28 (the cut limit).
    k_edge: float = K_EDGE
    edge_inner: float = EDGE_INNER
    edge_ramp_w: float = EDGE_RAMP_W
    # Reward (D-0036): minimum-time per-step penalty. Default 0 (inert); set > 0 to make speed +
    # the fast racing line emerge from the lap-time objective (not an imposed speed term).
    k_time: float = K_TIME
    # Reward modo carrera (D-0051): inertes en hotlap (default 0). Proximidad frontal, contacto,
    # bonus de pase por haces. Ver constantes arriba y docs/design/torcs-race-finetune.md §8.
    k_prox: float = K_PROX
    prox_trigger_m: float = PROX_TRIGGER_M
    prox_steepness: float = PROX_STEEPNESS
    k_contact: float = K_CONTACT
    k_pass: float = K_PASS
    pass_near_m: float = PASS_NEAR_M


DEFAULT_REWARD_PARAMS = RewardParams()

