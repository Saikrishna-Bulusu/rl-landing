# Wind study: learning to land in gusty wind

Sai, 26 Sept 2026: magpies aside; RL on gradually increasing wind; the learner in charge of the flying and the
deciding; a training curriculum; MuJoCo MJX; all of it for ACRA (4 Oct).

## 1. A GPU drone that flies like PX4 in Gazebo (`x500_mjx.py`)

The rig v2 x500 as PX4's Gazebo model defines it, one rigid body in MJX with every force applied by us:

| part | source | values |
|---|---|---|
| airframe | x500_base + gimbal + rig v2 camera link | 2.584 kg, inertia 0.0291/0.0291/0.0444 kg m^2 about the CoM |
| motors | Gazebo MulticopterMotorModel | T = 8.55e-6 w^2, w = 150 + 850 u rad/s, lag 12.5 ms up / 25 ms down |
| propeller drag | same | 8.06e-5 \|w\| v_perp per rotor, against the air |
| body drag | added (Gazebo had none) | 0.5 rho CdA \|v\| v, CdA 0.04 m^2 horizontal, 0.05 vertical |
| PX4 | PositionControl, AttitudeControl, RateControl, control allocation | default x500 gains; hover throttle 0.836 |

PX4's code paths emulated in JAX: velocity PID with acceleration feed-forward, thrust vector with the 45 deg tilt
limit, vertical priority with a 0.3 horizontal margin, both anti-windups; bodyzToAttitude; reduced-attitude control
with yaw weight 0.4; rate PID with PX4's integral factor and saturation freezing; the pseudo-inverse of the CA_ROTOR
effectiveness with PX4's normalisation and sequential desaturation (airmode off).

**Checked against PX4 in Gazebo** (`px4_steps.py`: the same velocity commands, Gazebo's true CoM in sim time;
`python x500_mjx.py --check`):

| command step | Gazebo 10/63/90% | MJX 10/63/90% | tilt Gazebo / MJX |
|---|---|---|---|
| 0 -> 3 m/s north | 0.22/0.68/1.24 s | 0.20/0.64/1.24 s | 26.2 / 26.8 deg |
| 0 -> 8 m/s north | 0.29/0.85/1.51 s | 0.28/0.80/1.52 s | 44.1 / 45.6 deg |
| 0 -> 1.5 m/s down | 0.06/0.20/0.32 s | 0.04/0.20/0.36 s | - |
| 0 -> 1.5 m/s up | 0.10/0.34/0.50 s | 0.08/0.32/0.48 s | - |

(Gazebo's model origin is at the drone's feet, which swing back when it pitches to accelerate: compare the CoM.)
4,096 drones step at ~21 M physics steps/s on the RTX 5070 Laptop; a PPO round of 4,096 drones x 64 decisions ~0.3 s.

## 2. The same wind in both simulators (`gz_winddrag/`, `land_env.wind_step`)

Mean wind W10 (at 10 m) with a log profile (z0 0.3 m, suburb); MIL-F-8785C low-altitude turbulence (first-order
filters, sigma_w = 0.1 W10, length scales from height); 1-cosine gusts (one per 15 s on average, 0.2-0.6 W10, 1.5-5 s,
within +-0.5 rad of the mean). The Gazebo system plugin sets the world's wind (which Gazebo's propeller drag uses) and
applies the body drag; it holds still air until switched on, then starts from its seed, so every decider meets the
same gusts. Hovering at 30 m in steady wind, tilt Gazebo / MJX: 5.76 / 5.73 deg (9.2 m/s, propeller drag only),
10.36 / 10.43 (9.2 m/s), 22.48 / 22.6 (15.8 m/s).

## 3. The task (`land_env.py`)

The last 30 m of a delivery: the drone hovers 30 m up, 15-25 m from a back-yard pad, a house (10 x 8 m, 4-7 m tall)
2.5-8 m from the pad, a tree beside it in 60% of yards (Logan LiDAR: 58%). Landed = touchdown within 0.75 m of the
centre, sinking < 1 m/s, sliding < 0.5 m/s; crashed = within 0.35 m of house or tree; otherwise hard/off the pad, or
timeout at 90 s.

**Today's rule** (gz_learner's landing): the approach law (0.69 s lead, sqrt and linear braking), descent 1.5 then
0.8 m/s over the pad, and below 6 m only when centred and still (the gate). 1,024 paired flights per wind:

| W10 (m/s) | 0 | 6 | 9 | 12 | 15 | 18 | 20 |
|---|---|---|---|---|---|---|---|
| landed | 100% | 100% | 100% | 100% | 78% | 5.5% | 0% |
| gave up (90 s) | 0 | 0 | 0 | 0 | 22% | 94% | 99% |
| mean time | 30 s | 32 s | 38 s | 51 s | 75 s | 89 s | 90 s |

The rule does not crash in strong wind: it waits for a stillness gusts never give.

## 4. The learner (`ppo.py`)

PPO in plain JAX (no RL library). It acts through the channel PX4 offers over MAVLink: the rule's velocity setpoint
plus an acceleration feed-forward (|a| <= 3 m/s^2 per axis, the flying), and a vertical choice every 0.04 s -- the
rule's own, down 1.5, down 0.6, hold, up 1 m/s (the deciding). It starts as the rule (its 'rule' option favoured,
feed-forward ~0) and learns where to leave it. Observation (49): pad offset (coarse and fine), height, velocity, lean
(where the wind shows), the rule's setpoint and gate state, house and tree, last action, last 4 (velocity, lean).
Reward: landed +10..15 (by precision), hard/off-pad -10, crash -15, gave up -5, -0.05/s; gamma 0.9998.
**Curriculum**: flights draw W10 from [0, W] and its top 3 m/s; W starts at 4 m/s and rises 1 m/s once 95% of the top
band lands (80% advanced too early: strong-wind results fell as it moved on).

## 5. Gazebo check (`gz_windland.py`, `gz_ladder.sh`)

MJX evaluation yard i is built in Gazebo (pad, house, tree), the drone flies out in still air, holds at the start,
the wind switches on, the rule or the learner (numpy, checked equal to the JAX policy) lands it. Flight-by-flight
agreement: `python figs.py --pairs <gz_dir> runs/rule_ladder.json`.
