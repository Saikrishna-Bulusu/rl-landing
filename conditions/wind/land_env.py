#!/usr/bin/env python3
"""
land_env.py -- the last 30 m of a delivery into a Logan back yard, in gusty wind, on the GPU: x500_mjx's drone and
PX4, the wind of gz_winddrag (the same model Gazebo now flies in), a yard with a house and often a tree, and today's
landing rule to measure learners against.

  yard     pad at the origin; a house (10 x 8 m, 4-7 m tall) whose nearest wall is 2.5-8 m from the pad centre; a tree
           (canopy radius 1.5-3 m, 5-12 m tall, its edge 1.5-4.5 m from the pad) in 60% of yards (Logan LiDAR: 58%)
  wind     W10 (m/s at 10 m) from the curriculum, any direction; log profile (z0 0.3 m), MIL-F-8785C turbulence,
           1-cosine gusts (1 per 15 s, 0.2-0.6 W10, 1.5-5 s)
  start    30 m up, 15-25 m from the pad, hovering, nose along a random heading (held: PX4 keeps it)
  rule     gz_learner's landing: Mission._approach to the pad estimate (lead 0.69 s, sqrt(2 A d) and K_LIN d laws,
           5 m/s far, 2.5 within 3 m); over the pad (within 2 m) descend as urban_agents.Rule does -- 2.5 m/s (PX4
           caps it at 1.5) above 8 m, 0.8 below -- and below 6 m only when centred (0.3 m + 0.1 m per m above 3 m) and
           still (< 0.3 m/s; < 0.1 below 2 m), gz_learner's gate
  sensing  the pad estimate: marker noise 0.03 + 0.004 h m and a 0.05 m bias per flight; velocity noise 0.05 m/s
  outcome  landed: touchdown within 0.75 m of the centre, sinking < 1 m/s, sliding < 0.5 m/s; crashed: within 0.35 m
           of the house or tree, or down off the pad; timeout 90 s
A learner acts through the same channel Gazebo offers PX4 (MAVLink velocity + acceleration setpoints):
  flying     acceleration feed-forward added to the rule's velocity setpoint, |a| <= 3 m/s^2 per axis
  deciding   the vertical command, all the way in: the rule's own / descend 1.5 / descend 0.6 / hold / climb 1.0 m/s
             (coming down early leaves the stronger wind aloft, near the house). The learner starts as the rule (its
             'rule' option favoured, no feed-forward) and learns where to leave it: copy the teacher first.
"""
import math

import jax
import jax.numpy as jnp
import numpy as np

import x500_mjx as X

DT_C = X.DT * X.N_SUB                     # 0.04 s per decision
T_MAX = 90.0
Z0 = 0.3
GUST_RATE = 1.0 / 15.0
PAD_R, R_DRONE, Z_TOUCH = 0.75, 0.35, 0.25
V_SINK_OK, V_SLIDE_OK = 1.0, 0.5
A_FF = 3.0
MODES = jnp.array([0.0, -1.5, -0.6, 0.0, 1.0])  # the deciding head: 0 = the rule's own vertical, then m/s (up +)
TAU_LAT, A_MAX, K_LIN = 0.69, 1.5, 0.8     # px4_mission
N_HIST = 4
CRASH_R, HARD_R = -15.0, -10.0   # a crash's price (ppo.py --crash-r): a lost drone costs far more than a failed delivery
OFFPAD_R = None                  # price of a GENTLE touchdown off the pad (ppo.py --offpad-r; Sai, 7 Oct: re-price the outcomes);
                                 # None = HARD_R, one price for every hard touchdown, as before. Too fast always costs HARD_R
SHAPE_G = 0.997                  # the shaping's discount (ppo.py --shape-g). Below the learner's (0.9998) it leaks
                                 # +0.0003 (d + 0.3 h) per step, which beats the 0.002 time cost away from the pad: hovering
                                 # at the start pays ~+15 over 90 s, as much as a landing (30 Sept, flip-priced runs hovered)
FLIP_R = 0.0                     # price of each climb<->descend reversal of the sent vertical command (ppo.py --flip-r). 30 Sept:
                                 # the hard -100 decider pulsed climb/descend at ~1.5 Hz, which held PX4 at its thrust limit;
                                 # in Gazebo that sank the drone (1.05 m/s against a +0.25 average command) instead of slowing it
VZ_FAST = -2.5                   # the rule's descent above 8 m; PX4 caps descents at 1.5 m/s (x500_mjx.VZ_DN), so -1.5 flies the same
GATE_SCALE = 1.0                 # tuned-rule baseline (6 Oct): the gate's centring and stillness limits times this
PATIENCE = 1e9                   # tuned-rule baseline: seconds the gate may hold before the rule creeps down at 0.6 m/s over the pad
RAND = frozenset()               # PLAN.md 2c, read at trace time: "sense" (sensing shaped like Gazebo's, randomized per flight),
                                 # "drone" (mass, thrust, motor lag, body drag, command latency), "gust" (the mean wind turns).
                                 # Empty: the environment of Phases 0-2a, bit for bit


# ---- wind: gz_winddrag/WindDrag.cc, updated once per decision (statistically the same process) ---------------------------
def wind_new(key, w10):
    th = jax.random.uniform(key, (), minval=0.0, maxval=2 * math.pi)
    return dict(w10=w10, th=th, uvw=jnp.zeros(3), gt=jnp.array(-1.0), gT=jnp.array(1.0), gA=jnp.array(0.0), gth=th,
                dt=jnp.array(-1.0), dT=jnp.array(1.0), dA=jnp.array(0.0), th0=th)


def wind_step(ws, h, key):
    k1, k2, k3 = jax.random.split(key, 3)
    w10, th = ws["w10"], ws["th"]
    hm = jnp.maximum(h, 1.0)
    wm = w10 * jnp.log(hm / Z0) / math.log(10.0 / Z0)
    hf = jnp.maximum(h, 3.0) * 3.281
    den = 0.177 + 0.000823 * hf
    sw = 0.1 * w10
    su = sw / den ** 0.4
    lu, lw, V = hf / den ** 1.2 / 3.281, jnp.maximum(h, 3.0), jnp.maximum(wm, 1.0)
    a = jnp.exp(-V * DT_C / jnp.array([lu, lu, lw]))
    sig = jnp.array([su, su, sw])
    uvw = a * ws["uvw"] + sig * jnp.sqrt(1 - a ** 2) * jax.random.normal(k1, (3,))
    r = jax.random.uniform(k2, (4,))
    start = (ws["gt"] < 0) & (w10 > 0) & (r[0] < GUST_RATE * DT_C)
    gt = jnp.where(start, 0.0, ws["gt"])
    gT = jnp.where(start, 1.5 + 3.5 * r[1], ws["gT"])
    gA = jnp.where(start, w10 * (0.2 + 0.4 * r[2]), ws["gA"])
    gth = jnp.where(start, th + r[3] - 0.5, ws["gth"])
    amp = jnp.where(gt >= 0, gA * 0.5 * (1 - jnp.cos(2 * math.pi * gt / gT)), 0.0)
    gt = jnp.where(gt >= 0, gt + DT_C, gt)
    gt = jnp.where(gt > gT, -1.0, gt)
    dt, dT, dA, th0 = ws["dt"], ws["dT"], ws["dA"], ws["th0"]
    if "gust" in RAND:            # the mean wind turns by up to 60 deg over 3-10 s, about once per 30 s, and stays turned
        q = jax.random.uniform(k3, (3,))
        turn = (dt < 0) & (w10 > 0) & (q[0] < DT_C / 30.0)
        dt, dT, th0 = jnp.where(turn, 0.0, dt), jnp.where(turn, 3.0 + 7.0 * q[1], dT), jnp.where(turn, th, th0)
        dA = jnp.where(turn, (q[2] - 0.5) * 2 * math.pi / 3, dA)
        th = jnp.where(dt >= 0, th0 + dA * 0.5 * (1 - jnp.cos(math.pi * jnp.minimum(dt / dT, 1.0))), th)
        dt = jnp.where(dt >= 0, dt + DT_C, dt)
        dt = jnp.where(dt > dT, -1.0, dt)
    ea, ec = jnp.array([jnp.cos(th), jnp.sin(th), 0.0]), jnp.array([-jnp.sin(th), jnp.cos(th), 0.0])
    wind = ea * (wm + uvw[0]) + ec * uvw[1] + jnp.array([0.0, 0.0, uvw[2]]) + amp * jnp.array([jnp.cos(gth), jnp.sin(gth), 0.0])
    return dict(w10=w10, th=th, uvw=uvw, gt=gt, gT=gT, gA=gA, gth=gth, dt=dt, dT=dT, dA=dA, th0=th0), wind


# ---- the yard ---------------------------------------------------------------------------------------------------------------------
def yard_new(key):
    k = jax.random.split(key, 8)
    ang = jax.random.uniform(k[0], (), minval=0, maxval=2 * math.pi)
    d_wall = jax.random.uniform(k[1], (), minval=2.5, maxval=8.0)
    half = jnp.array([5.0, 4.0])
    # house centre: nearest wall d_wall from the pad along ang (house axis-aligned in its own frame, rotated by ang)
    centre = jnp.array([jnp.cos(ang), jnp.sin(ang)]) * (d_wall + half[1])
    h_house = jax.random.uniform(k[2], (), minval=4.0, maxval=7.0)
    t_ang = ang + jax.random.uniform(k[3], (), minval=0.8, maxval=2 * math.pi - 0.8)     # not behind the house
    t_r = jax.random.uniform(k[5], (), minval=1.5, maxval=3.0)
    t_d = t_r + jax.random.uniform(k[4], (), minval=1.5, maxval=4.5)   # canopy edge 1.5-4.5 m from the pad centre:
                                                                       # pads are re-sited to the clearest spot
    t_top = jax.random.uniform(k[6], (), minval=5.0, maxval=12.0)
    has_tree = jax.random.uniform(k[7], ()) < 0.6
    return dict(hc=centre, hang=ang, hhalf=half, hh=h_house, tc=jnp.array([jnp.cos(t_ang), jnp.sin(t_ang)]) * t_d,
                tr=t_r, ttop=t_top, tree=has_tree)


def clearance(y, p):
    """Distance from the drone centre to the house box and the tree (canopy sphere + trunk), metres."""
    c, s = jnp.cos(-y["hang"]), jnp.sin(-y["hang"])
    rel = p[:2] - y["hc"]
    loc = jnp.array([c * rel[0] - s * rel[1], s * rel[0] + c * rel[1]])        # house frame: x from the pad to its centre
    q = jnp.abs(jnp.array([loc[1], loc[0]])) - y["hhalf"]                     # 10 m along the wall, 8 m deep
    dz = p[2] - y["hh"]
    out = jnp.linalg.norm(jnp.maximum(jnp.array([q[0], q[1], dz]), 0.0))
    d_house = out + jnp.minimum(jnp.maximum(jnp.maximum(q[0], q[1]), dz), 0.0)
    cz = y["ttop"] - y["tr"]
    d_can = jnp.linalg.norm(jnp.array([p[0] - y["tc"][0], p[1] - y["tc"][1], p[2] - cz])) - y["tr"]
    d_trunk = jnp.where(p[2] < cz, jnp.linalg.norm(p[:2] - y["tc"]) - 0.2, 1e3)
    d_tree = jnp.where(y["tree"], jnp.minimum(d_can, d_trunk), 1e3)
    return d_house, d_tree


# ---- the rule -------------------------------------------------------------------------------------------------------------------
def approach(d, v, cap):
    """Mission._approach: aim a lag-time ahead, arrive stopped (sqrt(2 A d)), proportional near (K_LIN d)."""
    d_eff = d - v * TAU_LAT
    n = jnp.linalg.norm(d_eff)
    return jnp.where(n > 1e-3, d_eff / jnp.maximum(n, 1e-6) * jnp.minimum(jnp.minimum(cap, jnp.sqrt(2 * A_MAX * n)), K_LIN * n), 0.0)


def gate(d_h, h, v_est):
    """The rule's gate as the learner sees it: would it hold, and how far and how fast against its limits."""
    tol = 0.3 + 0.1 * jnp.maximum(h - 3.0, 0.0)
    v_ok = jnp.where(h < 2.0, 0.1, 0.3)
    r_d, r_v = jnp.linalg.norm(d_h) / tol, jnp.linalg.norm(v_est[:2]) / v_ok
    hold = (h < 6.0) & ((r_d > 1.0) | (r_v > 1.0))
    return jnp.array([hold * 1.0, jnp.minimum(r_d, 3.0) / 3.0, jnp.minimum(r_v, 3.0) / 3.0])


def rule_hold(d_h, h, v_est):
    """The rule's landing gate: below 6 m, hold unless centred and still."""
    tol = (0.3 + 0.1 * jnp.maximum(h - 3.0, 0.0)) * GATE_SCALE
    v_ok = jnp.where(h < 2.0, 0.1, 0.3) * GATE_SCALE
    return (h < 6.0) & ((jnp.linalg.norm(d_h) > tol) | (jnp.linalg.norm(v_est[:2]) > v_ok))


def rule_cmd(d_h, h, v_est, held=0.0):
    """The rule's velocity setpoint (ENU) and whether it is in the landing phase. held: seconds the gate has held."""
    near = jnp.linalg.norm(d_h) < 3.0
    over = jnp.linalg.norm(d_h) < 2.0
    v_h = approach(d_h, v_est[:2], jnp.where(near, 2.5, 5.0))
    go = (held > PATIENCE) & (jnp.linalg.norm(d_h) < PAD_R)                     # waited long enough: creep down over the pad
    vz = jnp.where(over, jnp.where(rule_hold(d_h, h, v_est), jnp.where(go, -0.6, 0.0), jnp.where(h > 8.0, VZ_FAST, -0.8)), 0.0)
    return jnp.concatenate([v_h, vz[None]]), near


# ---- the episode --------------------------------------------------------------------------------------------------------------------
def drone_new(key):
    """One drone of the randomized fleet (x500_mjx.PRM0's layout): body drag x0.5-2 (the biggest unknown,
    papers/research/x500_wind_limit.md), mass and thrust coefficient +-10%, motor lag +-30%, command latency 0-80 ms."""
    u = jax.random.uniform(key, (5,))
    r_d = jnp.exp(math.log(0.5) + u[0] * math.log(4.0))
    return jnp.array([X.CDA_H * r_d, X.CDA_V * r_d, 0.9 + 0.2 * u[1], 0.9 + 0.2 * u[2], 0.7 + 0.6 * u[3], jnp.floor(21 * u[4])])


def sensors_new(key):
    """One flight's sensing, shaped like Gazebo's (gz_windland --log-obs against truth, 6 Oct). W10 9: pad estimate bias
    0.02-0.05 m, a slowly wandering error of sd 0.01-0.05 m lasting 0.2-1 s, velocity sd 0.06-0.12 m/s; the W10 15-18
    gate flights (the drone leaning 30-40 deg, the gimbal lagging): bias up to 0.37 m, sd up to 0.13 m lasting 1-10 s,
    velocity sd up to 0.29 m/s. Ranges here cover both, plus dropouts that let the estimate drift, rare one-frame
    outliers, a height bias and slow error, and 0-120 ms of delay (Gazebo showed none at 40 ms resolution)."""
    k = jax.random.split(key, 3)
    u = jax.random.uniform(k[0], (10,))
    return dict(b=jax.random.normal(k[1], (2,)) * (0.02 + 0.18 * u[0]), kp=jnp.exp(u[1] * math.log(6.0)),
                tp=jnp.exp(math.log(0.3) + u[2] * math.log(5.0 / 0.3)),
                sv=jnp.array([1.0, 1.0, 0.6]) * (0.05 + 0.25 * u[3]), tv=0.04 + 0.16 * u[4],
                hb=0.05 * jax.random.normal(k[2], ()), sh=0.01 + 0.04 * u[5], th=0.5 + 2.5 * u[6],
                lat=jnp.floor(4 * u[7]).astype(jnp.int32), rw=0.05 + 0.15 * u[8])


def reset(key, w10):
    k = jax.random.split(key, 6)
    b = jax.random.uniform(k[0], (), minval=0, maxval=2 * math.pi)
    r0 = jax.random.uniform(k[1], (), minval=15.0, maxval=25.0)
    yaw = jax.random.uniform(k[2], (), minval=-math.pi, maxval=math.pi)
    p0 = jnp.array([r0 * jnp.cos(b), r0 * jnp.sin(b), 30.0])
    prm = drone_new(jax.random.fold_in(k[5], 1)) if "drone" in RAND else X.PRM0      # new keys: the old draws stay as they were
    s = X.init_state(p0, yaw=yaw, prm=prm)
    return dict(sim=s, yaw=yaw, wind=wind_new(k[3], w10), yard=yard_new(k[4]),
                bias=0.05 * jax.random.normal(k[5], (2,)), t=jnp.array(0.0), key=k[5],
                prev_act=jnp.zeros(7), hist=jnp.zeros((N_HIST, 5)), done=jnp.array(False), ff_slow=jnp.zeros(2),
                outcome=jnp.array(0), miss=jnp.array(0.0), vz_sent=jnp.array(0.0), held=jnp.array(0.0),
                sen=sensors_new(jax.random.fold_in(k[5], 2)), truth=jnp.tile(jnp.concatenate([p0, jnp.zeros(3)]), (4, 1)),
                ep=jnp.zeros(2), ev=jnp.zeros(3), eh=jnp.array(0.0), drop=jnp.array(0.0), jump=jnp.zeros(2))


def estimates(st, key):
    if "sense" in RAND:            # the flight's own sensing (sensors_new), advanced once per decision in step(): the rule
        e, tr = st["sen"], st["truth"][st["sen"]["lat"]]    # and the learner see the same estimate, as in Gazebo
        return (tr[:3] + jnp.concatenate([e["b"] + st["ep"] + st["jump"], (e["hb"] + st["eh"])[None]]), tr[3:] + st["ev"])
    p, v = st["sim"]["d"].qpos[:3], st["sim"]["d"].qvel[:3]
    k1, k2 = jax.random.split(key)
    sd = 0.03 + 0.004 * p[2]
    p_est = p.at[:2].add(st["bias"] + sd * jax.random.normal(k1, (2,)))
    v_est = v + 0.05 * jax.random.normal(k2, (3,))
    return p_est, v_est


def observe(st, p_est, v_est, cmd):
    """What the learner sees: the pad estimate, its velocity, its lean (the wind shows there), the rule's setpoint, the
    house and tree relative to it (survey + voxel map), its last action, and the last N_HIST (velocity, lean)."""
    q = st["sim"]["d"].qpos[3:7]
    lean = X.qrot(q)[:2, 2]
    y = st["yard"]
    p = p_est
    rel_h = y["hc"] - p[:2]
    rel_t = jnp.where(y["tree"], y["tc"] - p[:2], jnp.array([30.0, 30.0]))
    dh, dt_ = clearance(y, p)
    # near the pad a 0.5 m offset is 0.05 of -p/10: the learner also gets it on a fine scale, and the rule's gate
    return jnp.concatenate([-p[:2] / 10, p[2:] / 10, v_est / 3, lean * 5, cmd / 3, rel_h / 10, rel_t / 10,
                            jnp.array([jnp.minimum(dh, 10) / 5, jnp.minimum(dt_, 10) / 5, y["tree"] * 1.0]),
                            st["prev_act"], st["hist"].ravel(), jnp.tanh(-p[:2] / 0.5), gate(-p[:2], p[2], v_est)])


HIST0 = 2 + 1 + 3 + 2 + 3 + 2 + 2 + 3 + 7            # the observation's index where the last N_HIST (v_est/3, lean*5) start
OBS_DIM = HIST0 + N_HIST * 5 + 2 + 3


def shield_block(d_h, h, v_est, k):
    """The safety layer (k > 0): below 6 m no descent unless within k times the rule's centring and stillness limits."""
    tol = k * (0.3 + 0.1 * jnp.maximum(h - 3.0, 0.0))
    v_ok = k * jnp.where(h < 2.0, 0.1, 0.3)
    return (k > 0) & (h < 6.0) & ((jnp.linalg.norm(d_h) > tol) | (jnp.linalg.norm(v_est[:2]) > v_ok))


def step(st, act, key, fly=True, decide=True, shield=0.0, hp_tau=0.0):
    """One decision. act: (a_x, a_y in [-1, 1], mode logits... as an index in act[2]). Returns the new state, the reward,
    the observation. fly/decide off -> the rule's own choice for that part."""
    X.PER_DRONE = "drone" in RAND                   # trace time: the physics reads each flight's drone (reset sampled it)
    k1, k2, k3 = jax.random.split(key, 3)
    p_est, v_est = estimates(st, k1)
    d_h = -p_est[:2]
    cmd, near = rule_cmd(d_h, p_est[2], v_est, st["held"])
    held = jnp.where(rule_hold(d_h, p_est[2], v_est), st["held"] + DT_C, 0.0)
    # hp_tau > 0: the push's slow part is taken out (a first-order high-pass): PX4's velocity integral cancels any
    # steady push anyway (26 Sept: a fixed push of 0-3 m/s^2 changed nothing), so the learner's range goes to reacting
    raw = jnp.clip(act[:2], -2, 2)
    a_ff = jnp.where(fly, jnp.clip(raw - st["ff_slow"] * (hp_tau > 0), -1, 1) * A_FF, 0.0)
    ff_slow = st["ff_slow"] + (DT_C / jnp.maximum(hp_tau, DT_C)) * (raw - st["ff_slow"])
    mode = act[2].astype(jnp.int32)
    vz = jnp.where(decide & (mode > 0), MODES[mode], cmd[2])
    vz = jnp.where(shield_block(d_h, p_est[2], v_est, shield), jnp.maximum(vz, 0.0), vz)
    vel_sp = cmd.at[2].set(vz)
    flip = ((st["vz_sent"] > 0.5) & (vz < -0.3)) | ((st["vz_sent"] < -0.3) & (vz > 0.5))    # the rule alone never climbs
    ws, wind = wind_step(st["wind"], st["sim"]["d"].qpos[2], k2)
    sim = X.control_step(st["sim"], vel_sp, jnp.array([a_ff[0], a_ff[1], 0.0]), st["yaw"], wind)
    p, v = sim["d"].qpos[:3], sim["d"].qvel[:3]
    t = st["t"] + DT_C
    dh, dtr = clearance(st["yard"], p)
    tilt = jnp.arccos(jnp.clip(X.qrot(sim["d"].qpos[3:7])[2, 2], -1, 1))
    miss = jnp.linalg.norm(p[:2])
    down = p[2] <= Z_TOUCH
    hit = (jnp.minimum(dh, dtr) < R_DRONE) | (tilt > 1.2)
    fast = (-v[2] >= V_SINK_OK) | (jnp.linalg.norm(v[:2]) >= V_SLIDE_OK)          # touchdown too fast: the damaging kind
    ok = down & (miss <= PAD_R) & ~fast
    timeout = t >= T_MAX
    outcome = jnp.where(hit, 2, jnp.where(ok, 1, jnp.where(down, 3, jnp.where(timeout, 4, 0))))   # 1 landed 2 crash 3 hard/off 4 timeout
    done = outcome > 0
    # reward: the goal priced first; shaping small and potential-based
    phi = lambda pp: -(jnp.linalg.norm(pp[:2]) + 0.3 * pp[2]) / 10.0
    r = SHAPE_G * phi(p) - phi(st["sim"]["d"].qpos[:3]) - 0.002 - 0.01 * jnp.sum(a_ff ** 2) / A_FF ** 2
    # a hard or off-pad touchdown is a failed delivery that may break something: as bad as missing the chance; a crash
    # worse; giving up (timeout) returns the parcel safely
    r = r + jnp.where(outcome == 1, 10.0 + 5.0 * (1 - miss / PAD_R), 0.0) + jnp.where(outcome == 2, CRASH_R, 0.0) \
        + jnp.where(outcome == 3, HARD_R if OFFPAD_R is None else jnp.where(fast, HARD_R, OFFPAD_R), 0.0) \
        + jnp.where(outcome == 4, -5.0, 0.0) - FLIP_R * flip
    lean = X.qrot(sim["d"].qpos[3:7])[:2, 2]
    hist = jnp.concatenate([st["hist"][1:], jnp.concatenate([v_est / 3, lean * 5])[None]], axis=0)
    prev = jnp.concatenate([a_ff / A_FF, jax.nn.one_hot(mode, 5)])
    st = dict(st, sim=sim, wind=ws, t=t, prev_act=prev, hist=hist, done=done, outcome=outcome, miss=miss, key=k3, ff_slow=ff_slow,
              vz_sent=vz, held=held, truth=jnp.concatenate([jnp.concatenate([p, v])[None], st["truth"][:-1]], axis=0))
    if "sense" in RAND:            # the flight's sensing errors move on once per decision (sensors_new for the shape)
        e, q = st["sen"], jax.random.split(jax.random.fold_in(k1, 7), 2)   # new keys: the old draws stay as they were
        n, u = jax.random.normal(q[0], (6,)), jax.random.uniform(q[1], (5,))
        a_p, a_v, a_h = jnp.exp(-DT_C / e["tp"]), jnp.exp(-DT_C / e["tv"]), jnp.exp(-DT_C / e["th"])
        lost = st["drop"] > 0                     # marker lost: the pad estimate drifts (random walk) until it is seen again
        ep = jnp.where(lost, st["ep"] + e["rw"] * math.sqrt(DT_C) * n[:2],
                       a_p * st["ep"] + e["kp"] * (0.01 + 0.001 * p[2]) * jnp.sqrt(1 - a_p ** 2) * n[:2])
        drop = jnp.where(lost, st["drop"] - DT_C, jnp.where(u[0] < DT_C / 20.0, 0.5 + 2.5 * u[1], 0.0))   # ~1 per 20 s, 0.5-3 s
        ang = 2 * math.pi * u[2]                    # a misread marker, one decision long, ~1 per 13 s
        st = dict(st, ep=ep, drop=drop, ev=a_v * st["ev"] + e["sv"] * jnp.sqrt(1 - a_v ** 2) * n[2:5],
                  eh=a_h * st["eh"] + e["sh"] * jnp.sqrt(1 - a_h ** 2) * n[5],
                  jump=jnp.where(u[3] < 0.003, (0.3 + 0.7 * u[4]) * jnp.array([jnp.cos(ang), jnp.sin(ang)]), 0.0))
    return st, r, done


def privileged(st):
    """What only the critic sees, in training (PLAN.md Phase 3, privileged critic; the actor never does): this flight's
    drone (drag, mass, thrust, motor lag, command delay), its sensing errors right now (pad estimate, height, velocity,
    delay, marker lost) and the wind (strength, direction, the gust now). Under randomization these hidden numbers decide
    whether a landing can work; a critic that sees them gives the actor advantages that are not noise. 18 numbers."""
    prm, e, w = st["sim"]["prm"], st["sen"], st["wind"]
    return jnp.concatenate([prm[:2] / jnp.array([X.CDA_H, X.CDA_V]), prm[2:5], prm[5:6] / 20.0,
                            (e["b"] + st["ep"] + st["jump"]) * 5.0, (e["hb"] + st["eh"])[None] * 10.0, st["ev"] * 3.0,
                            e["lat"][None] / 3.0, (st["drop"] > 0)[None] * 1.0,
                            jnp.array([w["w10"] / 20.0, jnp.cos(w["th"]), jnp.sin(w["th"]), w["gA"] * (w["gt"] >= 0) / 10.0])])


N_PRIV = 18


def obs_of(st, key):
    p_est, v_est = estimates(st, key)
    cmd, _ = rule_cmd(-p_est[:2], p_est[2], v_est, st["held"])
    return observe(st, p_est, v_est, cmd)


def rule_action(st, key):
    """The rule itself as an action: no feed-forward, and the gate's vertical choice (decide=False does that)."""
    return jnp.array([0.0, 0.0, 0.0])


def evaluate(policy_fn, w10, n=1024, seed=0, fly=True, decide=True, t_max=T_MAX, shield=0.0, hp_tau=0.0, per_flight=False):
    """n flights at wind w10 (paired: the same yards, starts and wind seeds for every policy). Returns outcome counts,
    miss distances of landed flights, times. w10 may be a list: one compiled program then flies every wind, and a list
    of results comes back (6 Oct: compiling took most of each call); module flags (RAND, GATE_SCALE...) must not change
    during the call. per_flight: also every flight's outcome (paired statistics, Phase 3)."""
    keys = jax.random.split(jax.random.PRNGKey(seed), n)

    def one(carry, k):
        st, acc = carry
        ks = jax.random.split(k, n * 2).reshape(n, 2, 2)
        obs = jax.vmap(obs_of)(st, ks[:, 0])
        act = policy_fn(obs, st)
        new, r, done = jax.vmap(lambda s, a, kk: step(s, a, kk, fly, decide, shield, hp_tau))(st, act, ks[:, 1])
        # how it flies, counted only while in the air (6 Oct: the DOB landed in MJX with its push at the cap ~80% of the
        # time, then landed hard in Gazebo; measure that before flying anything in Gazebo)
        lv = (~st["done"]).astype(jnp.float32)
        push, push0 = new["prev_act"][:, :2], st["prev_act"][:, :2]          # the applied push / A_FF, now and before
        v0, v1 = st["vz_sent"], new["vz_sent"]
        flip = ((v0 > 0.5) & (v1 < -0.3)) | ((v0 < -0.3) & (v1 > 0.5))
        h = st["sim"]["d"].qpos[:, 2]
        band = jnp.where(h > 8.0, 0, jnp.where(h > 6.0, 1, jnp.where(h > 2.0, 2, 3)))
        rule_vz = obs[:, 10] * 3.0                                            # the rule's own vertical (obs holds cmd/3)
        cat = jnp.where(rule_vz < -1.0, 0, jnp.where(rule_vz < -0.1, 1, 2))   # rule: fast down / slow down / hold
        mode = jnp.where(decide, act[:, 2].astype(jnp.int32), 0)
        acc = dict(live=acc["live"] + lv, cap=acc["cap"] + lv * (jnp.abs(push) > 0.97).any(-1),
                   slew=acc["slew"] + lv * jnp.linalg.norm(push - push0, axis=-1) * A_FF, flip=acc["flip"] + lv * flip,
                   msat=acc["msat"] + lv * (new["sim"]["w"] >= 0.995 * X.W_MAX).any(-1),
                   picks=acc["picks"].at[band, cat, mode].add(lv), ret=acc["ret"] + lv * r)
        keep = lambda a, b: jnp.where(st["done"].reshape((-1,) + (1,) * (a.ndim - 1)), a, b)
        st = jax.tree_util.tree_map(keep, st, new)
        return (st, acc), None

    n_steps = int(t_max / DT_C) + 1
    z = jnp.zeros(n)
    acc0 = dict(live=z, cap=z, slew=z, flip=z, msat=z, picks=jnp.zeros((4, 3, 5)), ret=z)
    run = jax.jit(lambda c, ks: jax.lax.scan(one, c, ks)[0])
    res = [summary(*run((jax.vmap(lambda k: reset(k, w))(keys), acc0), jax.random.split(jax.random.PRNGKey(seed + 1), n_steps)), w, n, per_flight)
           for w in (w10 if isinstance(w10, (list, tuple)) else [w10])]
    return res if isinstance(w10, (list, tuple)) else res[0]


def summary(st, acc, w10, n, per_flight=False):
    """evaluate()'s numbers for one wind."""
    acc = {k: np.array(v) for k, v in acc.items()}
    air = max(float(acc["live"].sum()), 1.0)
    out = np.array(st["outcome"])
    v = np.array(st["sim"]["d"].qvel[:, :3])                 # frozen at touchdown
    sink, slide = (out == 3) & (-v[:, 2] >= V_SINK_OK), (out == 3) & (np.hypot(v[:, 0], v[:, 1]) >= V_SLIDE_OK)
    fast = sink | slide
    return dict(w10=float(w10), n=n, landed=float((out == 1).mean()), crashed=float((out == 2).mean()),
                hard=float((out == 3).mean()), timeout=float(((out == 4) | (out == 0)).mean()),
                hard_fast=float(fast.mean()),                # of the hard ones, too fast (the rest: gentle, off the pad)
                miss_landed=float(np.median(np.array(st["miss"])[out == 1])) if (out == 1).any() else float("nan"),
                t_mean=float(np.array(st["t"]).mean()),
                t_land_median=float(np.median(np.array(st["t"])[out == 1])) if (out == 1).any() else float("nan"),
                push_cap=float(acc["cap"].sum() / air), push_step=float(acc["slew"].sum() / air),
                rev_per_s=float(acc["flip"].sum() / (air * DT_C)), motor_sat=float(acc["msat"].sum() / air),
                picks=acc["picks"].round().astype(int).tolist(),   # [height >8, 6-8, 2-6, <2 m][rule: fast, slow, hold][mode]
                ret_mean=float(acc["ret"].mean()),                 # a flight's summed reward at the prices traced (CRASH_R...)
                first40=dict(outcome=out[:40].tolist(), miss=np.array(st["miss"])[:40].round(3).tolist(),
                             t=np.array(st["t"])[:40].round(2).tolist()),    # episode i = Gazebo's yard i
                **({"outcome_all": out.tolist(), "sink_all": sink.tolist(), "slide_all": slide.tolist()} if per_flight else {}))


def check_rand(n=1024, steps=2250):
    """PLAN.md 2c, each randomization group alone, the rule flying 90 s at W10 15: the sensing errors span what Gazebo
    showed (and the rule and the learner see one estimate), every drone differs and still lands, and the wind turns."""
    global RAND
    hold = jnp.zeros((n, 3))
    for groups in (("sense",), ("drone",), ("gust",)):
        RAND = frozenset(groups)
        st = jax.vmap(lambda k: reset(k, 15.0))(jax.random.split(jax.random.PRNGKey(5), n))
        th0 = st["wind"]["th"][:64]

        def one(st, k):
            ks = jax.random.split(k, n * 3).reshape(n, 3, 2)
            pe, ve = jax.vmap(estimates)(st, ks[:, 0])
            pe2, _ = jax.vmap(estimates)(st, ks[:, 2])
            d = st["sim"]["d"]
            err = jnp.concatenate([pe - d.qpos[:, :3], ve - d.qvel[:, :3]], 1)[:64]
            new, _, _ = jax.vmap(lambda s, a, kk: step(s, a, kk, False, False))(st, hold, ks[:, 1])
            keep = lambda a, b: jnp.where(st["done"].reshape((-1,) + (1,) * (a.ndim - 1)), a, b)
            return jax.tree_util.tree_map(keep, st, new), (err, jnp.abs(pe - pe2).max(), st["drop"][:64] > 0,
                                                           st["wind"]["th"][:64], st["done"][:64], d.qpos[:64, 2])
        st, (err, same, lost, th, done, h) = jax.jit(lambda s, ks: jax.lax.scan(one, s, ks))(
            st, jax.random.split(jax.random.PRNGKey(6), steps))
        err, live, out = np.array(err), ~np.array(done), np.array(st["outcome"])
        if "sense" in groups:
            low = live & (np.array(h) > 2.0) & (np.array(h) < 20.0)
            bias = np.array([np.hypot(*err[low[:, i], i, :2].mean(0)) for i in range(64) if low[:, i].sum() > 50])
            vsd = np.array([err[live[:, i], i, 3:5].std() for i in range(64) if live[:, i].sum() > 50])
            print(f"sense: per-flight pad bias 5/50/95%: {np.percentile(bias, [5, 50, 95]).round(3)} m; velocity sd "
                  f"{np.percentile(vsd, [5, 50, 95]).round(3)} m/s; marker lost {100 * np.array(lost)[live].mean():.1f}% of the time; "
                  f"rule and learner see the same estimate: {float(np.array(same).max()) == 0.0}")
            assert float(np.array(same).max()) == 0.0, "the rule and the learner see different estimates"
            assert bias.min() < 0.05 < 0.2 < bias.max(), "pad bias does not span Gazebo's 0.02-0.37 m"
            assert vsd.min() < 0.1 < 0.25 < vsd.max(), "velocity error does not span Gazebo's 0.06-0.29 m/s"
            assert 0.02 < np.array(lost)[live].mean() < 0.2, "dropouts are not about 1 per 20 s"
        if "drone" in groups:
            prm = np.array(st["sim"]["prm"])
            print(f"drone: mass x {prm[:, 2].min():.2f}-{prm[:, 2].max():.2f}, thrust x {prm[:, 3].min():.2f}-{prm[:, 3].max():.2f}, "
                  f"drag x {prm[:, 0].min() / X.CDA_H:.2f}-{prm[:, 0].max() / X.CDA_H:.2f}, latency {prm[:, 5].min() * 4:.0f}-"
                  f"{prm[:, 5].max() * 4:.0f} ms; rule landed {100 * (out == 1).mean():.1f}%, crashed {100 * (out == 2).mean():.1f}% (nominal 65.5, 0.2)")
            assert prm[:, 2].std() > 0.04 and prm[:, 5].max() >= 16, "the drones do not differ"
            assert 0.2 < (out == 1).mean() < 0.95 and (out == 2).mean() < 0.05, "the randomized drones do not fly like x500s"
        if "gust" in groups:
            turned = np.abs(np.angle(np.exp(1j * (np.array(th) - np.array(th0)[None])))).max(0)
            print(f"gust: flights whose mean wind turned more than 0.3 rad in 90 s: {100 * (turned > 0.3).mean():.0f}%")
            assert (turned > 0.3).mean() > 0.5, "the wind does not turn"
    RAND = frozenset()


if __name__ == "__main__":
    import argparse, json, time
    ap = argparse.ArgumentParser()
    ap.add_argument("--rule-ladder", action="store_true", help="today's landing rule from calm to 18 m/s")
    ap.add_argument("--check-rand", action="store_true", help="PLAN.md 2c: each randomization group behaves as promised")
    ap.add_argument("--n", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=10_000, help="ppo.py evaluates on seed 10000: the same yards, starts, winds")
    ap.add_argument("--out", default=f"{X.R}/outputs/gazebo_wind/rule_ladder.json")
    a = ap.parse_args()
    if a.check_rand:
        check_rand()
    if a.rule_ladder:
        rule = lambda obs, st: jnp.tile(jnp.array([0.0, 0.0, 0.0]), (obs.shape[0], 1))
        res = []
        for w in (0, 3, 6, 9, 12, 15, 18):
            t0 = time.time()
            r = evaluate(rule, float(w), n=a.n, seed=a.seed, fly=False, decide=False)
            r["wall_s"] = round(time.time() - t0, 1)
            res.append(r)
            print(json.dumps(r), flush=True)
        json.dump(res, open(a.out, "w"), indent=1)
