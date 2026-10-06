#!/usr/bin/env python3
"""
gz_windland.py -- one land_env flight flown for real: Gazebo + PX4 with rig v2, the WindDrag wind, and the yard, start
and wind direction of MJX evaluation episode i (outputs/gazebo_wind/yards_s10000.json). The drone flies out from the depot in
still air, holds at the start point, the wind switches on (from its seed: the same gusts for every decider), and today's
rule or a trained policy (conditions/wind/runs/*.npz, run here in numpy) lands it. Judged on Gazebo's truth as land_env judges MJX.

    python3 conditions/wind/gz_windland.py --setup --episodes 20            worlds shared/px4_gazebo/gz_logan/windyard/e<i>/ (pad + house + tree)
    (conditions/wind/gz_ladder.sh)   ... gz_windland.py --episode 3 --w10 12 --decider rule | conditions/wind/runs/both_s0.npz
The learner acts as in MJX, through MAVLink: velocity setpoint (the rule's, its own vertical choice) plus an
acceleration feed-forward -- PX4 adds it to its velocity loop's output, as x500_mjx.position_control does.
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.dirname(os.path.dirname(HERE))
PAD = np.array(json.load(open(os.environ["GEO_MISSION"]))["pad"]["enu"], float) if os.environ.get("GEO_MISSION") \
    else np.array([120.0, 0.0])                   # the world's arucotag, ENU (px4_mission.PAD_XY is its NED); a mission's
DT_C = 0.04
T_MAX = 90.0
PAD_R, R_DRONE, V_SINK_OK, V_SLIDE_OK = 0.75, 0.35, 1.0, 0.5
COM_Z = 0.224                                     # CoM above the feet: MJX heights are the CoM's
A_FF, MODES = 3.0, np.array([0.0, -1.5, -0.6, 0.0, 1.0])     # land_env.MODES: 0 = the rule's own
TAU_LAT, A_MAX, K_LIN = 0.69, 1.5, 0.8
YARDS = f"{R}/outputs/gazebo_wind/yards_s10000.json"


# ---- the test yard: land_env.yard_new's house and tree around the world's pad ---------------------------------------------
def yard_world(y, i):
    base = open(f"{R}/shared/px4_gazebo/gz_logan/voxfast_v2/aruco.sdf").read()
    head = base[:base.index("    <include>\n      <uri>model://arucotag</uri>")]
    hc, ang, hh = np.array(y["hc"]) + PAD, y["hang"], y["hh"]
    parts = [head,
             f"    <include><uri>model://arucotag</uri><pose>{PAD[0]} {PAD[1]} 0.001 0 0 0</pose></include>\n",
             f"    <model name='yard_house'><static>true</static><pose>{hc[0]:.3f} {hc[1]:.3f} {hh / 2:.3f} 0 0 {ang:.4f}</pose>"
             f"<link name='l'><visual name='v'><geometry><box><size>8 10 {hh:.3f}</size></box></geometry><material>"
             f"<ambient>0.62 0.42 0.36 1</ambient><diffuse>0.72 0.50 0.42 1</diffuse></material></visual>"
             f"<collision name='c'><geometry><box><size>8 10 {hh:.3f}</size></box></geometry></collision></link></model>\n"]
    if y["tree"]:
        tc, tr, top = np.array(y["tc"]) + PAD, y["tr"], y["ttop"]
        cz = top - tr
        parts.append(f"    <model name='yard_tree'><static>true</static><pose>{tc[0]:.3f} {tc[1]:.3f} 0 0 0 0</pose><link name='l'>"
                     f"<visual name='can'><pose>0 0 {cz:.3f} 0 0 0</pose><geometry><sphere><radius>{tr:.3f}</radius></sphere></geometry>"
                     f"<material><ambient>0.20 0.42 0.18 1</ambient><diffuse>0.24 0.50 0.22 1</diffuse></material></visual>"
                     f"<collision name='cc'><pose>0 0 {cz:.3f} 0 0 0</pose><geometry><sphere><radius>{tr:.3f}</radius></sphere></geometry></collision>"
                     f"<visual name='trunk'><pose>0 0 {cz / 2:.3f} 0 0 0</pose><geometry><cylinder><radius>0.2</radius><length>{cz:.3f}</length></cylinder></geometry>"
                     f"<material><ambient>0.35 0.25 0.15 1</ambient><diffuse>0.40 0.28 0.17 1</diffuse></material></visual>"
                     f"<collision name='ct'><pose>0 0 {cz / 2:.3f} 0 0 0</pose><geometry><cylinder><radius>0.2</radius><length>{cz:.3f}</length></cylinder></geometry></collision>"
                     f"</link></model>\n")
    parts.append("    <wind><linear_velocity>0 0 0</linear_velocity></wind>\n"
                 "    <include><uri>model://x500_unified</uri><name>x500_mono_cam_down_0</name><pose>0 0 0 0 0 0</pose></include>\n"
                 "  </world>\n</sdf>\n")
    d = f"{R}/shared/px4_gazebo/gz_logan/windyard/e{i}"
    os.makedirs(d, exist_ok=True)
    open(f"{d}/aruco.sdf", "w").write("".join(parts))
    return d


# ---- land_env, in numpy ---------------------------------------------------------------------------------------------------------
def clearance(y, p):
    c, s = math.cos(-y["hang"]), math.sin(-y["hang"])
    rel = p[:2] - np.array(y["hc"])
    loc = np.array([c * rel[0] - s * rel[1], s * rel[0] + c * rel[1]])
    q = np.abs(np.array([loc[1], loc[0]])) - np.array(y["hhalf"])
    dz = p[2] - y["hh"]
    d_house = np.linalg.norm(np.maximum([q[0], q[1], dz], 0.0)) + min(max(q[0], q[1], dz), 0.0)
    if not y["tree"]:
        return d_house, 1e3
    cz = y["ttop"] - y["tr"]
    d_can = np.linalg.norm([p[0] - y["tc"][0], p[1] - y["tc"][1], p[2] - cz]) - y["tr"]
    d_trunk = np.linalg.norm(p[:2] - np.array(y["tc"])) - 0.2 if p[2] < cz else 1e3
    return d_house, min(d_can, d_trunk)


def approach(d, v, cap):
    d_eff = d - v * TAU_LAT
    n = np.linalg.norm(d_eff)
    return d_eff / n * min(cap, math.sqrt(2 * A_MAX * n), K_LIN * n) if n > 1e-3 else np.zeros(2)


def gate(d_h, h, v_est):
    tol = 0.3 + 0.1 * max(h - 3.0, 0.0)
    v_ok = 0.1 if h < 2.0 else 0.3
    r_d, r_v = np.linalg.norm(d_h) / tol, np.linalg.norm(v_est[:2]) / v_ok
    hold = h < 6.0 and (r_d > 1.0 or r_v > 1.0)
    return np.array([float(hold), min(r_d, 3.0) / 3.0, min(r_v, 3.0) / 3.0])


def rule_cmd(d_h, h, v_est):
    near, over = np.linalg.norm(d_h) < 3.0, np.linalg.norm(d_h) < 2.0
    v_h = approach(d_h, v_est[:2], 2.5 if near else 5.0)
    tol = 0.3 + 0.1 * max(h - 3.0, 0.0)
    v_ok = 0.1 if h < 2.0 else 0.3
    hold = h < 6.0 and (np.linalg.norm(d_h) > tol or np.linalg.norm(v_est[:2]) > v_ok)
    vz = (0.0 if hold else (-2.5 if h > 8.0 else -0.8)) if over else 0.0
    return np.r_[v_h, vz]


def observe(y, p_est, v_est, lean, cmd, prev_act, hist):
    """land_env.observe in numpy (checked against it: python conditions/wind/gz_windland.py --check-obs)."""
    dh, dt_ = clearance(y, p_est)
    rel_t = np.array(y["tc"]) - p_est[:2] if y["tree"] else np.array([30.0, 30.0])
    return np.r_[-p_est[:2] / 10, p_est[2:] / 10, v_est / 3, lean * 5, cmd / 3, (np.array(y["hc"]) - p_est[:2]) / 10,
                 rel_t / 10, min(dh, 10) / 5, min(dt_, 10) / 5, float(y["tree"]), prev_act, hist.ravel(),
                 np.tanh(-p_est[:2] / 0.5), gate(-p_est[:2], p_est[2], v_est)]


def check_obs(n=200):
    """Random MJX states: land_env.observe (JAX) against observe() here, and the rule's command too."""
    import jax
    import jax.numpy as jnp
    import land_env as E
    worst = 0.0
    for i in range(n):
        k = jax.random.split(jax.random.PRNGKey(i), 4)
        st = E.reset(k[0], 9.0)
        st = dict(st, prev_act=jax.random.normal(k[1], (7,)), hist=jax.random.normal(k[2], (4, 5)))
        p_est = np.array(st["sim"]["d"].qpos[:3]) + np.random.default_rng(i).normal(0, 3, 3) * np.array([1, 1, 0.3])
        p_est[2] = abs(p_est[2]) + 0.3
        v_est = np.random.default_rng(i + 1).normal(0, 1.5, 3)
        cmd_j, _ = E.rule_cmd(jnp.array(-p_est[:2]), jnp.array(p_est[2]), jnp.array(v_est))
        cmd_n = rule_cmd(-p_est[:2], p_est[2], v_est)
        o_j = np.array(E.observe(st, jnp.array(p_est), jnp.array(v_est), cmd_j))
        y = {kk: np.array(vv).tolist() for kk, vv in st["yard"].items()}
        lean = np.array(E.X.qrot(st["sim"]["d"].qpos[3:7])[:2, 2])
        o_n = observe(y, p_est, v_est, lean, cmd_n, np.array(st["prev_act"]), np.array(st["hist"]))
        assert o_n.shape == o_j.shape == (E.OBS_DIM,), (o_n.shape, o_j.shape)
        worst = max(worst, float(np.abs(o_n - o_j).max()), float(np.abs(cmd_n - np.array(cmd_j)).max()))
    assert worst < 1e-4, f"numpy observation differs from land_env's by {worst}"
    print(f"observation + rule command: numpy = land_env on {n} random states (worst {worst:.1e})")


class Policy:
    """A ppo.py checkpoint's greedy action, in numpy."""

    def __init__(self, path):
        z = np.load(path if os.path.exists(path) else os.path.join(HERE, "runs", os.path.basename(path)))   # flyer paths predate the reorg
        self.pi = [(z[f"pi_W{i}"], z[f"pi_b{i}"]) for i in range(len([k for k in z.files if k.startswith("pi_W")]))]
        self.mean, self.var = z["rms_mean"], z["rms_var"]
        self.meta = json.loads(str(z["meta"]))
        self.flyer = Policy(self.meta["flyer"]) if self.meta.get("flyer") else None     # stage 2: its frozen flyer

    def __call__(self, obs):
        x = np.clip((obs - self.mean) / np.sqrt(self.var + 1e-8), -10, 10)
        for W, b in self.pi[:-1]:
            x = np.tanh(x @ W + b)
        W, b = self.pi[-1]
        out = x @ W + b
        return (self.flyer(obs)[0] if self.flyer is not None else out[:2]), int(np.argmax(out[2:]))


def lean_enu(roll, pitch, yaw):
    """The body's up axis (FLU z) in ENU, x and y: land_env's 'lean' (PX4's roll/pitch/yaw are FRD/NED)."""
    cr, sr, cp, sp, cy, sy = math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch), math.cos(yaw), math.sin(yaw)
    r02 = cy * sp * cr + sy * sr
    r12 = sy * sp * cr - cy * sr
    return np.array([-r12, -r02])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--setup", action="store_true")
    ap.add_argument("--check-obs", action="store_true")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--episode", type=int)
    ap.add_argument("--w10", type=float, default=0.0)
    ap.add_argument("--decider", default="rule")
    ap.add_argument("--out", default=f"{R}/outputs/gazebo_wind/ladder/flights.jsonl")
    ap.add_argument("--geo", action="store_true", help="gnss_denied/geofly.sh: GeoAnchor's GNSS-free cruise first, then the "
                    "lander from 30 m over our real pad (its yard from the survey), the pad marker's fixes to PX4")
    ap.add_argument("--log-obs", action="store_true", help="save every decision's inputs and choice (sim-to-sim checks)")
    ap.add_argument("--video", default=None, help="--geo: the film (gnss_denied/geo_film.py)")
    a = ap.parse_args()
    if a.check_obs:
        return check_obs()
    yards = json.load(open(YARDS))
    if a.setup:
        for i in range(a.episodes):
            print(yard_world(yards[i]["yard"], i))
        return
    sys.path.insert(0, f"{R}/shared/px4_gazebo"); sys.path.insert(0, HERE)
    import px4_mission as PM
    from px4_steps import TruthLog
    from gz.transport13 import Node
    from gz.msgs10.double_pb2 import Double
    ep = yards[a.episode]
    y = ep["yard"]
    if a.geo:
        sys.path.insert(0, f"{R}/gnss_denied")
        import geo_cruise as GC
        y = GC.logan_yard()                                   # the real pad's nearest house and tree
        print("yard", json.dumps(y), flush=True)
    pol = None if a.decider == "rule" else Policy(a.decider)
    fly = pol is not None and (pol.meta.get("fly", True) or pol.flyer is not None)
    decide = pol is not None and pol.meta.get("decide", True)
    shield = 0.0 if pol is None else pol.meta.get("shield", 0.0)
    hp_tau = 0.0 if pol is None else (pol.meta.get("hp_tau", 0.0) or (pol.flyer.meta.get("hp_tau", 0.0) if pol.flyer else 0.0))
    ff_slow = np.zeros(2)                                         # land_env's high-pass of the push
    node = Node()
    wind_pub = node.advertise("/wind_scale", Double)
    truth = TruthLog()
    mi = PM.Mission(endpoint=f"udpin:0.0.0.0:{os.environ.get('MAV_PORT', 14540)}", expect_ids=PM.PAD_IDS)
    assert mi.arm_offboard(), "could not arm"
    mav = mi.mavutil.mavlink
    yaw_ned = math.pi / 2 - ep["yaw"]                        # land_env's heading (ENU) as PX4's

    def send(v_enu, a_enu):                                   # velocity + acceleration feed-forward (mask 2055)
        mi.m.mav.set_position_target_local_ned_send(0, mi.m.target_system, mi.m.target_component, mav.MAV_FRAME_LOCAL_NED,
                                                    2055, 0, 0, 0, float(v_enu[1]), float(v_enu[0]), -float(v_enu[2]),
                                                    float(a_enu[1]), float(a_enu[0]), 0.0, float(mi.yaw_hold), 0)

    start = PAD + np.array(ep["start"][:2])
    t_pre = None
    cr = None
    tag = "rule" if pol is None else os.path.basename(a.decider)[:-4]
    if a.geo:                                                 # GeoAnchor's cruise: GNSS denied from the suburb on
        film = None
        if a.video:
            from geo_film import GeoFilm
            import re                                         # the wind Gazebo runs (WindDrag's SDF): the film's arrow
            cfg = open(os.environ["SERVER_CONFIG"]).read() if os.environ.get("SERVER_CONFIG") else ""
            wd = re.search(r"<mean_speed>([-\d.]+)</mean_speed><direction_deg>([-\d.]+)</direction_deg>", cfg)
            film = GeoFilm(a.video, f"GeoAnchor + {'rule' if pol is None else 'flying learner'}"
                           + (f", wind {a.w10:g} m/s at 10 m" if a.w10 > 0 else ", calm"), PAD, [(0, 0), *GC.ROUTE, tuple(PAD)],
                           eye=mi.eye, wind=(float(wd[1]), float(wd[2])) if wd else None)
        cr = GC.Cruise(mi, truth, GC.Bridge(), film)
        cr.fly()
        start = None
    while start is not None:                                  # out from the depot in still air: up, across, settle
        p = mi.pos()
        if p is None:
            continue
        t_pre = mi.t_boot if t_pre is None else t_pre
        me = np.array([p[1], p[0], p[2]])
        d = start - me[:2]
        mi.yaw_hold = yaw_ned
        v = np.r_[approach(d, np.array([mi.vel_ne[1], mi.vel_ne[0]]), 6.0) if me[2] > 20 else np.zeros(2),
                  float(np.clip(0.8 * (30.0 - me[2]), -1.5, 3.0))]
        send(v, np.zeros(2)); mi.gimbal_down()
        if np.linalg.norm(d) < 0.5 and abs(me[2] - 30) < 0.5 and np.linalg.norm(mi.vel_ne) < 0.2:
            break
        if mi.t_boot - t_pre > 180:
            raise RuntimeError("never reached the start point")
        time.sleep(0.02)
    t_hold = mi.t_boot
    while cr is None and mi.t_boot - t_hold < 3.0:
        if mi.pos() is not None:
            send(np.zeros(3), np.zeros(2)); mi.gimbal_down()
        time.sleep(0.02)
    m = Double(); m.data = 1.0
    wind_pub.publish(m)                                       # the wind, from its seed
    t0 = mi.t_boot
    hist, prev_act = np.zeros((4, 5)), np.r_[0.0, 0.0, np.eye(5)[0]]
    mi.mark_ned, t_last = None, -1.0
    outcome, miss, min_clear, rows, g_prev, tilt_max = "timeout", float("nan"), 1e3, [], None, 0.0
    v_est_prev, pending = None, False
    obs_log = []
    while True:
        p = mi.pos()
        if p is None:
            continue
        if mi.t_boot - t_last < DT_C - 1e-3:
            continue
        t_last, t = mi.t_boot, mi.t_boot - t0
        seen, off, ids, mid = mi.eye.look()
        if seen:
            ray = PM.cam_ray(off, mi.roll, mi.pitch, mi.yaw, mi.gimbal, mi.eye.look_stamp)
            raw = p[:2] + ray[:2] * (p[2] / max(ray[2], 1e-3)) - PM.PAD_OFF_NED[mid]
            mi.mark_ned = raw if mi.mark_ned is None else mi.mark_ned + PM.MARK_ALPHA * (raw - mi.mark_ned)
            if cr is not None:
                cr.marker_fix(p, ray, mid)                    # ...and to PX4: no GNSS below GeoAnchor's 50 m
        pad_ne = np.array(PM.PAD_XY) if mi.mark_ned is None else mi.mark_ned
        p_est = np.array([p[1] - pad_ne[1], p[0] - pad_ne[0], p[2] + COM_Z])
        v_est = np.array([mi.vel_ne[1], mi.vel_ne[0], mi.climb])
        lean = lean_enu(mi.roll, mi.pitch, mi.yaw)
        if pending:                                           # land_env: history row = (v_est before, lean after)
            hist = np.vstack([hist[1:], np.r_[v_est_prev / 3, lean * 5]])
        cmd = rule_cmd(-p_est[:2], p_est[2], v_est)
        if pol is not None:
            ob = observe(y, p_est, v_est, lean, cmd, prev_act, hist)
            a_c, mode = pol(ob)
            if a.log_obs:
                obs_log.append(np.r_[t, ob, mode, p_est, v_est])
            raw = np.clip(a_c, -2, 2)
            a_ff = np.clip(raw - ff_slow * (hp_tau > 0), -1, 1) * A_FF if fly else np.zeros(2)
            ff_slow = ff_slow + (DT_C / max(hp_tau, DT_C)) * (raw - ff_slow)
            vz = MODES[mode] if decide and mode > 0 else cmd[2]
            if shield > 0 and p_est[2] < 6.0:                  # land_env.shield_block
                tol, v_ok = shield * (0.3 + 0.1 * max(p_est[2] - 3.0, 0.0)), shield * (0.1 if p_est[2] < 2.0 else 0.3)
                if np.linalg.norm(p_est[:2]) > tol or np.linalg.norm(v_est[:2]) > v_ok:
                    vz = max(vz, 0.0)
        else:
            a_ff, mode, vz = np.zeros(2), 0, cmd[2]
        send(np.r_[cmd[:2], vz], a_ff); mi.gimbal_down()
        if cr is not None:
            cr.observe(f"landing: {'rule' if pol is None else 'flying learner'}", p,
                       info={"wind": f"{a.w10:g} m/s at 10 m" if a.w10 > 0 else "calm", "marker": "locked" if mi.mark_ned is not None else "searching"})
        prev_act, v_est_prev, pending = np.r_[a_ff / A_FF, np.eye(5)[mode]], v_est, True
        # judged on Gazebo's truth, the drone's CoM relative to the pad
        with truth.lock:
            tr = truth.rows[-1] if truth.rows else None
        if tr is None:
            continue
        g = np.array([tr[1] - PAD[0], tr[2] - PAD[1], tr[3] + COM_Z])
        gv = np.zeros(3) if g_prev is None else (g - g_prev[1]) / max(tr[0] - g_prev[0], 1e-3)
        if g_prev is None or tr[0] > g_prev[0]:
            g_prev = (tr[0], g)
        w_, x_, y_, z_ = tr[4:8]
        tilt = math.degrees(math.acos(max(-1.0, min(1.0, 1 - 2 * (x_ * x_ + y_ * y_)))))
        tilt_max = max(tilt_max, tilt)
        dh, dt_ = clearance(y, g)
        min_clear = min(min_clear, dh, dt_)
        rows.append((t, *g, *gv, tilt, *a_ff, vz, float(mi.mark_ned is not None)))
        miss = float(np.linalg.norm(g[:2]))
        if min(dh, dt_) < R_DRONE or tilt > math.degrees(1.2):
            outcome = "crashed"; break
        if g[2] <= 0.25:
            outcome = "landed" if miss <= PAD_R and -gv[2] < V_SINK_OK and np.linalg.norm(gv[:2]) < V_SLIDE_OK else "hard"
            break
        if t >= T_MAX:
            break
    if cr is not None:                                        # the verdict on film for 3 s, then the record
        banner = {"landed": "DELIVERED", "crashed": "CRASHED", "hard": "LANDED OFF THE PAD"}.get(outcome, "GAVE UP")
        t_end = mi.t_boot
        while mi.t_boot - t_end < 3.0:
            p = mi.pos()
            send(np.array([0.0, 0.0, -0.5]), np.zeros(2))
            cr.observe("landed" if outcome == "landed" else outcome, p, banner=banner)
            time.sleep(0.02)
        if cr.film is not None:
            cr.film.close()
        with truth.lock:
            trows = list(truth.rows)
        geo_rep = GC.report(cr.rows, cr.bridge.fixes, cr.t_denied, trows)
        geo_rep.update(gnss_restored_at=cr.t_restored, marker_fixes=cr.mk.n if cr.mk else 0,
                       marker_skipped=cr.mk.skipped if cr.mk else 0)
        json.dump(dict(report=geo_rep, yard=y, rows=cr.rows, fixes=cr.bridge.fixes),
                  open(f"{os.path.dirname(a.out)}/geo_e{a.episode}_w{a.w10:g}_{tag}.json", "w"))
        print("geo", json.dumps(geo_rep), flush=True)
    m.data = 0.0
    wind_pub.publish(m)
    send(np.array([0.0, 0.0, -0.5]), np.zeros(2))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    if a.log_obs and obs_log:                             # t, obs (OBS_DIM), mode, p_est (3), v_est (3)
        np.save(f"{os.path.dirname(a.out)}/e{a.episode}_w{a.w10:g}_{tag}_obs.npy", np.array(obs_log))
    np.savetxt(f"{os.path.dirname(a.out)}/e{a.episode}_w{a.w10:g}_{tag}.csv", np.array(rows), delimiter=",", fmt="%.4f",
               header="t,x,y,z,vx,vy,vz,tilt,ax,ay,vz_cmd,locked", comments="")
    with open(a.out, "a") as f:
        f.write(json.dumps(dict(episode=a.episode, w10=a.w10, decider=tag, outcome=outcome, miss=miss, t=rows[-1][0] if rows else 0.0,
                                min_clear=min_clear, tilt_max=tilt_max, sink=float(-rows[-1][6]) if rows else None,
                                slide=float(np.hypot(rows[-1][4], rows[-1][5])) if rows else None)) + "\n")
    print(outcome, f"miss {miss:.2f} m, {rows[-1][0] if rows else 0:.1f} s")


if __name__ == "__main__":
    main()
