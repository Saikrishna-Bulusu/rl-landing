#!/usr/bin/env python3
"""
geo_cruise.py -- GeoAnchor's leg of a Logan delivery, flown for real (geo/geofly.sh: PX4 SITL + Gazebo on the photo
floor, GeoAnchor's own three layers on the live gimbal camera, its accepted fixes fused by PX4's EKF2 over AGP):

  1. straight up at the depot on GPS to 75 m
  2. the route at 75 m, 7 m/s, on PX4's own estimate; over the suburb, once GeoAnchor is steadily in PX4's loop (fixes
     arriving, cs_aux_gpos), GNSS is denied (EKF2_GPS_CTRL 0) -- from then on GeoAnchor alone knows where it is (if PX4
     drops it, GNSS comes back and the record says when)
  3. from 50 m out (by that estimate) down to 55 m, still inside GeoAnchor's envelope; settled over the pad, straight
     down to 30 m, the height the pad's 1.80 m marker reads from (geo/read_height.py, 26 Sept: 88% of frames at 30 m,
     13% at 50 m). From 50 m out every marker read is a fix for PX4 too (MarkerFix), so no stretch is unaided.

The landers take it from there (calm: the rule; wind: the flying learner; obstacles: rule + local 3D planner;
magpie: the rig-trained learner). On its own this is the smoke test: it stops at 30 m over the pad and reports
whether GeoAnchor was really in the loop -- PX4's flags, the fixes, and PX4's estimate against Gazebo's truth.

    OUT=smoke WORLD=geo_clean INSTANCE=5 geo/geofly.sh /usr/bin/python3 -u geo/geo_cruise.py
"""
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.dirname(HERE)
sys.path[:0] = [R, HERE, f"{R}/wind"]
from geo_world import gz_to_ll, ll_to_gz
from record_sweep import heading_deg

PAD = np.array([120.0, 0.0])                                          # our pad, Gazebo ENU
ROUTE = [(0.0, -120.0), (260.0, -120.0)]                              # then the pad: over the suburb, where the
ALT, V, DESCEND_AT, H_READ, HOVER_Z = 75.0, 7.0, 50.0, 30.0, 55.0                    # draped sweep matched (38% overall, the south
                                                                      # lane ~70%; the north lane's forest, 8%)
DENY = True
if os.environ.get("GEO_MISSION"):             # the dashboard's mission planner (geo/fly_mission.sh): its route and pad
    _m = json.load(open(os.environ["GEO_MISSION"]))
    PAD, ROUTE = np.array(_m["pad"]["enu"], float), [tuple(map(float, q)) for q in _m["route"]]
    ALT, V, DENY = float(_m["alt"]), float(_m["speed"]), bool(_m.get("deny_gnss", True))
HEALTHY = 4                    # accepted fixes in the last 15 s before GNSS goes (smoke2, 26 Sept: denied on the FIRST
                               # fused fix, none followed, EKF2 lost its position and PX4 blind-landed)
LOOK, A_SLEW, A_CORNER, YAW_RATE = 15.0, 2.0, 0.8, math.radians(30.0)   # 27 Sept (Sai: "turning right first, then a 180"): the
                               # route is FLOWN, not stopped at -- a carrot this far along it (m), the command's change
                               # capped (m/s per s), braking for corners (m/s per s; without it the 140 deg corner
                               # swung 13 m wide), the nose turned at this rate the short way. The old law aimed at
                               # each waypoint by the estimate: it hooked right toward a corner GeoAnchor put 5 m off,
                               # braked to 0.4 m/s at ~4 m/s^2, and the nose swung 144 -> 269 -> 310 deg the long way.
GNSS_BACK_S = 2.0              # PX4 has dropped GeoAnchor this long (its own 5 s timeout already passed): GNSS back,
                               # logged -- the flight goes on, and the record says how long GeoAnchor held


def carrot(poly, seg, p, look=LOOK):
    """Progress along the route polyline (the nearest segment, searched from `seg` on: never back), the point `look`
    metres further along it (clamped to its end), how far on the next corner is (m) and how sharply it turns (rad)."""
    best = None
    for i in range(seg, len(poly) - 1):
        a, ab = poly[i], poly[i + 1] - poly[i]
        s = float(np.clip(np.dot(p - a, ab) / (ab @ ab), 0.0, 1.0)) * float(np.linalg.norm(ab))
        dd = float(np.linalg.norm(a + s * ab / np.linalg.norm(ab) - p))
        if best is None or dd < best[0] - 1e-9:
            best = (dd, i, s)
    _, i0, s0 = best
    seg_len = lambda j: float(np.linalg.norm(poly[j + 1] - poly[j]))
    turn = 0.0 if i0 + 2 >= len(poly) else math.acos(float(np.clip(
        (poly[i0 + 1] - poly[i0]) @ (poly[i0 + 2] - poly[i0 + 1]) / (seg_len(i0) * seg_len(i0 + 1)), -1.0, 1.0)))
    i, s = i0, s0 + look
    while i < len(poly) - 2 and s > seg_len(i):
        s -= seg_len(i)
        i += 1
    return i0, poly[i] + min(s, seg_len(i)) * (poly[i + 1] - poly[i]) / seg_len(i), seg_len(i0) - s0, turn


def steer(poly, seg, p, v_prev, yaw, dt, v, want=None):
    """One step along the route (east, north): progress, the velocity command -- toward the carrot at v, or `want` --
    changed by at most A_SLEW*dt, and the nose (PX4 yaw, rad; None: held) turned toward it at YAW_RATE the short way."""
    if want is None:                   # slowed for the corner ahead: cos(turn/2) of v there (140 deg: 2.4 m/s), braking
        seg, c, to_corner, turn = carrot(poly, seg, p)                  # at A_CORNER to reach it LOOK/2 before it
        u = min(v, math.sqrt(max(v * math.cos(turn / 2), 2.0) ** 2 + 2 * A_CORNER * max(to_corner - LOOK / 2, 0.0)))
        want = u * (c - p) / max(float(np.linalg.norm(c - p)), 1e-6)
    dv = want - v_prev
    v_cmd = v_prev + dv * min(1.0, A_SLEW * dt / max(float(np.linalg.norm(dv)), 1e-9))
    if yaw is not None and np.linalg.norm(v_cmd) > 1.0:
        err = (math.atan2(v_cmd[0], v_cmd[1]) - yaw + math.pi) % (2 * math.pi) - math.pi
        yaw = (yaw + float(np.clip(err, -YAW_RATE * dt, YAW_RATE * dt)) + math.pi) % (2 * math.pi) - math.pi
    return seg, v_cmd, yaw


class Bridge:
    """geo/agp_sim_bridge.py's log, read as it grows: PX4's estimator flags and every fix GeoAnchor produced."""

    def __init__(self, path=os.environ.get("GEO_AGP_LOG")):
        self.path, self.pos, self.flags, self.fixes = path, 0, {}, []

    def poll(self):
        new = []
        if not self.path or not os.path.exists(self.path):
            return new
        with open(self.path) as f:
            f.seek(self.pos)
            for line in f:
                if not line.endswith("\n"):
                    break
                self.pos += len(line.encode())
                r = json.loads(line)
                if r["kind"] == "flags":
                    self.flags = r
                else:
                    if r["sent"]:
                        r["gz"] = ll_to_gz(r["lat"], r["lon"])
                    self.fixes.append(r)
                    new.append(r)
        return new


class Cruise:
    """Steps 1-4 on an armed px4_mission.Mission; `log(**row)` is called every step with the phase, PX4's estimate
    and Gazebo's truth, so the caller's film and records see the cruise too."""

    def __init__(self, mi, truth, bridge, film=None, log=None):
        self.mi, self.truth, self.bridge, self.film, self.log = mi, truth, bridge, film, log or (lambda **k: None)
        self.rows, self.t_denied, self.t_restored, self.me, self._agp_off, self.mk = [], None, None, None, None, None
        out = os.environ.get("GEO_OUT")                  # the dashboard's Flight log, Live: one row per 0.1 s of sim time
        self.live, self._t_live = (open(f"{out}/live.jsonl", "a") if out else None), -1.0

    def _truth(self):
        """Gazebo's newest pose (x, y, z, qw, qx, qy, qz) and velocity over the last ~0.2 s."""
        with self.truth.lock:
            if not self.truth.rows:
                return None, np.zeros(3)
            a, b = np.array(self.truth.rows[-1]), np.array(self.truth.rows[max(0, len(self.truth.rows) - 50)])
        return a[1:8], (a[1:4] - b[1:4]) / max(a[0] - b[0], 1e-3)

    def _step(self, phase, v_en, vz):
        mi = self.mi
        mi.send(float(v_en[1]), float(v_en[0]), float(vz)); mi.gimbal_down()
        me = self.observe(phase, mi.pos())
        time.sleep(0.02)
        return me

    def observe(self, phase, p, info=None, banner=None):
        """One step's record: the bridge polled, PX4's estimate p (NED north, east, up -- the caller's own pos(); a
        second read would block) against Gazebo's truth, GNSS back if PX4 has dropped GeoAnchor, the film ticked.
        The cruise calls it every step; the landers too, from their own loops."""
        mi = self.mi
        new = self.bridge.poll()
        if p is None:
            return self.me                                             # the last estimate (None before the first)
        me, (g, gv) = np.array([p[1], p[0], p[2]]), self._truth()
        self.me = me
        row = dict(t=mi.t_boot, wall=time.time(), phase=phase, est=me.tolist(), gz=None if g is None else g[:3].tolist(),
                   hdg_px4=round(math.degrees(mi.yaw) % 360.0, 2), hdg_true=None if g is None else round(heading_deg(g[3:7]), 2),
                   gnss=self.bridge.flags.get("gnss"), agp=self.bridge.flags.get("agp"),
                   fixes_sent=sum(f["sent"] for f in self.bridge.fixes))
        if self.t_denied is not None and self.t_restored is None:        # GeoAnchor dropped: GNSS back, logged --
            self._agp_off = None if row["agp"] else (self._agp_off or mi.t_boot)   # sim time (wall time made it 0.2 s
            if self._agp_off and mi.t_boot - self._agp_off > GNSS_BACK_S and me[2] > 5.0:   # at 0.1x); not below 5 m,
                                                                                        # where it would hide a failed landing
                self._set_gnss(7)
                self.t_restored = mi.t_boot
                print(f"GeoAnchor lost at t={mi.t_boot:.1f}: GNSS restored after {mi.t_boot - self.t_denied:.1f} s on GeoAnchor", flush=True)
        self.rows.append(row)
        self.log(**row)
        if self.live is not None and mi.t_boot - self._t_live >= 0.1:
            self._t_live = mi.t_boot
            self.live.write(json.dumps(dict(t=row["t"], est=[round(v, 2) for v in row["est"]], gz=None if g is None else
                                            [round(float(v), 2) for v in g[:3]], roll=round(math.degrees(mi.roll), 2),
                                            pitch=round(math.degrees(mi.pitch), 2), yaw=row["hdg_px4"],
                                            vel=[round(float(mi.vel_ne[1]), 2), round(float(mi.vel_ne[0]), 2), round(float(mi.climb), 2)],
                                            phase=phase, gnss=row["gnss"], agp=row["agp"])) + "\n")
            self.live.flush()
        if self.film is not None and g is not None:
            self.film.tick(mi.t_boot, g[:3], me, gv, phase, row["gnss"], row["agp"], info, banner, new, heading=row["hdg_px4"])
        return me

    def _wait(self, phase, secs, until, v_en=np.zeros(2)):
        """Step at v_en (east, north) until `until()`: the denial keeps flying -- a zero command here braked the
        cruise from 7 m/s at the moment GNSS went (calm4, 27 Sept: 30 deg of tilt in PX4's log)."""
        t0 = self.mi.t_boot
        while not until():
            if self.mi.t_boot - t0 > secs:
                raise RuntimeError(f"{phase}: not within {secs:.0f} s ({self.bridge.flags})")
            self._step(phase, v_en, 0.0)
        return self.me

    def _set_gnss(self, ctrl):
        mi = self.mi
        mi.m.mav.param_set_send(mi.m.target_system, mi.m.target_component, b"EKF2_GPS_CTRL", ctrl,
                                mi.mavutil.mavlink.MAV_PARAM_TYPE_INT32)

    def healthy(self):
        now = time.time()
        return self.bridge.flags.get("agp") is True and \
            sum(f["sent"] and now - f["t_wall"] < 15.0 for f in self.bridge.fixes) >= HEALTHY

    def fly(self, route=ROUTE, alt=ALT, v=V, descend_at=DESCEND_AT, h_read=H_READ, deny=DENY, eye=True, goal_out=0.0):
        """goal_out > 0: hand over that far short of the pad, still flying the last leg at cruise height -- the
        stage-3 flight: its planner takes it down, past an unmapped crane."""
        mi = self.mi
        ref = np.array(route[-1]) if route else np.zeros(2)            # where the last leg comes from (the depot)
        last_goal = PAD + goal_out * (ref - PAD) / np.linalg.norm(ref - PAD)
        while self.me is None or self.me[2] < alt - 1.0:                # 1. up, on GPS
            z = 0.0 if self.me is None else self.me[2]
            self._step("climb", np.zeros(2), float(np.clip(0.8 * (alt - z), 0.5, 3.0)))
        me, settled, seg, v_cmd, t_prev = self.me, False, 0, np.zeros(2), mi.t_boot
        poly = [me[:2].copy(), *map(np.array, route), last_goal]      # 3. the route on PX4's estimate, flown through
        while True:
            last = seg == len(poly) - 2
            # 2. GNSS goes once GeoAnchor is steadily in PX4's loop -- fixes arriving AND the EKF fusing them --
            # over the suburb (leg 2 on); by the last leg it must have happened
            if deny and self.t_denied is None and (seg >= 1 or not route) and self.healthy():
                self._set_gnss(0)
                self.t_denied = mi.t_boot
                me = self._wait("denying GNSS", 20.0, lambda: self.bridge.flags.get("gnss") is False, v_cmd)
                print(f"GNSS denied at t={mi.t_boot:.1f}, {me[0]:.0f} {me[1]:.0f}", flush=True)
            d = last_goal - me[:2]
            dist = float(np.linalg.norm(d))
            if deny and self.t_denied is None and last and (route or dist < descend_at):
                raise RuntimeError(f"GeoAnchor never steady enough to deny GNSS ({self.bridge.flags})")
            if goal_out and last and dist < 5.0:
                return me                                                # the planner's, at speed and height
            # 4. from 50 m out down to HOVER_Z, still inside GeoAnchor's envelope; settled over the pad, straight
            # down to where the marker reads (smoke6: descending while still closing, it hunted +-5 m/s over the
            # pad and dead-reckoned 10 s below 50 m -- PX4 then reset 12 m onto a fix)
            low = not goal_out and last and dist < descend_at
            if low and not settled and dist < 2.0 and np.linalg.norm(mi.vel_ne) < 0.5:
                settled = True
                self._param(b"MPC_Z_VEL_MAX_DN", 3.0)                    # the drop: less time between fixes
            z_goal = h_read if settled else (HOVER_Z if low else alt)
            want = None
            if low:                                                      # px4_mission's lag-aware law, arriving
                a = mi._approach(np.array([d[1], d[0]]), v)              # stopped (a plain P law hunted: PX4
                want = np.array([a[1], a[0]])                            # answers ~0.69 s late)
            dt, t_prev = float(np.clip(mi.t_boot - t_prev, 0.0, 0.2)), mi.t_boot
            # the nose along the way it flies; held from 50 m out -- smoke3/4 turned 150 deg over the pad and PX4's
            # heading (no GNSS) swung 6-10 deg with it
            seg, v_cmd, yaw = steer(poly, seg, me[:2], v_cmd, None if low else mi.yaw_hold, dt, v, want)
            mi.yaw_hold = mi.yaw_hold if yaw is None else yaw
            vz = float(np.clip(0.8 * (z_goal - me[2]), -3.0, 3.0))
            me = self._step("drop to the marker" if settled else "descent" if low else f"leg {seg + 1}", v_cmd, vz)
            if low and eye:
                self.marker()                                            # the pad marker: fixes from ~50 m down
            if settled and abs(me[2] - h_read) < 1.0:
                self._param(b"MPC_Z_VEL_MAX_DN", 1.5)                    # PX4's own, which the landers were tuned on
                return me
            if mi.t_boot - self.rows[0]["t"] > 900:
                raise RuntimeError("cruise timed out")

    def _param(self, name, value):
        mi = self.mi
        mi.m.mav.param_set_send(mi.m.target_system, mi.m.target_component, name, float(value),
                                mi.mavutil.mavlink.MAV_PARAM_TYPE_REAL32)

    def marker(self):
        """Read the pad marker; if seen, its fix to PX4. The cruise's own reads."""
        import px4_mission as PM
        mi = self.mi
        ok, off, ids, mid = mi.eye.look()
        if ok and self.rows:
            e = self.rows[-1]["est"]
            self.marker_fix(np.array([e[1], e[0], e[2]]), PM.cam_ray(off, mi.roll, mi.pitch, mi.yaw, mi.gimbal, mi.eye.look_stamp), mid)
        return ok

    def marker_fix(self, p, ray, mid):
        """A marker read the lander made itself (its p NED and px4_mission's ray): the same fix to PX4."""
        if self.mk is None:
            self.mk = MarkerFix()
        self.mk.send(p, ray, mid)

class MarkerFix:
    """The pad marker as a position fix for PX4 below GeoAnchor's envelope: the drone's place from the marker's ray and
    the pad's surveyed place (PM.PAD_XY), published like a GeoAnchor fix to geo/agp_sim_bridge.py. The landers call
    send() wherever they read the marker, with px4_mission's own projection (the same ray their centring uses)."""

    def __init__(self, endpoint=os.environ.get("GEO_MARKER_PUB")):
        ga = "/home/sai/GeoAnchor/geoanchor-rt"
        sys.path.append(ga)
        import glob
        for sp in sorted(glob.glob(f"{ga}/.venv/lib/python3.*/site-packages")):   # zmq: APPEND, never prepend
            if sp not in sys.path:
                sys.path.append(sp)
        from geoanchor.bus import Publisher
        from geoanchor import contracts as K
        import px4_mission as PM
        self.pub, self.topic, self.PM, self.seq, self.n = Publisher(endpoint), K.T_FIX, PM, 0, 0
        self.skipped, self.run = 0, 0                        # bad reads skipped; consecutive disagreements

    ANCHOR, TRUST = 20, 10         # fixes passed unchecked at hand-over; disagreements in a row that mean PX4 is wrong

    def send(self, p, ray, mid):
        """p: PX4's NED estimate (north, east, up); ray: PM.cam_ray's; mid: the marker read. All the way down, with a
        jump check: once the first ANCHOR fixes have pulled PX4 onto the pad, a read that disagrees with PX4 by more
        than max(0.5 m, 0.15 h) is taken for a bad read and skipped -- near the ground the camera sees part of the pad,
        and one bad read at 0.6 m (26 Sept, calm flight: 1.5 m off at sigma 0.5 m) threw the drone 2.8 m sideways --
        unless TRUST reads in a row disagree: then PX4 is what drifted. (27 Sept: stopping the fixes below 3 m instead
        let PX4 drift 0.3 m/s while the gate held it at 1.7 m in the gusts; 5 s later it dropped the aid, and the
        wind flight ran 10 m off the pad.)"""
        PM = self.PM
        rel = -np.asarray(ray[:2]) * (p[2] / max(ray[2], 1e-3)) + PM.PAD_OFF_NED[mid]     # drone - pad origin, NED
        n, e = np.array(PM.PAD_XY) + rel
        if self.n >= self.ANCHOR and math.hypot(n - p[0], e - p[1]) > max(0.5, 0.15 * max(p[2], 0.0)):
            self.run += 1
            if self.run < self.TRUST:
                self.skipped += 1
                return None
        self.run = 0
        lat, lon = gz_to_ll(e, n)
        self.seq += 1; self.n += 1
        self.pub.send(self.topic, dict(seq=self.seq, t_capture_unix=time.time(), accepted=True, lat=lat, lon=lon,
                                       sigma_m=0.5 + 0.03 * max(p[2], 0.0), source="marker"))
        return np.array([e, n])


def logan_yard():
    """land_env's yard for our real pad, from the 2023 survey the world's houses and trees were built from
    (urban/gz_logan.py): the nearest house as its footprint's minimum-area rectangle with a flat roof at its median
    height, the nearest tree as gz_logan plants it. 26 Sept: house wall 4.1 m north (21 x 13.5 m, roof 3.5 m), tree
    canopy edge 3.3 m east (top 5.2 m, crown 1.7 m) -- the tree inside land_env's training range, the house bigger."""
    import cv2
    from scipy.ndimage import label
    cwd = os.getcwd()
    os.chdir(f"{R}/urban"); sys.path.insert(0, f"{R}/urban")
    try:
        import gz_logan as GL
        lay, _road, X0, Y0 = GL.crop()
        tx, ty, th, tr = GL.treetops(lay["tree"], X0, Y0)
    finally:
        os.chdir(cwd)
    lab, n = label(lay["bldg"] > 2.5)
    best = None
    for k in range(1, n + 1):
        i, j = np.nonzero(lab == k)
        x, y = j + X0 + 0.5, i + Y0 + 0.5
        d = float(np.min(np.hypot(x - PAD[0], y - PAD[1])))
        if best is None or d < best[0]:
            best = (d, x, y, float(np.median(lay["bldg"][lab == k])))
    _d, x, y, hh = best
    (cx, cy), (w, h), ang = cv2.minAreaRect(np.stack([x, y], 1).astype(np.float32))
    m = int(np.argmin(np.hypot(tx - PAD[0], ty - PAD[1]) - tr))
    # land_env: local x along hang, hhalf = (half along local y, half along local x)
    return dict(hc=[float(cx - PAD[0]), float(cy - PAD[1])], hang=math.radians(ang), hhalf=[h / 2, w / 2], hh=hh,
                tree=True, tc=[float(tx[m] - PAD[0]), float(ty[m] - PAD[1])], tr=float(tr[m]), ttop=float(th[m]))


def report(rows, fixes, t_denied, truth_rows):
    """Was GeoAnchor really flying it? PX4's flags after the denial, the fixes against Gazebo's truth, and PX4's
    estimate against Gazebo's truth -- medians and tails (never the mean)."""
    tt = np.array([r[0] for r in truth_rows])
    xyz = np.array([r[1:4] for r in truth_rows])
    wall = np.array([r["wall"] for r in rows]); ts = np.array([r["t"] for r in rows])
    after = [r for r in rows if t_denied is not None and r["t"] > t_denied + 5 and r["gz"] is not None]
    est_err = np.array([np.hypot(*(np.array(r["est"][:2]) - np.array(r["gz"][:2]))) for r in after])
    fe = []
    for f in fixes:
        if not f["sent"] or f.get("source") == "marker":
            continue
        t_cap = np.interp(f["t_wall"] - f["latency_ms"] / 1e3, wall, ts)    # the frame's sim time
        j = int(np.clip(np.searchsorted(tt, t_cap), 0, len(tt) - 1))
        fe.append(float(np.hypot(f["gz"][0] - xyz[j, 0], f["gz"][1] - xyz[j, 1])))
    fe = np.array(fe)
    q = lambda a: dict(median=round(float(np.median(a)), 2), p90=round(float(np.percentile(a, 90)), 2),
                       max=round(float(a.max()), 2)) if len(a) else None
    sim_s = ts[-1] - ts[0] if len(ts) > 1 else 0.0
    return dict(
        fixes=len(fixes), sent=int(sum(f["sent"] and f.get("source") != "marker" for f in fixes)),
        marker_fixes=int(sum(f.get("source") == "marker" for f in fixes)),
        fixes_per_sim_s=round(sum(f["sent"] for f in fixes) / max(sim_s, 1e-6), 2),
        fix_err_m=q(fe), fix_within_10m_pct=round(100 * float((fe <= 10).mean()), 1) if len(fe) else None,
        latency_ms_wall_median=round(float(np.median([f["latency_ms"] for f in fixes])), 1) if fixes else None,
        rtf=round(sim_s / max(wall[-1] - wall[0], 1e-6), 2) if len(wall) > 1 else None,
        denied_s=round(float(ts[-1] - t_denied), 1) if t_denied is not None else None,
        agp_on_after_denial_pct=round(100 * float(np.mean([bool(r["agp"]) for r in after])), 1) if after else None,
        gnss_on_after_denial_pct=round(100 * float(np.mean([bool(r["gnss"]) for r in after])), 1) if after else None,
        est_err_m=q(est_err))


def main():
    import px4_mission as PM
    from px4_steps import TruthLog
    out = os.environ["GEO_OUT"]
    truth, bridge = TruthLog(), Bridge()
    mi = PM.Mission(endpoint=f"udpin:0.0.0.0:{os.environ.get('MAV_PORT', 14540)}", expect_ids=PM.PAD_IDS)
    assert mi.arm_offboard(), "could not arm"
    cr = Cruise(mi, truth, bridge)
    try:
        me = cr.fly()
        # 20 s over the pad at 30 m on the marker's fixes alone (GeoAnchor's envelope ends at 50 m): the landing's
        # navigation, checked -- PX4 must keep fusing, GNSS must stay off, its estimate must close on the truth
        seen, t_hold = 0, mi.t_boot
        while mi.t_boot - t_hold < 20.0:
            d = PAD - cr.me[:2]
            v_ne = mi._approach(np.array([d[1], d[0]]), 2.0)
            cr._step("over the pad (marker)", np.array([v_ne[1], v_ne[0]]), 0.0)
            seen += bool(cr.marker())
        ok = True
    except RuntimeError as e:
        print("FAILED:", e, flush=True)
        ok, seen = False, None
    with truth.lock:
        trows = list(truth.rows)
    last = [q for q in cr.rows if q["phase"] == "over the pad (marker)" and q["gz"]][-100:]
    r = dict(ok=ok, marker_reads=seen, gnss_restored_at=cr.t_restored,
             est_err_last2s_m=round(float(np.median([np.hypot(q["est"][0] - q["gz"][0], q["est"][1] - q["gz"][1]) for q in last])), 2) if last else None,
             **report(cr.rows, bridge.fixes, cr.t_denied, trows))
    json.dump(dict(report=r, rows=cr.rows, fixes=bridge.fixes), open(f"{out}/cruise.json", "w"))
    print(json.dumps(r), flush=True)


def check():
    """The route flown by steer() on a point mass that answers 0.69 s late (PX4, px4_response.json), no GeoAnchor
    error: it must never stop at a corner, turn the 140 deg corner LEFT (the short way), keep the command's change
    within A_SLEW, and stay near the route."""
    poly = [np.zeros(2), *map(np.array, ROUTE), PAD]
    p, vel, v_cmd, yaw, seg, dt = np.zeros(2), np.zeros(2), np.zeros(2), 0.0, 0, 0.05
    speeds, turn, off, dv = [], 0.0, [], []
    for k in range(4000):
        if np.linalg.norm(PAD - p) < DESCEND_AT:                        # the final approach: the approach law's
            break
        v0, y0 = v_cmd, yaw
        seg, v_cmd, yaw = steer(poly, seg, p, v_cmd, yaw, dt, V)
        dv.append(np.linalg.norm(v_cmd - v0) / dt)
        vel = vel + (v_cmd - vel) * dt / 0.69
        p = p + vel * dt
        speeds.append(np.linalg.norm(vel))
        if np.linalg.norm(p - poly[2]) < 40.0:                           # around the 140 deg corner
            turn += (yaw - y0 + math.pi) % (2 * math.pi) - math.pi
        off.append(min(np.linalg.norm(p - (a + np.clip(np.dot(p - a, b - a) / ((b - a) @ (b - a)), 0, 1) * (b - a)))
                       for a, b in zip(poly[:-1], poly[1:])))
    assert np.linalg.norm(PAD - p) < DESCEND_AT, f"never reached the final approach ({p})"
    assert min(speeds[200:]) > 2.0, f"stalled: {min(speeds[200:]):.2f} m/s"
    assert -160.0 < math.degrees(turn) < -120.0, f"corner turned {math.degrees(turn):.0f} deg (left is negative)"
    assert max(dv) <= A_SLEW + 1e-6 and max(off) < 6.0, (max(dv), max(off))
    print(f"route flown in {k * dt:.0f} s: slowest {min(speeds[200:]):.1f} m/s, corner {math.degrees(turn):.0f} deg "
          f"(left), command change <= {max(dv):.2f} m/s^2, at most {max(off):.1f} m off the route")


if __name__ == "__main__":
    check() if "--check" in sys.argv else main()
