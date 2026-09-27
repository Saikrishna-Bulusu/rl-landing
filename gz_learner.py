#!/usr/bin/env python3
"""
gz_learner.py -- the fast simulator's planner or learner flying PX4's x500 in the Logan Gazebo
world, with the magpie. Stage 5's first bridge: nothing about the decider changes, only where its
inputs come from.

    once:   python3 gz_learner.py --setup           writes gz_logan/birds/aruco.sdf
    SITL:   WORLDS_DIR=$PWD/gz_logan/birds EXTRA_MODELS=$PWD/gz_logan/models ./launch_px4.sh
    fly:    TREES_JSON=gz_logan/trees_logan.json python3 gz_learner.py --decider learner --seed 0 --video out.mp4
    pair:   python3 gz_learner.py --compose planner.mp4 learner.mp4 out.mp4
    all:    python3 gz_learner.py --final DIR/final.mp4   (DIR holds flights.jsonl and the flights)

Every 0.25 s of SIMULATED time (PX4's clock, lockstep with Gazebo) this builds the observation the
decider trained on -- position, velocity, the pad, the marker lock from px4_mission's perception,
the magpie when within 40 m -- sends it to urban/decider_server.py (PyTorch lives in the training
environment; PX4's and Gazebo's Python libraries live here) and flies the velocity it returns. The
magpie follows urban_env's rules, its constants read from urban_env.py itself, and is moved in
Gazebo by set_pose; so is a chase camera, for the video.

Frames: training map (x east, y north) = Gazebo (x east, y north) + OFFSET, a pure shift
(urban/gz_logan.py cut the world out of the same map); PX4 local NED (north, east) = Gazebo (y, x),
both from the depot.
"""
import argparse
import collections
import json
import math
import os
import queue
import re
import subprocess
import sys
import threading
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CONDA_PY = "/home/sai/miniconda3/envs/mujoco_drones/bin/python"


def _consts(path, *names):
    """Constants read from a source file's own top-level assignments, so they cannot drift."""
    ns = {"math": math, "os": os}
    for line in open(path).read().splitlines():
        m = re.match(r"^([A-Z_][A-Z0-9_, ]*) = ", line)
        if m and any(n in [s.strip() for s in m.group(1).split(",")] for n in names):
            exec(line.split("  #")[0], ns)
    return [ns[n] for n in names]


R_TERR, Z_TERR, V_SWOOP, REARM, HIT_BIRD, D_BIRD, MISS, DT, LOST_S = _consts(
    f"{HERE}/urban/urban_env.py", "R_TERR", "Z_TERR", "V_SWOOP", "REARM", "HIT_BIRD", "D_BIRD", "MISS", "DT", "LOST_S")
PAD_MAP, SITE = _consts(f"{HERE}/urban/gz_logan.py", "PAD", "SITE")
OFFSET = np.array(PAD_MAP) - np.array(SITE)              # Gazebo -> training map
WORLD, OBS_Z, OBS_HALF = "aruco", 60.0, 480              # the overhead observer camera
def VOX_BGR(z):
    """Turbo colour for a height (0 m blue ... 60 m red)."""
    import cv2
    return tuple(int(v) for v in cv2.applyColorMap(np.uint8([[int(np.clip(z / 60.0 * 255, 0, 255))]]), cv2.COLORMAP_TURBO)[0, 0])
R_HIT = 0.45                                             # x500 half-span with propellers, + a little
YAW_RATE = math.radians(90.0)                            # how fast the nose may turn toward the way it flies
V_COMMIT = 0.4                                           # m/s: the committed descent below 5 m (see the landing)
LAND_FIX = os.environ.get("LAND_FIX", "1") != "0"        # 0: the landing as before 25 Sept evening (comparison videos)


# ---- Gazebo: moving models -------------------------------------------------------------
class Mover:
    """set_pose on Gazebo models. asyn=True (flights): sent by a helper process, newest pose per model -- gz-transport's
    request() holds Python's GIL while it waits for the server (measured 25 Sept: another thread got 1.4% of its time
    during a 0.5 s request), so in the flight's own process every chase-camera move froze the flight loop (the film
    flights fell from 29 Hz to 2-10 Hz and flew into the crane). asyn=False (the benches): each move done before the
    next picture is taken."""

    def __init__(self, asyn=False):
        self.p = None
        if asyn:
            self.p = subprocess.Popen([sys.executable, "-u", os.path.abspath(__file__), "--mover"], stdin=subprocess.PIPE)
            return
        from gz.transport13 import Node
        from gz.msgs10.pose_pb2 import Pose
        from gz.msgs10.boolean_pb2 import Boolean
        self.node, self.Pose, self.Boolean = Node(), Pose, Boolean

    def put(self, name, xyz, yaw=0.0, pitch=0.0, roll=0.0):
        if self.p is not None:
            self.p.stdin.write((json.dumps([name, [float(v) for v in xyz], yaw, pitch, roll]) + "\n").encode())
            self.p.stdin.flush()
            return
        m = self.Pose(); m.name = name
        m.position.x, m.position.y, m.position.z = (float(v) for v in xyz)
        cy, sy, cp, sp = math.cos(yaw / 2), math.sin(yaw / 2), math.cos(pitch / 2), math.sin(pitch / 2)
        cr, sr = math.cos(roll / 2), math.sin(roll / 2)
        m.orientation.w, m.orientation.x = cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy
        m.orientation.y, m.orientation.z = cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy
        self.node.request(f"/world/{WORLD}/set_pose", m, self.Pose, self.Boolean, 200)


def mover_helper():
    """Mover(asyn=True)'s other half: reads [name, xyz, yaw, pitch, roll] lines, keeps the newest per model, sends them
    (never a stale pose queued ahead of a fresh one); ends when the flight closes the pipe."""
    import select
    m, buf, latest = Mover(), b"", {}
    while True:
        if not latest or select.select([0], [], [], 0)[0]:
            chunk = os.read(0, 1 << 16)
            if not chunk:
                return
            *lines, buf = (buf + chunk).split(b"\n")
            for line in lines:
                name, *args = json.loads(line)
                latest[name] = args
            continue
        name, args = latest.popitem()
        m.put(name, *args)


class Truth:
    """Where Gazebo really has the drone (the scene broadcaster's poses), kept with their sim-time stamps.
    PX4 flies on its own estimate (EKF: GPS + IMU + baro), which can sit a metre or more off; crashes and
    clearances are judged here, and the video's overlays are drawn here, never on the estimate."""

    def __init__(self, name="x500_mono_cam_down_0"):
        from gz.transport13 import Node
        from gz.msgs10.pose_v_pb2 import Pose_V
        self.name, self.hist, self.node = name, collections.deque(maxlen=200), Node()
        self.node.subscribe(Pose_V, f"/world/{WORLD}/dynamic_pose/info", self._on)

    def _on(self, m):
        for q in m.pose:
            if q.name == self.name:
                st = m.header.stamp
                self.hist.append((st.sec + st.nsec * 1e-9, np.array([q.position.x, q.position.y, q.position.z])))
                return

    def at(self, t_boot):
        """The true position nearest a PX4 time (PX4's clock is Gazebo's), or None if nothing close."""
        h = list(self.hist)
        if not h:
            return None
        ts, xyz = min(h, key=lambda q: abs(q[0] - t_boot))
        return xyz if abs(ts - t_boot) < 0.1 else None


# ---- the magpie: urban_env._bird for one bird -------------------------------------------
class Magpie:
    """Nest in a real Logan tree 20-90 m from the pad; while the drone is in its territory (within
    R_TERR of the nest, below Z_TERR) it waits, dives at where the drone WILL be (lead pursuit,
    committed, aimed to pass within 0..MISS), overshoots 15 m, waits REARM, dives again."""

    def __init__(self, seed, pad_gz, trees):
        self.rng = rng = np.random.default_rng(seed)
        cand = [t for t in trees if 20 <= math.dist(t["enu"], pad_gz) <= 90 and t["h"] >= 10]
        tr = cand[rng.integers(len(cand))]
        self.nest = np.array([*tr["enu"], min(tr["h"] - 1.0, 10.0 + 5.0 * rng.random())])
        self.pos, self.vel, self.aim = self.nest.copy(), np.zeros(3), self.nest.copy()
        self.dive, self.wait = False, 2.0 + 4.0 * rng.random()
        self.v = V_SWOOP[0] + (V_SWOOP[1] - V_SWOOP[0]) * rng.random()
        self.swoops, self.yaw = 0, 0.0

    def step(self, dt, me0, v0, me1):
        """Decide from where the drone was at the step's start (me0, v0), fly the step, and
        return the closest approach during it (both moved in straight lines)."""
        inside = me0[2] < Z_TERR and math.hypot(*(me0[:2] - self.nest[:2])) < R_TERR
        if inside and not self.dive:
            self.wait -= dt
            if self.wait <= 0:
                t_hit = np.linalg.norm(me0 - self.pos) / self.v
                for _ in range(3):
                    t_hit = np.linalg.norm(me0 + v0 * t_hit - self.pos) / self.v
                aim = me0 + v0 * t_hit
                line = aim - self.pos
                side = self.rng.standard_normal(3)
                side -= side @ line / max(line @ line, 1e-9) * line
                aim = aim + MISS * self.rng.random() * side / max(np.linalg.norm(side), 1e-9)
                self.aim, self.dive, self.swoops = aim, True, self.swoops + 1
                self.vel = self.v * (aim - self.pos) / max(np.linalg.norm(aim - self.pos), 1e-9)
        p0 = self.pos.copy()
        if self.dive:
            self.pos = self.pos + self.vel * dt
            self.pos[2] = max(self.pos[2], 1.0)
            self.yaw = math.atan2(self.vel[1], self.vel[0])
            if (self.pos - self.aim) @ self.vel / self.v > 15.0:
                self.dive, self.vel = False, np.zeros(3)
                self.wait = REARM[0] + (REARM[1] - REARM[0]) * self.rng.random()
        r0, r1 = p0 - me0, self.pos - me1
        d = r1 - r0
        s = min(max(-(r0 @ d) / max(d @ d, 1e-9), 0.0), 1.0)
        return float(np.linalg.norm(r0 + s * d))


BEHIND, STAGE_ANG, STAGE_BACK, STAGE_UP, V_STAGE, A_TURN, T_COMMIT, DIVE_R, DIVE_S = _consts(
    f"{HERE}/urban/urban_env.py", "BEHIND", "STAGE_ANG", "STAGE_BACK", "STAGE_UP", "V_STAGE", "A_TURN", "T_COMMIT",
    "DIVE_R", "DIVE_S")


class Hunter(Magpie):
    """urban_env's hunter magpie (magpie="hunter") for one bird, in numpy, line for line: waits
    behind and above the drone (a spot drawn anew for every dive), launches only from its rear
    sector, steers at where the drone will be at up to A_TURN, commits T_COMMIT before arrival,
    gives up after DIVE_S or when the drone leaves. Same nest, speeds and pass offsets as Magpie."""

    def __init__(self, seed, pad_gz, trees):
        super().__init__(seed, pad_gz, trees)
        self.off, self.t_dive, self.st = np.zeros(3), 0.0, self._stage_draw()

    def _stage_draw(self):
        u = self.rng.random(3)
        return np.array([STAGE_ANG * (2 * u[0] - 1), STAGE_BACK[0] + (STAGE_BACK[1] - STAGE_BACK[0]) * u[1],
                         STAGE_UP[0] + (STAGE_UP[1] - STAGE_UP[0]) * u[2]])

    def step(self, dt, me0, v0, me1, heading=(1.0, 0.0)):
        unit = lambda x: x / max(np.linalg.norm(x), 1e-9)
        heading = np.asarray(heading, float)
        inside = me0[2] < Z_TERR and math.hypot(*(me0[:2] - self.nest[:2])) < R_TERR
        rel = self.pos - me0
        dist, flat = np.linalg.norm(rel), np.linalg.norm(rel[:2])
        rear = rel[:2] @ heading < math.cos(BEHIND) * flat
        if inside and not self.dive:
            self.wait -= dt
        t_go = dist / self.v
        for _ in range(3):
            t_go = np.linalg.norm(me0 + v0 * t_go - self.pos) / self.v
        line = me0 + v0 * t_go - self.pos
        if inside and not self.dive and self.wait <= 0 and rear and DIVE_R[0] <= dist <= DIVE_R[1]:
            side = self.rng.standard_normal(3)
            side -= side @ line / max(line @ line, 1e-9) * line
            self.off = MISS * self.rng.random() * unit(side)
            self.dive, self.swoops, self.t_dive = True, self.swoops + 1, dt
            self.vel = self.v * unit(line + self.off)
        elif self.dive:
            if t_go > T_COMMIT:
                c, d = unit(self.vel), unit(line + self.off)
                th = math.acos(float(np.clip(c @ d, -1.0, 1.0)))
                f = min(A_TURN * dt / self.v / max(th, 1e-6), 1.0)
                turned = d if th < 1e-4 else (math.sin((1 - f) * th) * c + math.sin(f * th) * d) / math.sin(th)
                self.vel = self.v * unit(turned)
            self.t_dive += dt
        else:
            ca, sa = math.cos(self.st[0]), math.sin(self.st[0])
            tail = -np.array([ca * heading[0] - sa * heading[1], sa * heading[0] + ca * heading[1]])
            goal = me0 + np.array([*(self.st[1] * tail), self.st[2]]) if inside else self.nest.copy()
            goal[2] = min(goal[2], Z_TERR - 1.0)
            hop = goal - self.pos
            self.vel = hop * (V_STAGE / max(np.linalg.norm(hop), V_STAGE * dt))
        p0, diving = self.pos.copy(), self.dive
        self.pos = self.pos + self.vel * dt
        if not self.dive and np.linalg.norm(self.pos - me1) < 8.0:        # repositioning keeps 8 m clear
            self.pos = me1 + 8.0 * unit(self.pos - me1)
        self.pos[2] = max(self.pos[2], 1.0)
        if np.linalg.norm(self.vel[:2]) > 0.1:
            self.yaw = math.atan2(self.vel[1], self.vel[0])
        if self.dive and ((self.pos - me1) @ self.vel / self.v > 15.0 or self.t_dive > DIVE_S or not inside):
            self.dive = False
            self.wait = REARM[0] + (REARM[1] - REARM[0]) * self.rng.random()
            self.st = self._stage_draw()
        if not diving:
            return 1e9
        r0, r1 = p0 - me0, self.pos - me1
        d = r1 - r0
        s = min(max(-(r0 @ d) / max(d @ d, 1e-9), 0.0), 1.0)
        return float(np.linalg.norm(r0 + s * d))


# ---- the decider, in the training environment's Python ----------------------------------
class Decider:
    def __init__(self, kind, log, ckpt=None, voxel=False, local3d=False):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
        self.path = None                    # the local 3D planner's latest path (training map frame), for the video
        self.p = subprocess.Popen([CONDA_PY, "decider_server.py", "--decider", kind] + (["--ckpt", ckpt] if ckpt else [])
                                  + (["--voxel"] if voxel else []) + (["--local3d"] if local3d else []),
                                  cwd=f"{HERE}/urban",
                                  stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=open(log, "w"),
                                  text=True, bufsize=1, env=env)
        ready = self.p.stdout.readline()
        assert "ready" in ready, f"decider did not start -- see {log}"

    def ask(self, obs):
        self.send(obs)
        return self.recv()

    # ASKED WITHOUT WAITING (27 Sept): the local 3D planner's A* can take seconds once the rule heads down into the
    # clutter round the pad; waiting on it froze the flight loop 6 s beside the crane (GeoAnchor stage-3 flight) --
    # no new setpoints and no depth frames into the voxel map, so the crane was never mapped. The loop now goes on
    # (setpoints, depth, film) and takes the answer when it comes: the same decisions, never later than before.
    pending = False

    def send(self, obs):
        self.p.stdin.write(json.dumps(obs) + "\n"); self.p.stdin.flush()
        self.pending = True

    def ready(self):
        import select
        return self.pending and bool(select.select([self.p.stdout], [], [], 0)[0])

    def recv(self):
        r = json.loads(self.p.stdout.readline())
        self.pending, self.path = False, r.get("path")
        return np.array(r["cmd"])

    def close(self):
        self.p.stdin.close(); self.p.wait(timeout=10)


def _ahead(me, path, look=8.0):
    """Horizontal direction (east, north) from the drone to the point `look` m along the planned path (Gazebo frame),
    or None when that point is under 2 m away (hovering over the pad: keep the heading)."""
    p = np.asarray(path, float)
    if len(p) < 2:
        return None
    d = np.concatenate([np.linspace(a, b, max(2, int(np.linalg.norm(b - a) * 2) + 1)) for a, b in zip(p[:-1], p[1:])])
    d = d[int(np.argmin(np.linalg.norm(d - me, axis=1))):]
    s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(d, axis=0), axis=1))]
    q = d[min(int(np.searchsorted(s, look)), len(d) - 1)] - me
    return q[:2] if math.hypot(q[0], q[1]) > 2.0 else None


class NoBird:
    """Stage-3 flights have no magpie: one a thousand km away that never dives (not under the depot: the
    airspace test is horizontal distance to the nest, so a nest at 0,0 put the whole approach "in it")."""

    def __init__(self, *a):
        self.pos, self.vel, self.nest = np.array([1e6, 1e6, -100.0]), np.zeros(3), np.array([1e6, 1e6, -100.0])
        self.yaw, self.dive, self.swoops = 0.0, False, 0

    def step(self, *a):
        return 1e9


# ---- the six-camera rig, seen through the detector and the tracker ----------------------
class RigEye:
    """The rig's six cameras (gz_rig.py) streamed to urban/rig_perceive.py -- detector + tracker, in
    the training environment's PyTorch -- on a thread of its own; the flight loop reads the newest
    belief. Each frame set goes out with the drone's OWN position and attitude estimate (PX4's),
    not Gazebo's truth. If perception falls behind, older sets are skipped (and counted)."""
    def __init__(self, log):
        from gz.transport13 import Node
        from gz.msgs10.image_pb2 import Image
        import gz_rig as GR
        self.CAMS = tuple(GR.RIG)           # the six-camera rig; RIG=v2: the OAK-D's colour camera and three wide ones
        self.p = subprocess.Popen([CONDA_PY, "rig_perceive.py"], cwd=f"{HERE}/urban", stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE, stderr=open(log, "w"))
        assert b"ready" in self.p.stdout.readline(), f"perception did not start -- see {log}"
        self.last, self.cv = {}, threading.Condition()
        self.pose, self.belief, self.done_t, self.sets, self.skipped, self.log = None, None, -1.0, 0, 0, []
        self.inset = None                   # (t, camera, close-up RGB, score, distance): for the video
        self.node = Node()
        for n in self.CAMS:
            self.node.subscribe(Image, f"/rig/{n}", lambda m, n=n: self._on(n, m))
        self.th = threading.Thread(target=self._run, daemon=True); self.th.start()

    def _on(self, n, m):
        with self.cv:
            self.last[n] = (m.header.stamp.sec + m.header.stamp.nsec * 1e-9, m)
            self.cv.notify()

    def _set(self):
        """The newest complete set: all six cameras within half a frame of each other, newer than the last."""
        if len(self.last) < len(self.CAMS) or self.pose is None:
            return None
        st = [s for s, _ in self.last.values()]
        if max(st) - min(st) > 0.06 or min(st) <= self.done_t:      # one camera renders a frame apart (measured)
            return None
        return min(st), {n: m for n, (_, m) in self.last.items()}

    def _run(self):
        while True:
            with self.cv:
                while (got := self._set()) is None:
                    self.cv.wait(1.0)
                pose = self.pose
            t, ms = got
            self.skipped += max(0, round((t - self.done_t) * 20) - 1) if self.done_t > 0 else 0
            m0 = ms[self.CAMS[0]]
            hdr = dict(t=t, pos=[float(v) for v in pose[0]], rpy=[float(v) for v in pose[1]], cams=list(self.CAMS),
                       h=m0.height, w=m0.width, shapes={n: [ms[n].height, ms[n].width] for n in self.CAMS})
            try:
                self.p.stdin.write((json.dumps(hdr) + "\n").encode())
                for n in self.CAMS:
                    self.p.stdin.write(ms[n].data)
                self.p.stdin.flush()
                line = self.p.stdout.readline()
                for _ in range(5):                           # a stray line (a warning, a blank) is not the answer:
                    try:                                     # skip it (27 Sept: one killed this thread, and the
                        out = json.loads(line)               # magpie flight flew blind)
                        break
                    except ValueError:
                        if not line:
                            return
                        line = self.p.stdout.readline()
                else:
                    return
            except (BrokenPipeError, OSError, ValueError):     # closed at the end of the flight
                return
            out["t"] = t
            if out["dets"]:                                  # the surest sighting, cut out for the video
                cam, score, u, v, size = max(out["dets"], key=lambda d: d[1])
                m = ms[cam]
                img = np.frombuffer(m.data, np.uint8).reshape(m.height, m.width, 3)
                r = int(max(3 * size, 40))
                y0, x0 = int(np.clip(v - r, 0, m.height - 2 * r)), int(np.clip(u - r, 0, m.width - 2 * r))
                self.inset = (t, cam, img[y0:y0 + 2 * r, x0:x0 + 2 * r].copy(), score, out["dist"], (u - x0, v - y0, size, 2 * r))
            self.belief, self.done_t, self.sets = out, t, self.sets + 1
            self.log.append((t, out, pose))

    def close(self):
        try:
            self.p.stdin.close(); self.p.wait(timeout=10)
        except Exception:
            self.p.kill()


def rig_score(elog, truth, d_bird=40.0):
    """How the rig did in one flight, against Gazebo's truth: the detections' direction error (the
    frame check -- over 10 degrees means the attitude conversion or the mounting is wrong; PX4's own
    heading estimate was 5.8 deg off on the pad, rig_diag.py), how often a
    set of pictures found the magpie while it was within d_bird, the tracker's position error."""
    if not elog:
        return dict(sets=0)
    import gz_rig as GR
    tt = np.array([x[0] for x in truth])
    at = lambda t: truth[int(np.clip(np.searchsorted(tt, t), 0, len(tt) - 1))]
    ang, found, err = [], [], []
    for t, out, pose in elog:
        _, me, bp = at(t)
        rel = bp - me
        near = np.linalg.norm(rel) <= d_bird
        if out["dets"] and near and np.linalg.norm(rel) > 1.0:   # the detection nearest the true bird: clutter aside
            Rb = GR.rot(*pose[1])
            dirs = [Rb @ GR.rot(*GR.RIG[c][1]) @ (lambda r: r / np.linalg.norm(r))(
                        np.array([1.0, (GR.CAMSZ[c][0] / 2 - u) / GR.CAMSZ[c][2], (GR.CAMSZ[c][1] / 2 - v) / GR.CAMSZ[c][2]]))
                    for c, _, u, v, _ in out["dets"]]
            ang.append(min(math.degrees(math.acos(float(np.clip(dv @ rel / np.linalg.norm(rel), -1, 1)))) for dv in dirs))
        if near:            # found = a detection within 10 degrees of the bird (PX4's heading is ~6 deg off, measured)
            found.append(bool(out["dets"]) and np.linalg.norm(rel) > 1.0 and ang[-1] < 10.0)
        if out["seen"]:
            err.append(float(np.linalg.norm(np.array(out["rel"]) - rel)))
    med = float(np.median(ang)) if ang else None
    return dict(sets=len(elog), dir_err_deg=med, frame_check="ok" if med is None or med < 10.0 else "FAILED",
                found_within_40m=float(np.mean(found)) if found else None, sets_within_40m=len(found),
                track_err_m=float(np.median(err)) if err else None, tracked_sets=len(err))


def enu_rpy(roll, pitch, yaw):
    """PX4's attitude (NED world, forward-right-down body) as Gazebo roll, pitch, yaw (ENU, forward-
    left-up). Derived -- checked in every rig flight: the detections' directions must agree with
    where the magpie truly is (result rig_dir_err_deg)."""
    return roll, -pitch, math.pi / 2 - yaw


# ---- the video: chase camera beside the overhead one, in simulated time -----------------
class Film:
    """One video frame per chase-camera image (20 per simulated second), filmed by Gazebo's own
    camera thread, not the flight loop: the loop pauses whenever the decider thinks (0.1 s of
    simulated time and more), and the first magpie videos, framed from it, repeated most frames
    2-4 times, up to 16. After each image the chase camera and the magpie's model are put where
    the drone and the magpie will be at the NEXT image, extrapolated from the flight loop's latest
    state -- so the drone neither jumps about in the chase view nor the magpie freezes mid-dive."""
    FPS = 20                                             # the chase camera's update rate

    def __init__(self, path, title, mover, pad, eye=None, move_bird=True):
        import cv2
        from gz.transport13 import Node
        from gz.msgs10.image_pb2 import Image
        self.cv2, self.title, self.mover, self.pad, self.eye = cv2, title, mover, np.asarray(pad, float), eye
        self.move_bird = move_bird          # False when the flight loop places the magpie (rig flights: the
                                            # cameras must see it where it IS, not where a lagging video put it)
        self.vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"avc1"), self.FPS, (1500, 600))
        assert self.vw.isOpened(), "no H.264 writer"
        self.state, self.over, self.yaw, self.placed, self.errors = None, None, None, {}, 0
        self.s_last, self.s_first, self.n, self.n_img, self.hold = None, None, 0, 0, 0
        self.q = queue.Queue()
        self.th = threading.Thread(target=self._writer, daemon=True); self.th.start()
        self.next = queue.Queue()
        threading.Thread(target=self._placer, daemon=True).start()
        self.node = Node()
        for topic in ("/chase/image", "/observer/image"):
            self.node.subscribe(Image, topic, lambda m, t=topic: self._on(t, m))

    def _on(self, topic, m):
        """Gazebo's transport thread: keep it short, the writer thread does the work."""
        a = np.frombuffer(m.data, np.uint8).reshape(m.height, m.width, -1)[..., :3]
        if topic == "/observer/image":
            self.over = a
        elif self.state is not None:
            s = m.header.stamp.sec + m.header.stamp.nsec * 1e-9
            self.q.put((s, a, self.over, self.state))
            self.next.put((s, self.state))

    def _placer(self):
        """Puts the chase camera where the drone will be at the NEXT image as soon as an image arrives -- not in
        the writer, which falls behind once the map fills (800 voxel cubes take 0.14 s to draw): a late placement
        films from a pose the overlays don't know, and the cubes slid off the crane (25 Sept)."""
        while (item := self.next.get()) is not None:
            while not self.next.empty():                 # behind? only the newest image matters
                item = self.next.get()
                if item is None:
                    return
            t_next = item[0] + 1.0 / self.FPS
            self.placed[round(t_next * self.FPS)] = self.follow(t_next, item[1])

    def follow(self, t_next, st):
        """Chase camera 7 m behind the drone along its (smoothed) heading, 3 m up; magpie model on
        its straight-line flight -- both where they will be at t_next."""
        dt = t_next - st["t_abs"]
        me, bird = st["me"] + st["v"] * dt, st["bird"] + st["bird_v"] * dt
        v, to_pad = st["v"][:2], self.pad - me[:2]
        if np.linalg.norm(v) > 1.0:                      # slow (over the pad): keep the heading
            want = math.atan2(v[1], v[0])
            self.yaw = want if self.yaw is None else \
                self.yaw + 0.15 * ((want - self.yaw + math.pi) % (2 * math.pi) - math.pi)   # ~0.3 s lag
        elif self.yaw is None:
            self.yaw = math.atan2(to_pad[1], to_pad[0])
        hd = np.array([math.cos(self.yaw), math.sin(self.yaw)])
        cam = [me[0] - 7 * hd[0], me[1] - 7 * hd[1], me[2] + 3]
        pose = (cam, self.yaw, math.atan2(3, 7))
        if Film.PADCAM:                                  # landing films: 8 m straight above it, image up = east -- no
            cam = [me[0], me[1], me[2] + 8.0]            # tree or wall between; a turning nose turns in the picture,
            pose = (cam, 0.0, math.pi / 2)               # a sideways drift shows against the pad's markers
        self.mover.put("chase_cam", cam, yaw=pose[1], pitch=pose[2])
        if self.move_bird:
            self.mover.put("magpie", [bird[0], bird[1], max(bird[2], 1.0)], yaw=st["bird_yaw"])
        return pose

    OBS = (SITE[0], SITE[1], OBS_Z)             # where the overhead camera hangs (stage-3 films move it over the crane)

    @staticmethod
    def px(xyz):
        """Overhead camera pixel of a Gazebo point: OBS_Z m up over the pad (SITE) unless moved, 90 deg, 960 px;
        image up = east (+x), image right = south (-y)."""
        ox, oy, oz = Film.OBS
        s = OBS_HALF / max(oz - xyz[2], 1.0)
        return int(OBS_HALF - (xyz[1] - oy) * s), int(OBS_HALF - (xyz[0] - ox) * s)

    @staticmethod
    def px_chase(p, cam, yaw, pitch):
        """Chase camera pixel of a Gazebo point: 960x540, 1.2 rad horizontal field of view, looking
        along yaw, pitched down by pitch; None when behind it."""
        d = np.asarray(p) - np.asarray(cam)
        cy, sy, cp, sp = math.cos(yaw), math.sin(yaw), math.cos(pitch), math.sin(pitch)
        fw = d @ [cp * cy, cp * sy, -sp]
        if fw < 0.5:
            return None
        f = 480 / math.tan(0.6)
        return int(480 - f * (d @ [-sy, cy, 0]) / fw), int(270 - f * (d @ [sp * cy, sp * sy, cp]) / fw)

    PADCAM = False                              # landing films: the observer camera stands beside the pad (fly())
    vox = None                                  # the voxel map's occupied voxels (Gazebo frame), stage-3 flights
    path = None                                 # the local 3D planner's path (Gazebo frame), stage-3 flights
    CUBE = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]) * 0.25
    FACES = [((0, 1, 3, 2), (-1, 0, 0), 0.75), ((4, 5, 7, 6), (1, 0, 0), 0.75), ((0, 1, 5, 4), (0, -1, 0), 0.6),
             ((2, 3, 7, 6), (0, 1, 0), 0.6), ((0, 2, 6, 4), (0, 0, -1), 0.45), ((1, 3, 7, 5), (0, 0, 1), 1.0)]

    @staticmethod
    def px_chase_many(p, cam, yaw, pitch):
        """px_chase for an array of points (..., 3): pixels (..., 2) int32 and in-front mask (...)."""
        d = np.asarray(p, float) - np.asarray(cam, float)
        cy, sy, cp, sp = math.cos(yaw), math.sin(yaw), math.cos(pitch), math.sin(pitch)
        fw = d @ np.array([cp * cy, cp * sy, -sp])
        f, q = 480 / math.tan(0.6), np.maximum(fw, 0.5)
        uv = np.stack([480 - f * (d @ np.array([-sy, cy, 0.0])) / q, 270 - f * (d @ np.array([sp * cy, sp * sy, cp])) / q], -1)
        return uv.astype(np.int32), fw >= 0.5

    def _cubes(self, img, vox, pose, near=70.0, most=2500):
        """Each voxel a 0.5 m cube coloured by height, faces shaded, drawn far to near (painter's order).
        Projected all at once: per-point Python took 0.14 s for 800 cubes and, with the video thread holding
        Python that long, starved the flight loop (29 -> 4 Hz, 25 Sept)."""
        cv2 = self.cv2
        cam = np.asarray(pose[0], float)
        d = np.linalg.norm(vox - cam, axis=1)
        idx = np.argsort(d)[:most]
        idx = idx[d[idx] < near][::-1]
        if not len(idx):
            return
        c = np.asarray(vox, float)[idx]
        pts, front = self.px_chase_many(c[:, None, :] + self.CUBE[None], *pose)
        keep = front.all(axis=1)
        col = cv2.applyColorMap(np.uint8(np.clip(c[:, 2] / 60.0 * 255, 0, 255)).reshape(-1, 1), cv2.COLORMAP_TURBO)[:, 0].astype(float)
        nrm = np.array([n for _, n, _ in self.FACES], float)
        vis = np.einsum("fk,nfk->nf", nrm, cam - (c[:, None, :] + nrm[None] * 0.25)) > 0
        over = img.copy()
        for i in np.flatnonzero(keep):
            for j in np.flatnonzero(vis[i]):
                f, _, sh = self.FACES[j]
                poly = pts[i, list(f)]
                cv2.fillConvexPoly(over, poly, tuple(int(v) for v in col[i] * sh))
                cv2.polylines(over, [poly], True, (25, 25, 25), 1, cv2.LINE_AA)
        cv2.addWeighted(over, 0.8, img, 0.2, 0, img)

    def _ribbon(self, img, path, pose, me, w=0.9):
        """The planned path ahead of the drone as a translucent band, like a car's lane guidance."""
        cv2 = self.cv2
        p = np.asarray(path, float)
        pts = [p[0]]
        for a, b in zip(p[:-1], p[1:]):
            pts += list(np.linspace(a, b, max(2, int(np.linalg.norm(b - a))))[1:])
        pts = np.array(pts)
        pts = pts[int(np.argmin(np.linalg.norm(pts - me, axis=1))):]              # from where the drone is now
        over = img.copy()
        for a, b in zip(pts[:-1], pts[1:]):
            dxy = b[:2] - a[:2]
            L = np.linalg.norm(dxy)
            if L < 1e-3:
                continue
            nrm = np.array([-dxy[1], dxy[0], 0.0]) / L * w
            quad = [self.px_chase(q, *pose) for q in (a + nrm, b + nrm, b - nrm, a - nrm)]
            if any(q is None for q in quad):
                continue
            cv2.fillConvexPoly(over, np.array(quad, np.int32), (255, 190, 40))
        cv2.addWeighted(over, 0.45, img, 0.55, 0, img)

    def tick(self, t_abs, t, me, v, bird, info, banner=None):
        """The flight loop's latest state (PX4 time t_abs = Gazebo's simulated time, measured)."""
        self.state = dict(t_abs=t_abs, t=t, me=np.array(me), v=np.array(v), bird=bird.pos.copy(),
                          bird_v=bird.vel.copy(), bird_yaw=bird.yaw, info=dict(info), banner=banner)

    def _writer(self):
        while (item := self.q.get()) is not None:
            stamp, chase, over, st = item
            prev = self.placed.get(round(stamp * self.FPS))   # where the camera was put for this image
            if prev is None:                             # not placed for it (the first, or the placer skipped it)
                continue
            reps = 1 if self.s_last is None else min(round((stamp - self.s_last) * self.FPS), 10)
            if reps < 1:
                continue
            self.s_first = stamp if self.s_first is None else self.s_first
            self.s_last, self.n, self.n_img, self.hold = stamp, self.n + reps, self.n_img + 1, max(self.hold, reps)
            dt = stamp - st["t_abs"]
            try:
                canvas = self._compose(st["t"] + dt, st["me"] + st["v"] * dt, st["bird"] + st["bird_v"] * dt,
                                       st["info"], st["banner"], prev, chase, over, stamp)
            except Exception:                            # one bad frame must not end the video (25 Sept: an empty
                self.errors += 1                         # voxel map killed this thread on frame 1 -- six empty films)
                if self.errors == 1:
                    import traceback; traceback.print_exc()
                continue
            for _ in range(reps):
                self.vw.write(canvas)

    def _compose(self, t_sim, drone, bird, info, banner, chase_pose, chase, over, st_t=None):
        cv2 = self.cv2
        canvas = np.full((600, 1500, 3), 30, np.uint8)
        if chase is not None:
            c = cv2.cvtColor(cv2.resize(chase, (960, 540)), cv2.COLOR_RGB2BGR).copy()
            if chase_pose is not None:
                q = self.px_chase(bird, *chase_pose)
                if q is not None and 0 <= q[0] < 960 and 0 <= q[1] < 540:
                    cv2.circle(c, q, 16, (20, 20, 20), 2); cv2.circle(c, q, 18, (255, 255, 255), 1)
                    cv2.putText(c, "magpie", (q[0] + 20, q[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            if self.vox is not None and chase_pose is not None and len(self.vox) and not Film.PADCAM:
                self._cubes(c, self.vox, chase_pose)
            if self.path is not None and chase_pose is not None and not Film.PADCAM:
                self._ribbon(c, self.path, chase_pose, drone)
            ins = self.eye.inset if self.eye is not None else None
            if ins is not None and st_t is not None and 0 <= st_t - ins[0] < 0.5:   # what the rig saw, last 0.5 s
                tt, cam, crop, score, dist, (cu, cv_, sz, span) = ins
                k = 200 / span
                im = cv2.cvtColor(cv2.resize(crop, (200, 200), interpolation=cv2.INTER_NEAREST), cv2.COLOR_RGB2BGR)
                h2 = max(sz * k / 2, 6)
                cv2.rectangle(im, (int(cu * k - h2), int(cv_ * k - h2)), (int(cu * k + h2), int(cv_ * k + h2)), (0, 0, 255), 2)
                c[8:208, 752:952] = im
                cv2.rectangle(c, (751, 7), (952, 208), (255, 255, 255), 2)
                cv2.putText(c, f"{cam} camera: magpie?", (752, 228), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(c, f"score {score:.2f}, ~{dist:.0f} m", (752, 248), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            canvas[60:600, 0:960] = c
        if over is not None and Film.PADCAM:                  # a side view: no overhead marks
            canvas[60:600, 960:1500] = cv2.resize(cv2.cvtColor(over, cv2.COLOR_RGB2BGR), (540, 540))
        elif over is not None:
            o = cv2.cvtColor(over, cv2.COLOR_RGB2BGR).copy()
            red = info.get("in_airspace")
            cv2.circle(o, self.px(drone), 14, (40, 40, 220) if red else (220, 140, 40), 3)
            cv2.circle(o, self.px(bird), 11, (20, 20, 20), 3)
            if self.vox is not None and len(self.vox):
                ox, oy, oz = Film.OBS                    # px() for every voxel at once, on the ground
                v = np.asarray(self.vox, float).reshape(-1, 3); k = OBS_HALF / max(oz, 1.0)
                q = np.stack([OBS_HALF - (v[:, 1] - oy) * k, OBS_HALF - (v[:, 0] - ox) * k], -1).astype(int)
                col = cv2.applyColorMap(np.uint8(np.clip(v[:, 2] / 60.0 * 255, 0, 255)).reshape(-1, 1), cv2.COLORMAP_TURBO)[:, 0]
                for (u, w), cc in zip(q.tolist(), col.tolist()):
                    cv2.rectangle(o, (u - 3, w - 3), (u + 3, w + 3), tuple(cc), -1)
            if self.path is not None:
                cv2.polylines(o, [np.array([self.px(p) for p in self.path], np.int32)], False, (255, 190, 40), 3, cv2.LINE_AA)
            canvas[60:600, 960:1500] = cv2.resize(o, (540, 540))
        cv2.putText(canvas, self.title, (12, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
        if Film.PADCAM:
            txt = f"t {t_sim:4.1f} s  alt {drone[2]:4.1f} m  heading {info.get('heading', 0):3.0f} deg  off centre {info.get('off_m', 0):4.2f} m"
        else:
            txt = (f"t {t_sim:5.1f} s   alt {drone[2]:4.1f} m   swoops {info['swoops']}"
                   f"{'   IN MAGPIE AIRSPACE' if info.get('in_airspace') else ''}"
                   f"{'' if info.get('tracked') is None else '   RIG: TRACKING' if info['tracked'] else '   rig: no track'}")
        (tw, _), _ = cv2.getTextSize(self.title, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
        cv2.putText(canvas, txt, (max(600, tw + 40), 38), cv2.FONT_HERSHEY_SIMPLEX, 0.62 if Film.PADCAM else 0.7,
                    (80, 80, 240) if info.get("in_airspace")
                    else (220, 220, 220), 2, cv2.LINE_AA)
        cv2.putText(canvas, "camera 8 m above the drone, looking down (image up = east)" if Film.PADCAM else "chase camera",
                    (12, 590), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1, cv2.LINE_AA)
        cv2.putText(canvas, "pad camera: 4.5 m from the touchdown point, 2 m up" if Film.PADCAM else
                    "overhead: blue/red ring = drone, black ring = magpie" if self.vox is None else
                    "cubes: voxel map from the depth camera (colour = height); band: planned path", (970, 590),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1, cv2.LINE_AA)
        if banner:
            (w, _), _ = cv2.getTextSize(banner, cv2.FONT_HERSHEY_SIMPLEX, 1.4, 3)
            cv2.rectangle(canvas, (480 - w // 2 - 12, 280), (480 + w // 2 + 12, 336), (255, 255, 255), -1)
            cv2.putText(canvas, banner, (480 - w // 2, 322), cv2.FONT_HERSHEY_SIMPLEX, 1.4,
                        (40, 150, 40) if banner == "DELIVERED" else (40, 40, 220), 3, cv2.LINE_AA)
        return canvas

    def close(self):
        self.state = None; self.q.put(None); self.th.join(); self.vw.release()


# ---- the flight --------------------------------------------------------------------------
def vox_here(vm, p):
    """The tallest voxel column within 1 m of p (map frame): what the decider's look-ahead finds under the drone."""
    ix, iy = math.floor(p[0]), math.floor(p[1])
    return max((vm.top.get((ix + a, iy + b), 0.0) for a in (-1, 0, 1) for b in (-1, 0, 1)), default=0.0)


class NoFilm:
    """Bulk runs: no video, no chase camera (the voxfast world has none)."""
    vox = path = None
    n_img, s_first, s_last, hold = 1, 0.0, 1.0, 0

    def tick(self, *a, **k):
        pass

    def close(self):
        pass


def fly(kind, seed, video, tmax=300.0, magpie="straight", sense="omni", ckpt=None, voxel=False, local3d=False, fast=False,
        short=False, full=False, geo=False):
    """stage 3: short -- a quick climb, and the flight ends 15 m past the crane ("passed"); fast -- short, and no
    video (the voxfast world has no video cameras)."""
    short = (short or fast) and not full                     # --full: fast (no video) but all the way to the pad
    """magpie: "straight" (the first bird) or "hunter" (from behind, steering); sense: "omni" (the
    decider is handed the magpie's true state within D_BIRD) or "rig" (only what the six-camera rig's
    detector and tracker make of it -- needs the rigfly world, gz_rig.py)."""
    import px4_mission as PM
    # PX4_INSTANCE=N (several SITLs side by side): PX4 sends offboard MAVLink to 14540 + N
    mi = PM.Mission(seed=seed, expect_ids=PM.PAD_IDS, endpoint=f"udpin:0.0.0.0:{14540 + int(os.environ.get('PX4_INSTANCE', 0))}")
    pad_gz = np.array([mi.pad[1], mi.pad[0]])                # PX4 NED (n, e) -> Gazebo (e, n)
    assert np.allclose(pad_gz, SITE), (pad_gz, SITE)
    trees = PM.TREES
    bird, mover = (NoBird if magpie == "none" else Hunter if magpie == "hunter" else Magpie)(seed, pad_gz, trees), Mover(asyn=True)
    stage3 = magpie == "none"                                # gz_voxel.py's world: things the survey does not show
    if stage3:
        import gz_voxel as GV
        # depth trusted to 12 m (the OAK-D measures to 19 m; a 1.6 m mast still stands clear of the noise at 12)
        vm, depth, poses, t_depth, crash_obj = GV.VoxelMap(r_max=12.0), GV.Depth() if voxel else None, collections.deque(maxlen=90), -1.0, None
        off3, flown, err = np.array([*OFFSET, 0.0]), [], []
        cx, cy = GV.crane_at(seed)                       # this flight's crane: the world file's, moved
        mover.put("crane", (cx - GV.CRANE0[0], cy - GV.CRANE0[1], 0.0))
        if short and not fast:                           # films: the overhead camera over the crane, high enough
            Film.OBS = (cx + 5.0, 0.0, 130.0)            # to hold the drone at 50 m and the crane together
            mover.put("observer_cam", Film.OBS, pitch=math.pi / 2)
    if os.environ.get("FILM_LANDING") and not fast:      # landing films: that camera on a 2 m stand 4.5 m WSW of the
        Film.PADCAM = True                               # touchdown point, level (a clear sight line in the LiDAR map;
        a = math.radians(210.0)                          # the old side-to-side wobble shows left-right); the last ~6 m
        mover.put("observer_cam", (pad_gz[0] + 4.5 * math.cos(a), pad_gz[1] + 4.5 * math.sin(a), 2.0), yaw=a - math.pi)
    eye = RigEye(f"{HERE}/weather_runs/rig_perceive_{kind}_{seed}.log") if sense == "rig" else None
    cr = None
    if geo:                           # geo/geofly.sh: GeoAnchor's GNSS-free cruise first; its film (geo/geo_film.py)
        sys.path[:0] = [f"{HERE}/geo", f"{HERE}/wind"]
        import geo_cruise as GC
        from geo_film import GeoFilm
        from px4_steps import TruthLog
        who = ("learner (trained with birds)" if kind == "learner" else kind) + (f", magpie seen by rig {os.environ.get('RIG', 'v2')}" + (" (up camera)" if os.environ.get("RIG") == "v2l" else "")
                                                     if sense == "rig" else "")
        who += (", local 3D planner on the voxel map" if local3d else "") if stage3 else ""
        gfilm = None if fast else GeoFilm(video, f"GeoAnchor + {who}" + (", unmapped obstacles" if stage3 else ", magpie"),
                                          pad_gz, [(0, 0), *GC.ROUTE, tuple(pad_gz)], eye=mi.eye, rig=eye)
        cr = GC.Cruise(mi, TruthLog(), GC.Bridge(), gfilm)
    film = NoFilm() if fast or geo else Film(video, ("Learner" + ("" if stage3 else " (trained with birds)") if kind == "learner" else kind.capitalize())
                + (", 360-degree camera rig" if sense == "rig" else "")
                + ((", voxel map + local 3D planner" if local3d else ", voxel map from the depth camera" if voxel
                    else ", the 2023 survey only") if stage3 else "")
                + (("  |  NEW landing" if LAND_FIX else "  |  OLD landing") if "LAND_FIX" in os.environ else ""),
                Mover(asyn=True), pad_gz, eye, move_bird=eye is None and magpie != "none")
    log = open(os.path.splitext(video)[0] + ".csv", "w")
    log.write("t,x,y,z,vx,vy,vz,roll,pitch,yaw,bird_x,bird_y,bird_z,bird_dive,seen,locked,cmd_x,cmd_y,cmd_z,in_airspace,"
              "rig_seen,rig_x,rig_y,rig_z,gz_x,gz_y,gz_z,mk_ex,mk_ey,mk_id,cam_tilt,cq_w,cq_x,cq_y,cq_z,vox_here,landing\n")
    n_iter, hd = 0, (pad_gz - np.zeros(2)) / np.linalg.norm(pad_gz)
    dec = Decider(kind, f"{HERE}/weather_runs/decider_{kind}_{seed}.log", ckpt, voxel, local3d)
    truth, gz_truth = [], Truth()                            # (t_abs, drone, bird) for scoring the rig afterwards
    if magpie != "none":                                     # stage 3's worlds have no magpie to move
        mover.put("magpie_nest", bird.nest - [0, 0, 0.2]); mover.put("magpie", bird.pos)

    mi.return_to_depot(**({"v_home": 10.0, "v_down": 4.0} if short else {}),   # short flights: back quickly;
                       z_home=70.0 if stage3 else None,   # over the 58 m crane: at 50 m the way home hit its mast
                       tmax=200 if stage3 else 120)       # and the next flight could not arm (tipped over, 25 Sept)
    if not mi.arm_offboard():
        time.sleep(3.0)
        assert mi.arm_offboard(), "arming failed"
    if short:                             # climb at 6 m/s (PX4 caps offboard climb at MPC_Z_VEL_MAX_UP, 3 by default)
        mi.m.mav.param_set_send(mi.m.target_system, mi.m.target_component, b"MPC_Z_VEL_MAX_UP", 6.0,
                                mi.mavutil.mavlink.MAV_PARAM_TYPE_REAL32)
    t_up, w_up = mi.t_boot, time.time()
    if cr is not None:                    # 75 m, GNSS denied over the suburb, down to 30 m over the pad: the decider
        cr.fly(goal_out=float(os.environ.get("GEO_GOAL_OUT", 0.0)))      # takes it from there -- over the pad, or
                                                                         # (stage 3) short of it at 75 m: the planner
                                                                         # takes it down, past the crane
        if os.environ.get("GEO_LAND_SPEED"):          # the rig's detector needs the slow sim (rigfly.sh: 0.1)
            subprocess.run(["gz", "service", "-s", f"/world/{WORLD}/set_physics", "--reqtype", "gz.msgs.Physics",
                            "--reptype", "gz.msgs.Boolean", "--timeout", "5000",
                            "--req", f"real_time_factor: {os.environ['GEO_LAND_SPEED']}"], capture_output=True)
    while cr is None:                     # straight up to 50 m: the fast simulator's flights start at cruise
        p = mi.pos()
        if p is not None and p[2] > PM.CRUISE_Z - 0.5:
            break
        if mi.t_boot - t_up > 60.0 or time.time() - w_up > 300.0:   # never armed after a failsafe landing (25 Sept:
            raise RuntimeError(f"climb stalled at {'?' if p is None else f'{p[2]:.1f}'} m")   # two flights waited an hour)
        mi.send(0, 0, 6.0 if short else 3.0); time.sleep(0.05)

    t0, t_dec, cmd, t_yaw, landing, yaw_locked = mi.t_boot, -1.0, np.zeros(3), 0.0, False, False
    ever, t_seen, mi.mark_ned, prev = False, -1e9, None, None
    outcome, d_bird, banner_t = "timeout", 1e9, None
    while True:
        p = mi.pos()
        if p is None:
            continue
        t = mi.t_boot - t0
        me = np.array([p[1], p[0], p[2]])                    # Gazebo east, north, up: PX4's ESTIMATE
        g = gz_truth.at(mi.t_boot)                           # where Gazebo really has it
        g = np.full(3, np.nan) if g is None else g
        v = np.array([mi.vel_ne[1], mi.vel_ne[0], mi.climb])
        # camera scheduling: the landing camera's marker search only where a marker can matter -- within 60 m of the
        # pad, or once one has been read (stage 3; the magpie flights keep their validated always-on loop)
        if stage3 and not ever and math.hypot(*(me[:2] - pad_gz)) > 60.0:
            seen, off, ids, mid = False, None, None, None
        else:
            seen, off, ids, mid = mi.eye.look()
        if seen:                                             # px4_mission's marker projection, smoothed
            ray = PM.cam_ray(off, mi.roll, mi.pitch, mi.yaw, mi.gimbal, mi.eye.look_stamp)
            raw = p[:2] + ray[:2] * (p[2] / max(ray[2], 1e-3)) - PM.PAD_OFF_NED[mid]
            mi.mark_ned = raw if mi.mark_ned is None else mi.mark_ned + PM.MARK_ALPHA * (raw - mi.mark_ned)
            ever, t_seen = True, t
            if cr is not None:
                cr.marker_fix(p, ray, mid)                   # ...and to PX4: no GNSS below GeoAnchor's 50 m
        if np.linalg.norm(v[:2]) > 0.5:                      # which way it flies: where "behind" is
            hd = v[:2] / np.linalg.norm(v[:2])
        if eye is not None:
            eye.pose = (me, enu_rpy(mi.roll, mi.pitch, mi.yaw))
            mover.put("magpie", bird.pos + bird.vel * 0.02, yaw=bird.yaw)   # where the rig must see it: now
        truth.append((mi.t_boot, me.copy(), bird.pos.copy()))
        if stage3:
            # the depth camera's newest image, placed with PX4's OWN pose estimate at its stamp (its clock is
            # Gazebo's, measured), in the training map's frame: the decider reads it around that same estimate
            poses.append((mi.t_boot, me + off3, GV.rot(*enu_rpy(mi.roll, mi.pitch, mi.yaw))))
            if depth is not None and depth.last is not None and depth.last[0] > t_depth:
                t_depth, m = depth.last
                ts, pos_m, R_m = min(poses, key=lambda q: abs(q[0] - t_depth))
                if abs(ts - t_depth) < 0.1:
                    vm.insert(np.frombuffer(m.data, np.float32).reshape(m.height, m.width), pos_m, R_m, cam_off=GV.DEPTH_OFF)
                    film.vox = vm.occupied() - off3
            flown.append(me if np.isnan(g[0]) else g)        # judged where Gazebo has it, not on PX4's estimate
            err.append(np.linalg.norm((g - me)[:2]))
            hit, dist = GV.nearest(flown[-1], seed=seed)
            if dist < R_HIT and outcome == "timeout":
                outcome, banner_t, crash_obj = "crashed", t, hit
            if short and outcome == "timeout" and me[0] > cx + 15.0:    # short flights: the crane is behind it
                outcome, banner_t = "passed", t - 3.0
        if prev is not None and outcome == "timeout":
            d = bird.step(t - prev[0], prev[1], prev[2], me, hd) if magpie == "hunter" else \
                bird.step(t - prev[0], prev[1], prev[2], me)
            d_bird = min(d_bird, d)
            if d < HIT_BIRD:
                outcome, banner_t = "struck", t
        if not dec.pending and t - t_dec >= DT - 1e-3 and outcome == "timeout":
            mapxy = me[:2] + OFFSET
            fix = None if mi.mark_ned is None else np.array([mi.mark_ned[1], mi.mark_ned[0]]) + OFFSET
            vis = (fix if fix is not None else np.array(PAD_MAP)) - mapxy
            if eye is None:                                  # handed the truth within D_BIRD
                rel_b, vel_b = bird.pos - me, bird.vel
                seen_b = bool(np.linalg.norm(rel_b) <= D_BIRD)
            else:                                            # only what the rig's tracker believes
                bel = eye.belief or {"seen": False, "rel": [0.0, 0.0, 0.0], "vel": [0.0, 0.0, 0.0]}
                rel_b, vel_b, seen_b = np.array(bel["rel"]), np.array(bel["vel"]), bool(bel["seen"])
            obs = dict(t=t, pos_z=me[2], vel=v.tolist(), bel_xy=mapxy.tolist(), pad_xy=list(PAD_MAP),
                       pad_rel_bel=(np.array(PAD_MAP) - mapxy).tolist(), pad_rel_vis=vis.tolist(),
                       pad_rel_fix=(vis if ever else np.array(PAD_MAP) - mapxy).tolist(),
                       seen=bool(seen), ever_seen=ever, track=(t - t_seen) < LOST_S, hls_rel_bel=(-mapxy).tolist(),
                       bird_rel=rel_b.tolist(), bird_vel=vel_b.tolist(), bird_seen=seen_b)
            if stage3:
                obs["vox"] = vm.take()
            dec.send(obs); t_dec = t
        if dec.ready():                                      # the answer, whenever it comes (see Decider.send)
            cmd = dec.recv()
            if stage3:
                film.path = None if dec.path is None else np.asarray(dec.path) - off3
        if outcome == "timeout":
            # THE LANDING (Sai, 25 Sept: "face one direction only while landing ... important to not wobble"). Once its
            # own pad marker has been read the heading stays put, all the way down -- following the centring commands it
            # had turned ~3700 deg per descent (up to 200 deg/s); at >2 m/s only, still 75-140 deg, most of it as the
            # local planner's path ran out at the pad (65 deg in 2 s, 5-15 m out). And while the decider takes it down
            # within 3 m of the pad, the centring is px4_mission's lag-aware law (aim at d - v*0.69 s, linear near the
            # target): the rule's
            # sqrt(2ad) speed -- 0.8 m/s for 0.25 m of error, the airframe answering 0.7 s late -- swung it +-0.3 m
            # every 1.5 s. The descent rate, and whether to land at all, stay the decider's.
            d_land = None if not ever or mi.mark_ned is None else mi.mark_ned - p[:2]     # to the touchdown point, NED
            terminal = LAND_FIX and d_land is not None       # locked on its own pad: heading fixed from here on
            # COMMIT BELOW 5 m (Sai, 27 Sept): locked on its pad and under 5 m, it lands -- a constant V_COMMIT down,
            # centring on the way, the decider no longer asked (the magpie learner hovered at 2-5 m, dodging "magpies"
            # the rig saw in tree crowns)
            commit = terminal and p[2] < 5.0 and float(np.linalg.norm(d_land)) < 3.0
            landing = commit or (terminal and float(np.linalg.norm(d_land)) < 3.0 and cmd[2] < -0.1)
            ahead = _ahead(me, film.path) if stage3 and local3d and film.path is not None else None
            want = None
            # Elsewhere the nose (and cameras) turn the way it flies -- as the fast simulator assumes; PX4 had held its
            # spawn yaw and flown east tail first (25 Sept) -- or along the local planner's path, which keeps a mast in
            # the depth camera's view while braking or sidestepping. Only above 2 m/s, at most YAW_RATE: the planned
            # path's own turns had whipped the nose at 90-234 deg/s. Magpie flights too since rig v2 (its long front
            # camera); before, they held the take-off heading. PX4 yaw: 0 = north.
            if terminal:                                     # heading locked for the landing -- where the nose IS when
                if not yaw_locked:                           # the marker is first read, not the heading PX4 was still
                    mi.yaw_hold, yaw_locked = mi.yaw, True   # turning toward (it drifted another 18-58 deg after the lock)
            elif ahead is not None:
                want = math.atan2(ahead[0], ahead[1])
            elif math.hypot(cmd[0], cmd[1]) > 2.0:
                want = math.atan2(cmd[0], cmd[1])
            if want is not None:                             # ...and the nose winds down its turning on the way in, from
                dy = (want - mi.yaw_hold + math.pi) % (2 * math.pi) - math.pi   # 90 deg/s at 60 m from the pad to none at
                rate = YAW_RATE * float(np.clip((np.linalg.norm(pad_gz - me[:2]) - 35.0) / 25.0, 0.0, 1.0))   # 35 m: a
                mi.yaw_hold += float(np.clip(dy, -rate * (t - t_yaw), rate * (t - t_yaw)))   # lock that caught a 75 deg/s
                                                             # turn overshot 41 deg and swung back; freezing where the turn
                                                             # would stop (yaw + 0.35 s x rate) was worse on average
            t_yaw = t
            if landing:
                v_land = mi._approach(d_land, PM.V_CENTRE)   # north, east
                # Below 6 m it only keeps going down while centred (0.3 m below 3 m, +0.1 m per metre above) and
                # nearly still (< 0.3 m/s sideways); otherwise it holds its height and centres first -- as a precision
                # landing does. Live magpie flights (26 Sept): a swoop's dodge at 4-6 m left it 1-1.5 m off, the
                # descent went on, it touched down moving 0.3-0.5 m/s sideways and slid 0.9-1.3 m on the ground.
                # The last 2 m only below 0.1 m/s: the planner went down at 0.26 m/s and slid 0.4-0.7 m after touchdown,
                # the learner at 0.03-0.04 m/s and slid 0.1-0.2 m (26 Sept, magpie flights).
                vz, tol = cmd[2], 0.3 + 0.1 * max(p[2] - 3.0, 0.0)
                v_ok = 0.1 if p[2] < 2.0 else 0.3
                if p[2] < 6.0 and (float(np.linalg.norm(d_land)) > tol or float(np.linalg.norm(mi.vel_ne)) > v_ok):
                    vz = max(vz, 0.0)
                if commit:
                    vz = -V_COMMIT
                mi.send(float(v_land[0]), float(v_land[1]), vz)
            else:
                mi.send(cmd[1], cmd[0], cmd[2])
            mi.gimbal_down()                                 # rig v2: keeps Gazebo's gimbal bridge tracking PX4
        else:
            mi.send(0, 0, 0)
        inside = me[2] < Z_TERR and math.hypot(*(me[:2] - bird.nest[:2])) < R_TERR
        if p[2] < PM.TOUCHDOWN_ALT and outcome == "timeout":
            outcome, banner_t = "landed", t
        banner = None if banner_t is None else ("DELIVERED" if outcome == "landed" and
                                                np.linalg.norm(me[:2] - pad_gz) < 1.0 else
                                                "STRUCK BY THE MAGPIE" if outcome == "struck" else
                                                f"CRASHED INTO THE {crash_obj.split('_')[0].upper()}" if outcome == "crashed" else
                                                "LANDED OFF THE PAD")
        bel = eye.belief if eye is not None else None
        film.tick(mi.t_boot, t, me, v, bird, {"swoops": bird.swoops, "in_airspace": inside,
                                              "tracked": None if eye is None else bool(bel and bel["seen"]),
                                              "heading": math.degrees(mi.yaw) % 360,
                                              "off_m": float(np.linalg.norm((me if np.isnan(g[0]) else g)[:2] - pad_gz))}, banner)
        if cr is not None:
            if cr.film is not None:
                cr.film.vox, cr.film.path = film.vox, film.path          # the voxel map and the planner's path
            cr.observe(("local 3D planner" if local3d else f"landing: {kind}") if stage3 else f"landing: {kind}", p, banner=banner,
                       info=({"magpie": f"{bird.swoops} swoops, closest {d_bird:.1f} m", "bird": bird.pos[:2].copy(),
                              "rig": "tracking the magpie" if bel and bel["seen"] else "no magpie in view"}
                             if magpie != "none" else {"obstacles": "crane, lights, traffic light, trees, shed",
                                                       "voxels": len(film.vox) if film.vox is not None else 0}))
        if (banner_t is not None and t - banner_t > 3.0) or t > tmax:
            break
        est = (me + np.array(bel["rel"])) if bel and bel["seen"] else np.full(3, np.nan)
        log.write(",".join(f"{x:.3f}" for x in (t, *me, *v, mi.roll, mi.pitch, mi.yaw, *bird.pos, bird.dive, seen, ever,
                                                     *cmd, inside, bool(bel and bel["seen"]), *est, *g,
                                                     *(off if seen else (np.nan, np.nan)), mid if seen else -1,
                                                     *((np.nan,) * 5 if mi.gimbal is None or mi.gimbal.q is None
                                                       else (lambda q: (mi.gimbal.tilt_deg(q), *q))(
                                                           mi.gimbal.q_at(getattr(mi.eye, "look_stamp", None)))),
                                                     vox_here(vm, me + off3) if stage3 else np.nan, landing)) + "\n")
        n_iter += 1
        prev = (t, me, v)
        time.sleep(0.01)
    miss_est = float(np.linalg.norm(me[:2] - pad_gz))       # by PX4's own estimate (how runs before 25 Sept scored it)
    miss = miss_est if np.isnan(g[0]) else float(np.linalg.norm(g[:2] - pad_gz))   # where Gazebo has it
    if short and outcome == "passed":                        # left in the air: hand PX4 its own position hold, so the
        mi.m.set_mode("LOITER")                              # gap before the next flight is not an offboard failsafe
    mi.disarm(); dec.close(); film.close(); mi.close(); log.close()
    if cr is not None:
        if cr.film is not None:
            cr.film.close()
        with cr.truth.lock:
            trows = list(cr.truth.rows)
        geo_rep = GC.report(cr.rows, cr.bridge.fixes, cr.t_denied, trows)
        geo_rep.update(gnss_restored_at=cr.t_restored, marker_fixes=cr.mk.n if cr.mk else 0,
                       marker_skipped=cr.mk.skipped if cr.mk else 0)
        json.dump(dict(report=geo_rep, rows=cr.rows, fixes=cr.bridge.fixes), open(os.path.splitext(video)[0] + "_geo.json", "w"))
        print("geo", json.dumps(geo_rep), flush=True)
    rig = None
    if eye is not None:
        eye.close()
        rig = rig_score(eye.log, truth)
        with open(os.path.splitext(video)[0] + "_rig.jsonl", "w") as f:     # every frame set's detections + belief
            tt = np.array([x[0] for x in truth])
            for t_, out, pose in eye.log:
                _, me_, bp_ = truth[int(np.clip(np.searchsorted(tt, t_), 0, len(tt) - 1))]
                f.write(json.dumps(dict(t=t_, out=out, pose=[list(map(float, pose[0])), list(map(float, pose[1]))],
                                        drone=me_.tolist(), bird=bp_.tolist())) + "\n")
    r = dict(decider=kind, seed=seed, magpie=magpie, sense=sense, rig=rig,
             outcome=outcome if outcome != "landed" or miss < 1.0 else "landed_off_pad",
             miss_m=miss, miss_est_m=miss_est, swoops=bird.swoops, closest_bird_m=d_bird, t_s=t, nest=bird.nest.tolist(),
             loop_hz_sim=n_iter / max(t, 1e-6), video_new_per_s=(film.n_img - 1) / max((film.s_last or 0) - (film.s_first or 0), 1e-6),
             video_worst_hold=film.hold, video_frames=getattr(film, "n", 0), video_errors=getattr(film, "errors", 0))
    if stage3:
        path, e = np.array(flown), np.array(err)
        r.update(voxel=voxel, local3d=local3d, crane=[cx, cy], crashed_into=crash_obj, voxels=int(len(vm.occupied())),
                 est_err_m=[round(float(np.nanmedian(e)), 2), round(float(np.nanmax(e)), 2)],   # PX4 estimate vs Gazebo
                 closest_m={n: round(float(GV.nearest(path, only=n, seed=seed)[1].min()), 2) for n, _ in GV.objects(seed)})
    print(json.dumps(r), flush=True)
    return r


def bird_sdf():
    """The magpie (0.4 m long, 0.8 m wings, black with white wing patches) and its nest, as SDF.
    No shadows: the rig bench labels the bird by differencing renders with and without it, and its
    shadow on roofs and lawns came out as part of the 'bird', hundreds of pixels away (25 Sept)."""
    part = lambda name, geom, rgb, pose="0 0 0 0 0 0": (
        f"<visual name='{name}'><pose>{pose}</pose><cast_shadows>false</cast_shadows><geometry>{geom}</geometry>"
        f"<material><ambient>{rgb} 1</ambient><diffuse>{rgb} 1</diffuse></material></visual>")
    magpie = ("<model name='magpie'><static>true</static><link name='link'>"
              + part("body", "<ellipsoid><radii>0.2 0.06 0.06</radii></ellipsoid>", "0.05 0.05 0.05")
              + part("head", "<sphere><radius>0.05</radius></sphere>", "0.05 0.05 0.05", "0.2 0 0.03 0 0 0")
              + part("wings", "<box><size>0.14 0.8 0.015</size></box>", "0.05 0.05 0.05")
              + part("patches", "<box><size>0.08 0.5 0.017</size></box>", "0.95 0.95 0.95", "-0.02 0 0 0 0 0")
              + part("tail", "<box><size>0.16 0.08 0.01</size></box>", "0.05 0.05 0.05", "-0.26 0 0 0 0 0")
              + "</link></model>")
    nest = ("<model name='magpie_nest'><static>true</static><link name='link'>"
            + part("n", "<cylinder><radius>0.35</radius><length>0.2</length></cylinder>", "0.35 0.22 0.1")
            + "</link></model>")
    return magpie, nest


def chase_sdf():
    """The video's chase camera, moved by set_pose (Film.follow)."""
    return ("<model name='chase_cam'><static>true</static><pose>-14 0 57 0 0.46 0</pose><link name='link'>"
            "<sensor name='chase' type='camera'><topic>/chase/image</topic><update_rate>20</update_rate>"
            "<camera><horizontal_fov>1.2</horizontal_fov><image><width>960</width><height>540</height>"
            "<format>R8G8B8</format></image><clip><near>0.1</near><far>400</far></clip></camera>"
            "<always_on>1</always_on></sensor></link></model>")


def setup():
    """The Logan world plus a magpie, its nest and a chase camera, all moved by set_pose."""
    src = open(f"{HERE}/gz_logan/worlds/aruco.sdf").read()
    assert src.count("  </world>") == 1 and "<plugin" not in src
    magpie, nest = bird_sdf()
    chase = chase_sdf()
    os.makedirs(f"{HERE}/gz_logan/birds", exist_ok=True)
    open(f"{HERE}/gz_logan/birds/aruco.sdf", "w").write(src.replace("  </world>", f"    {magpie}\n    {nest}\n    {chase}\n  </world>"))
    print(f"{HERE}/gz_logan/birds/aruco.sdf")


def compose(a, b, out):
    """Two flights side by side, in step (both recorded in simulated time)."""
    import cv2
    ca, cb = cv2.VideoCapture(a), cv2.VideoCapture(b)
    vw = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"avc1"), Film.FPS, (1500, 1200))
    last_a = last_b = None
    while True:
        oa, fa = ca.read(); ob, fb = cb.read()
        if not oa and not ob:
            break
        last_a, last_b = (fa if oa else last_a), (fb if ob else last_b)
        vw.write(np.vstack([last_a, last_b]))
    vw.release(); print(out)


def final(out, folder=f"{HERE}/gazebo_birds"):
    """Every recorded flight, planner above learner, with a card of results before each."""
    import cv2
    rs = [json.loads(l) for l in open(f"{folder}/flights.jsonl")]
    by = {(r["decider"], r["seed"]): r for r in rs}
    seeds = sorted({s for (_, s) in by if ("planner", s) in by and ("learner", s) in by})
    W, H, F = 1500, 1200, cv2.FONT_HERSHEY_SIMPLEX
    vw = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"avc1"), Film.FPS, (W, H))

    def card(lines, secs=4.0):
        img = np.full((H, W, 3), (31, 24, 20), np.uint8); y = H // 2 - 40 * len(lines)
        for text, scale, col in lines:
            (tw, _), _ = cv2.getTextSize(text, F, scale, 2)
            cv2.putText(img, text, ((W - tw) // 2, y), F, scale, col, 2, cv2.LINE_AA); y += int(80 * scale)
        for _ in range(int(secs * Film.FPS)):
            vw.write(img)
    line = lambda r: (f"{r['decider']}: {r['outcome'].replace('_', ' ')} {r['miss_m']:.2f} m from centre, {r['swoops']} swoops, "
                      f"closest pass {r['closest_bird_m']:.1f} m")
    landed = sum(r["outcome"] == "landed" for r in rs); struck = sum(r["outcome"] == "struck" for r in rs)
    rig = any(r.get("sense") == "rig" for r in rs)          # the magpie known only through the drone's cameras
    card([("The magpie in Gazebo", 1.5, (255, 255, 255)),
          ("the fast simulator's planner and learner flying PX4, unchanged", 0.9, (210, 210, 210)),
          ("same nest and dives for both: planner above, learner below", 0.9, (210, 210, 210)),
          ("the drone knows the magpie ONLY through its cameras: OAK-D ahead, wide left/right/back, one looking up" if rig else
           "assumption: the drone knows the magpie within 40 m in every direction (a 360-degree sensor)", 0.72, (120, 200, 255)),
          *([("YOLO finds it in each picture, a tracker turns sightings into position and speed", 0.72, (120, 200, 255))] if rig else [])], 6)
    for k, s in enumerate(seeds):
        p, l = by[("planner", s)], by[("learner", s)]
        card([(f"Flight {k + 1} of {len(seeds)}", 1.3, (255, 255, 255)),
              (f"nest {math.dist(p['nest'][:2], SITE):.0f} m from the pad, {p['nest'][2]:.0f} m up a real Logan tree", 0.85, (210, 210, 210)),
              (line(p), 0.8, (180, 220, 180)), (line(l), 0.8, (180, 220, 180))])
        ca, cb = cv2.VideoCapture(f"{folder}/planner_s{s}.mp4"), cv2.VideoCapture(f"{folder}/learner_s{s}.mp4")
        fa = fb = None
        while True:
            oa, a = ca.read(); ob, b = cb.read()
            if not oa and not ob:
                break
            fa, fb = (a if oa else fa), (b if ob else fb)
            vw.write(np.vstack([fa, fb]))
    card([(f"{landed} of {len(rs)} landed, {struck} magpie strikes", 1.3, (255, 255, 255)),
          ("the decisions carry over to PX4 unchanged", 0.85, (210, 210, 210)),
          ("three flights each: too few to rank planner and learner -- that needs many more seeds" if rig else
           "not shown here: people (none in this Gazebo world) and the drone's real camera coverage", 0.75, (120, 200, 255))], 6)
    vw.release(); print(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--setup", action="store_true")
    ap.add_argument("--decider", choices=("planner", "learner", "rule"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--video", default="gz_birds.mp4")
    ap.add_argument("--compose", nargs=3, metavar=("A", "B", "OUT"))
    ap.add_argument("--final", metavar="OUT")
    ap.add_argument("--magpie", choices=("straight", "hunter", "none"), default="straight")
    ap.add_argument("--voxel", action="store_true", help="stage 3 (--magpie none, gz_voxel.py's world): the depth camera's voxel map")
    ap.add_argument("--local3d", action="store_true", help="stage 3: the rule flies a local 3D planner's path on that map")
    ap.add_argument("--fast", action="store_true", help="stage 3 bulk runs (voxfast world): no video, quick climb, end past the crane")
    ap.add_argument("--short", action="store_true", help="stage 3: quick climb, end past the crane (with video)")
    ap.add_argument("--full", action="store_true", help="stage 3 with --fast: no video, but the whole delivery (landing)")
    ap.add_argument("--sense", choices=("omni", "rig"), default="omni")
    ap.add_argument("--ckpt", default=None, help="learner checkpoint (urban/), default decider_server's")
    ap.add_argument("--geo", action="store_true", help="geo/geofly.sh: GeoAnchor's GNSS-free cruise first (geo/geo_cruise.py)")
    ap.add_argument("--mover", action="store_true", help=argparse.SUPPRESS)          # Mover(asyn=True)'s helper
    a = ap.parse_args()
    if a.mover:
        mover_helper(); os._exit(0)
    if a.setup:
        setup()
    elif a.compose:
        compose(*a.compose)
    elif a.final:
        final(a.final, os.path.dirname(os.path.abspath(a.final)))   # the flights sit beside it
    else:
        fly(a.decider, a.seed, a.video, magpie=a.magpie, sense=a.sense, ckpt=a.ckpt, voxel=a.voxel, local3d=a.local3d, fast=a.fast,
            short=a.short, full=a.full, geo=a.geo)
        os._exit(0)                       # gz transport's exit-time segfault is harmless; skip it
