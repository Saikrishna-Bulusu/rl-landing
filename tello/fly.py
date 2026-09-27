#!/usr/bin/env python3
"""
fly.py -- the cage delivery on the real Tello. The deciders the simulator scores (cage.py's rule and
planner, train.py's learner) fly Sai's DJI Tello over the Tello SDK -- plain UDP, nothing to install --
seeing the floor through the 45-degree mirror (bottom of the picture, flipped left-right) and a pole the
map lacks through the top of the picture.

    python fly.py --check-camera             video only, motors off: marker IDs + offsets, pole bearing/range
    python fly.py --check-axes               short hop: which way vgx/vgy and rc left/right really point
    python fly.py --sim [--decider planner]  the whole loop against the simulator instead of the drone
    python fly.py --decider planner          fly (Sai in the cage; ESC or Ctrl-C lands)
    python fly.py --decider learner --ckpt cage_rl_s0_greedy.pt

layout.json holds the cage as measured (metres; x along the 3 m side away from the hub, y along the 2 m side,
origin the corner left-behind the hub): the pads -- the customer's FIRST -- with their ArUco IDs, the houses
and trees. The pole is not in it: the map lacks it on purpose. See layout_example.json.

SAFETY: ESC / Ctrl-C lands now. Also lands if: the believed position leaves the cage minus 25 cm, battery
< 25 %, no video for 1 s, or 90 s in the air. The Tello itself lands 15 s after its last command.
Every [knob] must be checked in the cage before a delivery flight (--check-camera, --check-axes).
"""
import argparse
import json
import math
import os
import socket
import sys
import threading
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cage as C

# ---- [knobs] ------------------------------------------------------------------------------------------
MIRROR_FRAC = 0.40            # bottom share of the picture that the mirror turns downward
ROW_SIGN, COL_SIGN = +1, +1   # mirror picture (after the left-right flip): +row = +x (ahead)? +col = -y (right)?
VGX_SIGN, VGY_SIGN = +1, +1   # Tello state vgx/vgy (dm/s) -> +x ahead, +y left (check with --check-axes)
RC_PER_MPS = 100.0            # rc units per m/s (rc 30 ~ 0.3 m/s?): time a 2 s hop against a tape measure
POLE_D = 0.08                 # the unmapped pole's diameter (m)
POLE_HSV = ((0, 120, 70), (10, 255, 255), (170, 120, 70), (180, 255, 255))   # red, both ends of the hue circle
F_FWD = 480 / C.FWD_TAN       # forward picture focal length (px) from the 70-degree width
ARUCO = "DICT_4X4_50"
TELLO_IP, CMD_PORT, STATE_PORT, VIDEO = "192.168.10.1", 8889, 8890, "udp://@0.0.0.0:11111"


def load_layout(path):
    """The measured cage as a one-flight Cage: the deciders read pads, houses, trees from it as in simulation."""
    lay = json.load(open(path))
    w = C.Cage(1, dev="cpu"); w.reset()
    t = lambda k, m: torch.tensor(lay[k], dtype=torch.float32).reshape(1, -1, m)
    w.pads = t("pads", 3)[..., :2]
    w.ids = [int(p[2]) for p in lay["pads"]]
    w.houses, w.trees = t("houses", 5), t("trees", 4)
    w.pole_on[:] = True                      # a pole the deciders are told about (seen) counts as an obstacle
    return w, lay


# ---- seeing -------------------------------------------------------------------------------------------
class Eyes:
    def __init__(self, marker_m, ids):
        import cv2
        self.cv2, self.m, self.ids = cv2, marker_m, ids
        self.det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, ARUCO)),
                                           cv2.aruco.DetectorParameters())

    def markers(self, frame):
        """Mirror strip -> {aruco id: (dx, dy)} offset of each whole marker from the drone (m, cage axes)."""
        h = frame.shape[0]
        down = frame[int(h * (1 - MIRROR_FRAC)):, ::-1]                  # the mirror: flip left-right (Sai, 20 Sept)
        corners, ids, _ = self.det.detectMarkers(self.cv2.cvtColor(down, self.cv2.COLOR_BGR2GRAY))
        out = {}
        for c, i in zip(corners, [] if ids is None else ids.flatten()):
            c = c.reshape(4, 2)
            side = np.mean(np.linalg.norm(c - np.roll(c, 1, 0), axis=1))
            s = self.m / side                                              # metres per pixel, from the marker itself
            u, v = c.mean(0)
            out[int(i)] = (ROW_SIGN * (v - down.shape[0] / 2) * s, COL_SIGN * -(u - down.shape[1] / 2) * s)
        return out

    def pole(self, frame):
        """Top of the picture -> (range m, bearing rad, + = left) of the largest red blob, or None."""
        cv2 = self.cv2
        top = frame[: int(frame.shape[0] * (1 - MIRROR_FRAC))]
        hsv = cv2.cvtColor(top, cv2.COLOR_BGR2HSV)
        m = cv2.inRange(hsv, *POLE_HSV[:2]) | cv2.inRange(hsv, *POLE_HSV[2:])
        n, lab, st, cen = cv2.connectedComponentsWithStats(m)
        if n < 2:
            return None
        k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
        if st[k, cv2.CC_STAT_AREA] < 150:
            return None
        w = st[k, cv2.CC_STAT_WIDTH]
        return POLE_D * F_FWD / max(w, 1), math.atan((top.shape[1] / 2 - cen[k][0]) / F_FWD)


# ---- the drone ----------------------------------------------------------------------------------------
class Tello:
    """Tello SDK over UDP: commands answered "ok", state pushed ~10 Hz, H.264 video to port 11111."""

    def __init__(self):
        import cv2
        self.s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); self.s.bind(("", CMD_PORT)); self.s.settimeout(10)
        self.st = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); self.st.bind(("", STATE_PORT)); self.st.settimeout(1)
        self.state, self.frame, self.t_frame, self.alive = {}, None, 0.0, True
        threading.Thread(target=self._state, daemon=True).start()
        assert self.send("command") == "ok", "no answer: is this laptop on the Tello's Wi-Fi?"
        self.send("streamon")
        self.cap = cv2.VideoCapture(VIDEO, cv2.CAP_FFMPEG)
        threading.Thread(target=self._video, daemon=True).start()

    def send(self, cmd):
        self.s.sendto(cmd.encode(), (TELLO_IP, CMD_PORT))
        try:
            return self.s.recv(1024).decode(errors="ignore").strip()
        except socket.timeout:
            return "timeout"

    def rc(self, vx, vy, vz):
        """Velocity in cage axes (the Tello faces +x): rc a = right, b = forward, c = up."""
        c = lambda x: int(np.clip(round(x * RC_PER_MPS), -100, 100))
        self.s.sendto(f"rc {c(-vy)} {c(vx)} {c(vz)} 0".encode(), (TELLO_IP, CMD_PORT))

    def _state(self):
        while self.alive:
            try:
                raw = self.st.recv(1024).decode(errors="ignore")
                self.state = {k: float(v) for k, v in (kv.split(":") for kv in raw.strip().split(";") if ":" in kv)}
                self.state["t"] = time.time()
            except (socket.timeout, ValueError):
                pass

    def _video(self):
        while self.alive:
            ok, f = self.cap.read()
            if ok:
                self.frame, self.t_frame = f, time.time()

    def velocity(self):
        s = self.state
        return np.array([VGX_SIGN * s.get("vgx", 0.0), VGY_SIGN * s.get("vgy", 0.0), -s.get("vgz", 0.0)]) / 10.0

    def height(self):
        return self.state.get("tof", 0.0) / 100.0

    def close(self):
        self.alive = False


class SimTello:
    """The simulator behind the same interface: the loop below, run with the drone swapped out. Markers and
    the pole come from the simulator's own view (the real picture is what --check-camera tests)."""

    def __init__(self, seed=5):
        self.env = C.Cage(1, seed=seed, dev="cpu"); self.env.reset()
        self.cmd, self.land_now = torch.zeros(1, 3), False
        self.state = {"bat": 100.0}

    def send(self, cmd):
        if cmd == "land":
            self.land_now = True
        return "ok"

    def rc(self, vx, vy, vz):
        self.cmd = torch.tensor([[vx, vy, vz]], dtype=torch.float32)

    def tick(self):
        self.env.step(self.cmd, torch.tensor([self.land_now]))

    def velocity(self):
        return (self.env.v[0] * torch.cat([1 + self.env.k[0], torch.ones(1)])).numpy()      # the VPS's view

    def height(self):
        return float(self.env.p[0, 2])

    def markers(self):
        seen, rel = self.env.view(self.env.p)
        return {i: tuple(rel[0, j].tolist()) for j, i in enumerate(self.ids) if seen[0, j]}

    def pole(self):
        if not bool(self.env.pole_on[0]):
            return None
        d = self.env.pole[0, :2] - self.env.p[0, :2]
        if not (0 < d[0] < C.FWD_R and abs(d[1]) < d[0] * C.FWD_TAN):
            return None
        return float(d.norm()), math.atan2(float(d[1]), float(d[0]))

    def close(self):
        pass


# ---- the loop -----------------------------------------------------------------------------------------
def deliver(link, world, lay, decider, sim=False, log_path=None):
    eyes = None if sim else Eyes(lay.get("marker_m", C.MARKER), world.ids)
    b = np.array([lay["hub"][0], lay["hub"][1], 0.0])                   # believed position: takeoff from the hub
    t_fix, d_fix, t0, k = 0.0, 0.0, time.time(), 0
    pole = torch.full((1, 4), 1e3)
    decider.reset(world)
    cmd, land = torch.zeros(1, 3), torch.zeros(1, dtype=torch.bool)
    log = open(log_path, "w") if log_path else None
    if not sim:
        bat = link.state.get("bat", 0)
        assert bat >= 30, f"battery {bat}% -- charge it first"
        assert link.send("takeoff") == "ok", "takeoff refused"
    else:
        link.ids = world.ids
    while True:
        tick = time.time()
        if sim:
            link.tick()
        v = link.velocity()
        b[:2] += v[:2] * C.DT                                             # dead reckoning between fixes
        b[2] = link.height()
        t_fix += C.DT; d_fix += float(np.linalg.norm(v[:2])) * C.DT
        if sim:
            marks, pl = link.markers(), link.pole()
        else:
            if time.time() - link.t_frame > 1.0:
                print("no video for 1 s: landing"); break
            marks, pl = eyes.markers(link.frame), eyes.pole(link.frame)
        seen = torch.zeros(1, len(world.ids), dtype=torch.bool); rel = torch.zeros(1, len(world.ids), 2)
        fixes = []
        for j, i in enumerate(world.ids):
            if i in marks:
                seen[0, j] = True; rel[0, j] = torch.tensor(marks[i])
                fixes.append(world.pads[0, j].numpy() - np.array(marks[i]))
        if fixes:                                                          # a surveyed marker read: reset
            b[:2] = np.mean(fixes, 0); t_fix = d_fix = 0.0
        if pl is not None:                                                 # the pole, into the believed map
            r, a = pl
            pole = torch.tensor([[b[0] + r * math.cos(a), b[1] + r * math.sin(a), POLE_D / 2, 1.75]])
        sig = math.sqrt((C.FIX_SD + 0.01 * b[2]) ** 2 + C.DRIFT ** 2 * t_fix + (C.SCALE_SD * d_fix) ** 2)
        o = dict(b=torch.as_tensor(b[None], dtype=torch.float32), v=torch.as_tensor(v[None], dtype=torch.float32),
                 sig=torch.tensor([sig]), seen=seen, rel=rel, pole=pole)
        world.t = time.time() - t0 if not sim else k * C.DT
        if k % C.HOLD == 0:
            cmd, land = decider.act(o)
        out = cmd[0].numpy()
        inside = C.R_DRONE + 0.15 < b[0] < C.L - C.R_DRONE - 0.15 and C.R_DRONE + 0.15 < b[1] < C.B - C.R_DRONE - 0.15
        if log:
            log.write(json.dumps(dict(t=round(world.t, 2), b=b.round(3).tolist(), v=v.round(3).tolist(), sig=round(sig, 3),
                                      marks={str(i): list(m) for i, m in marks.items()}, pole=pl, cmd=out.round(3).tolist(),
                                      land=bool(land[0]))) + "\n")
        if bool(land[0]):
            print(f"t {world.t:5.1f} s: landing at believed ({b[0]:.2f}, {b[1]:.2f})"); break
        if not inside and world.t > 2.0:
            print(f"believed position ({b[0]:.2f}, {b[1]:.2f}) outside the cage box: landing"); break
        if world.t > C.T_MAX or (not sim and link.state.get("bat", 100) < 25):
            print("time or battery: landing"); break
        link.rc(*out)
        k += 1
        if not sim:
            time.sleep(max(0.0, C.DT - (time.time() - tick)))
    link.rc(0, 0, 0)
    link.send("land")
    if sim:
        while not bool(link.env.done[0]):
            link.tick()
        print("simulated outcome:", C.OUTCOMES[int(link.env.outcome[0])])
    if log:
        log.close()


def decider_for(name, ckpt=None):
    if name == "rule":
        return C.Rule()
    if name == "planner":
        return C.Planner()
    import train as T
    import urban_rl as R
    st = torch.load(ckpt, map_location="cpu", weights_only=True)
    net = R.Net(st["body.0.weight"].shape[1]); net.load_state_dict(st); net.eval()
    return T.Learned(net, greedy="greedy" in os.path.basename(ckpt))


def check_camera(lay):
    import cv2
    link = Tello(); eyes = Eyes(lay.get("marker_m", C.MARKER), [p[2] for p in lay["pads"]])
    print("markers: id -> (dx ahead, dy left) m; pole: (range m, bearing deg, + left). Ctrl-C to stop.")
    try:
        while True:
            if link.frame is not None:
                f = link.frame
                pl = eyes.pole(f)
                print({i: (round(a, 3), round(b_, 3)) for i, (a, b_) in eyes.markers(f).items()},
                      None if pl is None else (round(pl[0], 2), round(math.degrees(pl[1]), 1)), f"tof {link.height():.2f} m", flush=True)
                cv2.imshow("tello (top: ahead; bottom: mirror, unflipped)", f)
                if cv2.waitKey(1) == 27:
                    break
            time.sleep(0.2)
    finally:
        link.close()


def check_axes():
    """Take off, 1.5 s forward then 1.5 s right at rc 20, print what vgx/vgy said, land. Set the signs
    (and RC_PER_MPS against a tape measure) from it."""
    link = Tello()
    assert link.send("takeoff") == "ok"
    for name, rc in (("forward (+x)", "rc 0 20 0 0"), ("right (-y)", "rc 20 0 0 0")):
        vs = []
        t = time.time()
        while time.time() - t < 1.5:
            link.s.sendto(rc.encode(), (TELLO_IP, CMD_PORT)); time.sleep(0.1)
            vs.append((link.state.get("vgx", 0), link.state.get("vgy", 0)))
        link.rc(0, 0, 0); time.sleep(1.5)
        print(f"{name}: mean vgx {np.mean([a for a, _ in vs]):+.1f}  vgy {np.mean([b_ for _, b_ in vs]):+.1f} dm/s")
    link.send("land"); link.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--layout", default=f"{HERE}/layout.json")
    ap.add_argument("--decider", default="planner", choices=("rule", "planner", "learner"))
    ap.add_argument("--ckpt")
    ap.add_argument("--sim", action="store_true")
    ap.add_argument("--check-camera", action="store_true")
    ap.add_argument("--check-axes", action="store_true")
    a = ap.parse_args()
    if a.check_axes:
        check_axes(); sys.exit()
    lay_path = a.layout if os.path.exists(a.layout) else f"{HERE}/layout_example.json"
    if a.check_camera:
        check_camera(json.load(open(lay_path))); sys.exit()
    world, lay = load_layout(lay_path)
    dec = decider_for(a.decider, a.ckpt)
    link = SimTello() if a.sim else Tello()
    if a.sim:                                   # fly the simulator's world as if it were the measured cage
        link.env.pads, link.env.houses, link.env.trees = world.pads.clone(), world.houses.clone(), world.trees.clone()
        link.env.p[0, :2] = torch.tensor(lay["hub"])
    stamp = time.strftime("%d%H%M%S")
    os.makedirs(f"{HERE}/flights", exist_ok=True)
    try:
        deliver(link, world, lay, dec, sim=a.sim, log_path=f"{HERE}/flights/{a.decider}_{'sim_' if a.sim else ''}{stamp}.jsonl")
    except KeyboardInterrupt:
        print("Ctrl-C: landing"); link.rc(0, 0, 0); link.send("land")
    finally:
        link.close()
