#!/usr/bin/env python3
"""
gz_voxel.py -- stage 3 in Gazebo: things the 2023 survey does not show, PX4's x500 carrying a depth camera,
and a voxel map built from that camera's real depth images.

The world is Logan (gz_logan/worlds/aruco.sdf, built from the survey) plus what came since: a tower crane in
the open field on the way in (58 m, above the 50 m cruise), street lights, a traffic light whose arm reaches
toward the pad, a new tree and a shed. The drone is x500_mono_cam_down (the landing camera) plus the depth
camera of PX4's x500_depth (OAK-D Lite: 640x480, 73 deg, 0.2-19.1 m), added as its own link: both PX4
models name their camera link "camera_link", so they cannot simply be merged.

    python3 gz_voxel.py --setup     -> gz_logan/models/x500_depth_down, gz_logan/voxfly/aruco.sdf (flights),
                                       gz_logan/voxbench/aruco.sdf (bench), gz_logan/voxfly/unmapped.json
    python3 gz_voxel.py --bench     -> Gazebo in partition "vox", the depth camera flown along the approach
                                       by set_pose; its images -> voxel map, checked against the objects'
                                       true geometry (fails loudly); gz_logan/voxbench/voxels.npz, bench.png
Frames: Gazebo ENU (x east, y north, z up). The deciders' training map is Gazebo + OFFSET (gz_learner).
"""
import argparse
import functools
import json
import math
import os
import re
import subprocess
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PX4 = "/home/sai/thesis_2.0/PX4-Autopilot"
PARTITION = "vox"
DW, DH, DFOV, DHZ = 640, 480, 1.274, 15            # the OAK-D Lite depth stream, as PX4's OakD-Lite model
DFAR, DNEAR = 19.1, 0.2
FX = (DW / 2) / math.tan(DFOV / 2)                   # 432.7 px
CAM_OFF = np.array([0.13, 0.0, 0.07])               # on the front of the body plate (PX4's x500_depth puts it at
                                                    # z 0.242: floating 18 cm above the frame -- Sai, 25 Sept)
CRANE0 = (58.0, 1.0)                                # the crane's mast in the world file; flights move it per seed
YELLOW, GREY, DARK, GREEN, BROWN, SHED = "0.95 0.77 0.06", "0.55 0.55 0.58", "0.12 0.12 0.13", "0.18 0.45 0.17", "0.40 0.26 0.13", "0.62 0.50 0.40"
# RIG=v2 (gz_unified.py): the OAK-D hangs under the body plate -- its depth camera 0.095 m ahead, 0.045 m down
DEPTH_OFF = np.array([0.095, 0.0, -0.045]) if os.environ.get("RIG", "").startswith("v2") else CAM_OFF


# ---- the world ------------------------------------------------------------------------------------
def _link(name, parts):
    """parts: (kind, pose xyz, size, rgb); kind box (size = sx sy sz), cyl (r h), sph (r). Visual + collision."""
    geo = {"box": lambda s: f"<box><size>{s[0]} {s[1]} {s[2]}</size></box>",
           "cyl": lambda s: f"<cylinder><radius>{s[0]}</radius><length>{s[1]}</length></cylinder>",
           "sph": lambda s: f"<sphere><radius>{s[0]}</radius></sphere>"}
    out = []
    for k, (kind, p, s, rgb) in enumerate(parts):
        g = geo[kind](s)
        out.append(f"<visual name='v{k}'><pose>{p[0]} {p[1]} {p[2]} 0 0 0</pose><geometry>{g}</geometry>"
                   f"<material><ambient>{rgb} 1</ambient><diffuse>{rgb} 1</diffuse></material></visual>"
                   f"<collision name='c{k}'><pose>{p[0]} {p[1]} {p[2]} 0 0 0</pose><geometry>{g}</geometry></collision>")
    return f"<model name='{name}'><static>true</static><link name='link'>{''.join(out)}</link></model>"


@functools.lru_cache(maxsize=None)
def crane_at(seed=None):
    """Where the crane's mast stands on this flight: seed None -> as in the world file; otherwise a clear spot of
    the open field on the flight line (survey-free under the mast), 42-64 m from the depot, up to 1.5 m off
    the line either side -- always in the way, at a different distance and side each flight."""
    if seed is None and not os.environ.get("CRANE_AT"):
        return CRANE0
    m = np.load(f"{HERE}/urban/logan_map_lidar.npz")
    H, half = m["H"], float(m["half"])
    q0 = np.array([135.5 - 120.0, 1135.5])
    ok = lambda x, y: H[int(y + q0[1] + half) - 1:int(y + q0[1] + half) + 2, int(x + q0[0] + half) - 1:int(x + q0[0] + half) + 2].max() < 0.5
    if os.environ.get("CRANE_AT"):                     # a flight's own spot (geo/: on the planner's last stretch)
        x, y = map(float, os.environ["CRANE_AT"].split(","))
        assert ok(x, y), f"CRANE_AT {x},{y}: not clear ground in the survey"
        return (x, y)
    rng = np.random.default_rng(1000 + seed)
    while True:
        x, y = rng.uniform(42.0, 64.0), rng.uniform(-1.5, 1.5)
        if ok(x, y):
            return (round(float(x), 2), round(float(y), 2))


@functools.lru_cache(maxsize=None)
def objects(seed=None):
    """(name, [(kind, centre xyz in the world, size, rgb)]) -- one list drives the SDF and the truth file.
    seed: the crane where crane_at(seed) puts it (gz_learner moves the model there before the flight)."""
    cx, cy = crane_at(seed)
    sx, sy = cx - CRANE0[0], cy - CRANE0[1]
    crane = [("box", (58.0, 1.0, 29.0), (1.6, 1.6, 58.0), YELLOW),                 # mast
             ("box", (58.0, -11.0, 57.4), (1.2, 24.0, 1.2), YELLOW),              # jib, reaching south
             ("box", (58.0, 5.5, 57.4), (1.2, 9.0, 1.2), YELLOW),                 # counter-jib
             ("box", (58.0, 8.5, 55.8), (2.0, 3.0, 2.0), GREY),                   # counterweight
             ("box", (59.4, 1.0, 55.0), (1.2, 1.6, 2.0), GREY)]                   # cab
    crane = [(k, (c[0] + sx, c[1] + sy, c[2]), s, rgb) for k, c, s, rgb in crane]
    light = lambda x, y, arm: [("cyl", (x, y, 4.5), (0.1, 9.0), GREY), ("box", (x, y + arm / 2, 8.9), (0.1, abs(arm), 0.1), GREY),
                               ("box", (x, y + arm, 8.8), (0.4, 0.25, 0.15), DARK)]
    tree = lambda x, y, r, h: [("cyl", (x, y, (h - r) / 2), (0.25, h - r), BROWN), ("sph", (x, y, h - r), (r,), GREEN)]
    return [("crane", crane),
            ("light_1", light(110.0, -6.0, 1.5)), ("light_2", light(122.0, -6.0, 1.5)), ("light_3", light(132.0, -6.0, 1.5)),
            ("light_4", light(112.0, 5.0, -1.5)),
            ("traffic_light", [("cyl", (116.0, -5.0, 2.9), (0.15, 5.8), DARK), ("box", (116.0, -3.0, 5.7), (0.15, 4.0, 0.15), DARK),
                               ("box", (116.0, -1.2, 5.2), (0.4, 0.4, 1.1), DARK)]),
            ("tree_1", tree(112.0, 2.5, 2.2, 9.0)), ("tree_2", tree(125.0, -4.0, 1.8, 8.0)), ("tree_3", tree(60.0, -6.0, 3.0, 12.0)),
            ("shed", [("box", (130.0, -7.5, 1.3), (3.0, 2.5, 2.6), SHED)])]


def depth_link(pose=" ".join(f"{v:g}" for v in CAM_OFF) + " 0 0 0", bracket=True):
    return (f"<link name='depth_link'><pose relative_to='base_link'>{pose}</pose>"
            "<inertial><mass>0.001</mass><inertia><ixx>1e-7</ixx><iyy>1e-7</iyy><izz>1e-7</izz><ixy>0</ixy><ixz>0</ixz>"
            "<iyz>0</iyz></inertia></inertial>"
            "<visual name='oakd'><geometry><box><size>0.0175 0.091 0.028</size></box></geometry>"   # OAK-D Lite, 91 x 28 x 17.5 mm
            "<material><ambient>0.15 0.15 0.17 1</ambient><diffuse>0.15 0.15 0.17 1</diffuse></material></visual>"
            + ("<visual name='bracket'><pose>-0.025 0 -0.018 0 0 0</pose><geometry><box><size>0.05 0.03 0.012</size></box>"
               "</geometry><material><ambient>0.3 0.3 0.3 1</ambient><diffuse>0.3 0.3 0.3 1</diffuse></material></visual>"
               if bracket else "")
            + f"<sensor name='depth' type='depth_camera'><topic>/depth_camera</topic><update_rate>{DHZ}</update_rate>"
            f"<always_on>1</always_on><camera><horizontal_fov>{DFOV}</horizontal_fov><image><width>{DW}</width>"
            f"<height>{DH}</height><format>R_FLOAT32</format></image><clip><near>{DNEAR}</near><far>{DFAR}</far></clip>"
            "</camera></sensor></link><joint name='depth_joint' type='fixed'><parent>base_link</parent><child>depth_link</child></joint>")


def setup():
    import gz_learner as G
    d = f"{HERE}/gz_logan/models/x500_depth_down"
    os.makedirs(d, exist_ok=True)
    open(f"{d}/model.config", "w").write(
        "<?xml version='1.0'?><model><name>x500_depth_down</name><version>1.0</version><sdf version='1.9'>model.sdf</sdf>"
        "<description>PX4 x500 with its downward landing camera and the OAK-D Lite depth camera of x500_depth</description></model>\n")
    open(f"{d}/model.sdf", "w").write(
        "<?xml version='1.0'?>\n<sdf version='1.9'><model name='x500_depth_down'><self_collide>false</self_collide>"
        "<include merge='true'><uri>model://x500_mono_cam_down</uri></include>" + depth_link() + "</model></sdf>\n")
    things = "\n    ".join(_link(n, p) for n, p in objects())
    src = open(f"{HERE}/gz_logan/worlds/aruco.sdf").read()
    assert src.count("  </world>") == 1 and "<plugin" not in src
    # flights: Logan + the new things + the chase camera + the drone standing at the depot, PX4 attaches to it
    drone = "<include><uri>model://x500_depth_down</uri><name>x500_mono_cam_down_0</name><pose>0 0 0 0 0 0</pose></include>"
    # voxfast: the same world without the chase and overhead video cameras -- rendered for nobody in bulk runs
    for sub, extra in (("voxfly", f"{G.chase_sdf()}\n    {drone}"), ("voxfast", drone),
                       ("voxbench", "<model name='depthbench'><static>true</static><link name='base_link'/>"
                                    + depth_link("0 0 0 0 0 0", bracket=False) + "</model>")):
        os.makedirs(f"{HERE}/gz_logan/{sub}", exist_ok=True)
        s = src if sub == "voxfly" else re.sub(r'<model name="observer_cam">.*?</model>', "", src, flags=re.S)
        assert sub == "voxfly" or "observer_cam" not in s
        open(f"{HERE}/gz_logan/{sub}/aruco.sdf", "w").write(s.replace("  </world>", f"    {things}\n    {extra}\n  </world>"))
    json.dump({n: p for n, p in objects()}, open(f"{HERE}/gz_logan/voxfly/unmapped.json", "w"), indent=1)
    print(d, f"{HERE}/gz_logan/voxfly/aruco.sdf", f"{HERE}/gz_logan/voxbench/aruco.sdf")


# ---- depth images -> voxels -----------------------------------------------------------------------------
def rot(roll, pitch, yaw):
    cr, sr, cp, sp, cy, sy = math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch), math.cos(yaw), math.sin(yaw)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


class VoxelMap:
    """Occupied RES-m voxels from depth images, in the frame the drone's pose is given in (the drone's own
    estimate in flight: sensed things are relative to it, see voxel-map-frames). Depth past R_MAX is dropped
    (the fast simulator's 8 m: an OAK-D Lite's reliable range), stereo noise is added (sd grows with the
    square of the depth), and a voxel counts once HITS frames have put a point in it. tops(): the tallest
    occupied voxel per 1 m column -- what the deciders' map lookups read."""

    def __init__(self, res=0.5, r_max=8.0, hits=2, stride=4, sd_disp=0.15, baseline=0.075, seed=0, r_min=1.0):
        # r_min: rig v2's OAK-D hangs under the body plate and the front propellers' spinning tips cross the top of
        # its depth image 0.2-0.35 m away: voxels at the drone's own spot, "boxed in", climb -- to 239 m over the pad
        # (25 Sept). Nothing that close can be avoided anyway.
        self.res, self.r_max, self.hits, self.stride, self.r_min = res, r_max, hits, stride, r_min
        self.k_noise = sd_disp / (FX * baseline)
        self.count, self.rng = {}, np.random.default_rng(seed)
        self.top, self.dirty = {}, set()                          # 1 m column -> top of its occupied voxels
        v, u = np.mgrid[0:DH:stride, 0:DW:stride]
        self.ray = np.stack([np.ones(u.shape), (DW / 2 - u) / FX, (DH / 2 - v) / FX], -1).reshape(-1, 3)   # x ahead, y left, z up

    def points(self, depth, pos, R, cam_off=CAM_OFF):
        d = depth[::self.stride, ::self.stride].reshape(-1).astype(np.float64)
        ok = np.isfinite(d) & (d > max(DNEAR, self.r_min)) & (d <= self.r_max)
        d = d[ok] + self.rng.normal(0.0, 1.0, ok.sum()) * self.k_noise * d[ok] ** 2
        p = (R @ (cam_off[:, None] + (self.ray[ok] * d[:, None]).T)).T + pos
        return p[p[:, 2] > 0.3]                                  # the ground is not an obstacle

    def insert(self, depth, pos, R, cam_off=CAM_OFF):
        p = self.points(depth, pos, R, cam_off)
        keys, n = np.unique(np.floor(p / self.res).astype(np.int64), axis=0, return_counts=True)
        for k in map(tuple, keys):
            m = self.count[k] = self.count.get(k, 0) + 1          # one vote per frame, however many points
            if m == self.hits:                                    # just became occupied
                col, top = (math.floor(k[0] * self.res), math.floor(k[1] * self.res)), (k[2] + 1) * self.res
                if top > self.top.get(col, 0.0):
                    self.top[col] = top; self.dirty.add(col)
        return len(p)

    def take(self):
        """[[x, y, top]] for the 1 m columns whose top rose since the last call (what goes to the decider)."""
        out = [[c[0], c[1], self.top[c]] for c in self.dirty]
        self.dirty.clear()
        return out

    def occupied(self):
        k = np.array([k for k, c in self.count.items() if c >= self.hits], np.int64).reshape(-1, 3)
        return (k + 0.5) * self.res

    def tops(self):
        """{(ix, iy) 1 m column: top (m)} of the occupied voxels."""
        return dict(self.top)


# ---- the bench ---------------------------------------------------------------------------------------------
def bench_gz():
    env = dict(os.environ, GZ_PARTITION=PARTITION,
               GZ_SIM_RESOURCE_PATH=f"{PX4}/Tools/simulation/gz/models:{HERE}/gz_logan/models",
               GZ_SIM_SERVER_CONFIG_PATH=f"{PX4}/src/modules/simulation/gz_bridge/server.config",
               GZ_SIM_SYSTEM_PLUGIN_PATH=f"{PX4}/build/px4_sitl_default/src/modules/simulation/gz_plugins",
               __NV_PRIME_RENDER_OFFLOAD="1", __GLX_VENDOR_LIBRARY_NAME="nvidia",
               __EGL_VENDOR_LIBRARY_FILENAMES="/usr/share/glvnd/egl_vendor.d/10_nvidia.json")
    return subprocess.Popen(["bash", "-c", "source /opt/ros/jazzy/setup.bash; exec gz sim --verbose=1 -r -s "
                             f"{HERE}/gz_logan/voxbench/aruco.sdf"], env=env, start_new_session=True,
                            stdout=open(f"{HERE}/gz_logan/voxbench/gz.log", "w"), stderr=subprocess.STDOUT)


class Depth:
    """The newest depth image and its simulated-time stamp."""

    def __init__(self):
        from gz.transport13 import Node
        from gz.msgs10.image_pb2 import Image
        self.last, self.node = None, Node()
        self.node.subscribe(Image, "/depth_camera", self._on)

    def _on(self, m):
        self.last = (m.header.stamp.sec + m.header.stamp.nsec * 1e-9, m)

    def fresh(self, k=1, timeout=120.0):
        after = (self.last[0] if self.last else 0.0) + k / DHZ
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.last and self.last[0] >= after:
                m = self.last[1]
                return np.frombuffer(m.data, np.float32).reshape(m.height, m.width).copy()
            time.sleep(0.005)
        raise TimeoutError("no depth image")

    def warm(self, timeout=600.0):
        """Gazebo keeps adding Logan's houses and trees for a while after the first subscriber (the rig
        bench's lesson): wait until the image stays identical over a full simulated second."""
        t0 = time.time()
        while time.time() - t0 < timeout:
            a, b = self.fresh(1), self.fresh(DHZ)
            if np.array_equal(np.nan_to_num(a, posinf=99), np.nan_to_num(b, posinf=99)):
                return
        raise RuntimeError("the depth view never stopped changing")

    def settled(self, tries=20):
        a = self.fresh(4)
        for _ in range(tries):
            b = self.fresh(1)
            if np.array_equal(np.nan_to_num(a, posinf=99), np.nan_to_num(b, posinf=99)):
                return b
            a = b
        raise RuntimeError("the depth image never settled")


def truth_dist(p):
    """Distance from each point to the nearest added object's surface (0 inside)."""
    return np.min([_dist(p, parts) for _, parts in objects()], axis=0)


def nearest(p, only=None, seed=None):
    """(name, distance) of the added object nearest the point p (3,), or with p (n, 3): the distance of each
    point to object `only` (the flight's closest approach to each). seed: the crane where that flight put it."""
    p = np.atleast_2d(np.asarray(p, float))
    if only is not None:
        return only, _dist(p, dict(objects(seed))[only])
    ds = [(n, _dist(p, parts)[0]) for n, parts in objects(seed)]
    return min(ds, key=lambda x: x[1])


def _dist(p, parts):
    best = np.full(len(p), np.inf)
    for kind, c, s, _ in parts:
        q = p - np.array(c)
        if kind == "box":
            d = np.linalg.norm(np.maximum(np.abs(q) - np.array(s) / 2, 0), axis=1)
        elif kind == "cyl":
            d = np.hypot(np.maximum(np.hypot(q[:, 0], q[:, 1]) - s[0], 0), np.maximum(np.abs(q[:, 2]) - s[1] / 2, 0))
        else:
            d = np.maximum(np.linalg.norm(q, axis=1) - s[0], 0)
        best = np.minimum(best, d)
    return best


def survey_top(xy, r=1):
    """The survey's tallest obstacle within r m of each Gazebo xy (the Logan houses and trees Gazebo shows)."""
    m = np.load(f"{HERE}/urban/logan_map_lidar.npz")
    H, half, res = m["H"], float(m["half"]), float(m["res"])
    q = xy + np.array([135.5 - 120.0, 1135.5])                  # Gazebo -> training map (gz_learner.OFFSET)
    i, j = ((q[:, 1] + half) / res).astype(int), ((q[:, 0] + half) / res).astype(int)
    return np.max([H[i + di, j + dj] for di in range(-r, r + 1) for dj in range(-r, r + 1)], axis=0)


def bench(out=f"{HERE}/gz_logan/voxbench"):
    """Fly the depth camera along the approach as a flight would (50 m, gliding down near the pad), then
    look round the pad from 10 m; build the voxel map from the true poses and check it: every voxel must
    be explained by an added object or the survey, and the crane's mast must sit where it stands."""
    import gz_learner as G
    g = bench_gz()
    try:
        mover, cam = G.Mover(), Depth()
        cam.fresh(timeout=300)
        mover.put("depthbench", (100.0, 0.0, 12.0))             # looking at houses: something to settle on
        cam.warm()
        vm = VoxelMap()
        poses = [((x, 0.0, 50.0 if x < 60 else max(50.0 - (x - 60) * 0.6, 12.0)), 0.0) for x in np.arange(40.0, 118.0, 1.5)]
        poses += [((120.0, 0.0, 10.0), yaw) for yaw in np.radians(np.arange(0, 360, 30))]
        n_pts = 0
        for (x, y, z), yaw in poses:
            mover.put("depthbench", (x, y, z), yaw=float(yaw))
            d = cam.settled()
            n_pts += vm.insert(d, np.array([x, y, z]), rot(0.0, 0.0, float(yaw)), np.zeros(3))
        occ = vm.occupied()
        np.savez(f"{out}/voxels.npz", occupied=occ, poses=np.array([[*p, yw] for p, yw in poses]))
        dt, st = truth_dist(occ), survey_top(occ[:, :2])
        explained = (dt <= 0.75) | (occ[:, 2] <= st + 1.0)
        mast = occ[(np.abs(occ[:, 0] - 58.0) < 3) & (np.abs(occ[:, 1] - 1.0) < 3) & (occ[:, 2] > 35) & (occ[:, 2] < 54)]
        print(f"{len(poses)} depth images, {n_pts:,} points -> {len(occ):,} occupied voxels; explained by an added object "
              f"or the survey: {100 * explained.mean():.1f}%; crane mast voxels {len(mast)}", flush=True)
        per = {}
        for name, parts in objects():
            c = np.array([p[1] for p in parts])
            near = occ[np.min(np.linalg.norm(occ[:, None] - c[None], axis=-1), axis=1) < 6.0]
            per[name] = int((truth_dist(near) <= 0.75).sum()) if len(near) else 0
        print("voxels on each added object:", per, flush=True)
        assert len(mast) >= 5, "the depth camera never put the crane's mast on the map"
        cx, cy = mast[:, 0].mean(), mast[:, 1].mean()
        print(f"crane mast seen at ({cx:.2f}, {cy:.2f}); it stands at (58.00, 1.00), 1.6 m square", flush=True)
        assert math.hypot(cx - 58.0, cy - 1.0) < 1.0, "the mast is off where it stands -- a frame or depth-convention error"
        assert explained.mean() > 0.95, "voxels where nothing stands -- a frame or depth-convention error"
        render(occ, poses, f"{out}/bench.png")
    finally:
        os.killpg(g.pid, 15)


def render(occ, poses, path):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, Circle
    fig, (ax, bx) = plt.subplots(2, 1, figsize=(12, 8), height_ratios=[1, 1], layout="constrained")
    m = np.load(f"{HERE}/urban/logan_map_lidar.npz")
    H, half = m["H"], float(m["half"])
    q0 = np.array([135.5 - 120.0, 1135.5])
    x0, x1, y0, y1 = 35, 140, -20, 20
    i0, i1 = int(y0 + q0[1] + half), int(y1 + q0[1] + half)
    j0, j1 = int(x0 + q0[0] + half), int(x1 + q0[0] + half)
    ax.imshow(np.where(H[i0:i1, j0:j1] > 1, H[i0:i1, j0:j1], np.nan), origin="lower", extent=(x0, x1, y0, y1), cmap="Greys", vmin=-10, vmax=30)
    sc = ax.scatter(occ[:, 0], occ[:, 1], c=occ[:, 2], s=4, cmap="plasma", vmin=0, vmax=60)
    for _, parts in objects():
        for kind, c, s, _ in parts:
            if kind == "box":
                ax.add_patch(Rectangle((c[0] - s[0] / 2, c[1] - s[1] / 2), s[0], s[1], fill=False, ec="#27ae60", lw=0.8))
            else:
                ax.add_patch(Circle(c[:2], s[0], fill=False, ec="#27ae60", lw=0.8))
    p = np.array([pp for pp, _ in poses])
    ax.plot(p[:, 0], p[:, 1], "b.", ms=2)
    ax.plot(120, 0, "s", color="#1e8449", ms=6)
    ax.set(xlim=(x0, x1), ylim=(y0, y1), aspect="equal", title="Voxel map from the depth camera's images (colour = height, m); "
           "green outlines: where the added objects truly stand; blue: camera poses")
    fig.colorbar(sc, ax=ax, shrink=0.8)
    s = np.abs(occ[:, 1]) < 3
    bx.scatter(occ[s, 0], occ[s, 2], c=occ[s, 2], s=4, cmap="plasma", vmin=0, vmax=60)
    bx.plot(p[:, 0], p[:, 2], "b.", ms=2)
    bx.set(xlim=(x0, x1), ylim=(0, 62), xlabel="x (m, east)", ylabel="height (m)", title="Side view: voxels within 3 m of the flight line")
    fig.savefig(path, dpi=120)
    print(path)


def film(folder, out):
    """The voxel flights as one video: title card; the planner trusting the survey (top) against the planner on
    the voxel map (bottom), in step; the learner on the voxel map; a card of results."""
    import cv2
    rs = {(r["decider"], r["voxel"]): r for r in map(json.loads, open(f"{folder}/flights.jsonl"))}
    W, H, F, FPS = 1500, 1200, cv2.FONT_HERSHEY_SIMPLEX, 20
    vw = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"avc1"), FPS, (W, H))

    def panel(lines, h=H):
        img = np.full((h, W, 3), (31, 24, 20), np.uint8); y = h // 2 - 38 * len(lines) + 30
        for text, scale, col in lines:
            (tw, _), _ = cv2.getTextSize(text, F, scale, 2)
            cv2.putText(img, text, ((W - tw) // 2, y), F, scale, col, 2, cv2.LINE_AA); y += int(76 * scale)
        return img

    def card(lines, secs=5.0):
        img = panel(lines)
        for _ in range(int(secs * FPS)):
            vw.write(img)

    def say(r):
        out = {"crashed": f"crashed into the {str(r['crashed_into']).split('_')[0]} after {r['t_s'] - 3:.0f} s",
               "landed": f"delivered, {r['miss_m']:.2f} m from the pad centre, {r['t_s']:.0f} s"}.get(r["outcome"], r["outcome"])
        return out + (f"; passed {r['closest_m']['crane']:.1f} m from the crane" if r["outcome"] != "crashed" else "")

    def frames(path):
        c = cv2.VideoCapture(path)
        while True:
            ok, f = c.read()
            if not ok:
                return
            yield f
    card([("A crane the 2023 survey does not show", 1.4, (255, 255, 255)),
          ("PX4's x500 in Gazebo, flown by the fast simulator's decider, unchanged", 0.85, (210, 210, 210)),
          ("it carries a depth camera (OAK-D Lite, 73 deg); the voxel map is built from its images", 0.85, (210, 210, 210)),
          ("top: trusting the survey only      bottom: the survey + the voxel map", 0.85, (120, 200, 255))], 6)
    s, v = rs[("planner", False)], rs[("planner", True)]
    card([("The planner, twice", 1.3, (255, 255, 255)), ("survey only: " + say(s), 0.8, (180, 180, 240)),
          ("voxel map: " + say(v), 0.8, (180, 220, 180))])
    ta, tb, fa, fb = frames(f"{folder}/planner_survey_s0.mp4"), frames(f"{folder}/planner_voxel_s0.mp4"), None, None
    while True:
        a, b = next(ta, None), next(tb, None)
        if a is None and b is None:
            break
        fa, fb = (a if a is not None else fa), (b if b is not None else fb)
        vw.write(np.vstack([fa, fb]))
    if ("learner", True) in rs:
        lr = rs[("learner", True)]
        card([("The learner, on the same voxel map", 1.3, (255, 255, 255)),
              ("trained in the fast simulator with structures its survey lacks", 0.85, (210, 210, 210)),
              (say(lr), 0.8, (180, 220, 180))])
        info = panel([("The learner on the voxel map", 1.1, (255, 255, 255)), (say(lr), 0.8, (180, 220, 180)),
                      ("dots: voxels the depth camera has seen (yellow low, red high)", 0.7, (200, 200, 200))], H // 2)
        for f in frames(f"{folder}/learner_voxel_s0.mp4"):
            vw.write(np.vstack([f, info]))
    card([("What the depth camera changes", 1.3, (255, 255, 255)),
          ("the survey alone flies into what was built since 2023", 0.85, (210, 210, 210)),
          ("the voxel map, from the drone's own depth images, lets the same decider steer round it", 0.85, (210, 210, 210)),
          ("one seed each: a demonstration, not a statistic (the fast simulator's 4,096 flights are)", 0.75, (120, 200, 255))], 6)
    vw.release(); print(out)


def film_seeds(folder, out, show=3, table=None):
    """The crane-moved flights: a card per seed (where the crane stood, how each map fared), the survey-only
    flight (top) against the local 3D planner (bottom) for the first `show` seeds, and a table of all."""
    import cv2
    rs = [json.loads(l) for l in open(f"{folder}/flights.jsonl")]
    mode = lambda r: "local3d" if r.get("local3d") else "voxel" if r["voxel"] else "survey"
    fname = lambda r, s: f"{folder}/{'rule' if (s, 'local3d') in by and by[(s, 'local3d')]['decider'] == 'rule' and r == 'local3d' else 'planner'}_{r}_s{s}.mp4"
    by = {(r["seed"], mode(r)): r for r in rs}
    seeds = sorted({s for s, _ in by})
    W, H, F, FPS = 1500, 1200, cv2.FONT_HERSHEY_SIMPLEX, 20
    vw = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"avc1"), FPS, (W, H))
    name = {"survey": "survey only", "voxel": "voxel map, look-ahead", "local3d": "voxel map + local 3D planner"}

    def card(lines, secs=5.0):
        img = np.full((H, W, 3), (31, 24, 20), np.uint8); y = H // 2 - 38 * len(lines) + 30
        for text, scale, col in lines:
            (tw, _), _ = cv2.getTextSize(text, F, scale, 2)
            cv2.putText(img, text, ((W - tw) // 2, y), F, scale, col, 2, cv2.LINE_AA); y += int(76 * scale)
        for _ in range(int(secs * FPS)):
            vw.write(img)

    def say(r):
        if r is None:
            return "not flown"
        if r["outcome"] == "crashed":
            return f"crashed into the {str(r['crashed_into']).split('_')[0]}"
        if r["outcome"] == "landed":
            return f"delivered {r['miss_m']:.2f} m from centre, {r['t_s']:.0f} s, crane passed at {r['closest_m']['crane']:.1f} m"
        if r["outcome"] == "passed":
            return f"passed the crane {r['closest_m']['crane']:.1f} m away"
        return r["outcome"].replace("_", " ")

    def frames(path):
        c = cv2.VideoCapture(path)
        while True:
            ok, f = c.read()
            if not ok:
                return
            yield f
    card([(f"The crane, moved: {len(seeds[:show])} of the flights", 1.4, (255, 255, 255)),
          ("each flight puts a 58 m crane somewhere else on the way in (42-64 m from the depot, up to 1.5 m off the line)", 0.72, (210, 210, 210)),
          ("top: trusting the 2023 survey only      bottom: voxel map + local 3D planner (A* round what it shows)", 0.75, (120, 200, 255)),
          ("cubes: what the depth camera has seen, coloured by height      band: the planned path ahead", 0.75, (120, 200, 255))], 7)
    for s in seeds[:show]:
        c = by.get((s, "local3d"), by.get((s, "voxel"), by.get((s, "survey"))))
        card([(f"Seed {s}: the crane at x {c['crane'][0]:.0f} m, {abs(c['crane'][1]):.1f} m {'north' if c['crane'][1] > 0 else 'south'} of the line", 1.0, (255, 255, 255))]
             + [(f"{name[m]}: {say(by[(s, m)])}", 0.75, (180, 220, 180) if by[(s, m)]["outcome"] in ("landed", "passed") else (180, 180, 240))
                for m in ("survey", "voxel", "local3d") if (s, m) in by])
        ta, tb = frames(fname("survey", s)), frames(fname("local3d", s))
        fa = fb = None
        while True:
            a, b = next(ta, None), next(tb, None)
            if a is None and b is None:
                break
            fa, fb = (a if a is not None else fa), (b if b is not None else fb)
            vw.write(np.vstack([fa if fa is not None else np.zeros((600, W, 3), np.uint8), fb if fb is not None else np.zeros((600, W, 3), np.uint8)]))
    if table:                                                  # the table from a bigger (unfilmed) run of seeds
        tb = {(r["seed"], mode(r)): r for r in (json.loads(l) for l in open(f"{table}/flights.jsonl"))}
    else:
        tb = by
    cell = lambda r: "not flown" if r is None else ("crashed" if r["outcome"] == "crashed" else
                                                    f"{r['outcome']}, {r['closest_m']['crane']:.1f} m")
    rows = [(f"seed {s}", *[cell(tb.get((s, m))) for m in ("survey", "voxel", "local3d")]) for s in sorted({s for s, _ in tb})]
    img = np.full((H, W, 3), (31, 24, 20), np.uint8)
    cv2.putText(img, f"All {len(rows)} crane positions (how close the drone came to the crane)", (60, 120), F, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    for j, h in enumerate(("", "survey only", "voxel map, look-ahead", "+ local 3D planner")):
        cv2.putText(img, h, (60 + [0, 200, 620, 1060][j], 200), F, 0.75, (120, 200, 255), 2, cv2.LINE_AA)
    for i, row in enumerate(rows):
        for j, t in enumerate(row):
            cv2.putText(img, t, (60 + [0, 200, 620, 1060][j], 260 + 60 * i), F, 0.62, (220, 220, 220), 2, cv2.LINE_AA)
    y = 260 + 60 * len(rows) + 30
    cv2.putText(img, "crashed", (60, y), F, 0.7, (120, 200, 255), 2, cv2.LINE_AA)
    for j, m in enumerate(("survey", "voxel", "local3d")):
        n = [tb[k]["outcome"] == "crashed" for k in tb if k[1] == m]
        cv2.putText(img, f"{sum(n)} of {len(n)}", (60 + [200, 620, 1060][j], y), F, 0.7,
                    (180, 180, 240) if sum(n) else (180, 220, 180), 2, cv2.LINE_AA)
    if table:       # its own flights: a filmed seed's distance above can differ from its row here
        cv2.putText(img, "flown again for this table, without cameras for film, at full speed: a filmed seed's distance can differ",
                    (60, y + 70), F, 0.55, (170, 170, 170), 1, cv2.LINE_AA)
    for _ in range(8 * FPS):
        vw.write(img)
    vw.release(); print(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--setup", action="store_true")
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--film", nargs=2, metavar=("FOLDER", "OUT"), help="the voxel flights in FOLDER as one video")
    ap.add_argument("--film-seeds", nargs=2, metavar=("FOLDER", "OUT"), help="the crane-moved flights as one video")
    ap.add_argument("--table", default=None, help="--film-seeds: the folder whose flights fill the closing table")
    a = ap.parse_args()
    os.environ["GZ_PARTITION"] = PARTITION               # before any gz transport node exists
    if a.setup:
        setup()
    if a.bench:
        bench()
    if a.film:
        film(*a.film)
    if a.film_seeds:
        film_seeds(*a.film_seeds, table=a.table)
