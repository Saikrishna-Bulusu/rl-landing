#!/usr/bin/env python3
"""
gz_rig.py -- a 360-degree camera rig for the x500, and bench tests of what it actually sees.

Sai's rule (24 Sept): no bird result counts until the drone carries a camera rig that is shown to
see the magpie. The fast simulator had handed the decider the magpie's exact position and speed
within 40 m, in every direction -- no real sensor does that.

    once:   python3 gz_rig.py --setup
              gz_logan/models/x500_rig/   PX4's x500 + its landing camera + the six rig cameras
              gz_logan/rig/aruco.sdf      the bench: Logan, the magpie, and a STATIC drone with the rig
    bench:  python3 gz_rig.py --bench     Gazebo, headless, in its own transport partition ("rig"),
                                          so it never touches a PX4 simulation running alongside
    cover:  python3 gz_rig.py --cover DIR
              the magpie at 160 directions x 6 ranges, head-on and broadside, around the drone:
              which cameras see it and how many pixels it covers (render with and without the
              bird and difference them -- whatever blocks the view stays blocked)
    show:   python3 gz_rig.py --report DIR --views DIR     chart of the above; six-view contact sheet
    data:   python3 gz_rig.py --data DIR [--poses 600]     labelled pictures for the detector
    calib:  python3 gz_rig.py --calib COVER_DIR DATA_DIR    -> urban/rig_calib.json (after rig_detect.py --score)

Rig: six 1280x960 cameras with 110-degree lenses (93 degrees tall), 20 Hz, like the six navigation
cameras of a Skydio: a ring of four (front, left, back, right) and one looking up, on a mast 30 cm
above the body -- the propeller discs below their view; one looking down from a short boom under
the landing legs. Whatever of the drone the cameras do see stays in the pictures.
"""
import argparse
import json
import math
import os
import re
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PX4 = "/home/sai/thesis_2.0/PX4-Autopilot"
W, H, HFOV, HZ = 1280, 960, math.radians(110.0), 20
F = (W / 2) / math.tan(HFOV / 2)                          # focal length, pixels (448)
# On a 30 cm mast above the propellers (ring + up) and a short boom below the landing legs (down).
# The first rig sat 12 cm up, just above the propeller discs: spinning blades showed up at a new
# angle in every frame across the lower half of each ring camera, and the down camera saw the legs
# -- 55% of the detector's false alarms in the first good flight were the drone itself (25 Sept).
RIG = {                     # name: position (m) and roll, pitch, yaw (rad), on the drone's base_link
    "front": ((0.06, 0.0, 0.30), (0.0, 0.0, 0.0)),
    "left": ((0.0, 0.06, 0.30), (0.0, 0.0, math.pi / 2)),
    "back": ((-0.06, 0.0, 0.30), (0.0, 0.0, math.pi)),
    "right": ((0.0, -0.06, 0.30), (0.0, 0.0, -math.pi / 2)),
    "up": ((0.0, 0.0, 0.32), (0.0, -math.pi / 2, 0.0)),
    "down": ((0.0, 0.0, -0.23), (0.0, math.pi / 2, 0.0)),
}
BASE_Z = 0.24               # x500_base puts base_link this far above the model's origin (its feet)
BENCH_AT = (60.0, 0.0, 30.0)  # where the bench drone hovers: over Logan, between depot and pads
PARTITION = "rig"
CAMSZ = {n: (W, H, F) for n in RIG}     # each camera's picture: width, height (px) and focal length (px)
BENCH = "rig"                           # the bench world: gz_logan/<BENCH>/aruco.sdf
# RIG=v2 (gz_unified.py, Sai 25 Sept): the magpie is watched by the OAK-D Lite's colour camera ahead (1920x1080,
# 69 deg: 3.1x the wide cameras' focal length, so a bird ahead shows 3.1x bigger) and three wide cameras left,
# right and back, all under the body plate. The gimbal camera below only lands the drone; nothing looks up.
V2 = os.environ.get("RIG", "")                         # a rig v2 variant (gz_unified.VARIANTS): v2, v2f, v2t, v2u
if V2.startswith("v2"):
    import gz_unified as GU
    RIG = {n: (tuple(p), tuple(r)) for n, (p, r, w, h, hf) in GU.cams(V2).items()}          # as rig_link mounts them
    CAMSZ = {n: (w, h, (w / 2) / math.tan(hf / 2)) for n, (p, r, w, h, hf) in GU.cams(V2).items()}
    BENCH, PARTITION = f"rig_{V2}", {"v2": "rig2"}.get(V2, f"rig{V2}")


def prop_mask(r_blade=0.15, z_prop=0.05, grow=12):
    """Where each camera can see the spinning propellers: the four discs (x500_base: hubs at (+-0.174, +-0.174),
    0.05 m above base_link, 13-inch props scaled to ~0.14 m radius) projected into every camera, filled, grown by
    `grow` px. A blade at a new angle every frame looks like a small bird there, so the live tracker ignores
    detections inside (urban/rig_perceive.py). -> gz_logan/<BENCH>/prop_mask.npz, and a preview beside it."""
    import cv2
    out = {}
    a, r = np.meshgrid(np.linspace(0, 2 * math.pi, 180), np.linspace(0.0, r_blade, 12))
    for n in RIG:
        w, h, _ = CAMSZ[n]
        m = np.zeros((h, w), np.uint8)
        for sx in (1, -1):
            for sy in (1, -1):
                pts = np.c_[sx * 0.174 + (r * np.cos(a)).ravel(), sy * 0.174 + (r * np.sin(a)).ravel(), np.full(r.size, z_prop)]
                q = [project(p_, np.zeros(3), np.eye(3), n) for p_ in pts]
                q = np.array([v for v in q if v is not None and -w < v[0] < 2 * w and -h < v[1] < 2 * h], np.float32)
                if len(q) >= 3:
                    cv2.fillConvexPoly(m, cv2.convexHull(q.astype(np.int32)), 1)
        m = cv2.dilate(m, np.ones((2 * grow + 1, 2 * grow + 1), np.uint8)) > 0
        out[n] = m
        print(f"{n:6s} propellers cover {100 * m.mean():5.2f}% of the picture")
    np.savez_compressed(f"{HERE}/gz_logan/{BENCH}/prop_mask.npz", **out)
    print(f"{HERE}/gz_logan/{BENCH}/prop_mask.npz")
    return out


def rot(r, p, y):
    """SDF roll-pitch-yaw (fixed axes x, y, z) as a rotation matrix."""
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def project(p, base, R_body, cam):
    """Pixel (u, v) of world point p in rig camera `cam` (Gazebo cameras look along +x, +y is
    image left, +z image up); None if behind the camera."""
    t, rpy = RIG[cam]
    w, h, f = CAMSZ[cam]
    Rc = R_body @ rot(*rpy)
    d = Rc.T @ (np.asarray(p, float) - (np.asarray(base, float) + R_body @ np.array(t)))
    if d[0] <= 0.05:
        return None
    return w / 2 - f * d[1] / d[0], h / 2 - f * d[2] / d[0]


def rig_link():
    """The six cameras on a massless link bolted to base_link."""
    cams = "".join(
        f"<sensor name='rig_{n}' type='camera'><pose>{' '.join(f'{v:.4f}' for v in (*t, *rpy))}</pose>"
        f"<topic>/rig/{n}</topic><update_rate>{HZ}</update_rate><always_on>1</always_on>"
        f"<camera><horizontal_fov>{HFOV:.6f}</horizontal_fov><image><width>{W}</width><height>{H}</height>"
        f"<format>R8G8B8</format></image><clip><near>0.05</near><far>300</far></clip></camera></sensor>"
        for n, (t, rpy) in RIG.items())
    return ("<link name='rig_link'><pose relative_to='base_link'>0 0 0 0 0 0</pose>"
            "<inertial><mass>0.001</mass><inertia><ixx>1e-7</ixx><iyy>1e-7</iyy><izz>1e-7</izz>"
            "<ixy>0</ixy><ixz>0</ixz><iyz>0</iyz></inertia></inertial>" + cams + "</link>"
            "<joint name='rig_joint' type='fixed'><parent>base_link</parent><child>rig_link</child></joint>")


def setup():
    import gz_learner as G
    d = f"{HERE}/gz_logan/models/x500_rig"
    os.makedirs(d, exist_ok=True)
    open(f"{d}/model.config", "w").write(
        "<?xml version='1.0'?><model><name>x500_rig</name><version>1.0</version><sdf version='1.9'>model.sdf</sdf>"
        "<description>PX4 x500 with its downward landing camera and a six-camera 360-degree rig</description></model>\n")
    open(f"{d}/model.sdf", "w").write(
        "<?xml version='1.0'?>\n<sdf version='1.9'><model name='x500_rig'><self_collide>false</self_collide>"
        "<include merge='true'><uri>model://x500_mono_cam_down</uri></include>" + rig_link() + "</model></sdf>\n")
    src = open(f"{HERE}/gz_logan/worlds/aruco.sdf").read()
    src, n = re.subn(r'<model name="observer_cam">.*?</model>', "", src, flags=re.S)
    assert n == 1 and src.count("  </world>") == 1 and "<plugin" not in src
    magpie, nest = G.bird_sdf()
    bench = ("<model name='rig'><static>true</static><include merge='true'><uri>model://x500_base</uri></include>"
             + rig_link() + "</model>")
    os.makedirs(f"{HERE}/gz_logan/rig", exist_ok=True)
    open(f"{HERE}/gz_logan/rig/aruco.sdf", "w").write(
        src.replace("  </world>", f"    {magpie}\n    {nest}\n    {bench}\n  </world>"))
    # the world PX4 flies the rig drone in: Logan (overhead camera kept, for the video), the magpie,
    # its nest, the chase camera, and x500_rig already standing at the depot -- PX4 attaches to it
    # (PX4_GZ_MODEL_NAME) instead of spawning its own. It keeps the old instance name, so the
    # landing camera's topic (px4_perception.CAM_TOPIC) and every script that uses it are unchanged.
    src = open(f"{HERE}/gz_logan/worlds/aruco.sdf").read()
    drone = "<include><uri>model://x500_rig</uri><name>x500_mono_cam_down_0</name><pose>0 0 0 0 0 0</pose></include>"
    os.makedirs(f"{HERE}/gz_logan/rigfly", exist_ok=True)
    open(f"{HERE}/gz_logan/rigfly/aruco.sdf", "w").write(
        src.replace("  </world>", f"    {magpie}\n    {nest}\n    {G.chase_sdf()}\n    {drone}\n  </world>"))
    print(d, f"{HERE}/gz_logan/rig/aruco.sdf", f"{HERE}/gz_logan/rigfly/aruco.sdf")


def bench():
    """Headless Gazebo with the bench world, in its partition; returns at once (log: gz_logan/<BENCH>/gz.log)."""
    env = dict(os.environ, GZ_PARTITION=PARTITION,
               GZ_SIM_RESOURCE_PATH=f"{PX4}/Tools/simulation/gz/models:{HERE}/gz_logan/models",
               GZ_SIM_SERVER_CONFIG_PATH=f"{PX4}/src/modules/simulation/gz_bridge/server.config",
               GZ_SIM_SYSTEM_PLUGIN_PATH=f"{PX4}/build/px4_sitl_default/src/modules/simulation/gz_plugins",
               # this laptop's screen runs on the Intel chip: without these the bench rendered there, at
               # 0.12x real time with six cameras; on the NVIDIA card it is several times faster
               __NV_PRIME_RENDER_OFFLOAD="1", __GLX_VENDOR_LIBRARY_NAME="nvidia",
               __EGL_VENDOR_LIBRARY_FILENAMES="/usr/share/glvnd/egl_vendor.d/10_nvidia.json")
    p = subprocess.Popen(["bash", "-c", "source /opt/ros/jazzy/setup.bash; exec gz sim --verbose=1 -r -s "
                          f"{HERE}/gz_logan/{BENCH}/aruco.sdf"], env=env, start_new_session=True,
                         stdout=open(f"{HERE}/gz_logan/{BENCH}/gz.log", "w"), stderr=subprocess.STDOUT)
    print(p.pid)


class Cams:
    """The six rig cameras' newest images, with their simulated-time stamps."""

    def __init__(self):
        from gz.transport13 import Node
        from gz.msgs10.image_pb2 import Image
        self.last, self.node = {}, Node()
        for n in RIG:
            self.node.subscribe(Image, f"/rig/{n}", lambda m, n=n: self._on(n, m))

    def _on(self, n, m):
        self.last[n] = (m.header.stamp.sec + m.header.stamp.nsec * 1e-9, m)

    def now(self):
        return max((s for s, _ in self.last.values()), default=0.0)

    def fresh(self, k=1, timeout=60.0):
        """Six images all taken at least k frames after this call."""
        after, t0 = self.now() + k / HZ, time.time()
        while time.time() - t0 < timeout:
            got = dict(self.last)
            if len(got) == len(RIG) and min(s for s, _ in got.values()) >= after:
                return {n: np.frombuffer(m.data, np.uint8).reshape(m.height, m.width, 3) for n, (_, m) in got.items()}
            time.sleep(0.005)
        raise TimeoutError(f"rig cameras silent: {sorted(self.last)} -- is the bench running (--bench)?")

    def warm(self, timeout=600.0):
        """After a (re)start Gazebo keeps adding Logan's trees and houses to the camera views for a
        while after the first subscriber (two cover runs died on it: frames 'settled', then changed
        by 71 levels). Wait until the views stay identical over a full simulated second."""
        t0 = time.time()
        while time.time() - t0 < timeout:
            a, b = self.fresh(1), self.fresh(HZ)
            if all(np.array_equal(a[n], b[n]) for n in RIG):
                return
        raise RuntimeError("the bench never stopped changing")

    def settled(self, tries=20):
        """Six images of a scene that has stopped changing. A set_pose shows up 1-3 frames after the
        request (measured on the bench), so: wait 4 frames, then accept only a picture the next
        frame repeats exactly -- a half-applied move is never measured."""
        a = self.fresh(4)
        for _ in range(tries):
            b = self.fresh(1)
            if all(np.array_equal(a[n], b[n]) for n in RIG):
                return b
            a = b
        raise RuntimeError("the bench scene never settled -- something in it is moving")


def sphere(n):
    """n directions spread evenly over the sphere (Fibonacci lattice)."""
    k = np.arange(n) + 0.5
    z = 1 - 2 * k / n
    a = math.pi * (3 - math.sqrt(5)) * k
    return np.c_[np.sqrt(1 - z * z) * np.cos(a), np.sqrt(1 - z * z) * np.sin(a), z]


def cover(out, n_dir=160, ranges=(5, 10, 20, 30, 40, 60), thresh=12):
    import cv2
    import gz_learner as G
    os.makedirs(out, exist_ok=True)
    mover, cams = G.Mover(), Cams()
    cams.fresh()                                              # all six cameras are streaming
    base = np.array(BENCH_AT)
    mover.put("rig", base - [0, 0, BASE_Z])
    for name in ("magpie", "magpie_nest"):
        mover.put(name, [0, 0, -50])                          # out of sight, under the ground
    cams.warm()
    bg, bg2 = cams.settled(), cams.fresh(2)
    flicker = max(int(np.abs(bg[n].astype(np.int16) - bg2[n]).max()) for n in RIG)
    assert flicker <= 2, f"two renders of the same still scene differ by {flicker} -- differencing would lie"
    for n in RIG:
        cv2.imwrite(f"{out}/bg_{n}.jpg", cv2.cvtColor(bg[n], cv2.COLOR_RGB2BGR))
    rows, t0 = [], time.time()
    for r in ranges:
        for k, u in enumerate(sphere(n_dir)):
            p = base + r * u
            if p[2] < 1.5:
                continue
            for pose in ("head-on", "broadside"):
                to = base - p
                yaw, pitch = math.atan2(to[1], to[0]), -math.atan2(to[2], math.hypot(to[0], to[1]))
                if pose == "broadside":
                    yaw, pitch = yaw + math.pi / 2, 0.0
                mover.put("magpie", p, yaw=yaw, pitch=pitch)
                img = cams.settled()
                row = dict(range=r, dir=k, u=[round(float(v), 4) for v in u], pose=pose, cams={})
                for n in RIG:
                    m = np.abs(img[n].astype(np.int16) - bg[n]).max(axis=2) > thresh
                    q = project(p, base, np.eye(3), n)
                    c = dict(px=int(m.sum()), pred=None if q is None else [round(q[0], 1), round(q[1], 1)])
                    if c["px"]:
                        ys, xs = np.nonzero(m)
                        c.update(box=[int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
                                 mid=[round(float(xs.mean()), 1), round(float(ys.mean()), 1)])
                    row["cams"][n] = c
                rows.append(row)
        print(f"range {r:2d} m done: {len(rows)} placements, {time.time() - t0:.0f} s", flush=True)
    json.dump(dict(rig={n: [list(t), list(a)] for n, (t, a) in RIG.items()}, W=W, H=H, hfov_deg=110, cams=CAMSZ,
                   base=list(BENCH_AT), thresh=thresh, flicker=flicker, rows=rows), open(f"{out}/cover.json", "w"))
    check(rows)
    print(f"{out}/cover.json")


def check(rows):
    """The camera model must agree with Gazebo: where a camera sees a clear blob (>= 30 px, away from
    the image edge), the blob's centre must sit on the pixel project() predicts. Fails loudly."""
    err = [math.dist(c["mid"], c["pred"]) for row in rows for n, c in row["cams"].items()
           if c["px"] >= 30 and c["pred"] and 40 < c["pred"][0] < CAMSZ[n][0] - 40 and 40 < c["pred"][1] < CAMSZ[n][1] - 40]
    assert len(err) >= 20, f"only {len(err)} clear sightings to check the camera model with"
    med = float(np.median(err))
    print(f"camera model check: {len(err)} sightings, blob centre vs prediction median {med:.1f} px, "
          f"90% {np.percentile(err, 90):.1f} px")
    assert med < 5.0, f"camera frames are off by {med:.1f} px -- fix RIG/BASE_Z before trusting anything"


def negatives(out, n_places=200, seed=1, first=10000):
    """Hard negatives: the rig over the landing-pad area (black-and-white markers), 5-50 m above what
    is there, any attitude, NO magpie -- all six views, labelled empty. The first live rig flights
    tracked the pads' ArUco squares and tree shadows as birds (25 Sept). Places numbered from
    `first` so rig_detect.load splits them like the rest (every 10th held out)."""
    import cv2
    import gz_learner as G
    rng = np.random.default_rng(seed)
    mover, cams, lab = G.Mover(), Cams(), open(f"{out}/labels.jsonl", "a")
    cams.fresh(); cams.warm()
    m = np.load(f"{HERE}/urban/logan_map_lidar.npz")
    Hm, half, res = m["H"], float(m["half"]), float(m["res"])
    top = lambda xy: float(Hm[tuple(np.clip(((np.asarray(xy) + G.OFFSET + half) / res).astype(int), 0, Hm.shape[0] - 1)[::-1])])
    mover.put("magpie", [0, 0, -50])
    for i in range(n_places):
        base = np.array([rng.uniform(55, 185), rng.uniform(-90, 90), 0.0])
        base[2] = top(base[:2]) + rng.uniform(5, 50)
        roll, pitch, yaw = rng.uniform(-0.2, 0.2), rng.uniform(-0.2, 0.2), rng.uniform(-math.pi, math.pi)
        mover.put("rig", base - rot(roll, pitch, yaw) @ [0, 0, BASE_Z], yaw=yaw, pitch=pitch, roll=roll)
        img = cams.settled()
        rig = [round(float(v), 4) for v in (*base, roll, pitch, yaw)]
        for n in RIG:
            f = f"img/{first + i:05d}_neg_{n}.jpg"
            cv2.imwrite(f"{out}/{f}", cv2.cvtColor(img[n], cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
            lab.write(json.dumps(dict(file=f, cam=n, rig=rig, bird=None, box=None, px=0)) + "\n")
        lab.flush()
    print(f"{n_places} places x 6 empty views -> {out}/labels.jsonl")


def data(out, n_poses=600, per_pose=6, seed=0, thresh=12):
    """Detector training pictures, labelled by Gazebo itself. The bench drone goes to random places
    over Logan (3-45 m up, any heading, tilted up to 11 degrees as in flight); the magpie goes to
    random spots around it, 2-70 m away -- half anywhere, half behind and above (where the new
    magpie attacks from) -- mostly pointing at the drone as in a dive. Each camera image where the
    bird shows is saved with the box around the part that shows (render difference), plus two
    empty views per place: what the detector must NOT fire on. labels.jsonl: one line per image."""
    import cv2
    import gz_learner as G
    rng = np.random.default_rng(seed)
    os.makedirs(f"{out}/img", exist_ok=True)
    mover, cams, lab = G.Mover(), Cams(), open(f"{out}/labels.jsonl", "a")
    cams.fresh(); cams.warm()

    def save(tag, n, im, rig, bird=None, m=None):
        rec = dict(file=f"img/{tag}_{n}.jpg", cam=n, rig=rig, bird=bird, box=None, px=0)
        if m is not None:
            ys, xs = np.nonzero(m)
            rec.update(box=[int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())], px=int(m.sum()))
        cv2.imwrite(f"{out}/{rec['file']}", cv2.cvtColor(im, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
        lab.write(json.dumps(rec) + "\n")

    m = np.load(f"{HERE}/urban/logan_map_lidar.npz")                 # LiDAR heights, training-map frame
    Hm, half, res = m["H"], float(m["half"]), float(m["res"])
    top = lambda xy: float(Hm[tuple(np.clip(((np.asarray(xy) + G.OFFSET + half) / res).astype(int), 0, Hm.shape[0] - 1)[::-1])])
    t0 = time.time()
    for i in range(n_poses):
        base = np.array([rng.uniform(0, 180), rng.uniform(-90, 90), 0.0])
        base[2] = top(base[:2]) + rng.uniform(3, 45)                 # never inside a roof or a canopy
        roll, pitch, yaw = rng.uniform(-0.2, 0.2), rng.uniform(-0.2, 0.2), rng.uniform(-math.pi, math.pi)
        Rb = rot(roll, pitch, yaw)
        mover.put("rig", base - Rb @ [0, 0, BASE_Z], yaw=yaw, pitch=pitch, roll=roll)
        mover.put("magpie", [0, 0, -50])
        rig = [round(float(v), 4) for v in (*base, roll, pitch, yaw)]
        bg = cams.settled()
        for n in rng.choice(list(RIG), 2, replace=False):
            save(f"{i:05d}_bg", n, bg[n], rig)
        for j in range(per_pose):
            if rng.random() < 0.5:
                u = rng.normal(size=3)
            else:                                            # behind the nose, 0-45 degrees above
                az, el = yaw + math.pi + rng.uniform(-0.7, 0.7), rng.uniform(0, 0.8)
                u = np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
            u /= np.linalg.norm(u)
            p = base + math.exp(rng.uniform(math.log(2), math.log(70))) * u
            p[2] = max(p[2], top(p[:2]) + rng.uniform(0.5, 3))            # a magpie flies above what is there
            to = base - p
            b_yaw, b_pitch, b_roll = (math.atan2(to[1], to[0]) + rng.normal(0, 0.4),
                                      -math.atan2(to[2], math.hypot(to[0], to[1])) + rng.normal(0, 0.3), rng.normal(0, 0.3))
            if rng.random() < 0.2:                           # some not diving at the drone at all
                b_yaw, b_pitch, b_roll = rng.uniform(-math.pi, math.pi), rng.uniform(-0.8, 0.8), rng.uniform(-0.8, 0.8)
            mover.put("magpie", p, yaw=b_yaw, pitch=b_pitch, roll=b_roll)
            img = cams.settled()
            bird = [round(float(v), 4) for v in (*p, b_roll, b_pitch, b_yaw)]
            for n in RIG:                                    # label only pixels near where it projects
                w, h, f = CAMSZ[n]
                reach = f * 0.6 / np.linalg.norm(p - base) + 8   # the bird's half-size in pixels, plus slack
                q = project(p, base, Rb, n)
                if q is None or not (-reach < q[0] < w + reach and -reach < q[1] < h + reach):
                    continue
                x0, x1 = int(max(q[0] - reach, 0)), int(min(q[0] + reach, w))
                y0, y1 = int(max(q[1] - reach, 0)), int(min(q[1] + reach, h))
                m = np.zeros((h, w), bool)
                m[y0:y1, x0:x1] = np.abs(img[n][y0:y1, x0:x1].astype(np.int16) - bg[n][y0:y1, x0:x1]).max(axis=2) > thresh
                if m.any():
                    save(f"{i:05d}_{j}", n, img[n], rig, bird, m)
        lab.flush()
        if i % 20 == 19:
            print(f"{i + 1} places, {time.time() - t0:.0f} s", flush=True)
    print(f"{out}/labels.jsonl")


def calib(cover_dir, data_dir, out=f"{HERE}/urban/rig_calib.json", score="score.json"):
    """The numbers urban/rig_sense.Rig flies with, all measured: from the coverage bench, the size of
    the magpie by distance (head-on, side-on) and the directions the drone's own frame blinds; from
    the detector's held-out score, detection by size in pixels and its direction and size errors;
    from the labelled pictures, how far off a distance read from the bird's size is (poses vary).
    score: which detector's held-out score (score_yolo.json: rig_yolo.py) -- written to its own out file."""
    cv = json.load(open(f"{cover_dir}/cover.json"))
    rows, sc = cv["rows"], json.load(open(f"{data_dir}/{score}"))
    span = {}
    k_det = {n: min(1.0, 1280 / max(w, h)) for n, (w, h, _) in CAMSZ.items()}    # the detector's shrink (rig_detect.score)
    for r in rows:
        for n, c in r["cams"].items():
            w, h, f = CAMSZ[n]
            if c["px"] >= 20 and 20 < c["box"][0] and c["box"][2] < w - 20 and 20 < c["box"][1] and c["box"][3] < h - 20:
                span.setdefault(r["pose"], []).append((max(c["box"][2] - c["box"][0], c["box"][3] - c["box"][1]) + 1) * r["range"] / f)
    head, side = float(np.median(span["head-on"])), float(np.median(span["broadside"]))
    assumed = math.sqrt(head * side)
    # rig v2: a sighting counts only outside that camera's propeller mask -- live, detections inside it are dropped
    # (a blade looks like a bird), so there the rig is blind (the six-camera rig's published calibration ignored it)
    pm = dict(np.load(f"{HERE}/gz_logan/{BENCH}/prop_mask.npz")) if V2.startswith("v2") else None
    clear = lambda n, c: pm is None or not pm[n][min(int(c["mid"][1]), pm[n].shape[0] - 1), min(int(c["mid"][0]), pm[n].shape[1] - 1)]
    n_el, n_az = 6, 12                                        # blind spots, body frame (the bench rig faces +x)
    vis, cnt = np.zeros((n_el, n_az)), np.zeros((n_el, n_az))
    foc = [[[] for _ in range(n_az)] for _ in range(n_el)]    # the focal length, in detector pixels, of the camera
                                                              # that shows the bird biggest from each direction
    for r in rows:
        if r["range"] > 10:                                   # close: only the drone's own frame can hide it
            continue
        u = np.array(r["u"])
        ia = min(int((math.atan2(u[1], u[0]) + math.pi) / (2 * math.pi) * n_az), n_az - 1)
        ie = min(int((math.asin(np.clip(u[2], -1, 1)) + math.pi / 2) / math.pi * n_el), n_el - 1)
        cnt[ie, ia] += 1
        vis[ie, ia] += any(c["px"] > 0 and clear(n, c) for n, c in r["cams"].items())
        seen = [(c["px"], n) for n, c in r["cams"].items() if c["px"] > 0 and clear(n, c)]
        if seen:
            n = max(seen)[1]
            foc[ie][ia].append(CAMSZ[n][2] * k_det[n])
    p_vis = np.where(cnt > 0, vis / np.maximum(cnt, 1), 1.0)
    f_wide = min(f * k_det[n] for n, (_, _, f) in CAMSZ.items())
    focal_map = [[float(np.median(v)) if v else f_wide for v in row] for row in foc]
    recs = [json.loads(l) for l in open(f"{data_dir}/labels.jsonl")]
    lr = [math.log(math.dist(q["bird"][:3], q["rig"][:3]) / (CAMSZ[q["cam"]][2] * assumed / (max(q["box"][2] - q["box"][0], q["box"][3] - q["box"][1]) + 1)))
          for q in recs if q["box"] and q["px"] >= 20 and 0 < q["box"][0] and q["box"][2] < CAMSZ[q["cam"]][0] - 1
          and 0 < q["box"][1] and q["box"][3] < CAMSZ[q["cam"]][1] - 1]
    sd_pose, sd_det = float(np.std(lr)), sc["log_size_error"]["sd"]
    # detection by size: only bins with >= 20 held-out sightings (the smallest were barely tested:
    # 2-4 px had 3, 4-7 px had 11), and none at 4 px (~110 m) -- not extrapolated from a handful
    pd = sc["p_detect"]
    n_bin = [v["n"] for v in sc["found_by_size_px"].values() if v["n"]]
    keep = [i for i, n_ in enumerate(n_bin) if n_ >= 20]
    p_detect = {k: [4.0 if k == "size_px" else 0.0] + [pd[k][i] for i in keep] for k in ("size_px", "p", "p_sky", "p_ground")}
    p_detect["sightings"] = [None] + [n_bin[i] for i in keep]
    # focal_px: the size the simulator gives the bird (detector pixels = p_detect's); focal_map: per direction, the
    # camera that shows it biggest (rig v2: the OAK-D ahead). The direction error: measured as an angle when the score
    # has it (per camera), else from pixels at the one focal length (the six-camera rig)
    sig_ang = sc["centre_error_rad"]["median"] / 1.1774 if sc.get("centre_error_rad") else sc["centre_error_px"]["median"] / 1.1774 / F
    c = dict(focal_px=f_wide, focal_map=focal_map, n_cameras=len(RIG), span_head_m=head, span_side_m=side, span_assumed_m=assumed,
             blind=dict(n_el=n_el, n_az=n_az, p_visible=p_vis.round(3).tolist(), samples=int(cnt.sum())),
             p_detect=p_detect, sig_ang_rad=sig_ang,
             sig_logsize=sd_det, sig_rel_dist=math.sqrt(sd_pose ** 2 + sd_det ** 2),
             source=dict(cover=cover_dir, data=data_dir, score=score, dist_from_size_log_sd_pose=sd_pose, n_pose=len(lr),
                         operating_threshold=sc["operating_threshold"],
                         false_alarms_per_image=sc["by_threshold"][str(sc["operating_threshold"])]["false_alarms_per_image"]))
    json.dump(c, open(out, "w"), indent=1)
    print(json.dumps({k: v for k, v in c.items() if k != "blind"}, indent=1))
    print(f"blind spots (share visible, rows = elevation from below): min {p_vis.min():.2f}")
    print(out)


def report(out):
    """cover.json -> rig_pixels.png: how big the magpie looks, and how often some camera sees it."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = json.load(open(f"{out}/cover.json"))
    rows = d["rows"]
    size = lambda row: max((max(c["box"][2] - c["box"][0], c["box"][3] - c["box"][1]) + 1
                            for c in row["cams"].values() if c["px"]), default=0)
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(13, 5), layout="constrained")
    ranges = sorted({r["range"] for r in rows})
    for pose, col in (("broadside", "#2e86c1"), ("head-on", "#c0392b")):
        pts = [(r["range"], size(r)) for r in rows if r["pose"] == pose and size(r)]
        x = np.array([p[0] for p in pts]) * np.exp(np.random.default_rng(0).normal(0, 0.03, len(pts)))
        ax.scatter(x, [p[1] for p in pts], s=6, alpha=0.25, color=col)
        med = [np.median([s for rr, s in pts if rr == r]) for r in ranges]
        ax.plot(ranges, med, "-o", color=col, lw=2, label=f"{pose} (median)")
        for r, m in zip(ranges, med):
            print(f"{pose:9s} {r:2d} m: magpie spans {m:5.1f} px (median)")
    ax.axhline(10, color="gray", ls="--", lw=1)
    ax.text(ranges[-1], 10.5, "10 px", ha="right", va="bottom", color="gray")
    ax.set(xscale="log", yscale="log", xlabel="Distance from the drone (m)", ylabel="Magpie's longest side in the image (px)",
           title="How big the magpie looks to the rig\n" + ("(its biggest view; 0.8 m wings)" if V2.startswith("v2") else
                                                              "(1280x960 cameras, 110-degree lenses; 0.8 m wings)"))
    if V2.startswith("v2"):                                   # rig v2: the OAK-D's own view, when the bird is ahead
        fr = [(r["range"], max(r["cams"]["front"]["box"][2] - r["cams"]["front"]["box"][0],
                               r["cams"]["front"]["box"][3] - r["cams"]["front"]["box"][1]) + 1)
              for r in rows if r["cams"]["front"]["px"] > 0]
        rg_f = [rg for rg in ranges if any(a == rg for a, _ in fr)]
        ax.plot(rg_f, [np.median([s_ for a, s_ in fr if a == rg]) for rg in rg_f], "--s", color="#27ae60", lw=2,
                label="ahead, OAK-D 1920x1080 69 deg (median)")
        ax.legend(fontsize=8)
    ax.set_xticks(ranges, [str(r) for r in ranges]); ax.grid(alpha=0.3, which="both"); ax.legend()
    w = 0.38
    # rig v2: also counted only outside the propeller masks (the live tracker drops detections inside them)
    pm = dict(np.load(f"{HERE}/gz_logan/{BENCH}/prop_mask.npz")) if V2.startswith("v2") else None
    clear_seen = lambda r: any(c["px"] > 0 and not pm[n][min(int(c["mid"][1]), pm[n].shape[0] - 1), min(int(c["mid"][0]), pm[n].shape[1] - 1)]
                               for n, c in r["cams"].items())
    for j, (band, sel, col) in enumerate((("magpie above the drone", lambda r: r["u"][2] > 0.05, "#16a085"),
                                          ("magpie below the drone", lambda r: r["u"][2] < -0.05, "#8e6e3b"))):
        frac = [100 * np.mean([size(r) > 0 for r in rows if r["range"] == rg and sel(r)] or [np.nan]) for rg in ranges]
        bx.bar(np.arange(len(ranges)) + (j - 0.5) * w, frac, w * 0.95, color=col, label=band, alpha=0.35 if pm else 1.0)
        print(band, " ".join(f"{rg}m {f:.0f}%" for rg, f in zip(ranges, frac)))
        if pm is not None:
            fm = [100 * np.mean([clear_seen(r) for r in rows if r["range"] == rg and sel(r)] or [np.nan]) for rg in ranges]
            bx.bar(np.arange(len(ranges)) + (j - 0.5) * w, fm, w * 0.95, color=col, label=band + ", outside the propellers")
            print(band, "outside the propellers", " ".join(f"{rg}m {f:.0f}%" for rg, f in zip(ranges, fm)))
    bx.set_xticks(range(len(ranges)), [f"{r} m" for r in ranges])
    bx.set(ylim=(0, 105), ylabel="Placements where some camera sees it (%)",
           title="Coverage: seen by at least one rig camera\n(below the drone, trees and roofs can hide it)")
    bx.grid(axis="y", alpha=0.3); bx.legend(loc="upper left", fontsize=8)
    both = np.mean([sum(c["px"] > 0 for c in r["cams"].values()) >= 2 for r in rows if size(r)])
    print(f"seen by two or more cameras: {100 * both:.0f}% of sightings")
    fig.savefig(f"{out}/rig_pixels.png", dpi=150)
    print(f"{out}/rig_pixels.png")


def views(out, cases=(("20 m behind", (-20, 0, 0)), ("20 m behind, 30 deg above", (-17.3, 0, 10)),
                      ("30 m to the left, level", (0, 30, 0)), ("12 m below, ahead", (6, 0, -12)))):
    """Contact sheet: the six rig views for a few magpie positions (relative to the drone, the
    magpie diving at it), the magpie ringed where the difference image finds it."""
    import cv2
    import gz_learner as G
    mover, cams = G.Mover(), Cams()
    cams.fresh()
    base = np.array(BENCH_AT)
    mover.put("rig", base - [0, 0, BASE_Z]); mover.put("magpie", [0, 0, -50])
    cams.warm()
    bg, rows = cams.settled(), []
    for label, rel in cases:
        p = base + rel
        to = -np.asarray(rel, float)
        mover.put("magpie", p, yaw=math.atan2(to[1], to[0]), pitch=-math.atan2(to[2], math.hypot(to[0], to[1])))
        img, tiles = cams.settled(), []
        for n in RIG:
            t = cv2.cvtColor(img[n], cv2.COLOR_RGB2BGR).copy()
            m = np.abs(img[n].astype(np.int16) - bg[n]).max(axis=2) > 12
            if m.any():
                ys, xs = np.nonzero(m)
                cv2.circle(t, (int(xs.mean()), int(ys.mean())), 40, (0, 0, 255), 4)
            t = cv2.resize(t, (320, 240))
            cv2.putText(t, n, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            tiles.append(t)
        head = np.full((34, 320 * len(RIG), 3), 30, np.uint8)
        cv2.putText(head, f"magpie {label}", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
        rows += [head, np.hstack(tiles)]
    cv2.imwrite(f"{out}/rig_views.jpg", np.vstack(rows))
    print(f"{out}/rig_views.jpg")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--setup", action="store_true")
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--cover", metavar="DIR")
    ap.add_argument("--report", metavar="DIR")
    ap.add_argument("--views", metavar="DIR")
    ap.add_argument("--data", metavar="DIR")
    ap.add_argument("--poses", type=int, default=600)
    ap.add_argument("--calib", nargs=2, metavar=("COVER_DIR", "DATA_DIR"))
    ap.add_argument("--score", default="score.json", help="--calib: the detector score to use (score_yolo.json)")
    ap.add_argument("--out", default=f"{HERE}/urban/rig_calib.json", help="--calib: where to write it")
    ap.add_argument("--negatives", metavar="DIR")
    ap.add_argument("--prop-mask", action="store_true", help="where each camera sees the propellers (gz_logan/<BENCH>)")
    a = ap.parse_args()
    os.environ["GZ_PARTITION"] = PARTITION               # before any gz transport node exists
    if a.setup:
        setup()
    if a.bench:
        bench()
    if a.cover:
        cover(a.cover)
    if a.report:
        report(a.report)
    if a.views:
        views(a.views)
    if a.data:
        data(a.data, a.poses)
    if a.calib:
        calib(*a.calib, out=a.out, score=a.score)
    if a.negatives:
        negatives(a.negatives)
    if a.prop_mask:
        prop_mask()
    sys.stdout.flush()
    os._exit(0)                                          # gz transport's exit-time segfault is harmless; skip it
