#!/usr/bin/env python3
"""
gz_unified.py -- one sensor rig for the birds and the obstacles (Sai, 25 Sept): the x500's landing camera looking
down, an OAK-D Lite looking ahead (its colour camera for the magpie, its stereo depth for the voxel map), and
three wide cameras looking left, right and back. The four horizontal cameras sit on a mast above the propellers
(the first rig, at body height, saw its own propellers and legs); each is held by a bracket to a hub on the mast.
No camera looks straight up (the six-camera rig had one). v2: all under the body, bottom camera on a gimbal.

    python3 gz_unified.py --setup   -> gz_logan/models/x500_unified, gz_logan/photo/studio.sdf
    python3 gz_unified.py --photo   -> gz_logan/photo/rig_*.png (Gazebo renders of the drone, two angles)
"""
import argparse
import math
import os
import re
import subprocess
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PX4 = "/home/sai/thesis_2.0/PX4-Autopilot"
# v2 (Sai, 25 Sept): no mast -- it raised the drone's box and the magpie's target; every camera now hangs on a
# short bracket UNDER the body plate, below the propellers' plane; the bottom camera is PX4's stabilised CGO3
# gimbal (model://gimbal, as x500_gimbal mounts it), not a box stuck to the belly.
PLATE_Z, CAM_Z = -0.018, -0.045    # the x500's lower body plate; the camera housings' centres just under it
ALU, BLACK = "0.72 0.72 0.75", "0.08 0.08 0.09"
# name: (position on base_link, yaw) -- the wide cameras keep the six-camera rig's names and optics
WIDE = {"left": ((0.0, 0.07, CAM_Z), math.pi / 2), "right": ((0.0, -0.07, CAM_Z), -math.pi / 2),
        "back": ((-0.07, 0.0, CAM_Z), math.pi)}
OAKD = (0.085, 0.0, CAM_Z)
# Variants (Sai, 25 Sept evening), each answering one question:
#   v2   as built: PX4's CGO3 gimbal landing camera; OAK-D ahead; wide cameras left, right, back, level
#   v2f  x500_mono_cam_down's FIXED downward camera instead of the gimbal -- did the gimbal pay for itself?
#   v2t  the back camera tilted up 30 deg -- the magpie attacks from behind and above
#   v2u  v2t plus a wide camera on the top plate looking straight up
#   v2l  that up camera with the back camera left level (the tilt bought 3 points above and cost 8 below)
VARIANTS = {"v2": dict(gimbal=True, back_up=0.0, up=False), "v2f": dict(gimbal=False, back_up=0.0, up=False),
            "v2t": dict(gimbal=True, back_up=30.0, up=False), "v2u": dict(gimbal=True, back_up=30.0, up=True),
            "v2l": dict(gimbal=True, back_up=0.0, up=True)}
TOP_Z = 0.051                         # the x500 frame's top at its centre (NXP-HGD-CF mesh: 0.0264 + its 0.025 pose)
UP_CAM = (0.0, 0.0, TOP_Z + 0.019)    # a 38 mm cube camera standing on it; the props are ~110 deg off its axis


def model_name(v):
    return "x500_unified" if v == "v2" else f"x500_unified_{v}"


def cams(v="v2"):
    """A variant's magpie cameras exactly as rig_link mounts their sensors: {name: ((x, y, z) on base_link, (roll,
    pitch, yaw), width, height, horizontal fov)}. gz_rig.py builds its tables from this: the bench can't drift."""
    V = VARIANTS[v]
    x, y, z = OAKD
    out = {"front": ((x + 0.01, 0.0, z), (0.0, 0.0, 0.0), 1920, 1080, 1.204)}              # the OAK-D's colour camera
    for n, ((x_, y_, z_), yw) in WIDE.items():
        pitch = -math.radians(V["back_up"]) if n == "back" else 0.0                      # negative pitch: tilted up
        out[n] = ((x_ + 0.02 * math.cos(yw), y_ + 0.02 * math.sin(yw), z_), (0.0, pitch, yw), 1280, 960, 1.919862)
    if V["up"]:
        out["up"] = ((UP_CAM[0], UP_CAM[1], UP_CAM[2] + 0.02), (0.0, -math.pi / 2, 0.0), 1280, 960, 1.919862)
    return out


def _vis(name, xyz, size, rgb, rpy=(0, 0, 0), shape="box"):
    geo = f"<box><size>{size[0]} {size[1]} {size[2]}</size></box>" if shape == "box" else \
        f"<cylinder><radius>{size[0]}</radius><length>{size[1]}</length></cylinder>"
    return (f"<visual name='{name}'><pose>{xyz[0]} {xyz[1]} {xyz[2]} {rpy[0]} {rpy[1]} {rpy[2]}</pose><geometry>{geo}</geometry>"
            f"<material><ambient>{rgb} 1</ambient><diffuse>{rgb} 1</diffuse><specular>0.3 0.3 0.3 1</specular></material></visual>")


def _cam(name, xyz, yaw, topic, w, h, hfov, hz=20, kind="camera", fmt="R8G8B8", clip=(0.05, 300), pitch=0.0):
    return (f"<sensor name='{name}' type='{kind}'><pose>{xyz[0]} {xyz[1]} {xyz[2]} 0 {pitch} {yaw}</pose><topic>{topic}</topic>"
            f"<update_rate>{hz}</update_rate><always_on>1</always_on><camera><horizontal_fov>{hfov}</horizontal_fov>"
            f"<image><width>{w}</width><height>{h}</height><format>{fmt}</format></image>"
            f"<clip><near>{clip[0]}</near><far>{clip[1]}</far></clip></camera></sensor>")


def rig_link(var="v2"):
    v, C = [], cams(var)
    for n, ((x, y, z), yaw) in WIDE.items():                  # a 38 mm cube camera under the plate, on an L-bracket
        c, s = math.cos(yaw), math.sin(yaw)
        pt = C[n][1][1]                                       # its pitch (the back camera tilted up in v2t, v2u)
        v += [_vis(f"{n}_bracket", (x - 0.012 * c, y - 0.012 * s, (PLATE_Z + z) / 2 + 0.004), (0.012, 0.034, abs(PLATE_Z - z) + 0.008),
                   ALU, (0, 0, yaw)),
              _vis(f"{n}_cam", (x, y, z), (0.03, 0.038, 0.038), BLACK, (0, pt, yaw)),
              _vis(f"{n}_lens", (x + 0.017 * math.cos(pt) * c, y + 0.017 * math.cos(pt) * s, z - 0.017 * math.sin(pt)), (0.007, 0.006),
                   "0.02 0.02 0.05", (0, math.pi / 2 + pt, yaw), "cyl")]
    if "up" in C:                                             # on the top plate, on a small base, lens up
        x, y, z = UP_CAM
        v += [_vis("up_base", (x, y, TOP_Z + 0.002), (0.05, 0.05, 0.004), ALU),
              _vis("up_cam", (x, y, z), (0.038, 0.038, 0.03), BLACK),
              _vis("up_lens", (x, y, z + 0.017), (0.007, 0.006), "0.02 0.02 0.05", (0, 0, 0), "cyl")]
    x, y, z = OAKD                                             # OAK-D Lite, 91 x 28 x 17.5 mm, on its bracket
    v += [_vis("oakd_bracket", (x - 0.012, 0, (PLATE_Z + z) / 2 + 0.004), (0.012, 0.06, abs(PLATE_Z - z) + 0.008), ALU),
          _vis("oakd", (x, 0, z), (0.0175, 0.091, 0.028), "0.15 0.15 0.17"),
          _vis("oakd_face", (x + 0.009, 0, z), (0.002, 0.085, 0.022), "0.04 0.04 0.05")]
    s = [_cam(f"rig_{n}", pos, rpy[2], f"/rig/{n}", w, h, hf, pitch=rpy[1]) for n, (pos, rpy, w, h, hf) in C.items() if n != "front"]
    s += [_cam("oakd_rgb", (x + 0.01, 0, z), 0.0, "/rig/front", 1920, 1080, 1.204, hz=20),   # IMX214, 69 deg
          _cam("oakd_depth", (x + 0.01, 0, z), 0.0, "/depth_camera", 640, 480, 1.274, hz=15, kind="depth_camera",
               fmt="R_FLOAT32", clip=(0.2, 19.1))]
    return ("<link name='rig_link'><pose relative_to='base_link'>0 0 0 0 0 0</pose><inertial><mass>0.12</mass><inertia>"
            "<ixx>1e-4</ixx><iyy>1e-4</iyy><izz>1e-4</izz><ixy>0</ixy><ixz>0</ixz><iyz>0</iyz></inertia></inertial>"
            + "".join(v) + "".join(s) + "</link><joint name='rig_joint' type='fixed'><parent>base_link</parent>"
            "<child>rig_link</child></joint>")


def photo_cam(name, eye, at=(0.0, 0.0, 0.3), hfov=0.62):
    d = np.subtract(at, eye)
    yaw, pitch = math.atan2(d[1], d[0]), math.atan2(-d[2], math.hypot(d[0], d[1]))
    return (f"<model name='{name}'><static>true</static><pose>{eye[0]} {eye[1]} {eye[2]} 0 {pitch} {yaw}</pose><link name='l'>"
            + _cam(name, (0, 0, 0), 0.0, f"/photo/{name}", 1600, 1200, hfov, hz=5) + "</link></model>")


def setup():
    for var, V in VARIANTS.items():
        name = model_name(var)
        d = f"{HERE}/gz_logan/models/{name}"
        os.makedirs(d, exist_ok=True)
        open(f"{d}/model.config", "w").write(
            f"<?xml version='1.0'?><model><name>{name}</name><version>1.0</version><sdf version='1.9'>model.sdf</sdf>"
            f"<description>x500 ({var}): {'CGO3 gimbal' if V['gimbal'] else 'fixed'} camera down; OAK-D Lite ahead (RGB + depth); "
            f"wide cameras left, right, back{' (tilted up %g deg)' % V['back_up'] if V['back_up'] else ''}, all under the body"
            f"{'; a wide camera on top looking up' if V['up'] else ''}</description></model>\n")
        body = ("<include merge='true'><uri>model://x500</uri></include>"
                "<include merge='true'><uri>model://gimbal</uri><pose>0 0 0.26 0 0 3.14</pose></include>"   # as x500_gimbal
                "<joint name='GimbalAttachJoint' type='fixed'><parent>base_link</parent><child>cgo3_mount_link</child></joint>"
                if V["gimbal"] else "<include merge='true'><uri>model://x500_mono_cam_down</uri></include>")
        open(f"{d}/model.sdf", "w").write(
            f"<?xml version='1.0'?>\n<sdf version='1.9'><model name='{name}'><self_collide>false</self_collide>"
            + body + rig_link(var) + "</model></sdf>\n")
    os.makedirs(f"{HERE}/gz_logan/photo", exist_ok=True)
    open(f"{HERE}/gz_logan/photo/studio.sdf", "w").write(f"""<?xml version="1.0"?>
<sdf version="1.9"><world name="studio">
  <scene><ambient>0.55 0.55 0.58 1</ambient><background>0.85 0.88 0.92 1</background><shadows>true</shadows><grid>false</grid></scene>
  <light type="directional" name="sun"><cast_shadows>true</cast_shadows><pose>0 0 10 0 0 0</pose>
    <diffuse>0.9 0.9 0.88 1</diffuse><specular>0.3 0.3 0.3 1</specular><direction>-0.4 0.3 -0.85</direction></light>
  <model name="floor"><static>true</static><link name="l"><visual name="v"><geometry><plane><normal>0 0 1</normal><size>20 20</size></plane></geometry>
    <material><ambient>0.62 0.63 0.6 1</ambient><diffuse>0.66 0.67 0.64 1</diffuse></material></visual></link></model>
  <include><uri>model://x500_unified</uri><name>drone</name><pose>0 0 0 0 0 0</pose><static>true</static></include>
  {photo_cam("front_left", (0.95, 0.78, 0.55), at=(0.0, 0.0, 0.2))}
  {photo_cam("rear_right", (-0.9, -0.85, 0.6), at=(0.0, 0.0, 0.2))}
  {photo_cam("side", (0.0, 1.2, 0.22), at=(0.0, 0.0, 0.18), hfov=0.55)}
  {photo_cam("under", (0.75, 0.35, 0.05), at=(0.0, 0.0, 0.16), hfov=0.6)}
</world></sdf>
""")
    print(d, f"{HERE}/gz_logan/photo/studio.sdf")
    worlds()


def worlds():
    """The flight worlds again with x500_unified as the drone PX4 attaches to: gz_logan/{voxfly,voxfast,rigfly}_v2
    (voxfly.sh / rigfly.sh with RIG=v2). Rebuild after gz_voxel.py / gz_rig.py --setup rewrites the originals."""
    for var in VARIANTS:
        for w in ("voxfly", "voxfast", "rigfly"):
            src = open(f"{HERE}/gz_logan/{w}/aruco.sdf").read()
            src, n = re.subn(r"<uri>model://x500_(depth_down|rig)</uri>(<name>x500_mono_cam_down_0</name>)",
                             rf"<uri>model://{model_name(var)}</uri>\2", src)
            assert n == 1, (w, n)
            os.makedirs(f"{HERE}/gz_logan/{w}_{var}", exist_ok=True)
            open(f"{HERE}/gz_logan/{w}_{var}/aruco.sdf", "w").write(src)
        # the magpie bench (gz_rig.py under RIG=<var>): Logan, the magpie, its nest, this drone standing static as "rig"
        src = open(f"{HERE}/gz_logan/rig/aruco.sdf").read()
        src, n = re.subn(r"<model name='rig'>.*?</model>",
                         f"<include><uri>model://{model_name(var)}</uri><name>rig</name><pose>0 0 0 0 0 0</pose><static>true</static></include>",
                         src, count=1, flags=re.S)
        assert n == 1
        os.makedirs(f"{HERE}/gz_logan/rig_{var}", exist_ok=True)
        open(f"{HERE}/gz_logan/rig_{var}/aruco.sdf", "w").write(src)
        print(f"{var}: gz_logan/{{voxfly,voxfast,rigfly,rig}}_{var}/aruco.sdf")


def check(folder):
    """Rig v2's landing camera, measured on flights: where each pad marker SHOULD appear, from where Gazebo had the
    drone and the gimbal camera's own IMU attitude (px4_mission.cam_ray's model, inverted), against where the detector
    found it. Fails loudly if the model is off (median > 10 px). Also how far PX4 kept the camera from straight down."""
    import glob
    os.environ["RIG"] = "v2"
    import px4_mission as PM
    res, gnd, tilt = [], [], []
    for f in sorted(glob.glob(f"{folder}/*.csv")):
        d = np.genfromtxt(f, delimiter=",", names=True)
        if "cq_w" not in d.dtype.names:
            continue
        up = d["gz_z"] > 3.0
        tilt += list(d["cam_tilt"][up & np.isfinite(d["cam_tilt"])])
        for r in d[up & np.isfinite(d["mk_ex"]) & np.isfinite(d["gz_x"]) & np.isfinite(d["cq_w"])]:
            g = PM.GimbalAtt.__new__(PM.GimbalAtt); g.q = (r["cq_w"], r["cq_x"], r["cq_y"], r["cq_z"])
            n, e = np.array(PM.PAD_XY, float) + PM.PAD_OFF_NED[int(r["mk_id"])]
            dc = g.R().T @ (np.array([e, n, 0.0]) - np.array([r["gz_x"], r["gz_y"], r["gz_z"] + 0.08]))  # camera ~8 cm over the feet
            px = np.array([-PM.F_PX * dc[1] / dc[0], -PM.F_PX * dc[2] / dc[0]])
            res.append(np.linalg.norm(px - [r["mk_ex"], r["mk_ey"]]))
            gnd.append(res[-1] * r["gz_z"] / PM.F_PX)
    tilt, res, gnd = np.array(tilt), np.array(res), np.array(gnd)
    print(f"{folder}: {len(res)} marker frames; predicted vs detected: median {np.median(res):.1f} px, 95% "
          f"{np.percentile(res, 95):.1f} px = {np.median(gnd):.2f} / {np.percentile(gnd, 95):.2f} m on the ground")
    print(f"camera from straight down, airborne: median {np.median(tilt):.2f}, 95% {np.percentile(tilt, 95):.2f}, "
          f"max {tilt.max():.2f} deg ({len(tilt)} rows)")
    assert np.median(res) < 10, "the gimbal camera model does not match what the camera sees"


def photo(var="v2"):
    """Studio renders of a variant (v2: gz_logan/photo/rig_<angle>.png; others: rig_<var>_<angle>.png)."""
    import cv2
    studio = f"{HERE}/gz_logan/photo/studio.sdf"
    if var != "v2":                                            # the same studio with that variant's drone
        src = open(studio).read().replace("<uri>model://x500_unified</uri>", f"<uri>model://{model_name(var)}</uri>")
        studio = f"{HERE}/gz_logan/photo/studio_{var}.sdf"
        open(studio, "w").write(src)
    env = dict(os.environ, GZ_PARTITION="photo",
               GZ_SIM_RESOURCE_PATH=f"{PX4}/Tools/simulation/gz/models:{HERE}/gz_logan/models",
               GZ_SIM_SERVER_CONFIG_PATH=f"{PX4}/src/modules/simulation/gz_bridge/server.config",
               GZ_SIM_SYSTEM_PLUGIN_PATH=f"{PX4}/build/px4_sitl_default/src/modules/simulation/gz_plugins",
               __NV_PRIME_RENDER_OFFLOAD="1", __GLX_VENDOR_LIBRARY_NAME="nvidia",
               __EGL_VENDOR_LIBRARY_FILENAMES="/usr/share/glvnd/egl_vendor.d/10_nvidia.json")
    g = subprocess.Popen(["bash", "-c", f"source /opt/ros/jazzy/setup.bash; exec gz sim --verbose=1 -r -s {studio}"],
                         env=env, start_new_session=True, stdout=open(f"{HERE}/gz_logan/photo/gz.log", "w"), stderr=subprocess.STDOUT)
    os.environ["GZ_PARTITION"] = "photo"
    try:
        from gz.transport13 import Node
        from gz.msgs10.image_pb2 import Image
        got, node = {}, Node()
        names = ("front_left", "rear_right", "side", "under")
        for n in names:
            node.subscribe(Image, f"/photo/{n}", lambda m, n=n: got.__setitem__(n, m))
        t0 = time.time()
        while len(got) < len(names) and time.time() - t0 < 240:
            time.sleep(0.5)
        time.sleep(8.0)                                        # let the scene finish loading, then take the newest
        for n in names:
            m = got[n]
            img = np.frombuffer(m.data, np.uint8).reshape(m.height, m.width, 3)
            f = f"{HERE}/gz_logan/photo/rig_{n}.png" if var == "v2" else f"{HERE}/gz_logan/photo/rig_{var}_{n}.png"
            cv2.imwrite(f, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            print(f)
    finally:
        os.killpg(g.pid, 15)


def selfview():
    """Each rig camera's own view with the drone hung in an empty sky: every pixel that is not sky is the drone
    itself -- propellers, arms, legs, the gimbal -- blocking that camera. -> gz_logan/photo/self_*.png + a summary."""
    import cv2
    os.makedirs(f"{HERE}/gz_logan/photo", exist_ok=True)
    open(f"{HERE}/gz_logan/photo/sky.sdf", "w").write(f"""<?xml version="1.0"?>
<sdf version="1.9"><world name="sky">
  <scene><ambient>0.6 0.6 0.6 1</ambient><background>0.2 0.9 0.2 1</background><shadows>false</shadows><grid>false</grid></scene>
  <light type="directional" name="sun"><pose>0 0 10 0 0 0</pose><diffuse>0.9 0.9 0.9 1</diffuse><direction>-0.3 0.2 -0.9</direction></light>
  <include><uri>model://x500_unified</uri><name>drone</name><pose>0 0 30 0 0 0</pose><static>true</static></include>
</world></sdf>
""")
    env = dict(os.environ, GZ_PARTITION="selfview",
               GZ_SIM_RESOURCE_PATH=f"{PX4}/Tools/simulation/gz/models:{HERE}/gz_logan/models",
               GZ_SIM_SERVER_CONFIG_PATH=f"{PX4}/src/modules/simulation/gz_bridge/server.config",
               GZ_SIM_SYSTEM_PLUGIN_PATH=f"{PX4}/build/px4_sitl_default/src/modules/simulation/gz_plugins",
               __NV_PRIME_RENDER_OFFLOAD="1", __GLX_VENDOR_LIBRARY_NAME="nvidia",
               __EGL_VENDOR_LIBRARY_FILENAMES="/usr/share/glvnd/egl_vendor.d/10_nvidia.json")
    g = subprocess.Popen(["bash", "-c", f"source /opt/ros/jazzy/setup.bash; exec gz sim --verbose=1 -r -s {HERE}/gz_logan/photo/sky.sdf"],
                         env=env, start_new_session=True, stdout=open(f"{HERE}/gz_logan/photo/gz_sky.log", "w"), stderr=subprocess.STDOUT)
    os.environ["GZ_PARTITION"] = "selfview"
    try:
        from gz.transport13 import Node
        from gz.msgs10.image_pb2 import Image
        cams = {"front (OAK-D colour)": "/rig/front", "left": "/rig/left", "right": "/rig/right", "back": "/rig/back"}
        got, node = {}, Node()
        for n, t in cams.items():
            node.subscribe(Image, t, lambda m, n=n: got.__setitem__(n, m))
        t0 = time.time()
        while len(got) < len(cams) and time.time() - t0 < 240:
            time.sleep(0.5)
        time.sleep(8.0)
        for n in cams:
            m = got[n]
            img = np.frombuffer(m.data, np.uint8).reshape(m.height, m.width, 3).astype(int)
            cols, cnt = np.unique(img.reshape(-1, 3) // 8, axis=0, return_counts=True)    # the sky: the picture's
            sky = (np.abs(img // 8 - cols[cnt.argmax()]).max(-1) <= 2)                    # most common colour
            top, bottom = (~sky[: m.height // 2]).mean(), (~sky[m.height // 2:]).mean()
            print(f"{n:22s} blocked by the drone itself: {100 * (~sky).mean():5.1f}% of the picture "
                  f"(upper half {100 * top:4.1f}%, lower half {100 * bottom:4.1f}%)")
            cv2.imwrite(f"{HERE}/gz_logan/photo/self_{n.split()[0]}.png", cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2BGR))
    finally:
        os.killpg(g.pid, 15)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--setup", action="store_true")
    ap.add_argument("--photo", action="store_true")
    ap.add_argument("--selfview", action="store_true")
    ap.add_argument("--worlds", action="store_true", help="only the _v2 flight worlds")
    ap.add_argument("--check", metavar="FOLDER", help="rig v2 flights: the gimbal camera's model against its pictures")
    a = ap.parse_args()
    if a.worlds:
        worlds()
    if a.check:
        check(a.check)
    if a.setup:
        setup()
    if a.photo:
        photo(os.environ.get("RIG", "v2") if os.environ.get("RIG", "").startswith("v2") else "v2")
    if a.selfview:
        selfview()
