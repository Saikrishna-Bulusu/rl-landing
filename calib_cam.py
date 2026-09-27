#!/usr/bin/env python3
"""
calib_cam.py -- solve the pixel -> world mapping for the downward camera.

WHY THIS REPLACES THE OLD FITTED MATRIX

The previous mapping was a single 2x2 matrix fitted by flying to six offsets
and solving least squares. It worked where it was fitted and nowhere else,
because a body-fixed camera's pixel-to-world mapping DEPENDS ON YAW, and the
matrix silently baked in whatever yaw the aircraft happened to hold that day.
Check it against the model and it is not even a valid rotation -- its singular
values are 0.81 and 0.62 instead of 1.0 and 1.0, so it also skewed and shrank
every offset it converted. On a mission flown at a different heading it drove
the aircraft 40 m in the wrong direction.

THE MODEL

For a level aircraft with a camera looking straight down:

    dNED = (h / f) * R(yaw) * B * (ex, ey)

    h   altitude, f = (W/2)/tan(hfov/2) = 539.3 px
    R   the live yaw rotation -- read from the vehicle every frame, not baked
    B   a FIXED camera-to-body rotation: mounting only, the one unknown

So B is solved once here and R(yaw) is applied at run time. B is forced
orthonormal via SVD, because a mounting rotation cannot skew or shrink -- that
constraint is exactly what the old free-form fit was missing.
"""

import math
import time

import numpy as np

from px4_bridge import PX4Link
from px4_perception import ArucoEye

# The tag's SDF pose is "20 0 0", and Gazebo's world frame here is ENU
# (see <world_frame_orientation> in aruco.sdf), so that is 20 m EAST. PX4's
# LOCAL_POSITION_NED is north-east-down, so in the frame every number in this
# file is written in, the tag is at north 0, east 20. Getting this backwards
# rotated the whole problem 90 degrees: the calibration solved for a camera
# mapping that had to explain a 28 m error it could not see, and returned a
# matrix with singular values 0.48 and 0.14 instead of 1.0.
PAD = np.array([0.0, 20.0])          # NED north, east
ALT = 12.0                           # 1.5 m tag spans 67 px here
F_PX = (1280 / 2) / math.tan(math.radians(99.7) / 2)
OFFSETS = [(0, 0), (5, 0), (-5, 0), (0, 5), (0, -5), (4, 4), (-4, 4), (4, -4)]


def rot(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s], [s, c]])


def goto(link, eye, target_ne, alt, tmax=25, tol=1.5):
    """
    Move toward a world point, sampling the whole way.

    The first version only sampled once the aircraft had settled inside 0.6 m
    at under 0.25 m/s, and a velocity-controlled multirotor never quite does
    that -- it produced ZERO samples at all eight offsets. Nothing about the
    fit needs the aircraft to be AT a commanded offset: it needs the position
    to be KNOWN, and the EKF reports that continuously. So sample every frame
    the marker is visible and the airframe is near level, which is the model's
    only real assumption.
    """
    t0, hits = time.time(), []
    while time.time() - t0 < tmax:
        s = link.state(timeout=0.5)
        if s is None:
            continue
        d = np.array(target_ne) - np.array([s["x"], s["y"]])
        v = np.clip(0.8 * d, -2.0, 2.0)
        link.send_velocity(v[0], v[1], 0.6 * (alt - s["alt"]))
        if np.linalg.norm(d) < tol:
            break
        time.sleep(0.05)

    # HOLD STILL, THEN SAMPLE. Sampling while translating was the second thing
    # that wrecked the fit: a multirotor tilts to accelerate, and at 12 m a 5
    # degree tilt walks the camera's nadir 1.05 m across the ground -- 47 px of
    # pure error injected into every sample. Stopping first costs three seconds
    # per point and removes the whole term.
    t0 = time.time()
    while time.time() - t0 < 6.0:
        s = link.state(timeout=0.5)
        if s is None:
            continue
        link.send_velocity(0, 0, 0.6 * (alt - s["alt"]))
        still = (math.hypot(s["vx"], s["vy"]) < 0.15 and
                 max(abs(s["roll"]), abs(s["pitch"])) < math.radians(2))
        seen, off, _, _ = eye.look()
        if seen and still and s["alt"] > 5.0:
            hits.append((np.array([s["x"], s["y"]]), s["alt"], s["yaw"],
                         np.array(off, float)))
            if len(hits) >= 5:
                break
        time.sleep(0.05)
    return hits


def main():
    link = PX4Link()
    eye = ArucoEye()
    assert link.start_offboard(), "could not arm"
    samples = []
    for dn, de in OFFSETS:
        tgt = PAD + np.array([dn, de], float)
        hits = goto(link, eye, tgt, ALT)
        print(f"  offset ({dn:+.0f},{de:+.0f}): {len(hits)} detections")
        samples += hits
    link.send_velocity(0, 0, -1.5)

    if len(samples) < 12:
        print(f"only {len(samples)} samples -- cannot fit"); return

    # y = B x, with the yaw rotated out so B is mounting-only
    X, Y = [], []
    for pos, alt, yaw, px in samples:
        d_ned = PAD - pos                       # where the tag is, from here
        Y.append(rot(-yaw) @ d_ned * F_PX / alt)
        X.append(px)
    X, Y = np.array(X).T, np.array(Y).T          # 2xN
    B = Y @ np.linalg.pinv(X)
    u, sv, vt = np.linalg.svd(B)
    # Nearest SCALED rotation. The scale is left in deliberately: the model
    # predicts it should come out at 1.0, so it is a free consistency check on
    # the focal length and the altitude reference rather than a fudge factor.
    B_orth = (u @ vt) * sv.mean()
    resid = np.linalg.norm(Y - B_orth @ X, axis=0)
    px_scale = np.array([np.linalg.norm(p) for _, _, _, p in samples])

    print(f"\n  raw fit singular values {np.round(sv, 3)}  (both 1.0 = model holds)")
    print(f"  B (camera px -> body metres * f/h), orthonormal:")
    print("   ", np.round(B_orth, 4).tolist())
    print(f"  residual {resid.mean():.2f} px-equivalent over {len(samples)} "
          f"samples, offsets up to {px_scale.max():.0f} px")
    np.save("cam_B.npy", B_orth)
    print("  saved cam_B.npy")


if __name__ == "__main__":
    main()
