#!/usr/bin/env python3
"""
rig_perceive.py -- the magpie as the drone would really know it in Gazebo: the six rig cameras'
pictures -> the detector (rig_detect.py) -> directions (the drone's OWN attitude estimate, the rig's
mounting) and distances (from the bird's size in the picture) -> the tracker (rig_sense.BirdTrack,
the same code the fast simulator uses). No true bird position anywhere.

gz_learner.py (system Python, Gazebo's libraries) streams frames in; this (training environment,
PyTorch) answers each with the tracker's belief. Protocol, per frame set on stdin:
    one JSON line {"t": sim s, "pos": [x, y, z] Gazebo, "rpy": [roll, pitch, yaw], "cams": [...], "h": H, "w": W}
    then the images, raw RGB, in the order of "cams"
and on stdout one JSON line {"seen": bool, "rel": [..], "vel": [..], "dets": [[cam, score, u, v, size], ...]}.

    python rig_perceive.py [--det ../../shared/px4_gazebo/gz_logan/rig/det.pt] [--thresh 0.5]
"""
import argparse
import json
import math
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [f"{ROOT}/shared/px4_gazebo", f"{ROOT}/shared/fast_sim"]
import gz_rig as GR                  # the rig's cameras: mounting, focal length
import rig_detect as RD              # the detector
import rig_sense as RS               # the tracker and the calibration


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--det", default=f"{ROOT}/shared/px4_gazebo/gz_logan/rig/det.pt")      # the current detector
    ap.add_argument("--thresh", type=float, default=None, help="default: the calibration's operating threshold")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(2)
    c = json.load(open(RS.CALIB))
    a.thresh = a.thresh or float(os.environ.get("RIG_THRESH", 0)) or c["source"]["operating_threshold"]   # as calibrated
    if os.environ.get("RIG_YOLO"):            # the fine-tuned YOLO (rig_yolo.py) instead of the small CNN; its
        sys.path[:0] = [f"{ROOT}/.yolo_pkgs"]      # packages live apart from this environment's; rig_yolo is beside this file
        import rig_yolo
        detect_batch = rig_yolo.detector(os.environ["RIG_YOLO"])
        print(f"YOLO {os.environ['RIG_YOLO']} at {a.thresh}", file=sys.stderr)
    else:
        net = RD.Det().to(dev)
        net.load_state_dict(torch.load(a.det, map_location=dev, weights_only=True)); net.eval()
        detect_batch = lambda imgs, t: RD.detect_batch(net, imgs, t, dev)
    track, t_prev = RS.BirdTrack(1, "cpu"), None
    rng = np.random.default_rng(0)            # a real camera's pictures: noise, blur, JPEG -- as the detector was scored
    from concurrent.futures import ThreadPoolExecutor
    pool = ThreadPoolExecutor(6)              # the six pictures worsened side by side (OpenCV releases the GIL)
    cam_R = {n: GR.rot(*rpy) for n, (_, rpy) in GR.RIG.items()}
    # the spinning propellers' tips reach into the ring cameras' bottom rows even on the mast: a blade
    # at a new angle every frame looks like a small bird there. Ignore detections inside that sweep
    # (shared/px4_gazebo/gz_logan/rig/prop_mask.npz, projected from the geometry) -- the down camera covers those directions
    prop = dict(np.load(f"{ROOT}/shared/px4_gazebo/gz_logan/{GR.BENCH}/prop_mask.npz"))   # RIG=v2: shared/px4_gazebo/gz_logan/rig_v2
    # the landing pads are surveyed (the delivery addresses, in the drone's map): their black-and-white
    # markers seen at an angle scored up to 0.82 as "magpie" in Gazebo even after hard-negative
    # training. Ignore detections pointing at a known pad (its angular size + 4 deg of heading error)
    import re
    world = open(f"{ROOT}/shared/px4_gazebo/gz_logan/rigfly/aruco.sdf").read()
    pads = [tuple(map(float, m.split()[:3])) for m in re.findall(r"model://(?:logan_pad_\d+|arucotag)</uri>(?:<name>[^<]*</name>)?\s*(?:<!--.*?-->\s*)?<pose>([^<]*)</pose>", world, re.S)]
    pads = np.array(pads)
    print(f"{len(pads)} known pads", file=sys.stderr)

    def on_pad(d, pos):
        rel = pads - pos
        dist = np.linalg.norm(rel, axis=1)
        ang = np.degrees(np.arccos(np.clip(rel @ d / np.maximum(dist, 1e-6), -1, 1)))
        return bool(np.any(ang < np.degrees(np.arctan2(2.5, dist)) + 4.0))
    print(json.dumps({"ready": dev}), flush=True)
    inp = sys.stdin.buffer
    while (line := inp.readline()):
        hdr = json.loads(line)
        shape = {n: hdr["shapes"][n] if "shapes" in hdr else (hdr["h"], hdr["w"]) for n in hdr["cams"]}   # rig v2: sizes differ
        Rb = GR.rot(*hdr["rpy"])
        pos = np.array(hdr["pos"])
        dets = []
        raw = [np.frombuffer(inp.read(shape[n][0] * shape[n][1] * 3), np.uint8).reshape(shape[n][0], shape[n][1], 3) for n in hdr["cams"]]
        if dev == "cuda" and os.environ.get("RIG_WORSEN") != "cpu":             # the frame set on the GPU: 6x faster
            imgs = RD.worsen_gpu(raw, rng)
        else:
            seeds = rng.integers(1 << 31, size=len(raw))                          # worsen works in BGR
            imgs = list(pool.map(lambda k: RD.worsen(np.ascontiguousarray(raw[k][..., ::-1]), np.random.default_rng(seeds[k]))[..., ::-1],
                                 range(len(raw))))
        for n, found in zip(hdr["cams"], detect_batch(imgs, a.thresh)):
            h, w = shape[n]
            f = GR.CAMSZ[n][2]                                                    # this camera's focal length, px
            for score, u, v, bw, bh in found:
                if prop[n][min(int(v), h - 1), min(int(u), w - 1)]:
                    continue
                ray = np.array([1.0, (w / 2 - u) / f, (h / 2 - v) / f])            # camera: x ahead, y left, z up
                d = Rb @ cam_R[n] @ (ray / np.linalg.norm(ray))
                if on_pad(d, pos):
                    continue
                size = max(bw, bh)
                dets.append((score, n, u, v, size, d, f * c["span_assumed_m"] / size))
        t = float(hdr["t"])
        me = torch.tensor(pos, dtype=torch.float32)[None]
        track.predict(0.0 if t_prev is None else t - t_prev)
        zr = lambda x: (torch.tensor(pos + x[5] * x[6], dtype=torch.float32)[None],
                        RS.measure_cov(torch.tensor(x[5], dtype=torch.float32)[None], torch.tensor([x[6]], dtype=torch.float32),
                                       c["sig_ang_rad"], c["sig_rel_dist"]))
        if dets and bool(track.confirmed[0]):          # a confirmed track takes the detection that fits it best
            best = min(dets, key=lambda x: float(track.angle(zr(x)[0], me)[0]))
        elif dets:                                     # otherwise the surest sighting
            best = max(dets, key=lambda x: x[0])
        det = torch.tensor([bool(dets)])
        score, n, u, v, size, d, dist = best if dets else (0.0, None, 0.0, 0.0, 0.0, np.array([1.0, 0.0, 0.0]), 1.0)
        z, R = zr((score, n, u, v, size, d, dist))
        track.update(t, det, z, R, me)
        t_prev = t
        x = track.x[0].tolist()
        print(json.dumps({"seen": bool(track.reported[0]), "rel": [x[0] - pos[0], x[1] - pos[1], x[2] - pos[2]],
                          "vel": x[3:], "dir": d.tolist() if dets else None, "dist": dist if dets else None,
                          "dets": [[n_, round(s_, 3), round(u_, 1), round(v_, 1), round(z_, 1)]
                                   for s_, n_, u_, v_, z_, _, _ in dets]}), flush=True)


if __name__ == "__main__":
    main()
