#!/usr/bin/env python3
"""
rig_replay.py -- replay a Gazebo rig flight's logged detections (gz_learner --sense rig writes
*_rig.jsonl: every frame set's detections, the drone's own attitude, and Gazebo's truth) through the
tracker offline, to test a change without flying: phantom reports while the real magpie is far,
and how often a diving magpie close by is reported.

    python rig_replay.py FLIGHT_rig.jsonl [...] [--static]
"""
import argparse
import json
import math
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path[:0] = [f"{ROOT}/shared/px4_gazebo", f"{ROOT}/shared/fast_sim"]
import gz_rig as GR
import rig_sense as RS


def replay(path, static, min_age=0.0):
    c = json.load(open(RS.CALIB))
    track, t_prev = RS.BirdTrack(1, "cpu"), None
    cam_R = {n: GR.rot(*rpy) for n, (_, rpy) in GR.RIG.items()}
    hist = []                                  # (drone position, direction, size) of the track's detections
    t_start = None
    out = []
    for l in open(path):
        r = json.loads(l)
        pos, Rb, t = np.array(r["pose"][0]), GR.rot(*r["pose"][1]), r["t"]
        dets = []
        for cam, score, u, v, size in r["out"]["dets"]:
            ray = np.array([1.0, (GR.W / 2 - u) / GR.F, (GR.H / 2 - v) / GR.F])
            d = Rb @ cam_R[cam] @ (ray / np.linalg.norm(ray))
            dets.append((score, d, size, GR.F * c["span_assumed_m"] / size))
        me = torch.tensor(pos, dtype=torch.float32)[None]
        track.predict(0.0 if t_prev is None else t - t_prev)
        zr = lambda x: (torch.tensor(pos + x[1] * x[3], dtype=torch.float32)[None],
                        RS.measure_cov(torch.tensor(x[1], dtype=torch.float32)[None], torch.tensor([x[3]], dtype=torch.float32),
                                       c["sig_ang_rad"], c["sig_rel_dist"]))
        if dets and bool(track.confirmed[0]):
            best = min(dets, key=lambda x: float(track.angle(zr(x)[0], me)[0]))
        elif dets:
            best = max(dets, key=lambda x: x[0])
        hits0 = int(track.hits[0])
        if dets:
            z, R = zr(best)
            track.update(t, torch.tensor([True]), z, R, me)
        else:
            track.update(t, torch.tensor([False]), torch.zeros(1, 3), torch.eye(3)[None], me)
        t_prev = t
        h = int(track.hits[0])
        if h == 1 and dets and h != hits0 or (h == 1 and hits0 > 1):
            t_start = t
        if h == 0:
            hist = []
            t_start = None
        elif h == 1 and dets:
            hist = [(pos, best[1], best[2], t)]
        elif dets and h > hits0:
            hist = (hist + [(pos, best[1], best[2], t)])[-12:]
        rep = bool(track.reported[0])
        if static and rep and len(hist) >= 6:
            rep = not is_static(hist)
        if min_age and rep and (t_start is None or t - t_start < min_age):
            rep = False
        rel_t = np.array(r["bird"]) - pos
        est = track.x[0, :3].numpy() - pos
        out.append((t, np.linalg.norm(rel_t), rep, float(np.degrees(np.arccos(np.clip(est @ rel_t / max(np.linalg.norm(est) * np.linalg.norm(rel_t), 1e-9), -1, 1)))),
                    r.get("dive", None)))
    return out


def is_static(hist, max_res_deg=1.5, max_loom=0.35):
    """Do the lines of sight meet at one fixed point (least squares), with the object not growing?
    Static clutter: yes. A flying bird: its sight lines miss any single point -- unless it comes
    straight at the drone, when it grows fast (looming)."""
    P = np.array([h[0] for h in hist]); D = np.array([h[1] for h in hist]); S = np.array([h[2] for h in hist])
    A, b = np.zeros((3, 3)), np.zeros(3)
    for p, d in zip(P, D):
        M = np.eye(3) - np.outer(d, d)
        A += M; b += M @ p
    X = np.linalg.lstsq(A, b, rcond=None)[0]
    res = [math.degrees(math.acos(np.clip(d @ (X - p) / max(np.linalg.norm(X - p), 1e-9), -1, 1))) for p, d in zip(P, D)]
    ts = np.array([h[3] for h in hist])
    loom = np.polyfit(ts - ts[0], np.log(S), 1)[0] if ts[-1] > ts[0] else 0.0   # growth rate of log size, 1/s
    return float(np.median(res)) < max_res_deg and loom < max_loom


def summary(rows, name):
    a = np.array([(d, rep, ang) for _, d, rep, ang, _ in rows], float)
    far = a[:, 0] > 40
    close = a[:, 0] < 30
    ph_far = np.mean(a[far, 1]) if far.any() else float("nan")
    rep_close = np.mean(a[close, 1]) if close.any() else float("nan")
    ok_close = np.mean((a[close, 1] > 0) & (a[close, 2] < 20)) if close.any() else float("nan")
    print(f"  {name:16s} magpie over 40 m away: something reported {ph_far:5.1%} of frames (all phantom) | "
          f"magpie within 30 m: reported {rep_close:5.1%}, and pointing at it {ok_close:5.1%}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+")
    a = ap.parse_args()
    for p in a.logs:
        print(os.path.basename(p))
        summary(replay(p, False), "as flown")
        for age in (0.3, 0.5, 0.8):
            summary(replay(p, False, age), f"track age >= {age} s")


if __name__ == "__main__":
    main()
