#!/usr/bin/env python3
"""
join_videos.py -- stitch the mission recordings into one file to watch.

There is no ffmpeg on this machine, so the concatenation is done frame by frame
with OpenCV. A title card before each mission carries the numbers that matter --
which scene, how wrong GeoAnchor was, and how close it landed -- because a
video of a drone landing looks the same whether the fix was 3 m out or 34 m.
"""
import glob
import re
import sys

import cv2
import numpy as np

W = H = None
FPS = 20


def card(text_lines, w, h, secs=2.5):
    f = np.zeros((h, w, 3), np.uint8)
    y = h // 2 - 14 * len(text_lines)
    for i, (t, scale, col) in enumerate(text_lines):
        size = cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0]
        cv2.putText(f, t, ((w - size[0]) // 2, y + i * 34),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, col, 1, cv2.LINE_AA)
    return [f] * int(FPS * secs)


def main():
    log = sys.argv[1] if len(sys.argv) > 1 else None
    info = {}
    if log:
        txt = open(log).read()
        blocks = txt.split("--- mission ")
        for b in blocks[1:]:
            i = int(b.split("/")[0])
            sc = re.search(r"scene (\S+): sigma ([\d.]+) m, actual error ([\d.]+)", b)
            rs = re.search(r"=> (\w+)\s+miss\s+([\d.]+) m", b)
            if sc:
                info[i] = dict(scene=sc.group(1), sigma=float(sc.group(2)),
                               err=float(sc.group(3)),
                               outcome=rs.group(1) if rs else "?",
                               miss=float(rs.group(2)) if rs else float("nan"))
    files = sorted(glob.glob("mission_*.mp4"),
                   key=lambda p: int(re.search(r"(\d+)", p).group(1)))
    if not files:
        print("no mission_*.mp4 found"); return
    c = cv2.VideoCapture(files[0])
    w = int(c.get(cv2.CAP_PROP_FRAME_WIDTH)); h = int(c.get(cv2.CAP_PROP_FRAME_HEIGHT))
    c.release()
    out = cv2.VideoWriter("landing_rule.mp4", cv2.VideoWriter_fourcc(*"mp4v"),
                          FPS, (w, h))
    out.write(np.zeros((h, w, 3), np.uint8))
    for f in card([("GeoAnchor -> PX4 -> ArUco landing", 0.8, (255, 255, 255)),
                   ("analytical rule: aim at the averaged fix", 0.55, (0, 255, 0)),
                   ("PX4 SITL + Gazebo, real firmware", 0.5, (180, 180, 180))],
                  w, h, 3.0):
        out.write(f)
    total = 0
    for p in files:
        i = int(re.search(r"(\d+)", p).group(1))
        d = info.get(i)
        lines = [(f"mission {i}", 0.8, (255, 255, 255))]
        if d:
            lines += [(f"{d['scene']}   sigma {d['sigma']:.1f} m", 0.55, (200, 200, 200)),
                      (f"GeoAnchor was {d['err']:.1f} m wrong", 0.55, (0, 165, 255)),
                      (f"{d['outcome']}  {d['miss']:.2f} m from the pad", 0.6,
                       (0, 255, 0) if d["outcome"] == "landed" else (0, 0, 255))]
        for fr in card(lines, w, h, 2.5):
            out.write(fr)
        c = cv2.VideoCapture(p)
        while True:
            ok, fr = c.read()
            if not ok:
                break
            out.write(fr); total += 1
        c.release()
    out.release()
    print(f"landing_rule.mp4  {len(files)} missions, {total} frames, "
          f"{total/FPS:.0f} s of flight")


if __name__ == "__main__":
    main()
