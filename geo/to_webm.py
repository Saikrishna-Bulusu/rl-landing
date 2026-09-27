#!/usr/bin/env python3
"""
to_webm.py -- a flight film for the deck: 1280 x 720 VP8 webm under the deck's 20 MB clip limit, plus a still.
The only ffmpeg here is Playwright's: VP8 out, and in only piped JPEGs (no H.264, no raw video) -- so OpenCV
decodes, resizes and writes each frame as a JPEG into a named pipe, which ffmpeg reads as its file (no stdin
protocol in that build either).
    python3 geo/to_webm.py film.mp4 out.webm [--still-at 0.9] [--kbps 750]
"""
import argparse
import os
import subprocess

import cv2

FF = "/home/sai/.cache/ms-playwright/ffmpeg-1011/ffmpeg-linux"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src"); ap.add_argument("dst")
    ap.add_argument("--still-at", type=float, default=0.9, help="the still: this far into the film (0-1)")
    ap.add_argument("--kbps", type=int, default=750)
    a = ap.parse_args()
    v = cv2.VideoCapture(a.src)
    n, fps = int(v.get(cv2.CAP_PROP_FRAME_COUNT)), v.get(cv2.CAP_PROP_FPS) or 20.0
    w, h = int(v.get(cv2.CAP_PROP_FRAME_WIDTH)), int(v.get(cv2.CAP_PROP_FRAME_HEIGHT))
    import tempfile
    fifo = os.path.join(tempfile.mkdtemp(dir=os.path.dirname(os.path.abspath(a.dst))), "frames.mjpeg")
    os.mkfifo(fifo)
    p = subprocess.Popen([FF, "-y", "-loglevel", "error", "-f", "image2pipe", "-c:v", "mjpeg", "-framerate", f"{fps:g}",
                          "-i", fifo, "-c:v", "libvpx", "-b:v", f"{a.kbps}k", "-crf", "10", "-an", a.dst])
    pipe = open(fifo, "wb")
    k, still = 0, None
    while True:
        ok, im = v.read()
        if not ok:
            break
        if k == int(n * a.still_at):
            still = im
        pipe.write(cv2.imencode(".jpg", cv2.resize(im, (1280, 720), interpolation=cv2.INTER_AREA),
                                [cv2.IMWRITE_JPEG_QUALITY, 92])[1].tobytes()); k += 1
    pipe.close(); p.wait()
    os.remove(fifo); os.rmdir(os.path.dirname(fifo))
    jpg = os.path.splitext(a.dst)[0] + "_still.jpg"
    cv2.imwrite(jpg, cv2.resize(still, (1280, 720)), [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(a.dst, f"{os.path.getsize(a.dst) / 1e6:.1f} MB, {k} frames ({k / fps:.0f} s);", jpg)


if __name__ == "__main__":
    main()
