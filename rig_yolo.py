#!/usr/bin/env python3
"""
rig_yolo.py -- a pretrained YOLO (Ultralytics YOLO26n, COCO weights) fine-tuned on the rig's pictures,
scored exactly like rig_detect.py's small network: same held-out drone places, same worsened images.

    python rig_yolo.py --data  gz_logan/rig/data2   -> gz_logan/rig/yolo/ds (YOLO-format copy, worsened)
    python rig_yolo.py --train                      -> gz_logan/rig/yolo/run/weights/best.pt
    python rig_yolo.py --score gz_logan/rig/data2   -> data2/score_yolo.json, detect_vs_size_yolo.png

Runs in mujoco_drones with PYTHONPATH=.yolo_pkgs: ultralytics + torchvision live there, installed
--no-deps, so the training environment's numpy/torch are never touched. YOLO_AUTOINSTALL=false stops
ultralytics pip-installing anything itself.
"""
import argparse
import os

os.environ.setdefault("YOLO_AUTOINSTALL", "false")
import cv2
import numpy as np

import rig_detect as RD

HERE = os.path.dirname(os.path.abspath(__file__))
Y = f"{HERE}/gz_logan/{'rig_' + os.environ['RIG'] if os.environ.get('RIG', '').startswith('v2') else 'rig'}/yolo"
BEST = f"{Y}/run/weights/best.pt"


def build(root, out=f"{Y}/ds"):
    """Same place split as rig_detect.load. Every picture worsened once (noise, blur, JPEG, brightness),
    as the live pipeline and the score see them; one class, the magpie; no box = background."""
    for k, recs in RD.load(root).items():
        os.makedirs(f"{out}/images/{k}", exist_ok=True); os.makedirs(f"{out}/labels/{k}", exist_ok=True)
        rng = np.random.default_rng({"train": 1, "val": 2, "test": 3}[k])
        for r in recs:
            name = os.path.basename(r["file"])[:-4]
            img = cv2.imread(f"{root}/{r['file']}")
            h, w = img.shape[:2]
            cv2.imwrite(f"{out}/images/{k}/{name}.jpg", RD.worsen(img, rng), [cv2.IMWRITE_JPEG_QUALITY, 95])
            with open(f"{out}/labels/{k}/{name}.txt", "w") as f:
                if r["box"]:
                    x0, y0, x1, y1 = r["box"]
                    f.write(f"0 {(x0 + x1 + 1) / 2 / w:.6f} {(y0 + y1 + 1) / 2 / h:.6f} {(x1 - x0 + 1) / w:.6f} {(y1 - y0 + 1) / h:.6f}\n")
        print(k, len(recs), flush=True)
    with open(f"{out}/data.yaml", "w") as f:
        f.write(f"path: {out}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: magpie\n")


def model(path):
    from ultralytics import YOLO, settings
    settings.update(sync=False)                    # no usage analytics sent to Ultralytics
    return YOLO(path)


def train(epochs, imgsz=1280, batch=8, init="yolo26n.pt"):
    """init: the starting weights -- COCO's yolo26n.pt, or (rig v2) the first rig's fine-tune: the magpie is the same
    bird, only the cameras changed."""
    os.makedirs(Y, exist_ok=True); os.chdir(Y)     # the pretrained weights download here
    m = model(init)
    # scale 0.3 (default 0.5): mosaic already shrinks pictures, and the far magpie is only 5-10 px
    m.train(data=f"{Y}/ds/data.yaml", epochs=epochs, imgsz=imgsz, batch=batch, project=Y, name="run", exist_ok=True,
            single_cls=True, scale=0.3, degrees=0.0, close_mosaic=10, patience=25, workers=6, seed=0, amp=True)


def detector(path=BEST, imgsz=1280):
    """det(img_rgb, thresh) -> [(score, cx, cy, w, h)], rig_detect's format; a list of pictures -> one list each
    (one batch: the rig's six cameras)."""
    m = model(path)

    def det(imgs_rgb, thresh):
        one = not isinstance(imgs_rgb, list)
        res = m.predict([np.ascontiguousarray(i[..., ::-1]) for i in ([imgs_rgb] if one else imgs_rgb)], imgsz=imgsz,
                        conf=thresh, verbose=False, half=RD.torch.cuda.is_available())   # fp16: 34 vs 59 ms, same finds
        out = [[(float(c), float(x), float(y), float(w), float(h)) for c, (x, y, w, h) in zip(r.boxes.conf.tolist(), r.boxes.xywh.tolist())]
               for r in res]
        return out[0] if one else out
    return det


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", metavar="DIR")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--init", default="yolo26n.pt", help="--train: starting weights")
    ap.add_argument("--score", metavar="DIR")
    ap.add_argument("--weights", default=BEST)
    ap.add_argument("--op", type=float, default=0.5, help="operating threshold (score.json's by-size curves)")
    a = ap.parse_args()
    if a.data:
        build(a.data)
    if a.train:
        train(a.epochs, init=a.init)
    if a.score:
        RD.score(a.score, thresholds=tuple(sorted({0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, a.op})), op=a.op, det=detector(a.weights), tag="_yolo")
