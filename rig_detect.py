#!/usr/bin/env python3
"""
rig_detect.py -- the magpie detector for the six-camera rig (gz_rig.py), and its honest score.

A small fully-convolutional network (~90k weights, plain PyTorch -- nothing downloaded): full
1280x960 camera image in, a heat map at 1/4 resolution out (where a magpie's centre is) plus the
size of the box around it. Trained on the bench's pictures, which Gazebo labels itself (render
with and without the bird, difference). Scored only on drone PLACES it never trained on (every
10th place), with the images made worse the way a real camera would: sensor noise, slight blur,
JPEG, brightness changes.

    train:  python rig_detect.py --train gz_logan/rig/data          -> gz_logan/rig/data/det.pt
    score:  python rig_detect.py --score gz_logan/rig/data          -> score.json, detect_vs_size.png

Runs in the training environment (mujoco_drones: PyTorch + CUDA).
"""
import argparse
import json
import math
import os

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn

S = 4                    # heat map stride
CROP = 320               # training crop, pixels


class Det(nn.Module):
    def __init__(self, c=32):
        super().__init__()
        cbr = lambda i, o, s=1, d=1: nn.Sequential(nn.Conv2d(i, o, 3, s, d, d, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))
        self.stem = nn.Sequential(cbr(3, 16, 2), cbr(16, c, 2))
        self.body = nn.ModuleList(nn.Sequential(cbr(c, c, 1, d), cbr(c, c)) for d in (1, 2, 4, 8))
        self.head = nn.Sequential(cbr(c, c), nn.Conv2d(c, 3, 1))
        nn.init.constant_(self.head[1].bias, 0.0); self.head[1].bias.data[0] = -4.6   # start at p = 1%

    def forward(self, x):                                  # x: (B, 3, H, W) in 0..1
        x = self.stem(x - 0.5)
        for b in self.body:
            x = x + b(x)
        return self.head(x)                                # (B, 3, H/4, W/4): heat logit, log w, log h


def load(root):
    recs = [json.loads(l) for l in open(f"{root}/labels.jsonl")]
    place = lambda r: int(os.path.basename(r["file"])[:5])
    split = lambda r: "test" if place(r) % 10 == 0 else "val" if place(r) % 10 == 1 else "train"
    return {k: [r for r in recs if split(r) == k] for k in ("train", "val", "test")}


def worsen(img, rng):
    """What a real camera does to the picture: brightness/contrast, slight blur, sensor noise, JPEG."""
    img = img.astype(np.float32) * rng.uniform(0.75, 1.2) + rng.uniform(-20, 20)
    if rng.random() < 0.5:
        img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.3, 1.0))
    img = np.clip(img + rng.normal(0, rng.uniform(1, 6), img.shape), 0, 255).astype(np.uint8)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(60, 95))])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def worsen_gpu(imgs, rng):
    """worsen() for a whole frame set on the GPU (torch + nvjpeg), RGB in and out, same steps and parameter draws.
    26 Sept, rig v2l held-out pictures: 15 ms vs 98 ms per 5-picture set (CPU, 6 threads); YOLO found the magpie in
    90.5% vs 90.0% of 369, same false alarms (fp16 too)."""
    sys_path = f"{os.path.dirname(os.path.abspath(__file__))}/.yolo_pkgs"      # torchvision lives there (rig_yolo.py)
    if sys_path not in __import__("sys").path:
        __import__("sys").path.insert(0, sys_path)
    from torchvision.io import encode_jpeg, decode_jpeg
    from torchvision.transforms.functional import gaussian_blur
    g = torch.Generator(device="cuda").manual_seed(int(rng.integers(1 << 31)))
    enc = []
    for img in imgs:
        x = torch.from_numpy(np.ascontiguousarray(img)).cuda().permute(2, 0, 1).float() * rng.uniform(0.75, 1.2) + rng.uniform(-20, 20)
        if rng.random() < 0.5:
            s = rng.uniform(0.3, 1.0)
            x = gaussian_blur(x[None], 2 * math.ceil(3 * s) + 1, s)[0]
        x = (x + torch.randn(x.shape, device="cuda", generator=g) * rng.uniform(1, 6)).clamp_(0, 255).to(torch.uint8)
        enc.append(encode_jpeg(x, quality=int(rng.integers(60, 95))).cpu())      # the JPEG bytes are small
    return [d.permute(1, 2, 0).cpu().numpy() for d in decode_jpeg(enc, device="cuda")]


def target(box, h, w):
    """Heat map (Gaussian at the box centre, width from its size) and the log box size there."""
    heat, size, mask = np.zeros((h // S, w // S), np.float32), np.zeros((2, h // S, w // S), np.float32), np.zeros((h // S, w // S), np.float32)
    if box is None:
        return heat, size, mask
    x0, y0, x1, y1 = box
    cx, cy, bw, bh = (x0 + x1) / 2 / S, (y0 + y1) / 2 / S, x1 - x0 + 1, y1 - y0 + 1
    sig = max(0.8, 0.15 * max(bw, bh) / S)
    yy, xx = np.mgrid[0:h // S, 0:w // S]
    heat = np.exp(-((xx + 0.5 - cx) ** 2 + (yy + 0.5 - cy) ** 2) / (2 * sig * sig)).astype(np.float32)
    i, j = min(int(cy), h // S - 1), min(int(cx), w // S - 1)
    heat[i, j], mask[i, j], size[:, i, j] = 1.0, 1.0, (math.log(bw), math.log(bh))
    return heat, size, mask


def batch(root, recs, n, rng):
    """n training crops: half around a magpie (anywhere in the crop), half empty sky/trees/roofs."""
    pos = [r for r in recs if r["box"]]
    neg = [r for r in recs if not r["box"]]
    X, Y, Z, M = [], [], [], []
    for k in range(n):
        r = pos[rng.integers(len(pos))] if k % 2 == 0 else (neg + pos)[rng.integers(len(neg) + len(pos))]
        img = cv2.imread(f"{root}/{r['file']}")
        h, w = img.shape[:2]
        box = r["box"] if k % 2 == 0 else None
        if box is not None:
            cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
            x0 = int(np.clip(cx - rng.uniform(16, CROP - 16), 0, w - CROP)); y0 = int(np.clip(cy - rng.uniform(16, CROP - 16), 0, h - CROP))
        else:
            for _ in range(20):                          # a crop that does not contain the bird
                x0, y0 = int(rng.integers(0, w - CROP + 1)), int(rng.integers(0, h - CROP + 1))
                if not r["box"] or not (x0 - 8 < r["box"][2] and r["box"][0] < x0 + CROP + 8 and
                                        y0 - 8 < r["box"][3] and r["box"][1] < y0 + CROP + 8):
                    break
            else:
                continue
        crop = worsen(img[y0:y0 + CROP, x0:x0 + CROP], rng)
        b = None
        if box is not None:
            b = [box[0] - x0, box[1] - y0, box[2] - x0, box[3] - y0]
            b = [max(b[0], 0), max(b[1], 0), min(b[2], CROP - 1), min(b[3], CROP - 1)]
        if rng.random() < 0.5:
            crop = crop[:, ::-1]
            b = None if b is None else [CROP - 1 - b[2], b[1], CROP - 1 - b[0], b[3]]
        heat, size, mask = target(b, CROP, CROP)
        X.append(np.ascontiguousarray(crop[..., ::-1]).transpose(2, 0, 1)); Y.append(heat); Z.append(size); M.append(mask)
    t = lambda a: torch.from_numpy(np.stack(a))
    return t(X).float() / 255, t(Y), t(Z), t(M)


def loss_fn(out, heat, size, mask):
    """CenterNet: penalty-reduced focal loss on the heat map, L1 on log size at the centres."""
    p = torch.sigmoid(out[:, 0]).clamp(1e-4, 1 - 1e-4)
    pos = heat.eq(1).float()
    focal = -(pos * (1 - p) ** 2 * torch.log(p) + (1 - pos) * (1 - heat) ** 4 * p ** 2 * torch.log(1 - p)).sum()
    n = pos.sum().clamp(min=1)
    l1 = (mask.unsqueeze(1) * (out[:, 1:] - size).abs()).sum() / n
    return focal / n + 0.1 * l1


class Crops(torch.utils.data.IterableDataset):
    """Endless training batches, made by worker processes (reading and worsening JPEGs is the slow part)."""

    def __init__(self, root, recs, n):
        super().__init__()
        self.root, self.recs, self.n = root, recs, n

    def __iter__(self):
        wi = torch.utils.data.get_worker_info()
        cv2.setNumThreads(1)
        rng = np.random.default_rng(1000 + (wi.id if wi else 0))
        while True:
            yield batch(self.root, self.recs, self.n, rng)


def train(root, steps=8000, dev="cuda", bs=16, workers=6):
    torch.manual_seed(0)
    sp = load(root)
    print({k: (len(v), sum(bool(r["box"]) for r in v)) for k, v in sp.items()}, "(images, with a magpie)", flush=True)
    net = Det().to(dev)
    opt = torch.optim.AdamW(net.parameters(), 2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=steps, pct_start=0.05)
    feed = iter(torch.utils.data.DataLoader(Crops(root, sp["train"], bs), batch_size=None, num_workers=workers,
                                            prefetch_factor=4, persistent_workers=True))
    for it in range(steps):
        x, y, z, m = (a.to(dev) for a in next(feed))
        loss = loss_fn(net(x), y, z, m)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if it % 500 == 0 or it == steps - 1:
            print(f"step {it:5d}  loss {loss.item():.3f}", flush=True)
    torch.save(net.state_dict(), f"{root}/det.pt")
    print(f"{root}/det.pt")


@torch.no_grad()
def detect(net, img_rgb, thresh, dev="cuda"):
    """Full image -> list of (score, cx, cy, w, h) in pixels, peaks of the heat map above thresh."""
    x = torch.from_numpy(np.ascontiguousarray(img_rgb.transpose(2, 0, 1))).float().div(255)[None].to(dev)
    out = net(x)[0]
    p = torch.sigmoid(out[0])
    peak = (p == Fn.max_pool2d(p[None, None], 5, 1, 2)[0, 0]) & (p > thresh)
    ys, xs = torch.nonzero(peak, as_tuple=True)
    return [(float(p[i, j]), (float(j) + 0.5) * S, (float(i) + 0.5) * S, math.exp(float(out[1, i, j])), math.exp(float(out[2, i, j])))
            for i, j in zip(ys.tolist(), xs.tolist())]


@torch.no_grad()
def detect_batch(net, imgs_rgb, thresh, dev="cuda"):
    """detect() for several pictures in one forward pass (the rig's six cameras)."""
    x = torch.from_numpy(np.ascontiguousarray(np.stack(imgs_rgb).transpose(0, 3, 1, 2))).float().div(255).to(dev)
    out = net(x)
    p = torch.sigmoid(out[:, 0])
    peak = (p == Fn.max_pool2d(p[:, None], 5, 1, 2)[:, 0]) & (p > thresh)
    res = [[] for _ in imgs_rgb]
    for b, i, j in torch.nonzero(peak).tolist():
        res[b].append((float(p[b, i, j]), (j + 0.5) * S, (i + 0.5) * S, math.exp(float(out[b, 1, i, j])), math.exp(float(out[b, 2, i, j]))))
    return res


def score(root, dev="cuda", thresholds=(0.3, 0.5, 0.7), op=0.5, det=None, tag=""):
    """Held-out places only, images worsened as in training (fixed seed): how often the magpie is
    found, by its size in the image and its distance; false alarms per image; centre error.
    det(img_rgb, thresh) -> [(score, cx, cy, w, h)] scores another detector the same way (rig_yolo.py);
    tag names its outputs (score{tag}.json)."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rng = np.random.default_rng(123)
    if det is None:
        net = Det().to(dev); net.load_state_dict(torch.load(f"{root}/det.pt", map_location=dev, weights_only=True)); net.eval()
        det = lambda img, t: detect(net, img, t, dev)
    import gz_rig as GR                          # each camera's focal length (RIG=v2: the OAK-D's is 3.1x the others')
    recs = load(root)["test"]
    rows = []
    for r in recs:
        img = worsen(cv2.imread(f"{root}/{r['file']}"), rng)[..., ::-1]
        dets = det(img, min(thresholds))
        # sizes in the DETECTOR's pixels: YOLO shrinks a picture to 1280 wide (the OAK-D's 1920x1080 to 2/3), so the
        # simulator's detection-by-size curve holds for every camera; the same bird at 1280 wide is unchanged
        k_det = min(1.0, 1280 / max(img.shape[:2]))
        row = dict(file=r["file"], px=r["px"], dist=None, size=None, dets=dets, hit={}, fa={}, err=None)
        if r["box"]:
            b = r["box"]; cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
            size = max(b[2] - b[0], b[3] - b[1]) + 1
            row["size"] = size * k_det
            row["dist"] = math.dist(r["bird"][:3], r["rig"][:3])
            row["above"] = r["bird"][2] > r["rig"][2]
            near = [d for d in dets if math.hypot(d[1] - cx, d[2] - cy) <= max(8.0, 0.6 * size)]
            for t in thresholds:
                row["hit"][t] = any(d[0] >= t for d in near)
            if near:
                best = max(near)
                row["err"] = math.hypot(best[1] - cx, best[2] - cy)
                row["err_rad"] = row["err"] / GR.CAMSZ[r["cam"]][2]              # as a direction error
                row["logsize"] = math.log(max(best[3], best[4]) / size)          # size error -> distance error
            far = [d for d in dets if d not in near]
        else:
            far = dets
        for t in thresholds:
            row["fa"][t] = sum(d[0] >= t for d in far)
        rows.append(row)
    pos = [r for r in rows if r["size"]]
    out = dict(images=len(rows), with_magpie=len(pos), by_threshold={})
    for t in thresholds:
        out["by_threshold"][t] = dict(found=np.mean([r["hit"][t] for r in pos]),
                                      false_alarms_per_image=np.mean([r["fa"][t] for r in rows]))
    bins = [(1, 4), (4, 7), (7, 10), (10, 15), (15, 25), (25, 50), (50, 2000)]
    # the operating threshold, 0.5: with 200 places of pad/shadow hard negatives in training (the first
    # detector at 0.5 tracked the pads' ArUco squares as birds in Gazebo), and the tracker reporting only
    # MOVING birds. At 0.7 phantoms vanished but a diving magpie was reported only 66% of the time
    out["operating_threshold"] = op
    out["found_by_size_px"] = {f"{a}-{b}": dict(n=len(s := [r for r in pos if a <= r["size"] < b]),
                                                found=float(np.mean([r["hit"][op] for r in s])) if s else None) for a, b in bins}
    dbins = [(0, 5), (5, 10), (10, 20), (20, 30), (30, 45), (45, 70)]
    out["found_by_distance_m"] = {f"{a}-{b}": dict(n=len(s := [r for r in pos if a <= r["dist"] < b]),
                                                   found=float(np.mean([r["hit"][op] for r in s])) if s else None) for a, b in dbins}
    e = [r["err"] for r in pos if r["err"] is not None]
    out["centre_error_px"] = dict(median=float(np.median(e)), p90=float(np.percentile(e, 90))) if e else None
    ea = [r["err_rad"] for r in pos if r["err"] is not None]
    out["centre_error_rad"] = dict(median=float(np.median(ea)), p90=float(np.percentile(ea, 90))) if ea else None
    ls = np.array([r["logsize"] for r in pos if r.get("logsize") is not None])
    out["log_size_error"] = dict(mean=float(ls.mean()), sd=float(ls.std()), n=len(ls)) if len(ls) else None
    sizes = [(a + b) / 2 if b < 2000 else 80 for a, b in bins]              # bin centres, for the simulator
    keep = [k for k, (a, b) in enumerate(bins) if out["found_by_size_px"][f"{a}-{b}"]["n"]]
    pooled = [out["found_by_size_px"][f"{bins[k][0]}-{bins[k][1]}"]["found"] for k in keep]

    def curve(sel):
        """Detection by size for a subset (bird above the drone: sky behind it; below: trees, roofs,
        ground); bins with under 20 sightings fall back to the pooled value."""
        res = []
        for k, pk in zip(keep, pooled):
            a, b = bins[k]
            s_ = [r for r in pos if a <= r["size"] < b and sel(r)]
            res.append(float(np.mean([r["hit"][op] for r in s_])) if len(s_) >= 20 else pk)
        return res
    out["p_detect"] = dict(size_px=[sizes[k] for k in keep], p=pooled,
                           p_sky=curve(lambda r: r["above"]), p_ground=curve(lambda r: not r["above"]))
    json.dump(out, open(f"{root}/score{tag}.json", "w"), indent=1, default=float)
    print(json.dumps(out, indent=1, default=float))
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    for a, key, lab in ((ax, "found_by_size_px", "Magpie's longest side in the image (px)"),
                        (bx, "found_by_distance_m", "Distance from the drone (m)")):
        ks = [k for k, v in out[key].items() if v["n"]]
        a.bar(range(len(ks)), [100 * out[key][k]["found"] for k in ks], color="#2e86c1")
        for i, k in enumerate(ks):
            a.text(i, 100 * out[key][k]["found"] + 1, f"n={out[key][k]['n']}", ha="center", fontsize=8)
        a.set_xticks(range(len(ks)), ks); a.set(xlabel=lab, ylabel="Found in a single frame (%)", ylim=(0, 108))
        a.grid(axis="y", alpha=0.3)
    fa = out["by_threshold"][op]["false_alarms_per_image"]
    fig.suptitle(f"Detector on drone places it never trained on (noisy, blurred, compressed images; "
                 f"{fa:.3f} false alarms per image)", fontsize=11)
    fig.savefig(f"{root}/detect_vs_size{tag}.png", dpi=150)
    print(f"{root}/detect_vs_size{tag}.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", metavar="DIR")
    ap.add_argument("--score", metavar="DIR")
    ap.add_argument("--steps", type=int, default=8000)
    a = ap.parse_args()
    if a.train:
        train(a.train, a.steps)
    if a.score:
        score(a.score)
