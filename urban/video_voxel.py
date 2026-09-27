#!/usr/bin/env python3
"""
video_voxel.py -- stage 3 on video: the same deliveries flown trusting only the 2023 survey, and by the
learner that flies on the voxel map (the survey plus what its depth camera and rangefinder have seen).
Top: from above, around the pad. Bottom: from the side, altitude against distance to the pad. Things the
survey does not show are dashed while unseen and fill in red once the drone's sensors have put them on
its map; the wedge ahead of the drone is the depth camera (8 m, 73 deg).

CPU only -- the GPU belongs to the live flights:
    CUDA_VISIBLE_DEVICES="" nice -n 15 python video_voxel.py [--seed 7] [--n 512]
"""
import argparse
import math
import subprocess

import numpy as np
import torch

import urban_agents as A
import urban_env as U
import urban_rl as R

MAP = "logan_map_lidar.npz"
HALF = 75.0           # view: +/- this many metres around the pad
FPS = 12              # 0.25 s steps at 12 fps: three times real time


class SurveyOnly(A.Rule):
    """The rule flying on the 2023 survey alone: no voxel map, no look-ahead for what it lacks."""

    def reset(self, env):
        super().reset(env)
        self.voxel = False


def fly(agent, n, seed):
    env = U.UrbanEnv(n, "corridor", seed=seed, map_path=f"{U.HERE}/{MAP}", nav="gps", p_new=U.P_NEW)
    o = env.reset(); agent.reset(env)
    tr = {k: [] for k in ("pos", "head", "known", "done", "outcome")}
    while not bool(env.done.all()) and env.t < U.T_MAX - 1e-6:
        o = env.step(agent.act(o))
        tr["pos"].append(env.pos.clone()); tr["head"].append(env.heading.clone()); tr["known"].append(env.new["known"].clone())
        tr["done"].append(env.done.clone()); tr["outcome"].append(env.outcome.clone())
    return env, {k: torch.stack(v).numpy() for k, v in tr.items()}


def pick(es, el, tl):
    """Flights with the tall crane on the approach where the survey alone flies into it and the voxel-map
    learner delivers clean -- and where the learner first SEES the crane while below its top, so that it
    had to react to its map (passing over it before seeing it would show the flight profile, not the
    map). The two where it was deepest below the top when it saw it."""
    tall = es.new["used"][:, -1].numpy()
    ok = np.nonzero(tall & (es.outcome.numpy() == 8) & el.summary()["clean"].numpy())[0]
    h = el.new["h"][:, -1].numpy()
    depth = []
    for i in ok:
        seen = np.nonzero(tl["known"][:, i, -1])[0]
        if len(seen):
            depth.append((h[i] - tl["pos"][seen[0], i, 2], int(i)))
    return [i for d, i in sorted(depth, reverse=True)[:2] if d > 0]


def render(i, envs, trs, names, vw, maps=(False, True)):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, Wedge
    import cv2
    e0 = envs[0]
    pad = e0.pad[i].numpy()
    ij = lambda v: int((v + e0.half) / e0.res)
    hmap = e0.H[ij(pad[1] - HALF):ij(pad[1] + HALF), ij(pad[0] - HALF):ij(pad[0] + HALF)].numpy()
    ext = (pad[0] - HALF, pad[0] + HALF, pad[1] - HALF, pad[1] + HALF)
    nb = e0.new
    used, c, half, h = nb["used"][i].numpy(), nb["c"][i].numpy(), nb["half"][i].numpy(), nb["h"][i].numpy()
    near = [np.nonzero(np.hypot(*(t["pos"][:, i, :2] - pad).T) < 110)[0] for t in trs]
    k0 = min(n[0] for n in near if len(n))
    ends = [int(np.argmax(t["done"][:, i])) if t["done"][:, i].any() else len(t["done"]) - 1 for t in trs]
    k1 = min(max(ends) + 10, min(len(t["pos"]) for t in trs) - 1)
    fig, axs = plt.subplots(2, 2, figsize=(12.8, 7.2), dpi=100, height_ratios=[1.9, 1])
    fig.subplots_adjust(left=0.05, right=0.98, top=0.9, bottom=0.15, wspace=0.1, hspace=0.2)
    fig.text(0.5, 0.04, "grey: the 2023 LiDAR survey    red dashed: built since, not yet seen    red filled: seen, now on the drone's "
             "voxel map", ha="center", fontsize=10)
    fig.text(0.5, 0.012, "blue: the drone and its trail    wedge: depth camera, 8 m    green: the pad    bottom: side view, "
             "altitude against distance to the pad", ha="center", fontsize=10)
    n = 0
    for k in range(k0, k1 + 1):
        for col, (t, name, end, voxel) in enumerate(zip(trs, names, ends, maps)):
            kk = min(k, end)
            known = t["known"][kk, i] & voxel             # the survey-only drone never puts anything on its map
            p = t["pos"][: kk + 1, i]
            ax, bx = axs[0, col], axs[1, col]
            ax.clear(); bx.clear()
            ax.set_facecolor("#e8efe4")
            ax.imshow(np.where(hmap > 1.0, hmap, np.nan), origin="lower", extent=ext, cmap="Greys", vmin=-10, vmax=40)
            for j in np.nonzero(used)[0]:
                x0, y0 = c[j] - half[j]
                ax.add_patch(Rectangle((x0, y0), 2 * half[j][0], 2 * half[j][1], fill=bool(known[j]), fc="#e74c3c",
                                       ec="#c0392b", lw=1.6, ls="-" if known[j] else "--"))
                if j == len(used) - 1:
                    ax.text(c[j][0] + 3, c[j][1] + 3, f"crane {h[j]:.0f} m", fontsize=9, color="#c0392b")
            ax.add_patch(Rectangle(pad - 1.5, 3, 3, color="#1e8449"))
            ax.plot(p[max(0, kk - 80):, 0], p[max(0, kk - 80):, 1], color="#2e86c1", lw=1.5, alpha=0.8)
            hd = t["head"][kk, i]
            ang = math.degrees(math.atan2(hd[1], hd[0]))
            ax.add_patch(Wedge(p[-1, :2], U.DEPTH_R, ang - 36.5, ang + 36.5, color="#5dade2", alpha=0.35, lw=0))
            ax.plot(*p[-1, :2], "o", color="#2e86c1", ms=8, mec="white")
            oc = int(t["outcome"][kk, i])
            state = {0: "", 1: "DELIVERED", 8: "CRASHED INTO THE CRANE"}.get(oc, U.OUT.get(oc, "").replace("_", " ").upper())
            ax.set_title(f"{name}\naltitude {p[-1, 2]:4.1f} m    " + (f"on its voxel map: {int(known.sum())} of {int(used.sum())} "
                         f"things the survey lacks" if voxel else "its map: the survey alone"), fontsize=10.5)
            if state:
                ax.text(0.5, 0.06, state, transform=ax.transAxes, ha="center", fontsize=16, fontweight="bold",
                        color="#1e8449" if oc == 1 else "#c0392b", bbox=dict(facecolor="white", alpha=0.85, edgecolor="none"))
            ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3]); ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
            # side view
            d = np.hypot(*(p[:, :2] - pad).T)
            for j in np.nonzero(used)[0]:
                dj = max(float(np.hypot(*(c[j] - pad)) - half[j].max()), 0.0)
                bx.add_patch(Rectangle((dj, 0), 2 * half[j].max(), h[j], fill=bool(known[j]), fc="#e74c3c", ec="#c0392b",
                                       lw=1.4, ls="-" if known[j] else "--"))
            bx.plot(d, p[:, 2], color="#2e86c1", lw=1.4)
            bx.plot(d[-1], p[-1, 2], "o", color="#2e86c1", ms=7, mec="white")
            bx.add_patch(Rectangle((0, 0), 1.5, 0.8, color="#1e8449"))
            bx.set_xlim(110, -3); bx.set_ylim(0, 70); bx.set_facecolor("#f4f6f7")
            bx.set_xlabel("distance to the pad (m)", fontsize=9); bx.set_ylabel("altitude (m)", fontsize=9)
            bx.tick_params(labelsize=8); bx.grid(alpha=0.3)
        fig.suptitle(f"Stage 3: what the survey misses   (t = {(k + 1) * U.DT:5.1f} s, three times real time)", fontsize=13)
        fig.canvas.draw()
        vw.write(cv2.cvtColor(np.asarray(fig.canvas.buffer_rgba())[..., :3], cv2.COLOR_RGB2BGR)); n += 1
    plt.close(fig)
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--ckpt", default="rl_corridor_eyes_lidar_gps_unmapped_s0_mean.pt")
    ap.add_argument("--out", default="voxel_video_h264.mp4")
    a = ap.parse_args()
    torch.set_num_threads(8)
    R.MAP, R.NAV, R.UNMAPPED = MAP, "gps", True
    probe = U.UrbanEnv(8, "corridor", seed=1, map_path=f"{U.HERE}/{MAP}", nav="gps", p_new=U.P_NEW)
    o = probe.reset(); g = A.MPC("obs"); g.reset(probe); g.act(o)
    net = R.Net(R.inputs(o, g, True).shape[-1]).to(R.DEV)
    net.load_state_dict(torch.load(a.ckpt, map_location=R.DEV, weights_only=True)); net.eval()
    read = "mean" if "_mean" in a.ckpt else ("sampled" not in a.ckpt)
    es, ts = fly(SurveyOnly(), a.n, a.seed)
    el, tl = fly(R.Learned(net, greedy=read, eyes=True), a.n, a.seed)
    f = lambda s: 100 * s.float().mean().item()
    tall = es.new["used"][:, -1]
    print(f"crane on the way in: {int(tall.sum())} of {a.n} flights | into something the survey lacks: survey only "
          f"{f(es.outcome == 8):.1f}%, learner on the voxel map {f(el.outcome == 8):.1f}% | clean: {f(es.summary()['clean']):.1f}% "
          f"vs {f(el.summary()['clean']):.1f}%", flush=True)
    import cv2
    tmp = a.out.replace(".mp4", "_mp4v.mp4")
    vw = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (1280, 720))
    assert vw.isOpened()
    n = 0
    for i in pick(es, el, tl):
        print(f"flight {i}: crane {float(es.new['h'][i, -1]):.0f} m tall, "
              f"{float((es.new['c'][i, -1] - es.pad[i]).norm()):.0f} m from the pad", flush=True)
        k = np.nonzero(tl["known"][:, i, -1])[0][0]
        print(f"   the learner saw it at {float(tl['pos'][k, i, 2]):.0f} m altitude, "
              f"{float(np.hypot(*(tl['pos'][k, i, :2] - es.new['c'][i, -1].numpy()))):.0f} m from it", flush=True)
        n += render(i, [es, el], [ts, tl], ["Trusting the 2023 survey only", "Learner on the voxel map (survey + depth camera)"], vw)
    vw.release()
    subprocess.run(["/usr/bin/python3", "-c", f"import cv2; s = cv2.VideoCapture('{tmp}'); "
                    f"w = cv2.VideoWriter('{a.out}', cv2.VideoWriter_fourcc(*'avc1'), {FPS}, (1280, 720)); "
                    f"ok, fr = s.read()\nwhile ok: w.write(fr); ok, fr = s.read()\nw.release()"], check=True)
    print(f"{a.out}: {n} frames, {n / FPS:.0f} s")


if __name__ == "__main__":
    main()
