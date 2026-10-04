#!/usr/bin/env python3
"""
video_birds.py -- the same magpie flights flown by the planner and by the learner trained with
birds, side by side, top-down: the pad, the nest and its 100 m territory, the drone's trail, the
magpie and its dive lines.

CPU only -- the GPU belongs to training:
    CUDA_VISIBLE_DEVICES="" nice -n 15 python video_birds.py [--seed 71] [--n 256] [--ckpt ...]

Both batches use the same seed, so flight i is the same delivery, people, traffic and nest for
both deciders; the magpie then reacts to whatever each drone does.
"""
import argparse

import numpy as np
import torch

import urban_agents as A
import urban_env as U
import urban_rl as R

MAP = "logan_map_lidar.npz"
HALF = 110.0          # view: +/- this many metres around the pad
FPS = 12              # 0.25 s steps at 12 fps: three times real time


def fly(agent, n, seed):
    env = U.UrbanEnv(n, "corridor", seed=seed, map_path=f"{U.HERE}/{MAP}", nav="gps", p_swoop=U.P_SWOOP)
    o = env.reset(); agent.reset(env)
    tr = {k: [] for k in ("pos", "bird", "dive", "done", "outcome", "swoops", "people")}
    while not bool(env.done.all()) and env.t < U.T_MAX - 1e-6:
        o = env.step(agent.act(o))
        tr["pos"].append(env.pos.clone()); tr["bird"].append(env.bird["pos"].clone())
        tr["dive"].append(env.bird["dive"].clone()); tr["done"].append(env.done.clone())
        tr["outcome"].append(env.outcome.clone()); tr["swoops"].append(env.bird["swoops"].clone())
        gp, _ = env.ground_at(env.gr, torch.full((env.n, 1), env.t))
        tr["people"].append(gp[:, 0, :U.N_PEOPLE, :2].clone())            # FAR when not out
    return env, {k: torch.stack(v).numpy() for k, v in tr.items()}


def pick(ep, el, tp, tl):
    """A flight the planner lost to the magpie and the learner flew clean; one with the most swoops
    the learner came through clean."""
    on = ep.bird["on"].numpy()
    sp, sl = ep.bird["swoops"].numpy(), el.bird["swoops"].numpy()
    cp, cl = ep.summary()["clean"].numpy(), el.summary()["clean"].numpy()
    out = []
    a = np.nonzero(on & (ep.outcome.numpy() == 7) & cl)[0]
    if len(a):
        out.append((int(a[0]), "The planner is struck; the learner is not"))
    b = np.nonzero(on & cl & (sl >= 2))[0]
    if len(b):
        i = int(b[np.argmax(sl[b])])
        out.append((i, f"The learner comes through {int(sl[i])} swoops"))
    return out


def render(i, title, envs, trs, names, vw):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    env0 = envs[0]
    pad = env0.pad[i].numpy(); nest = env0.bird["nest"][i, :2].numpy()
    ij = lambda v: int((v + env0.half) / env0.res)
    r0, r1, c0, c1 = ij(pad[1] - HALF), ij(pad[1] + HALF), ij(pad[0] - HALF), ij(pad[0] + HALF)
    hmap = env0.H[r0:r1, c0:c1].numpy()
    ext = (pad[0] - HALF, pad[0] + HALF, pad[1] - HALF, pad[1] + HALF)
    # from the first moment either drone is within 150 m of the pad, to 2 s after both are done
    near = [np.nonzero(np.hypot(*(t["pos"][:, i, :2] - pad).T) < 150)[0] for t in trs]
    k0 = min(n[0] for n in near if len(n))
    ends = [np.argmax(t["done"][:, i]) if t["done"][:, i].any() else len(t["done"]) - 1 for t in trs]
    k1 = min(max(ends) + 8, min(len(t["pos"]) for t in trs) - 1)
    import cv2
    n = 0
    fig, axs = plt.subplots(1, 2, figsize=(12.8, 7.2), dpi=100)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.86, bottom=0.1, wspace=0.04)
    fig.text(0.5, 0.055, "drone: blue, turning RED inside the magpie's airspace (below 40 m, within 100 m of the nest)",
             ha="center", fontsize=10)
    fig.text(0.5, 0.02, "black triangle: magpie (red dots: its dive)    brown dot: nest    dashed circle: territory    "
             "green square: pad    orange: people", ha="center", fontsize=10)
    for k in range(k0, k1 + 1):
        for ax, t, name, end in zip(axs, trs, names, ends):
            ax.clear()
            ax.imshow(np.where(hmap > 1.0, hmap, np.nan), origin="lower", extent=ext, cmap="Greys", vmin=-10, vmax=30)
            ax.set_facecolor("#e8efe4")
            ax.add_patch(plt.Circle(nest, U.R_TERR, fill=False, ls="--", color="#b9770e", lw=1.2))
            ax.plot(*nest, "o", color="#6e2c00", ms=8)
            ax.add_patch(plt.Rectangle(pad - 1.5, 3, 3, color="#1e8449"))
            kk = min(k, end)
            p = t["pos"][: kk + 1, i]
            ax.plot(p[max(0, kk - 60):, 0], p[max(0, kk - 60):, 1], color="#2e86c1", lw=1.5, alpha=0.7)
            z = p[-1, 2]
            inside = z < U.Z_TERR and np.hypot(*(p[-1, :2] - nest)) < U.R_TERR
            ax.plot(*p[-1, :2], "o", color="#c0392b" if inside else "#2e86c1", ms=9, mec="white")
            pp = t["people"][kk, i]
            pp = pp[np.abs(pp).max(-1) < 1e5]
            if len(pp):
                ax.plot(pp[:, 0], pp[:, 1], "o", color="#e67e22", ms=6, mec="white")
            b = t["bird"][kk, i]
            ax.plot(*b[:2], marker="v", color="black", ms=9)
            if t["dive"][kk, i]:
                ax.plot([b[0], p[-1, 0]], [b[1], p[-1, 1]], ls=":", color="#c0392b", lw=1)
            oc = int(t["outcome"][kk, i])
            state = {0: "", 1: "DELIVERED", 7: "STRUCK BY THE MAGPIE"}.get(oc, U.OUT.get(oc, "").upper())
            ax.set_title(f"{name}\naltitude {z:4.1f} m    swoops so far {int(t['swoops'][kk, i])}"
                         f"{'    in its airspace' if inside else ''}",
                         fontsize=11, color="#c0392b" if inside else "black")
            if state:
                ax.text(0.5, 0.06, state, transform=ax.transAxes, ha="center", fontsize=16, fontweight="bold",
                        color="#1e8449" if oc == 1 else "#c0392b",
                        bbox=dict(facecolor="white", alpha=0.85, edgecolor="none"))
            ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3]); ax.set_xticks([]); ax.set_yticks([])
        fig.suptitle(f"{title}   (t = {(k + 1) * U.DT:5.1f} s, three times real time)", fontsize=13)
        fig.canvas.draw()
        vw.write(cv2.cvtColor(np.asarray(fig.canvas.buffer_rgba())[..., :3], cv2.COLOR_RGB2BGR)); n += 1
    plt.close(fig)
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=71)
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--ckpt", default="rl_corridor_eyes_lidar_gps_birds_s1_mean.pt")
    ap.add_argument("--out", default="birds_video_h264.mp4")
    a = ap.parse_args()
    torch.set_num_threads(8)
    R.MAP, R.NAV, R.BIRDS = MAP, "gps", True
    probe = U.UrbanEnv(8, "corridor", seed=1, map_path=f"{U.HERE}/{MAP}", nav="gps", p_swoop=U.P_SWOOP)
    o = probe.reset(); g = A.MPC("obs"); g.reset(probe); g.act(o)
    net = R.Net(R.inputs(o, g, True).shape[-1]).to(R.DEV)
    net.load_state_dict(torch.load(a.ckpt, map_location=R.DEV, weights_only=True)); net.eval()
    read = "mean" if "_mean" in a.ckpt else ("sampled" not in a.ckpt)
    ep, tp = fly(A.MPC("obs"), a.n, a.seed)
    el, tl = fly(R.Learned(net, greedy=read, eyes=True), a.n, a.seed)
    f = lambda s: 100 * s.float().mean().item()
    on = ep.bird["on"]
    print(f"with a magpie ({int(on.sum())} flights): planner clean {f(ep.summary()['clean'][on]):.1f}% struck "
          f"{f((ep.outcome == 7)[on]):.1f}% | learner clean {f(el.summary()['clean'][on]):.1f}% struck {f((el.outcome == 7)[on]):.1f}%")
    # Frames stream to mp4v (this OpenCV has no H.264) -- a few hundred 1280x720 frames held in
    # memory would be ~2 GB next to two trainings -- then /usr/bin/python3's OpenCV re-encodes.
    import cv2, subprocess
    tmp = a.out.replace(".mp4", "_mp4v.mp4")
    vw = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (1280, 720))
    assert vw.isOpened()
    n = 0
    for i, title in pick(ep, el, tp, tl):
        print(f"flight {i}: {title}", flush=True)
        n += render(i, title, [ep, el], [tp, tl], ["Planner", "Learner (trained with birds)"], vw)
    vw.release()
    subprocess.run(["/usr/bin/python3", "-c", f"import cv2; s = cv2.VideoCapture('{tmp}'); "
                    f"w = cv2.VideoWriter('{a.out}', cv2.VideoWriter_fourcc(*'avc1'), {FPS}, (1280, 720)); "
                    f"ok, fr = s.read()\nwhile ok: w.write(fr); ok, fr = s.read()\nw.release()"], check=True)
    print(f"{a.out}: {n} frames, {n / FPS:.0f} s")


if __name__ == "__main__":
    main()
