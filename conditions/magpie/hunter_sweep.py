#!/usr/bin/env python3
"""
hunter_sweep.py -- does dodging work at all against a magpie that keeps re-aiming? It depends on
two guesses about the bird: how late it stops correcting (T_COMMIT, its reaction time) and how hard
it can turn (A_TURN). Sweep the reaction time, rule decider, magpie on every flight, corridor world,
GPS, LiDAR map; the bird known exactly (omni) or through the rig (rig). CPU, one process per point
(the hunter's constants are read at import).

    python hunter_sweep.py [--n 1024] [--out hunter_sweep.json]
"""
import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
T_COMMITS = (0.1, 0.3, 0.5, 0.8, 1.2)


def one(sense, n):
    import torch
    sys.path.insert(0, HERE)
    import urban_agents as A
    import urban_env as U
    torch.set_num_threads(8)
    env = U.UrbanEnv(n, "corridor", seed=2026, map_path=f"{U.HERE}/logan_map_lidar.npz", nav="gps", p_swoop=1.0,
                     bird_sense=sense, magpie="hunter", device="cpu")
    s = U.run(env, A.Rule())
    struck = (s["outcome"] == 7).float()
    print(json.dumps(dict(sense=sense, t_commit=U.T_COMMIT, a_turn=U.A_TURN, n=n, struck=100 * struck.mean().item(),
                          clean=100 * s["clean"].float().mean().item(), swoops=env.bird["swoops"].mean().item(),
                          per_dive=100 * struck.sum().item() / max(env.bird["swoops"].sum().item(), 1))))


def plot(path, out):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = json.load(open(path))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), layout="constrained")
    for ax, key, lab in ((axes[0], "per_dive", "Dives that touch the drone (%)"), (axes[1], "struck", "Flights struck (%)")):
        for sense, col, name in (("omni", "#7f8c8d", "handed the truth (all-round, exact)"), ("rig", "#16a085", "360-degree rig (detector + tracker)")):
            r = sorted((x for x in rows if x["sense"] == sense), key=lambda x: x["t_commit"])
            ax.plot([x["t_commit"] for x in r], [x[key] for x in r], "-o", color=col, lw=2, label=name)
        ax.set(xlabel="How late the magpie stops correcting its dive (s before arrival)", ylabel=lab)
        ax.axvline(0.3, color="black", ls=":", lw=1); ax.grid(alpha=0.3)
    axes[0].axhline(12.0, color="#c0392b", ls="--", lw=1); axes[0].text(1.2, 12.3, "no dodging at all: 12%", ha="right", color="#c0392b", fontsize=9)
    axes[0].legend(fontsize=9); axes[1].text(0.32, axes[1].get_ylim()[1] * 0.95, "default", fontsize=8, va="top")
    fig.suptitle("Can a drone dodge a magpie that keeps re-aiming? The rule decider, a magpie on every flight (1,024 flights a point)", fontsize=11)
    fig.savefig(out, dpi=150)
    print(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1024)
    ap.add_argument("--out", default=f"{HERE}/hunter_sweep.json")
    ap.add_argument("--one", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--plot", metavar="PNG", default=None, help="chart an existing --out file")
    a = ap.parse_args()
    if a.plot:
        return plot(a.out, a.plot)
    if a.one:
        return one(a.one, a.n)
    rows = []
    for tc in T_COMMITS:
        for sense in ("omni", "rig"):
            env = dict(os.environ, HUNTER_T_COMMIT=str(tc), CUDA_VISIBLE_DEVICES="")
            out = subprocess.run([sys.executable, __file__, "--one", sense, "--n", str(a.n)], env=env,
                                 capture_output=True, text=True, check=True).stdout.strip().splitlines()[-1]
            rows.append(json.loads(out))
            print(rows[-1], flush=True)
    json.dump(rows, open(a.out, "w"), indent=1)
    print(a.out)


if __name__ == "__main__":
    main()
