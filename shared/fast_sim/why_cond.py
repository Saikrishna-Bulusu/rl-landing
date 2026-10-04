#!/usr/bin/env python3
"""
why_cond.py -- what did practice fix, per condition? Each seed's kept learner (dice read-out, the round
chosen on validation) and the planner it copied fly the ladder's 4,096 test flights; failures are broken
down by first reason (urban_headroom.failures). Descriptive only: nothing here is chosen on test flights.

    python why_cond.py [--n 4096] [clear unmapped fog rig]      -> why_cond.json (merged per condition)
"""
import argparse
import json
import os

import torch

import urban_agents as A
import urban_env as U
import urban_rl as R
from urban_headroom import failures

# condition -> (the trainer's globals, the checkpoint tag's suffix); the map and GPS nav are shared
CONDS = {"clear": ({}, ""), "unmapped": ({"UNMAPPED": True}, "_unmapped"), "fog": ({"FOG": True}, "_fog"),
         "rig": ({"BIRDS": True, "HUNTER": True, "RIG": True}, "_birds_hunter_rig")}
OUT = f"{U.HERE}/why_cond.json"


def run(name, n):
    flags, suf = CONDS[name]
    for k in ("BIRDS", "FOG", "UNMAPPED", "CAMERAS", "HUNTER", "RIG"):
        setattr(R, k, flags.get(k, False))
    R.MAP, R.NAV = "logan_map_lidar.npz", "gps"
    e, g = R.world_env(8, "corridor", 1), A.MPC("obs"); o = e.reset(); g.reset(e); g.act(o)
    dim = R.inputs(o, g, True).shape[-1]
    fly = lambda agent: U.run(R.world_env(n, "corridor", 2026), agent)           # the ladder's test flights
    f = lambda s: 100 * s["clean"].float().mean().item()
    rows = {}
    for label, agent in (("rule", A.Rule()), ("planner", A.MPC("obs"))):
        s = fly(agent); rows[label] = dict(clean=f(s), fail=failures(s))
        print(f"  {name:<9} {label:<10} clean {rows[label]['clean']:.1f}%", flush=True)
    for sd in range(3):
        path = f"{U.HERE}/rl_corridor_eyes_lidar_gps{suf}_s{sd}_sampled.pt"
        net = R.Net(dim).to(R.DEV); net.load_state_dict(torch.load(path, weights_only=True)); net.eval()
        torch.manual_seed(sd)
        s = fly(R.Learned(net, greedy=False, eyes=True)); rows[f"learner_s{sd}"] = dict(clean=f(s), fail=failures(s))
        print(f"  {name:<9} learner s{sd} clean {rows[f'learner_s{sd}']['clean']:.1f}%", flush=True)
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("conds", nargs="*", default=list(CONDS))
    ap.add_argument("--n", type=int, default=4096)
    a = ap.parse_args()
    out = json.load(open(OUT)) if os.path.exists(OUT) and a.n == 4096 else {}
    for c in a.conds:
        out[c] = run(c, a.n)
        if a.n == 4096:
            json.dump(out, open(OUT, "w"), indent=1)
    for c in a.conds:
        print(f"\n{c}: first failure reason, % of {a.n} test flights (planner -> learner mean)")
        ls = [out[c][k]["fail"] for k in out[c] if k.startswith("learner")]
        for reason, v in out[c]["planner"]["fail"].items():
            m = sum(l[reason] for l in ls) / len(ls)
            if max(v, m) >= 0.3:
                print(f"  {reason:<26} {v:5.1f} -> {m:5.1f}")
