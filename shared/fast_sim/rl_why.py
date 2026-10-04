#!/usr/bin/env python3
"""
What did practice fix? For each world and seed, fly the read-out chosen on
VALIDATION (the kept checkpoint whose validation score is highest) on the ladder's
4,096 test flights and break the failures down by first reason, next to the
same-sensor planner's. Descriptive only: nothing here is chosen on test flights.
Writes rl_why.json.
"""
import glob
import json

import torch

import urban_agents as A
import urban_env as U
import urban_rl as R
from urban_headroom import failures

head = json.load(open("headroom.json"))
out = {}
for world in ("corridor", "random"):
    e, g = U.UrbanEnv(8, world, seed=1), A.MPC("obs"); o = e.reset(); g.reset(e); g.act(o)
    rows = {"planner": head[world]["fail_best_obs"]}
    for path in sorted(glob.glob(f"rl_eyes_{world}_*.json")):
        for tag, r in json.load(open(path)).items():
            if tag == "baselines":
                continue
            kept = r["log"][-1]
            k = max(R.READOUTS, key=lambda m: kept[f"kept_{m}"]["validation"])
            net = R.Net(R.inputs(o, g, True).shape[-1]).to(R.DEV)
            net.load_state_dict(torch.load(f"rl_{tag}_{k}.pt", weights_only=True))
            s = U.run(U.UrbanEnv(4096, world, seed=2026), R.Learned(net, greedy=R.READOUTS[k], eyes=True))
            rows[f"{tag}:{k}"] = failures(s)
            print(f"  {tag:<20} read-out {k:<8} clean {100 * s['clean'].float().mean().item():.1f}%", flush=True)
    out[world] = rows
    print(f"\n{world}: first failure reason, % of 4,096 test flights")
    cols = list(rows)
    for reason in rows["planner"]:
        rl = [rows[c][reason] for c in cols[1:]]
        print(f"  {reason:26s} planner {rows['planner'][reason]:5.2f}   RL mean {sum(rl) / len(rl):5.2f}  "
              f"({', '.join(f'{x:.2f}' for x in rl)})")
json.dump(out, open("rl_why.json", "w"), indent=1)
