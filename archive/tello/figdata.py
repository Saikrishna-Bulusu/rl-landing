#!/usr/bin/env python3
"""
figdata.py -- what the paper's Tello figure draws (shared/fast_sim/paper_figs.py reads it; that script has no torch):
the ladder's outcomes on the 4,096 test flights, and one flight where the unmapped pole stands on the way --
the rule's path into it and the planner's around it.        python figdata.py  -> cage_fig.json
"""
import json
import os

import torch

import cage as C

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    out = {"ladder": {}}
    for name, ag in (("rule", C.Rule()), ("planner", C.Planner()), ("all_knowing", C.Planner("full"))):
        s = C.run(C.Cage(4096, seed=2026), ag)
        out["ladder"][name] = {o: 100 * (s["outcome"] == i).float().mean().item() for i, o in enumerate(C.OUTCOMES) if i}
    import train as T, urban_rl as R                                    # the kept learners (dice read-out), same flights
    seeds = []
    for sd in range(3):
        path = f"{HERE}/cage_rl_s{sd}_sampled.pt"
        if os.path.exists(path):
            st = torch.load(path, weights_only=True)
            net = R.Net(st["body.0.weight"].shape[1]).to(C.DEV); net.load_state_dict(st); net.eval()
            torch.manual_seed(sd)
            s = C.run(C.Cage(4096, seed=2026), T.Learned(net, greedy=False))
            seeds.append({o: 100 * (s["outcome"] == i).float().mean().item() for i, o in enumerate(C.OUTCOMES) if i})
    if seeds:
        out["ladder"]["learner"] = {o: sum(x[o] for x in seeds) / len(seeds) for o in seeds[0]}
        out["learner_seeds"] = seeds
    e = C.Cage(256, seed=11)
    sr, pr = C.run(e, C.Rule(), record=True)
    pole_on = e.pole_on.clone()
    sp, pp = C.run(C.Cage(256, seed=11), C.Planner(), record=True)          # same seed: the same 256 cages
    k = int(((sr["outcome"] == 4) & (sp["outcome"] == 1) & pole_on).float().argmax())
    cut = lambda s, p: p[k, : int(round(float(s["t_done"][k]) / C.DT))].tolist()
    out["flight"] = dict(pads=e.pads[k].tolist(), houses=e.houses[k].tolist(), trees=e.trees[k].tolist(),
                         pole=e.pole[k].tolist(), hub=list(C.HUB), rule=cut(sr, pr), planner=cut(sp, pp),
                         outcome=dict(rule=C.OUTCOMES[int(sr["outcome"][k])], planner=C.OUTCOMES[int(sp["outcome"][k])]))
    json.dump(out, open(f"{HERE}/cage_fig.json", "w"))
    print({k_: round(v["delivered"], 1) for k_, v in out["ladder"].items()}, "flight", k, out["flight"]["outcome"])


if __name__ == "__main__":
    main()
