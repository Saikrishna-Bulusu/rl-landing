#!/usr/bin/env python3
"""Does refusing to descend through a canopy actually pay?"""
import numpy as np
import sensing_compare as C
import rule_agent as R


def score(make_agent, n=400, scale=3.5):
    env = C.make("sectors"); env.sigma_scale = scale
    rew, ok, cr = [], [], []
    for i in range(n):
        o, _ = env.reset(seed=7_500_000 + i)
        ag = make_agent(env); ag.reset(); tot = 0.0
        while True:
            o, r, t, tr, info = env.step(np.clip(ag.act(o), -1, 1)); tot += r
            if t or tr: break
        rew.append(tot); ok.append(info["success"]); cr.append(info.get("outcome") == "crash")
    return float(np.mean(rew)), 100*np.mean(ok), 100*np.mean(cr)


if __name__ == "__main__":
    print("  agent                                reward   delivered   crash")
    for name, f in (("aim at the averaged fix", lambda e: R.SigmaRuleMean(e, 0.5)),
                    ("+ refuse to descend into canopy", lambda e: R.CanopyRule(e, 0.5))):
        rew, ok, cr = score(f)
        print(f"  {name:<34} {rew:+7.1f}     {ok:5.1f}%   {cr:5.1f}%")
