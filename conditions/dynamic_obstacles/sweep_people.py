#!/usr/bin/env python3
"""
sweep_people.py -- does the headroom ladder survive the people assumption?

The 3.2-3.5% that no decider avoids comes mostly from people lingering at the
pad, and how long they linger (uniform 30-240 s) is an assumption, not a
measurement. Rerun the ladder at an impatient, the current, and a patient
setting. If only the ceiling moves, the ladder is a property of the deciders;
if its shape changes, it was a property of the assumption.
"""
import os, sys
import torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "shared", "fast_sim"))   # run from shared/fast_sim
import urban_agents as A
import urban_env as U
from urban_headroom import paired

N, SEED = 4096, 2026
for leave in ((0.0, 60.0), (30.0, 240.0), (60.0, 300.0)):
    S = {k: U.run(U.UrbanEnv(N, "corridor", seed=SEED, leave=leave), a) for k, a in
         {"rule": A.Rule(), "shield_obs": A.MPC("obs"), "safeplan_obs": A.MPC("obs", "safeplan"),
          "shield_full": A.MPC("full"), "safeplan_full": A.MPC("full", "safeplan")}.items()}
    f = lambda x: 100 * x.float().mean().item()
    bo = max(("shield_obs", "safeplan_obs"), key=lambda k: f(S[k]["clean"]))
    bf = max(("shield_full", "safeplan_full"), key=lambda k: f(S[k]["clean"]))
    dec, fut, tot = (paired(S[bo]["clean"], S["rule"]["clean"]), paired(S[bf]["clean"], S[bo]["clean"]),
                     paired(S[bf]["clean"], S["rule"]["clean"]))
    viol = lambda s: f(s["crash"] | s["nmac"] | (s["viol_s"] > 0).any(-1))
    print(f"people leave {leave[0]:3.0f}-{leave[1]:3.0f} s | clean: rule {f(S['rule']['clean']):4.1f}  "
          f"planner {f(S[bo]['clean']):4.1f}  oracle {f(S[bf]['clean']):4.1f} | decisions {dec[0]:+.1f}+/-{dec[1]:.1f}  "
          f"future {fut[0]:+.1f}+/-{fut[1]:.1f}  total {tot[0]:+.1f} | unavoidable {100 - f(S[bf]['clean']):.1f}% | "
          f"violations {viol(S['rule']):.1f} -> {viol(S[bo]):.1f} -> {viol(S[bf]):.1f}%")
