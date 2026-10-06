#!/usr/bin/env python3
"""
phase1_eval.py -- PLAN.md Phase 1: how each arm flies, not only whether it lands. Every arm on the same test flights
(seed 10000, 1,024 per wind) from W10 9 to 22 (22 is beyond anything trained on), with evaluate()'s flying metrics
(push at its cap, push step change, climb/descend reversals, motor saturation, time to land), the paper decider's
pick table (its vertical choice by height band and by what the rule would have done), and greedy vs sampled read-out.

    python conditions/wind/phase1_eval.py   -> runs/phase1.json, outputs/wind/phase1.png, ClearML (~15 min)
"""
import json
import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_deterministic_ops=true")   # reported numbers repeat exactly
import jax.numpy as jnp
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import land_env as E
import ppo

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.dirname(os.path.dirname(HERE))
WINDS = (9.0, 12.0, 15.0, 18.0, 20.0, 22.0)
BANDS, CATS = ("above 8 m", "8-6 m", "6-2 m", "below 2 m"), ("rule: fast down", "rule: slow down", "rule: hold")
MODES = ("rule's own", "down 1.5", "down 0.6", "hold", "up 1")


def ladder(pol, fly, decide, shield=0.0):
    return [E.evaluate(pol, w, n=1024, seed=10_000, fly=fly, decide=decide, shield=shield) for w in WINDS]


def main():
    rule = lambda obs, st: jnp.zeros((obs.shape[0], 3))
    dob = lambda obs, st: jnp.zeros((obs.shape[0], 3)).at[:, :2].set(ppo.dob_ff(1.0, obs))
    arms = {"rule": ladder(rule, False, False)}
    tuned = json.load(open(f"{HERE}/runs/rule_tuned.json"))["chosen"]
    E.GATE_SCALE, E.PATIENCE = tuned["scale"], tuned["patience"]
    arms["tuned rule"] = ladder(rule, False, False)
    E.GATE_SCALE, E.PATIENCE = 1.0, 1e9
    arms["DOB"] = ladder(dob, True, False)
    for s in (0, 1, 2):
        for greedy in (True, False):
            pol, fly, decide, sh = ppo.load_policy(f"{HERE}/runs/stage2_c100h100g_s{s}_best.npz", greedy=greedy)
            arms[f"paper decider s{s}" + ("" if greedy else " sampled")] = ladder(pol, fly, decide, sh)
    pol, fly, decide, sh = ppo.load_policy(f"{HERE}/runs/stage2_dob_c100h100g_s0_best.npz")
    arms["DOB decider s0"] = ladder(pol, fly, decide, sh)
    for rows in arms.values():
        for r in rows:
            r.pop("first40", None)
    json.dump(arms, open(f"{HERE}/runs/phase1.json", "w"), indent=1)

    print("landed % / crashed % on the test flights")
    print(f"{'':28s}" + "".join(f"   W10 {w:<4g}" for w in WINDS))
    for name, rows in arms.items():
        print(f"{name:28s}" + "".join(f" {100*r['landed']:5.1f}/{100*r['crashed']:4.1f}" for r in rows))
    print("\nflying at W10 18: push at cap %, push step m/s^2, reversals /s, motor saturation %, median time to land s")
    for name, rows in arms.items():
        r = rows[WINDS.index(18.0)]
        print(f"{name:28s} {100*r['push_cap']:5.1f}  {r['push_step']:5.2f}  {r['rev_per_s']:5.2f}  {100*r['motor_sat']:5.1f}  {r['t_land_median']:5.1f}")
    for w in (12.0, 18.0):
        P = sum(np.array(arms[f"paper decider s{s}"][WINDS.index(w)]["picks"], float) for s in (0, 1, 2))
        print(f"\npick table, paper decider s0-s2 together, W10 {w:g}: % of its choices in each situation (decisions)")
        for b, band in enumerate(BANDS):
            for c, cat in enumerate(CATS):
                n = P[b, c].sum()
                if n >= 100:
                    print(f"  {band:10s} {cat:16s} n={int(n):8d}  " + "  ".join(f"{m} {100*P[b, c, k]/n:5.1f}" for k, m in enumerate(MODES)))

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    col = {"rule": "#99a3a4", "tuned rule": "#16a085", "DOB": "#8e44ad", "DOB decider s0": "#d35400"}
    for name, rows in arms.items():
        if "sampled" in name:
            continue
        c = col.get(name, "#2e86c1")
        lab = name if not name.startswith("paper") else ("paper decider (3 seeds)" if name.endswith("s0") else None)
        ax[0].plot(WINDS, [100 * r["landed"] for r in rows], "o-", color=c, ms=4, label=lab)
        ax[1].plot(WINDS, [100 * r["push_cap"] for r in rows], "o-", color=c, ms=4)
        ax[2].plot(WINDS, [100 * r["crashed"] for r in rows], "o-", color=c, ms=4)
    ax[0].axvspan(20.5, 22.5, color="gray", alpha=0.12); ax[0].text(20.6, 50, "beyond\ntraining", fontsize=8, color="gray")
    ax[0].set(title="landed (%), 1,024 test flights per wind", xlabel="W10 (m/s)", ylim=(0, 102)); ax[0].legend(fontsize=8)
    ax[1].set(title="push held at its 3 m/s² cap (% of flying time)", xlabel="W10 (m/s)", ylim=(0, 100))
    ax[2].set(title="crashed (%)", xlabel="W10 (m/s)")
    for a in ax: a.grid(alpha=0.3); a.set_xticks(WINDS)
    fig.tight_layout(); os.makedirs(f"{R}/outputs/wind", exist_ok=True); fig.savefig(f"{R}/outputs/wind/phase1.png", dpi=120)

    try:
        from clearml import Task
        t = Task.init(project_name="rl-landing/wind", task_name="phase 1: how each arm flies", reuse_last_task_id=False,
                      auto_connect_frameworks=False, auto_connect_arg_parser=False)
        lg = t.get_logger()
        for k in ("landed", "crashed", "push_cap", "push_step", "rev_per_s", "motor_sat"):
            lg.report_table(k, "all arms", iteration=0, table_plot=[["arm"] + [f"W10 {w:g}" for w in WINDS]]
                            + [[n] + [round(float(r[k]) * (100 if k in ("landed", "crashed", "push_cap", "motor_sat") else 1), 2) for r in rows] for n, rows in arms.items()])
        lg.report_matplotlib_figure("landed, push at cap, crashed", "all arms", fig, iteration=0)
        t.close()
    except Exception as e:
        print("clearml:", e)


if __name__ == "__main__":
    main()
