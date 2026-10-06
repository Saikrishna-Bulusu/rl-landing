#!/usr/bin/env python3
"""
robustness.py -- PLAN.md 2c's table: how brittle is each controller, without retraining, when the world varies per
flight? Every arm flies the same 1,024 test flights (seed 10000) per wind in five conditions:

  nominal   the environment of Phases 0-2a (the corrected twin)
  sensing   land_env.sensors_new: pad estimate bias, slow error, dropouts, outliers, height error, 0-120 ms delay
  drone     land_env.drone_new: mass and thrust coefficient +-10%, motor lag +-30%, body drag x0.5-2, 0-80 ms command delay
  gusts     the mean wind turns up to 60 deg, about once per 30 s
  all       the three together

Arms: standard rule, hand-tuned rule (rule_tuned.py's choice), DOB with gain 0.5 (no learning; PLAN.md: it returns
smoothed), and the paper decider's three training runs (stage2_c100h100g_s0-s2).

    python conditions/wind/robustness.py     -> runs/robustness_2c.json, outputs/wind/robustness_2c.png, ClearML (~25 min)
"""
import json
import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_deterministic_ops=true")
import jax.numpy as jnp
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

import land_env as E
import ppo

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.dirname(os.path.dirname(HERE))
WINDS = [9.0, 12.0, 15.0, 18.0, 20.0]
CONDS = {"nominal": (), "sensing": ("sense",), "drone": ("drone",), "gusts": ("gust",), "all": ("sense", "drone", "gust")}
KEYS = ("landed", "crashed", "hard", "timeout", "push_cap", "motor_sat")


def arms():
    rule = lambda obs, st: jnp.zeros((obs.shape[0], 3))
    dob = lambda obs, st: jnp.zeros((obs.shape[0], 3)).at[:, :2].set(ppo.dob_ff(0.5, obs))
    tuned = json.load(open(f"{HERE}/runs/rule_tuned.json"))["chosen"]
    out = [("standard rule", rule, False, False, 0.0, (1.0, 1e9)), ("hand-tuned rule", rule, False, False, 0.0, (tuned["scale"], tuned["patience"])),
           ("DOB 0.5, no learning", dob, True, False, 0.0, (1.0, 1e9))]
    for s in (0, 1, 2):
        pol, fly, decide, sh = ppo.load_policy(f"{HERE}/runs/stage2_c100h100g_s{s}_best.npz")
        out.append((f"learned run {s + 1}", pol, fly, decide, sh, (1.0, 1e9)))
    return out


def main():
    res = {}
    for cname, groups in CONDS.items():
        E.RAND = frozenset(groups)
        res[cname] = {}
        for name, pol, fly, decide, sh, (gs, pa) in arms():
            E.GATE_SCALE, E.PATIENCE = gs, pa
            rows = E.evaluate(pol, WINDS, n=1024, seed=10_000, fly=fly, decide=decide, shield=sh)
            res[cname][name] = [{k: r[k] for k in ("w10",) + KEYS} for r in rows]
            print(f"{cname:8s} {name:22s} landed " + " ".join(f"{100 * r['landed']:5.1f}" for r in rows)
                  + "   crashed " + " ".join(f"{100 * r['crashed']:4.1f}" for r in rows), flush=True)
        E.GATE_SCALE, E.PATIENCE = 1.0, 1e9
    E.RAND = frozenset()
    json.dump(res, open(f"{HERE}/runs/robustness_2c.json", "w"), indent=1)

    col = {"standard rule": "#8a939e", "hand-tuned rule": "#1d8475", "DOB 0.5, no learning": "#7656c0",
           "learned run 1": "#1f5ea8", "learned run 2": "#4f86c9", "learned run 3": "#8cb0de"}
    fig, ax = plt.subplots(1, len(CONDS), figsize=(4 * len(CONDS), 3.8), sharey=True)
    for a, (cname, arm) in zip(ax, res.items()):
        for name, rows in arm.items():
            a.plot(WINDS, [100 * r["landed"] for r in rows], "o-", ms=3, color=col[name], label=name)
        a.set(title=cname, xlabel="W10 (m/s)", xticks=WINDS, ylim=(0, 102)); a.grid(alpha=0.3)
    ax[0].set_ylabel("landed, % of 1,024 flights"); ax[0].legend(fontsize=7, loc="lower left")
    fig.tight_layout(); os.makedirs(f"{R}/outputs/wind", exist_ok=True); fig.savefig(f"{R}/outputs/wind/robustness_2c.png", dpi=120)

    try:
        from clearml import Task
        t = Task.init(project_name="rl-landing/wind", task_name="phase 2c: robustness without retraining", reuse_last_task_id=False,
                      auto_connect_frameworks=False, auto_connect_arg_parser=False)
        lg = t.get_logger()
        for k in ("landed", "crashed", "hard", "timeout"):
            lg.report_table(f"{k} % by condition", "all arms", iteration=0, table_plot=[["condition", "arm"] + [f"W10 {w:g}" for w in WINDS]]
                            + [[c, n] + [round(100 * r[k], 1) for r in rows] for c, arm in res.items() for n, rows in arm.items()])
        lg.report_matplotlib_figure("landed vs wind, each condition", "all arms", fig, iteration=0)
        t.close()
    except Exception as e:
        print("clearml:", e)


if __name__ == "__main__":
    main()
