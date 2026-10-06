#!/usr/bin/env python3
"""
compare_baselines.py -- what learning adds in wind (Sai, 6 Oct): every arm on the SAME test flights (seed 10000,
1,024 per wind), scored with today's code, so every difference is paired.

  rule            the paper's landing rule alone
  tuned rule      rule_tuned.py's winner (gate scale and patience chosen on the validation yards)
  DOB             the disturbance observer pushes, the rule decides: no learning at all
  paper decider   stage2_c100h100g s0-s2: the PPO decider on the learned flyer fly_s0 (the ACRA draft's)
  DOB + decider   stage2_dob_c100h100g s0-s2: the same PPO recipe, the observer as the frozen flyer

    python conditions/wind/compare_baselines.py   -> runs/baselines_6oct.json, outputs/wind/baselines_6oct.png, ClearML
"""
import json
import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_deterministic_ops=true")   # 6 Oct: default kernels flipped ~3 of 1,024 gusty flights between runs
import jax.numpy as jnp
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

import land_env as E
import ppo

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.dirname(os.path.dirname(HERE))
WINDS = (9.0, 12.0, 15.0, 18.0, 20.0)
KEYS = ("landed", "crashed", "hard", "timeout")


def ladder(pol, fly, decide, shield=0.0):
    return [{k: r[k] for k in ("w10",) + KEYS} for r in
            (E.evaluate(pol, w, n=1024, seed=10_000, fly=fly, decide=decide, shield=shield) for w in WINDS)]


def main():
    rule = lambda obs, st: jnp.zeros((obs.shape[0], 3))
    dob = lambda obs, st: jnp.zeros((obs.shape[0], 3)).at[:, :2].set(ppo.dob_ff(1.0, obs))
    arms = {"rule": ladder(rule, False, False)}
    tuned = json.load(open(f"{HERE}/runs/rule_tuned.json"))["chosen"]
    E.GATE_SCALE, E.PATIENCE = tuned["scale"], tuned["patience"]
    arms[f"tuned rule (gate x{tuned['scale']:g}, patience {tuned['patience']:g} s)"] = ladder(rule, False, False)
    E.GATE_SCALE, E.PATIENCE = 1.0, 1e9
    arms["DOB (no learning)"] = ladder(dob, True, False)
    for tag, name in (("stage2_c100h100g", "paper decider"), ("stage2_dob_c100h100g", "DOB + decider")):
        for s in (0, 1, 2):
            pol, fly, decide, sh = ppo.load_policy(f"{HERE}/runs/{tag}_s{s}_best.npz")
            arms[f"{name} s{s}"] = ladder(pol, fly, decide, sh)
            print(name, s, " ".join(f"W{r['w10']:.0f} {r['landed']:.3f}" for r in arms[f"{name} s{s}"]), flush=True)
    json.dump(arms, open(f"{HERE}/runs/baselines_6oct.json", "w"), indent=1)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.3))
    style = {"rule": ("#99a3a4", "o-"), "tuned": ("#16a085", "s-"), "DOB (": ("#8e44ad", "^-"), "paper": ("#2e86c1", "-"), "DOB +": ("#d35400", "-")}
    for name, rows in arms.items():
        c, m = next(v for k, v in style.items() if name.startswith(k))
        seed = name[-2:] in ("s0", "s1", "s2")
        lab = None if seed and not name.endswith("s0") else (name[:-3] + " (3 seeds)" if seed else name)
        for i, k in enumerate(("landed", "crashed")):
            ax[i].plot(WINDS, [100 * r[k] for r in rows], m if not seed else "o-", color=c, ms=4, lw=1.5 if not seed else 1.0,
                       alpha=1.0 if not seed else 0.8, label=lab if i == 0 else None)
    ax[0].set(title="landed (%), 1,024 test flights per wind, the same flights for every arm", xlabel="W10 (m/s)", ylim=(0, 102))
    ax[1].set(title="crashed (%)", xlabel="W10 (m/s)")
    ax[0].legend(fontsize=8, loc="lower left")
    for a in ax: a.grid(alpha=0.3); a.set_xticks(WINDS)
    fig.tight_layout()
    os.makedirs(f"{R}/outputs/wind", exist_ok=True)
    fig.savefig(f"{R}/outputs/wind/baselines_6oct.png", dpi=120)

    try:
        from clearml import Task
        t = Task.init(project_name="rl-landing/wind", task_name="baselines 6 Oct: what learning adds", reuse_last_task_id=False,
                      auto_connect_frameworks=False, auto_connect_arg_parser=False)
        lg = t.get_logger()
        for k in KEYS:
            lg.report_table(f"{k} (%) on the test flights", "all arms", iteration=0,
                            table_plot=[["arm"] + [f"W10 {w:g}" for w in WINDS]] + [[n] + [round(100 * r[k], 1) for r in rows] for n, rows in arms.items()])
        lg.report_matplotlib_figure("landed and crashed vs wind", "all arms", fig, iteration=0)
        t.close()
    except Exception as e:
        print("clearml:", e)


if __name__ == "__main__":
    main()
