#!/usr/bin/env python3
"""
phase3_compare.py -- PLAN.md Phase 3 upgrades against the retrained control: 1, the DOB's estimate of the wind's push
as two more decider inputs (Sai, 6 Oct: "DOB first"); 2, a privileged critic; 3-4 (Sai, 7 Oct), the paper decider
fine-tuned in the randomized world with that critic, on a leash (queue/phase3_warm*.sh), and the same re-priced (a gentle
touchdown off the pad -10 instead of -100). Arms appear when their checkpoints exist. Every arm flies the same 1,024 test
flights (seed 10000) per wind, in the Phase 2 world (sensing, drone and gusts randomized per flight) and in the nominal one.

  standard rule, hand-tuned rule     no learning
  paper decider s0-s2                trained before the 6 Oct twin fix, no randomization
  base_rand s0-s2                    the same recipe retrained in the corrected, randomized world: the control
  dobin_rand s0-s2                   the control plus the DOB estimate as input (same frozen flyer)
  dobin_rand, wind input zeroed      Tasim & Sun's test (arXiv 2607.01528): the same policies without their wind channel
  priv_rand s0-s2                    the control plus a privileged critic
  WARM s0-s2, WARM_P s0-s2           the paper decider fine-tuned (seed s from its seed s); WARM_P re-priced

Kept only if the paired gain over the control is above zero: per flight, the landed share over the three seeds, the
difference to the control's, a bootstrap 95% interval over the 1,024 flights. The fine-tuned arms are judged against the
paper decider and the hand-tuned rule, landed AND damaging (hard + crashed; 'too fast' is the hard kind that breaks things).
    python conditions/wind/phase3_compare.py   -> runs/phase3_dobin.json, outputs/wind/phase3_dobin.png, ClearML (~20 min)
    python conditions/wind/phase3_compare.py leash   -> runs/phase3_leash.json: the warm start's leash, chosen (~3 min)
"""
import json
import os
import sys

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_deterministic_ops=true")
import jax.numpy as jnp
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import land_env as E
import ppo

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.dirname(os.path.dirname(HERE))
WINDS = [9.0, 12.0, 15.0, 18.0, 20.0]
CONDS = {"randomized": ("sense", "drone", "gust"), "nominal": ()}
WARM, WARM_P = "warm_leash03", "warmp_leash03"      # the leash chosen on validation flights (queue/phase3_warm.sh, seed 0)


def arms():
    rule = lambda obs, st: jnp.zeros((obs.shape[0], 3))
    tuned = json.load(open(f"{HERE}/runs/rule_tuned.json"))["chosen"]
    yield "standard rule", [(rule, False, False, 0.0)], (1.0, 1e9)
    yield "hand-tuned rule", [(rule, False, False, 0.0)], (tuned["scale"], tuned["patience"])
    # the fine-tuned arms at their last checkpoint: 'best' is chosen on validation LANDED, which prefers the checkpoint
    # nearest the paper decider (it lands most and breaks most); the fine-tuning's job is that trade
    for name, tag, wind_in, ck in (("paper decider", "stage2_c100h100g", True, "best"), ("retrained (control)", "base_rand", True, "best"),
                                   ("retrained + DOB input", "dobin_rand", True, "best"), ("retrained + DOB input, zeroed", "dobin_rand", False, "best"),
                                   ("retrained + privileged critic", "priv_rand", True, "best"),
                                   ("fine-tuned + privileged critic", WARM, True, "it600"), ("fine-tuned, re-priced", WARM_P, True, "it600")):
        if all(os.path.exists(f"{HERE}/runs/{tag}_s{s}_{ck}.npz") for s in (0, 1, 2)):
            yield name, [ppo.load_policy(f"{HERE}/runs/{tag}_s{s}_{ck}.npz", wind_in=wind_in) for s in (0, 1, 2)], (1.0, 1e9)


def ci(d, rng=np.random.default_rng(0), n_boot=10_000):
    """Mean of paired per-flight differences and its bootstrap 95% interval (percentage points)."""
    b = d[rng.integers(0, len(d), (n_boot, len(d)))].mean(1)
    return round(100 * float(d.mean()), 1), round(100 * float(np.percentile(b, 2.5)), 1), round(100 * float(np.percentile(b, 97.5)), 1)


def main():
    res, per = {}, {}
    for cname, groups in CONDS.items():
        E.RAND = frozenset(groups)
        res[cname], per[cname] = {}, {}
        for name, pols, (gs, pa) in arms():
            E.GATE_SCALE, E.PATIENCE = gs, pa
            runs = [E.evaluate(pol, WINDS, n=1024, seed=10_000, fly=fly, decide=dec, shield=sh, per_flight=True) for pol, fly, dec, sh in pols]
            E.GATE_SCALE, E.PATIENCE = 1.0, 1e9
            out = np.array([[r["outcome_all"] for r in rows] for rows in runs])          # seed x wind x flight
            fast = np.array([[r["fast_all"] for r in rows] for rows in runs])
            per[cname][name] = out
            pct = lambda m: [round(100 * float(m[:, i].mean()), 1) for i in range(len(WINDS))]
            res[cname][name] = dict(landed=pct(out == 1), crashed=pct(out == 2), hard=pct(out == 3), too_fast=pct(fast),
                                    gentle_off_pad=pct((out == 3) & ~fast), gave_up=pct((out == 4) | (out == 0)),
                                    damaging=pct((out == 2) | (out == 3)))
            print(f"{cname:10s} {name:30s} landed " + " ".join(f"{v:5.1f}" for v in res[cname][name]["landed"])
                  + "   damaging " + " ".join(f"{v:4.1f}" for v in res[cname][name]["damaging"])
                  + "   too fast " + " ".join(f"{v:4.1f}" for v in res[cname][name]["too_fast"]), flush=True)
    E.RAND = frozenset()
    # paired: the upgrade against the control, and Tasim & Sun's split (rule -> policy without wind -> with wind)
    for cname in CONDS:
        P = per[cname]
        land = {k: (v == 1).mean(0) for k, v in P.items()}                                # wind x flight, seed-averaged
        crash = {k: (v == 2).mean(0) for k, v in P.items()}
        met = dict(landed=land, damaging={k: ((v == 2) | (v == 3)).mean(0) for k, v in P.items()})
        res[cname]["paired"] = {
            "landed: DOB input - control": [ci(land["retrained + DOB input"][i] - land["retrained (control)"][i]) for i in range(len(WINDS))],
            "crashed: DOB input - control": [ci(crash["retrained + DOB input"][i] - crash["retrained (control)"][i]) for i in range(len(WINDS))],
            "landed: control - paper decider": [ci(land["retrained (control)"][i] - land["paper decider"][i]) for i in range(len(WINDS))],
            **({"landed: privileged critic - control": [ci(land["retrained + privileged critic"][i] - land["retrained (control)"][i]) for i in range(len(WINDS))],
                "crashed: privileged critic - control": [ci(crash["retrained + privileged critic"][i] - crash["retrained (control)"][i]) for i in range(len(WINDS))]}
               if "retrained + privileged critic" in land else {}),
            "landed, split: policy without wind - standard rule": [ci(land["retrained + DOB input, zeroed"][i] - land["standard rule"][i]) for i in range(len(WINDS))],
            "landed, split: wind channel on - off": [ci(land["retrained + DOB input"][i] - land["retrained + DOB input, zeroed"][i]) for i in range(len(WINDS))],
            **{f"{m}: {a} - {b}": [ci(met[m][a][i] - met[m][b][i]) for i in range(len(WINDS))]
               for m in met for a, b in (("fine-tuned + privileged critic", "paper decider"),
                                         ("fine-tuned + privileged critic", "hand-tuned rule"),
                                         ("fine-tuned, re-priced", "fine-tuned + privileged critic"),
                                         ("fine-tuned, re-priced", "hand-tuned rule")) if a in P and b in P}}
        for k, rows in res[cname]["paired"].items():
            print(f"{cname:10s} {k:52s} " + "  ".join(f"W{w:g} {m:+5.1f} [{lo:+5.1f},{hi:+5.1f}]" for w, (m, lo, hi) in zip(WINDS, rows)), flush=True)
    json.dump(res, open(f"{HERE}/runs/phase3_dobin.json", "w"), indent=1)

    col = {"standard rule": "#8a939e", "hand-tuned rule": "#1d8475", "paper decider": "#9db7d9", "retrained (control)": "#1f5ea8",
           "retrained + DOB input": "#c9520c", "retrained + DOB input, zeroed": "#e8a87c", "retrained + privileged critic": "#7656c0",
           "fine-tuned + privileged critic": "#b3245c", "fine-tuned, re-priced": "#e07ba5"}
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    for a, cname in zip(ax, CONDS):
        for name, r in res[cname].items():
            if name != "paired":
                a.plot(WINDS, r["landed"], "o-", ms=3, color=col[name], label=name, ls="--" if "zeroed" in name else "-")
        a.set(title=f"{cname} world", xlabel="W10 (m/s)", xticks=WINDS, ylim=(0, 102)); a.grid(alpha=0.3)
    ax[0].set_ylabel("landed, % of 1,024 flights (3 seeds averaged)"); ax[0].legend(fontsize=7, loc="lower left")
    fig.tight_layout(); os.makedirs(f"{R}/outputs/wind", exist_ok=True); fig.savefig(f"{R}/outputs/wind/phase3_dobin.png", dpi=120)

    try:
        from clearml import Task
        t = Task.init(project_name="rl-landing/wind", task_name="phase 3: every arm, paired (landed and damaging)", reuse_last_task_id=False,
                      auto_connect_frameworks=False, auto_connect_arg_parser=False)
        lg = t.get_logger()
        for cname in CONDS:
            for k in ("landed", "damaging", "too_fast", "gave_up"):
                lg.report_table(f"{k} % ({cname} world)", "all arms", iteration=0, table_plot=[["arm"] + [f"W10 {w:g}" for w in WINDS]]
                                + [[n] + r[k] for n, r in res[cname].items() if n != "paired"])
            lg.report_table(f"paired differences, points, 95% CI ({cname} world)", "paired", iteration=0,
                            table_plot=[["comparison"] + [f"W10 {w:g}" for w in WINDS]]
                            + [[k] + [f"{m:+.1f} [{lo:+.1f}, {hi:+.1f}]" for m, lo, hi in rows] for k, rows in res[cname]["paired"].items()])
        lg.report_matplotlib_figure("landed vs wind", "all arms", fig, iteration=0)
        t.close()
    except Exception as e:
        print("clearml:", e)


def pick_leash():
    """The warm start's leash (ppo.py --anchor), chosen on seed 0's VALIDATION flights (seed 20000; the test flights stay
    unseen): each sweep run's last checkpoint, randomized world, W10 9-18, by the summed reward per flight at the
    training prices (crash and hard -100), the number training maximizes. Landing alone would pick the paper decider."""
    E.RAND = frozenset(CONDS["randomized"])
    E.CRASH_R, E.HARD_R, E.SHAPE_G = -100.0, -100.0, 0.9998
    tuned = json.load(open(f"{HERE}/runs/rule_tuned.json"))["chosen"]
    cands = [("hand-tuned rule", (lambda obs, st: jnp.zeros((obs.shape[0], 3)), False, False, 0.0), (tuned["scale"], tuned["patience"])),
             ("paper decider", ppo.load_policy(f"{HERE}/runs/stage2_c100h100g_s0_best.npz"), (1.0, 1e9))]
    cands += [(f"leash {b}", ppo.load_policy(f"{HERE}/runs/warm_leash{b.replace('.', '')}_s0_it600.npz"), (1.0, 1e9)) for b in ("0", "0.1", "0.3", "1")]
    out = {}
    for name, (pol, fly, dec, sh), (gs, pa) in cands:
        E.GATE_SCALE, E.PATIENCE = gs, pa
        rs = E.evaluate(pol, WINDS[:4], n=1024, seed=20_000, fly=fly, decide=dec, shield=sh)
        E.GATE_SCALE, E.PATIENCE = 1.0, 1e9
        out[name] = {k: [round(r[k], 3) for r in rs] for k in ("landed", "crashed", "hard", "hard_fast", "timeout", "ret_mean")}
        out[name]["mean reward"] = round(float(np.mean(out[name]["ret_mean"])), 2)
        print(f"{name:16s} landed {out[name]['landed']}  damaging {[round(c + h, 3) for c, h in zip(out[name]['crashed'], out[name]['hard'])]}"
              f"  reward {out[name]['ret_mean']}  mean {out[name]['mean reward']}", flush=True)
    out["chosen"] = max((k for k in out if k.startswith("leash")), key=lambda k: out[k]["mean reward"])
    print("chosen:", out["chosen"])
    json.dump(out, open(f"{HERE}/runs/phase3_leash.json", "w"), indent=1)


if __name__ == "__main__":
    pick_leash() if sys.argv[1:] == ["leash"] else main()
