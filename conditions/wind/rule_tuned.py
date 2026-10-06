#!/usr/bin/env python3
"""
rule_tuned.py -- the no-learning baseline a reviewer will ask for (Sai, 6 Oct): the landing rule with its two obvious
hand fixes, tuned the way the learner is chosen, so any gain left over is what learning adds.

  GATE_SCALE  the gate's centring and stillness limits times this (1 = the paper's rule)
  PATIENCE    seconds the gate may hold before the rule creeps down at 0.6 m/s while within the landing radius
              (1e9 = never: the paper's rule)

Every pair of the grid is scored on the VALIDATION yards (seed 20000, 512 flights per wind) by the mean landed share
over W10 9, 12, 15, 18, the same yards and the same measure that pick the learner's checkpoint; the winner is then
flown once on the TEST yards (seed 10000, 1,024 flights per wind). The rule's own 2.5 m/s descent request is left as
it is: PX4 caps descents at 1.5 m/s, so asking for 1.5 flies the same flights (checked 6 Oct).

    python conditions/wind/rule_tuned.py          -> conditions/wind/runs/rule_tuned.json  (~10 min on the RTX 5070)
"""
import json
import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_deterministic_ops=true")   # 6 Oct: default kernels flipped ~3 of 1,024 gusty flights between runs
import jax.numpy as jnp

import land_env as E

HERE = os.path.dirname(os.path.abspath(__file__))
SCALES, PATIENCES = (1.0, 1.5, 2.0, 3.0), (1e9, 5.0, 10.0, 20.0)
VAL_W, TEST_W = (9.0, 12.0, 15.0, 18.0), (9.0, 12.0, 15.0, 18.0, 20.0)
RULE = lambda obs, st: jnp.zeros((obs.shape[0], 3))          # no push, the rule's own vertical: the rule alone


def score(scale, patience, winds, n, seed):
    E.GATE_SCALE, E.PATIENCE = scale, patience               # read when evaluate() traces the step
    return [E.evaluate(RULE, w, n=n, seed=seed, fly=False, decide=False) for w in winds]


def main():
    grid = []
    for s in SCALES:
        for p in PATIENCES:
            rs = score(s, p, VAL_W, 512, 20_000)
            grid.append(dict(scale=s, patience=p, val=[round(r["landed"], 4) for r in rs], crashed=[round(r["crashed"], 4) for r in rs]))
            print(f"scale {s:3.1f} patience {p:6.0f}: val landed " + " ".join(f"{r['landed']:.3f}" for r in rs)
                  + "  crashed " + " ".join(f"{r['crashed']:.3f}" for r in rs), flush=True)
    best = max(grid, key=lambda g: sum(g["val"]))
    test = score(best["scale"], best["patience"], TEST_W, 1024, 10_000)
    E.GATE_SCALE, E.PATIENCE = 1.0, 1e9
    out = dict(chosen=dict(scale=best["scale"], patience=best["patience"]), grid=grid,
               test=[{k: r[k] for k in ("w10", "landed", "crashed", "hard", "timeout")} for r in test])
    json.dump(out, open(f"{HERE}/runs/rule_tuned.json", "w"), indent=1)
    print("chosen on validation:", out["chosen"], "| test landed:", " ".join(f"W{r['w10']:.0f} {r['landed']:.3f}" for r in test))


if __name__ == "__main__":
    main()
