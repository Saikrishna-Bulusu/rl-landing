"""Is a decider fragile to height and vertical-speed estimate errors, which MJX's estimate model leaves out (exact height,
5 cm/s velocity noise) but PX4's estimator in Gazebo has? Same 1024 test flights (seed 10000), MJX's estimates() with
extra height noise (sd, per decision) and vertical-speed noise added.  python estimate_check.py CKPT --w 9"""
import argparse, json
import numpy as np
import jax, jax.numpy as jnp
import land_env as E, ppo as P
from latency_check import replay, flips

BASE = E.estimates
def make(h_sd, vz_sd):
    def est(st, key):
        p_est, v_est = BASE(st, key)
        k1, k2 = jax.random.split(jax.random.fold_in(key, 7))
        return p_est.at[2].add(h_sd * jax.random.normal(k1)), v_est.at[2].add(vz_sd * jax.random.normal(k2))
    return est

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("ckpt"); ap.add_argument("--w", type=float, nargs="+", default=[9.0]); ap.add_argument("--n", type=int, default=1024)
    a = ap.parse_args()
    pol, fly, decide, sh = P.load_policy(a.ckpt); hp = P.hp_of(a.ckpt)
    for h_sd, vz_sd in ((0.0, 0.0), (0.1, 0.0), (0.3, 0.0), (0.0, 0.2), (0.1, 0.2), (0.3, 0.4)):
        E.estimates = make(h_sd, vz_sd)
        for w in a.w:
            out, mode, z, vz, done = replay(pol, w, a.n, 10_000, fly, decide, hp, 0)
            fr = np.array([flips(mode[:, i], z[:, i], done[:, i]) for i in range(a.n)])
            print(json.dumps(dict(ckpt=a.ckpt.split("/")[-1].replace("_best.npz", ""), w10=w, h_sd=h_sd, vz_sd=vz_sd, landed=round(float((out == 1).mean()), 3),
                                  hard=round(float((out == 3).mean()), 3), crashed=round(float((out == 2).mean()), 3),
                                  flips_p90=round(float(np.percentile(fr, 90)), 2))), flush=True)
