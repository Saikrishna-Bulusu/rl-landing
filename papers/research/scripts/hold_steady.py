"""Closed loop in MJX: hold (0,0,30) in steady wind U for 30 s, no turbulence. Batch over U with vmap.
Variants: vel0 = velocity setpoint 0 (pure PX4 velocity hold); rule = land_env's horizontal approach law to the origin.
CdA variants via X.control_step(cda=...)."""
import sys, math, json, argparse
sys.path.insert(0, "/home/sai/RL_projects/rl-landing/conditions/wind")
import numpy as np, jax, jax.numpy as jnp
import x500_mjx as X
import land_env as L
from xcs import control_step

def run(Us, cda, mode, T=30.0, ramp=0.0, z0=30.0):
    cd = jnp.array([cda, X.CDA_V if cda else 0.0])
    n = int(T / X.DT / X.N_SUB)
    Ua = jnp.asarray(Us, jnp.float32)
    s0 = jax.tree_util.tree_map(lambda a: jnp.broadcast_to(a, (len(Us),) + a.shape), X.init_state((0.0, 0.0, z0)))
    def one(s, U, i):
        t = i * X.DT * X.N_SUB
        wind = jnp.array([U * jnp.minimum(1.0, t / ramp) if ramp > 0 else U, 0.0, 0.0])
        p, v = s["d"].qpos[:3], s["d"].qvel[:3]
        if mode == "vel0":
            vsp = jnp.zeros(3)
        else:
            vh = L.approach(-p[:2], v[:2], 5.0); vsp = jnp.concatenate([vh, jnp.zeros(1)])
        thr, _ = X.position_control(v, vsp, jnp.zeros(3), s["vel_int"], s["vdot"])
        s = control_step(s, vsp, jnp.zeros(3), 0.0, wind, cd)
        q = s["d"].qpos[3:7]
        tilt = jnp.degrees(jnp.arccos(jnp.clip(X.qrot(q)[2, 2], -1, 1)))
        return s, jnp.concatenate([s["d"].qpos[:3], s["d"].qvel[:3], jnp.array([tilt, jnp.linalg.norm(thr), s["w"].max(), s["w"].mean()])])
    def body(S, i):
        return jax.vmap(one, in_axes=(0, 0, None))(S, Ua, i)
    tr = jax.jit(lambda S: jax.lax.scan(body, S, jnp.arange(n))[1])(s0)
    return np.array(tr).transpose(1, 0, 2)

def summarise(Us, tr):
    out = []
    for U, t in zip(Us, tr):
        xy = np.linalg.norm(t[:, :2], axis=1)
        sp = np.linalg.norm(t[:, 3:5], axis=1)
        k = len(t) // 6        # last 5 s
        out.append(dict(U=float(U), maxdrift=float(xy.max()), finaldrift=float(xy[-1]), speed_end=float(sp[-k:].mean()),
                        dz=float(t[-1, 2] - 30.0), minz=float(t[:, 2].min()), tilt_max=float(t[:, 6].max()), tilt_end=float(t[-k:, 6].mean()),
                        coll_max=float(t[:, 7].max()), coll_end=float(t[-k:, 7].mean()),
                        sat_frac=float((t[:, 7] >= 0.995).mean()), wtop_max=float(t[:, 8].max()), wmean_end=float(t[-k:, 9].mean()),
                        top_sat_frac=float((t[:, 8] >= 999).mean())))
    return out

def holds(r, tol=1.0):
    return r["maxdrift"] <= tol and r["tilt_max"] < 68 and r["minz"] > 27.0

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--T", type=float, default=30.0); a = ap.parse_args()
    Us = np.arange(0, 40.01, 0.5)
    res = {}
    for cda in (0.0, 0.04, 0.08, 0.12):
        for mode in ("vel0", "rule"):
            for ramp in (0.0,):
                tr = run(Us, cda, mode, a.T, ramp)
                S = summarise(Us, tr)
                res[f"cda{cda}_{mode}_ramp{ramp}"] = S
                okU = [r["U"] for r in S]
                # largest U such that every U' <= U holds within 1 m (and first failure)
                last = None
                for r in S:
                    if holds(r): last = r["U"]
                    else: break
                fail = next((r["U"] for r in S if not holds(r)), None)
                print(f"CdA {cda} {mode:5s}: holds <=1 m up to U={last}  first fail U={fail}", flush=True)
    json.dump(res, open("hold_steady.json", "w"))
