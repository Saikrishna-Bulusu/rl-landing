"""Does a decider depend on MJX's instant command path? Replays the test ladder (seed 10000: flight i = Gazebo's yard i)
with the policy's commands delayed by k decisions (0.04 s each; MAVLink put Gazebo ~60 ms behind MJX in the step check),
and records, per flight, how often the vertical command flips between climb and descend below 10 m and the touchdown sink.
    python latency_check.py runs/stage2_c100h100_s1_best.npz --w 9 18 --delay 0 1 2"""
import argparse, json
import numpy as np
import jax, jax.numpy as jnp
import land_env as E, ppo as P

def replay(pol, w10, n, seed, fly, decide, hp, delay):
    keys = jax.random.split(jax.random.PRNGKey(seed), n)
    st = jax.vmap(lambda k: E.reset(k, w10))(keys)
    buf = jnp.zeros((delay + 1, n, 3))
    def one(carry, k):
        st, buf = carry
        ks = jax.random.split(k, n * 2).reshape(n, 2, 2)
        obs = jax.vmap(E.obs_of)(st, ks[:, 0])
        buf = jnp.concatenate([buf[1:], pol(obs, st)[None]], 0)          # the command sent now arrives `delay` steps later
        act = buf[0]
        new, r, done = jax.vmap(lambda s, a, kk: E.step(s, a, kk, fly, decide, 0.0, hp))(st, act, ks[:, 1])
        keep = lambda a, b: jnp.where(st["done"].reshape((-1,) + (1,) * (a.ndim - 1)), a, b)
        z = new["sim"]["d"].qpos[:, 2]; vz = new["sim"]["d"].qvel[:, 2]
        return (jax.tree_util.tree_map(keep, st, new), buf), (act[:, 2], z, vz, st["done"])
    n_steps = int(E.T_MAX / E.DT_C) + 1
    (st, _), (mode, z, vz, done) = jax.jit(lambda s, b, ks: jax.lax.scan(one, (s, b), ks))(st, buf, jax.random.split(jax.random.PRNGKey(seed + 1), n_steps))
    return np.array(st["outcome"]), np.array(mode), np.array(z), np.array(vz), np.array(done)

def flips(mode_col, z_col, done_col):
    live = (~done_col) & (z_col < 10.0)
    vz_cmd = np.where(mode_col > 0, np.array(E.MODES)[mode_col.astype(int)], np.nan)     # nan = the rule's own vertical
    up, dn = vz_cmd > 0.5, vz_cmd < -0.3
    f = (up[1:] & dn[:-1]) | (dn[1:] & up[:-1])
    return f[live[1:]].sum() / max(live.sum() * E.DT_C, 1e-6)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("ckpt"); ap.add_argument("--w", type=float, nargs="+", default=[9.0, 18.0])
    ap.add_argument("--delay", type=int, nargs="+", default=[0, 1, 2]); ap.add_argument("--n", type=int, default=1024)
    a = ap.parse_args()
    pol, fly, decide, sh = P.load_policy(a.ckpt); hp = P.hp_of(a.ckpt)
    for w in a.w:
        for d in a.delay:
            out, mode, z, vz, done = replay(pol, w, a.n, 10_000, fly, decide, hp, d)
            fr = np.array([flips(mode[:, i], z[:, i], done[:, i]) for i in range(a.n)])
            print(json.dumps(dict(ckpt=a.ckpt.split("/")[-1], w10=w, delay_ms=40 * d, landed=round(float((out == 1).mean()), 3),
                                  hard=round(float((out == 3).mean()), 3), crashed=round(float((out == 2).mean()), 3),
                                  flips_per_s_median=round(float(np.median(fr)), 2), flips_per_s_p90=round(float(np.percentile(fr, 90)), 2),
                                  first20=out[:20].tolist())), flush=True)
