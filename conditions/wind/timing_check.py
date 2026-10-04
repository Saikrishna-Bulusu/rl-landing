"""Does a decider chatter when Gazebo's two timing gaps come TOGETHER? Gazebo's loop decided every ~0.068 s (not 0.04 s)
and MAVLink put PX4 ~60 ms behind: rate_check and latency_check tried each alone. Same 1024 test flights (seed 10000),
the command held for `hold` steps (mix: 1 or 2 at random) and delivered `delay` steps late. Reports landed/hard and,
below 6 m, the vertical-speed swing (std of vz) and the share of decisions that climb (Gazebo, 29 Sept: 0.27-0.31 and
50-57% on yards 4, 6, 9 at 9 m/s; latency_check's flips miss that cycle, which passes through the rule's own choice).
    python timing_check.py runs/stage2_c100h100_s1_best.npz --w 9"""
import argparse, json
import numpy as np
import jax, jax.numpy as jnp
import land_env as E, ppo as P

def replay(pol, w10, n, seed, fly, decide, hp, hold, delay):
    st = jax.vmap(lambda k: E.reset(k, w10))(jax.random.split(jax.random.PRNGKey(seed), n))
    def one(carry, k):
        st, act, age, buf = carry
        ks = jax.random.split(k, n * 2 + 1)
        kd, ks = ks[0], ks[1:].reshape(n, 2, 2)
        obs = jax.vmap(E.obs_of)(st, ks[:, 0])
        due = age >= (jax.random.randint(kd, (n,), 1, 3) if hold == "mix" else hold)
        act = jnp.where(due[:, None], pol(obs, st), act); age = jnp.where(due, 1, age + 1)
        buf = jnp.concatenate([buf[1:], act[None]], 0)
        new, r, done = jax.vmap(lambda s, a, kk: E.step(s, a, kk, fly, decide, 0.0, hp))(st, buf[0], ks[:, 1])
        keep = lambda a, b: jnp.where(st["done"].reshape((-1,) + (1,) * (a.ndim - 1)), a, b)
        qv = new["sim"]["d"].qvel
        return (jax.tree_util.tree_map(keep, st, new), act, age, buf), (buf[0][:, 2], new["sim"]["d"].qpos[:, 2], qv[:, 2], st["done"], new["vz_sent"])
    ns = int(E.T_MAX / E.DT_C) + 1
    init = (st, jnp.zeros((n, 3)), jnp.full((n,), 99), jnp.zeros((delay + 1, n, 3)))
    (st, *_), (mode, z, vz, done, sent) = jax.jit(lambda c, ks: jax.lax.scan(one, c, ks))(init, jax.random.split(jax.random.PRNGKey(seed + 1), ns))
    return np.array(st["outcome"]), np.array(mode), np.array(z), np.array(vz), np.array(done), np.array(sent)

def reversals(sent_col, done_col):
    """Climb<->descend reversals of the sent vertical command per second of flight (land_env's flip)."""
    live = ~done_col
    f = ((sent_col[:-1] > 0.5) & (sent_col[1:] < -0.3)) | ((sent_col[:-1] < -0.3) & (sent_col[1:] > 0.5))
    return f[live[1:]].sum() / max(live.sum() * E.DT_C, 1e-6)

def low(x_col, z_col, done_col, f):
    m = (~done_col) & (z_col > 1) & (z_col < 6)
    return f(x_col[m]) if m.sum() > 10 else np.nan

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("ckpt", nargs="+"); ap.add_argument("--w", type=float, nargs="+", default=[9.0])
    ap.add_argument("--n", type=int, default=1024)
    ap.add_argument("--kt", type=float, nargs="+", default=[1.0], help="rotor thrust scale (x500_mjx.KT; PX4's hover estimate follows)")
    ap.add_argument("--timing", default="1:0,mix:1,mix:2,2:1,2:2", help="hold:delay pairs")
    a = ap.parse_args()
    import x500_mjx as X, math
    KT0 = X.KT
    for kt in a.kt:
      X.KT = KT0 * kt; X.W_HOVER = math.sqrt(X.MASS * X.G / 4 / X.KT); X.U_HOVER = (X.W_HOVER - X.W_MIN) / (X.W_MAX - X.W_MIN)
      for ck in a.ckpt:
        pol, fly, decide, sh = P.load_policy(ck); hp = P.hp_of(ck)
        for w in a.w:
            for hd in a.timing.split(","):
                hold, delay = hd.split(":"); hold = hold if hold == "mix" else int(hold); delay = int(delay)
                out, mode, z, vz, done, sent = replay(pol, w, a.n, 10_000, fly, decide, hp, hold, delay)
                cl = np.array([low(mode[:, i], z[:, i], done[:, i], lambda x: float((x == 4).mean())) for i in range(a.n)])
                sw = np.array([low(vz[:, i], z[:, i], done[:, i], lambda x: float(x.std())) for i in range(a.n)])
                rv = np.array([reversals(sent[:, i], done[:, i]) for i in range(a.n)])
                print(json.dumps(dict(ckpt=ck.split("/")[-1].replace("_best.npz", ""), kt=kt, w10=w, hold=hold, delay_ms=40 * delay,
                                      landed=round(float((out == 1).mean()), 3), hard=round(float((out == 3).mean()), 3),
                                      crashed=round(float((out == 2).mean()), 3), climb_share_median=round(float(np.nanmedian(cl)), 2),
                                      vz_swing_median=round(float(np.nanmedian(sw)), 2),
                                      reversals_per_s_median=round(float(np.median(rv)), 2), reversals_per_s_p90=round(float(np.percentile(rv, 90)), 2), yards_4_6_9=out[[4, 6, 9]].tolist())), flush=True)
