"""Does a decider depend on deciding every 0.04 s? Gazebo's loop decided every ~0.066 s (15 Hz). Same 1024 test flights,
the policy's command held for `hold` steps (sample-and-hold; hold=2 -> 12.5 Hz, 'mix' -> 1 or 2 at random, ~16.7 Hz)."""
import argparse, json
import numpy as np
import jax, jax.numpy as jnp
import land_env as E, ppo as P
from latency_check import flips

def replay_hold(pol, w10, n, seed, fly, decide, hp, hold):
    keys = jax.random.split(jax.random.PRNGKey(seed), n)
    st = jax.vmap(lambda k: E.reset(k, w10))(keys)
    def one(carry, k):
        st, act, age = carry
        ks = jax.random.split(k, n * 2 + 1)
        kd, ks = ks[0], ks[1:].reshape(n, 2, 2)
        obs = jax.vmap(E.obs_of)(st, ks[:, 0])
        new_act = pol(obs, st)
        due = age >= (jax.random.randint(kd, (n,), 1, 3) if hold == "mix" else hold)
        act = jnp.where(due[:, None], new_act, act); age = jnp.where(due, 1, age + 1)
        new, r, done = jax.vmap(lambda s, a, kk: E.step(s, a, kk, fly, decide, 0.0, hp))(st, act, ks[:, 1])
        keep = lambda a, b: jnp.where(st["done"].reshape((-1,) + (1,) * (a.ndim - 1)), a, b)
        return (jax.tree_util.tree_map(keep, st, new), act, age), (act[:, 2], new["sim"]["d"].qpos[:, 2], st["done"])
    n_steps = int(E.T_MAX / E.DT_C) + 1
    init = (st, jnp.zeros((n, 3)), jnp.full((n,), 99))
    (st, _, _), (mode, z, done) = jax.jit(lambda c, ks: jax.lax.scan(one, c, ks))(init, jax.random.split(jax.random.PRNGKey(seed + 1), n_steps))
    return np.array(st["outcome"]), np.array(mode), np.array(z), np.array(done)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("ckpt"); ap.add_argument("--w", type=float, nargs="+", default=[9.0, 18.0]); ap.add_argument("--n", type=int, default=1024)
    a = ap.parse_args()
    pol, fly, decide, sh = P.load_policy(a.ckpt); hp = P.hp_of(a.ckpt)
    for hold in (1, "mix", 2):
        for w in a.w:
            out, mode, z, done = replay_hold(pol, w, a.n, 10_000, fly, decide, hp, hold)
            fr = np.array([flips(mode[:, i], z[:, i], done[:, i]) for i in range(a.n)])
            print(json.dumps(dict(ckpt=a.ckpt.split("/")[-1].replace("_best.npz", ""), w10=w, hold=hold, landed=round(float((out == 1).mean()), 3),
                                  hard=round(float((out == 3).mean()), 3), crashed=round(float((out == 2).mean()), 3), flips_p90=round(float(np.percentile(fr, 90)), 2),
                                  first20=out[:20].tolist())), flush=True)
