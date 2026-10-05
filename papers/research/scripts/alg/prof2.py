import sys, time, os
sys.path.insert(0, "/home/sai/RL_projects/rl-landing/conditions/wind")
import jax, jax.numpy as jnp, numpy as np
import land_env as E, ppo as P
N, T = 4096, 64
rollout, gae, update, vreset, vobs = P.make_fns(N, T, True, False)   # fly only, like stage 1
key = jax.random.PRNGKey(0); key, k1, k2, k3 = jax.random.split(key, 4)
p = P.init_params(k1); opt = P.adam_init(p)
w10 = jax.vmap(P.draw_w10, in_axes=(0, None))(jax.random.split(k2, N), 15.0)
st = vreset(jax.random.split(k3, N), w10); obs = vobs(st, jax.random.split(k1, N))
rms = dict(mean=jnp.zeros(E.OBS_DIM), var=jnp.ones(E.OBS_DIM), count=1e-4)
ts = []
for it in range(1, 41):
    t0 = time.time(); key, kr, ku = jax.random.split(key, 3)
    st, obs, tr, lv = rollout(p, rms, st, obs, kr, jnp.float32(15.0), 0.0)
    adv, ret = gae(tr, lv)
    p, opt, info = update(p, opt, rms, tr, adv, ret, ku, jnp.float32(3e-4))
    rms = P.rms_update(rms, tr["obs"].reshape(-1, E.OBS_DIM))
    done = np.array(tr["done"]).ravel(); oc = np.array(tr["outcome"]).ravel()[done]
    _ = [float(x) for x in info]
    ts.append(time.time() - t0)
ts = np.array(ts[10:]); print(f"fly-only full loop (host reads included), 4096x64: {ts.mean():.3f} s/round -> {N*T/ts.mean():.0f} decisions/s; episodes done per round {done.sum()}")
