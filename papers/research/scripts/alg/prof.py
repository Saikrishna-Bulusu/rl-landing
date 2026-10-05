import sys, time, os
sys.path.insert(0, "/home/sai/RL_projects/rl-landing/conditions/wind")
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
import jax, jax.numpy as jnp, numpy as np
import land_env as E, ppo as P
N, T = 4096, 64
rollout, gae, update, vreset, vobs = P.make_fns(N, T, True, True)
key = jax.random.PRNGKey(0); key, k1, k2, k3 = jax.random.split(key, 4)
p = P.init_params(k1); opt = P.adam_init(p)
w10 = jax.vmap(P.draw_w10, in_axes=(0, None))(jax.random.split(k2, N), 15.0)
st = vreset(jax.random.split(k3, N), w10)
obs = vobs(st, jax.random.split(k1, N))
rms = dict(mean=jnp.zeros(E.OBS_DIM), var=jnp.ones(E.OBS_DIM), count=1e-4)
def blk(x): jax.tree_util.tree_map(lambda a: a.block_until_ready(), x)
tr_t = up_t = ga_t = 0.0
for i in range(8):
    key, kr, ku = jax.random.split(key, 3)
    t0 = time.time(); st, obs, tr, lv = rollout(p, rms, st, obs, kr, jnp.float32(15.0), 0.0); blk((st, tr)); t1 = time.time()
    adv, ret = gae(tr, lv); blk(adv); t2 = time.time()
    p, opt, info = update(p, opt, rms, tr, adv, ret, ku, jnp.float32(3e-4)); blk(p); t3 = time.time()
    if i >= 2: tr_t += t1 - t0; ga_t += t2 - t1; up_t += t3 - t2
n = 6
print(f"per round (4096x64): rollout {tr_t/n:.3f}s  gae {ga_t/n:.3f}s  update {up_t/n:.3f}s  -> decisions/s {N*T/((tr_t+ga_t+up_t)/n):.0f}")
print(f"rollout-only decisions/s {N*T/(tr_t/n):.0f}")
