import sys, jax, jax.numpy as jnp
sys.path.insert(0, "/home/sai/RL_projects/rl-landing/conditions/wind")
import land_env as E
G = 9.81
def make(k, att_noise=0.0, tilt_bias=0.0):
    def pol(obs, st):
        h = st["hist"]
        v = h[:, :, :2] * 3.0
        a_meas = (v[:, -1] - v[:, 0]) / (3 * E.DT_C)
        key = jax.random.fold_in(jax.random.PRNGKey(7), (st["t"][0] / E.DT_C).astype(jnp.int32))
        zh = h[:, -2:, 3:5].mean(1) / 5.0 + att_noise * jax.random.normal(key, (h.shape[0], 2)) + tilt_bias
        zz = jnp.sqrt(jnp.maximum(1.0 - (zh ** 2).sum(-1), 0.25))
        d = a_meas - G * zh / zz[:, None]
        ff = jnp.clip(-k * d / E.A_FF, -1, 1)
        return jnp.concatenate([ff, jnp.zeros((ff.shape[0], 1))], -1)
    return pol
for name, kw in [("k=-0.5 (wrong sign)", dict(k=-0.5)), ("k=1.0 clean", dict(k=1.0)), ("k=1.0 attitude noise 0.6 deg", dict(k=1.0, att_noise=0.01)),
                 ("k=1.0 attitude noise 1.7 deg", dict(k=1.0, att_noise=0.03)), ("k=1.0 attitude bias 0.6 deg", dict(k=1.0, tilt_bias=0.01))]:
    r = E.evaluate(make(**kw), 18.0, n=1024, seed=20000, fly=True, decide=False)
    print(f"{name:34s} W18 val: landed {r['landed']:.3f} crash {r['crashed']:.3f} timeout {r['timeout']:.3f}", flush=True)
