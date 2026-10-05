"""land_env's wind (log profile z0 0.3 + MIL-F-8785C turbulence + 1-cosine gusts) at a fixed height: peak wind a drone sees in 90 s."""
import sys, json, math
sys.path.insert(0, "/home/sai/RL_projects/rl-landing/conditions/wind")
import numpy as np, jax, jax.numpy as jnp
import land_env as L
N, T = 512, 90.0
def stats(w10, h):
    n = int(T / L.DT_C)
    keys = jax.random.split(jax.random.PRNGKey(7), N)
    ws0 = jax.vmap(lambda k: L.wind_new(k, w10))(keys)
    def body(ws, k):
        ks = jax.random.split(k, N)
        ws, wind = jax.vmap(lambda w, kk: L.wind_step(w, h, kk))(ws, ks)
        return ws, jnp.stack([jnp.linalg.norm(wind[:, :2], axis=1), jnp.linalg.norm(wind, axis=1)], 1)
    _, tr = jax.jit(lambda w: jax.lax.scan(body, w, jax.random.split(jax.random.PRNGKey(8), n)))(ws0)
    tr = np.array(tr)                      # (n, N, 2)
    pk = tr[:, :, 0].max(0)                # peak horizontal wind per run
    mean_h = tr[:, :, 0].mean()
    return dict(w10=w10, h=h, mean=float(mean_h), profile=float(w10 * math.log(max(h, 1.0) / 0.3) / math.log(10 / 0.3)),
                peak_med=float(np.median(pk)), peak_p95=float(np.percentile(pk, 95)), peak_max=float(pk.max()),
                peak3d_max=float(tr[:, :, 1].max()), std=float(tr[:, :, 0].std(0).mean()))
if __name__ == "__main__":
    out = []
    print("W10  h   profile  mean(sim) sd   peak_median  peak_p95  peak_max")
    for h in (30.0, 15.0, 6.0):
        for w in (6.0, 9.0, 12.0, 15.0, 18.0, 20.0):
            r = stats(w, h); out.append(r)
            print(f"{w:4.0f} {h:4.0f} {r['profile']:7.2f} {r['mean']:7.2f} {r['std']:5.2f} {r['peak_med']:8.2f} {r['peak_p95']:9.2f} {r['peak_max']:8.2f}")
    json.dump(out, open("wind_stats.json", "w"))
    print("ratio U(30)/W10 =", math.log(30 / 0.3) / math.log(10 / 0.3), " U(6)/W10 =", math.log(6 / 0.3) / math.log(10 / 0.3))
