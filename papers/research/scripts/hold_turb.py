"""Hold the pad position (rule's horizontal approach law, vz = 0) at height h for 90 s in land_env's wind (turbulence + gusts), n drones
per W10, wind direction random. Reports survival, drift, tilt, time at the 45 deg limit, time with collective >= 0.995."""
import sys, json
sys.path.insert(0, "/home/sai/RL_projects/rl-landing/conditions/wind")
import numpy as np, jax, jax.numpy as jnp
import x500_mjx as X, land_env as L
from xcs import control_step
N, T = 128, 90.0
def run(w10, h, cda):
    cd = jnp.array([cda, X.CDA_V])
    n = int(T / L.DT_C)
    keys = jax.random.split(jax.random.PRNGKey(3), N)
    s0 = jax.tree_util.tree_map(lambda a: jnp.broadcast_to(a, (N,) + a.shape), X.init_state((0.0, 0.0, h)))
    ws0 = jax.vmap(lambda k: L.wind_new(k, w10))(keys)
    def one(c, k):
        s, ws = c
        p, v = s["d"].qpos[:3], s["d"].qvel[:3]
        ws, wind = L.wind_step(ws, p[2], k)
        vh = L.approach(-p[:2], v[:2], 5.0); vsp = jnp.concatenate([vh, jnp.zeros(1)])
        thr, _ = X.position_control(v, vsp, jnp.zeros(3), s["vel_int"], s["vdot"])
        s = control_step(s, vsp, jnp.zeros(3), 0.0, wind, cd)
        tilt = jnp.degrees(jnp.arccos(jnp.clip(X.qrot(s["d"].qpos[3:7])[2, 2], -1, 1)))
        return (s, ws), jnp.concatenate([s["d"].qpos[:3], jnp.array([tilt, jnp.linalg.norm(thr), jnp.linalg.norm(wind[:2])])])
    def body(C, k):
        return jax.vmap(one)(C, jax.random.split(k, N))
    _, tr = jax.jit(lambda C: jax.lax.scan(body, C, jax.random.split(jax.random.PRNGKey(4), n)))((s0, ws0))
    tr = np.array(tr)   # (n, N, 7)
    xy = np.linalg.norm(tr[:, :, :2], axis=2)
    drift = xy.max(0); tilt = tr[:, :, 3]
    lost = (xy.max(0) > 30) | (tr[:, :, 2].min(0) < h - 10) | (tilt.max(0) > 68)
    return dict(w10=w10, h=h, cda=cda, lost=float(lost.mean()), drift_med=float(np.median(drift)), drift_p95=float(np.percentile(drift, 95)),
                rms_med=float(np.median(np.sqrt((xy ** 2).mean(0)))), tilt_med_max=float(np.median(tilt.max(0))), tilt_mean=float(tilt.mean()),
                at_limit=float((tilt >= 44.5).mean()), coll_sat=float((tr[:, :, 4] >= 0.995).mean()), zmin_med=float(np.median(tr[:, :, 2].min(0))),
                wind_peak_med=float(np.median(tr[:, :, 5].max(0))))
if __name__ == "__main__":
    out = []
    for cda in (0.04, 0.08):
        for h in (30.0, 6.0):
            for w in (9.0, 12.0, 15.0, 18.0, 20.0):
                r = run(w, h, cda); out.append(r)
                print(f"CdA {cda} h {h:4.0f} W10 {w:4.0f}: lost {r['lost']*100:5.1f}%  drift med {r['drift_med']:6.2f} p95 {r['drift_p95']:7.2f}  rms {r['rms_med']:5.2f}  tilt max(med) {r['tilt_med_max']:5.1f}  at 45deg {r['at_limit']*100:5.1f}%  coll>=.995 {r['coll_sat']*100:5.1f}%  peak wind(med) {r['wind_peak_med']:5.1f}", flush=True)
    json.dump(out, open("hold_turb.json", "w"))
