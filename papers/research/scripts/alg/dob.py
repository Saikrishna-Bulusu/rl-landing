# Quick prototype of the non-learning baseline: a disturbance-observer feed-forward (DOB-FF) on the same channel as the learned flyer.
# Uses ONLY what the learner sees: the last 4 (velocity estimate, lean) readings in st["hist"]. Stateless: slope of velocity over 4 readings.
import sys, os, time, json
sys.path.insert(0, "/home/sai/RL_projects/rl-landing/conditions/wind")
import jax, jax.numpy as jnp
import land_env as E
G = 9.81
def make(k, clip=1.0):
    def pol(obs, st):
        h = st["hist"]                                    # (n,4,5): v_est/3 (x,y,z), lean*5 (x,y); newest last
        v = h[:, :, :2] * 3.0
        a_meas = (v[:, -1] - v[:, 0]) / (3 * E.DT_C)      # m/s^2 over the last 3 steps (noise ~0.6 m/s^2)
        zh = h[:, -2:, 3:5].mean(1) / 5.0                 # horizontal part of body z
        zz = jnp.sqrt(jnp.maximum(1.0 - (zh ** 2).sum(-1), 0.25))
        a_thrust = G * zh / zz[:, None]                   # horizontal acceleration the thrust vector gives
        d = a_meas - a_thrust                             # what the wind (and drag) adds
        ff = jnp.clip(-k * d / E.A_FF, -clip, clip)
        return jnp.concatenate([ff, jnp.zeros((ff.shape[0], 1))], -1)
    return pol
if __name__ == "__main__":
    for k in (0.0, 0.5, 1.0):
        row = []
        for w in (15.0, 18.0):
            t0 = time.time(); r = E.evaluate(make(k), w, n=1024, seed=20000, fly=True, decide=False)
            row.append((w, round(r["landed"], 3), round(r["crashed"], 3), round(r["timeout"], 3), round(time.time() - t0, 1)))
        print("k", k, row, flush=True)
