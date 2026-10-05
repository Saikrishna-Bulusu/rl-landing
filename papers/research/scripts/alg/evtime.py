import sys, time, os
sys.path.insert(0, "/home/sai/RL_projects/rl-landing/conditions/wind")
import jax, jax.numpy as jnp
import land_env as E
rule = lambda obs, st: jnp.zeros((obs.shape[0], 3))
for n in (512, 1024):
    for rep in range(2):
        t0 = time.time(); r = E.evaluate(rule, 18.0, n=n, seed=20000, fly=False, decide=False); print(n, rep, round(time.time() - t0, 1), "s", r["landed"], flush=True)
