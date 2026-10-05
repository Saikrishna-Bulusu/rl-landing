# independent re-implementation of the report's DOB spec (written from the report text, not copied from alg/dob.py)
import sys, os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"
sys.path.insert(0, "/home/sai/RL_projects/rl-landing/conditions/wind")
import jax.numpy as jnp, land_env as E
def dob(k):
    def pol(obs, st):
        H = st["hist"]                                   # (n,4,5)
        vel = H[..., 0:2] * 3.0
        acc = (vel[:, 3] - vel[:, 0]) / (3 * 0.04)
        zx = 0.5 * (H[:, 2, 3] + H[:, 3, 3]) / 5.0; zy = 0.5 * (H[:, 2, 4] + H[:, 3, 4]) / 5.0
        zh = jnp.stack([zx, zy], -1)
        n2 = jnp.sum(zh * zh, -1, keepdims=True)
        thrust_acc = 9.81 * zh / jnp.sqrt(jnp.maximum(1 - n2, 0.25))
        ff = jnp.clip(-k * (acc - thrust_acc), -3, 3) / 3.0
        return jnp.concatenate([ff, jnp.zeros((ff.shape[0], 1))], -1)
    return pol
rule = lambda obs, st: jnp.zeros((obs.shape[0], 3))
def show(tag, r): print(f"{tag}: landed {100*r['landed']:.1f} crash {100*r['crashed']:.1f} timeout {100*r['timeout']:.1f} hard {100*r['hard']:.1f}", flush=True)
show("DOB k=1.0 TEST W18", E.evaluate(dob(1.0), 18.0, n=1024, seed=10000, fly=True, decide=False))
show("DOB k=1.0 TEST W20", E.evaluate(dob(1.0), 20.0, n=1024, seed=10000, fly=True, decide=False))
show("DOB k=-0.5 VAL W18", E.evaluate(dob(-0.5), 18.0, n=1024, seed=20000, fly=True, decide=False))
show("DOB k=1.0 VAL W18", E.evaluate(dob(1.0), 18.0, n=1024, seed=20000, fly=True, decide=False))
show("RULE TEST W18", E.evaluate(rule, 18.0, n=1024, seed=10000, fly=False, decide=False))
