# Micro-benchmark: cost of one FastTD3-sized update on this GPU (random data; timing only, not learning).
import sys, time, os
sys.path.insert(0, "/home/sai/RL_projects/rl-landing/conditions/wind")
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
import jax, jax.numpy as jnp, numpy as np
import ppo as P
OBS, ACT = 49, 2
def mk(key, sizes): return P.mlp_init(key, sizes, last=1.0)
def relu_mlp(ps, x, dt):
    for W, b in ps[:-1]: x = jax.nn.relu(x.astype(dt) @ W.astype(dt) + b.astype(dt))
    W, b = ps[-1]; return (x.astype(dt) @ W.astype(dt) + b.astype(dt)).astype(jnp.float32)
def bench(B, n_atoms, dt, hid=(1024, 512, 256), ahid=(512, 256, 128), N=2_000_000):
    k = jax.random.split(jax.random.PRNGKey(0), 6)
    q = [mk(k[0], [OBS + ACT, *hid, n_atoms]), mk(k[1], [OBS + ACT, *hid, n_atoms])]
    qt = jax.tree_util.tree_map(lambda x: x, q)
    pi = mk(k[2], [OBS, *ahid, ACT])
    buf = dict(o=jax.random.normal(k[3], (N, OBS)), a=jnp.tanh(jax.random.normal(k[4], (N, ACT))), r=jax.random.normal(k[5], (N,)), o2=jax.random.normal(k[3], (N, OBS)))
    oq, op = P.adam_init(q), P.adam_init(pi)
    def qf(qs, o, a): return jnp.stack([relu_mlp(m, jnp.concatenate([o, a], -1), dt) for m in qs])   # (2,B,atoms)
    @jax.jit
    def upd(q, qt, pi, oq, op, key, do_actor):
        idx = jax.random.randint(key, (B,), 0, N)
        b = jax.tree_util.tree_map(lambda x: x[idx], buf)
        a2 = jnp.tanh(relu_mlp(pi, b["o2"], dt))
        tq = qf(qt, b["o2"], a2).mean(-1).min(0)
        y = b["r"] + 0.99 * tq
        def lq(q):
            v = qf(q, b["o"], b["a"])
            if n_atoms > 1:                       # stand-in for the C51 cross-entropy: same matmul cost, softmax over atoms
                lp = jax.nn.log_softmax(v, -1); tgt = jax.nn.softmax(jnp.broadcast_to(y[None, :, None], v.shape), -1)
                return -(tgt * lp).sum(-1).mean()
            return ((v[..., 0] - y[None]) ** 2).mean()
        g = jax.grad(lq)(q)
        q2, oq2 = P.adam(q, g, oq, 3e-4)
        def lp_(pi): return -qf(q2[:1], b["o"], jnp.tanh(relu_mlp(pi, b["o"], dt))).mean()
        gp = jax.grad(lp_)(pi)
        pi2, op2 = P.adam(pi, gp, op, 3e-4)
        pi = jax.tree_util.tree_map(lambda x, y_: jnp.where(do_actor, y_, x), pi, pi2)
        return q2, pi, oq2, op2
    key = jax.random.PRNGKey(1)
    for i in range(3): q, pi, oq, op = upd(q, qt, pi, oq, op, jax.random.fold_in(key, i), i % 2 == 0)
    jax.block_until_ready(q)
    t0 = time.time(); n = 20
    for i in range(n): q, pi, oq, op = upd(q, qt, pi, oq, op, jax.random.fold_in(key, 10 + i), i % 2 == 0)
    jax.block_until_ready(q); return (time.time() - t0) / n * 1000
for B, atoms, dt in [(32768, 101, jnp.float32), (32768, 101, jnp.bfloat16), (32768, 1, jnp.float32), (8192, 101, jnp.float32), (8192, 101, jnp.bfloat16), (4096, 1, jnp.float32)]:
    print(f"batch {B:6d} atoms {atoms:3d} {jnp.dtype(dt).name:8s}: {bench(B, atoms, dt):7.1f} ms per update", flush=True)
