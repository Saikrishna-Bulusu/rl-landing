"""control_step with the 10 physics steps unrolled: the lax.scan form trips an XLA GPU (Triton) compile error when the
setpoint is batched and the initial state is not (found 4 Oct, jax on this machine). Same physics."""
import sys; sys.path.insert(0, "/home/sai/RL_projects/rl-landing/conditions/wind")
import jax, jax.numpy as jnp, x500_mjx as X
def control_step(s, vel_sp, acc_ff, yaw_sp, wind, cda=jnp.array([X.CDA_H, X.CDA_V])):
    s = dict(s, prm=s["prm"].at[..., :2].set(cda))                  # the drag areas live in the drone's state (6 Oct)
    return jax.lax.scan(lambda c, _: (X.physics_step(c, vel_sp, acc_ff, yaw_sp, wind), None), s, None, length=X.N_SUB, unroll=True)[0]
