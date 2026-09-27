#!/usr/bin/env python3
"""
export_policy.py -- a trained PPO policy as plain numpy weights.

The flight process cannot import stable-baselines3. The gz-transport and
pymavlink bindings live in the system python; torch and SB3 live in a conda
env, and neither interpreter has both. Rather than force them together, take
the only part that flies -- the weights -- and leave the training stack behind.

The policy is an MLP: 6 inputs, two tanh layers, a linear head, and for PPO a
deterministic action is simply the distribution mean. That is four matrix
multiplies. It needs no framework, it starts instantly, and it is what would
have to happen anyway to run this on a Pixhawk.
"""

import argparse
import pickle

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = a.out or a.model.replace(".zip", "_weights.npz")

    from stable_baselines3 import PPO
    m = PPO.load(a.model, device="cpu")
    sd = {k: v.detach().cpu().numpy() for k, v in m.policy.state_dict().items()}

    layers = []
    for i in (0, 2):                     # Linear, Tanh, Linear, Tanh
        layers.append((sd[f"mlp_extractor.policy_net.{i}.weight"],
                       sd[f"mlp_extractor.policy_net.{i}.bias"]))
    layers.append((sd["action_net.weight"], sd["action_net.bias"]))

    d = {}
    for i, (w, b) in enumerate(layers):
        d[f"w{i}"] = w.astype(np.float32)
        d[f"b{i}"] = b.astype(np.float32)
    d["n_layers"] = np.array(len(layers))
    d["low"] = m.action_space.low.astype(np.float32)
    d["high"] = m.action_space.high.astype(np.float32)
    np.savez(out, **d)

    # Prove the exported weights reproduce SB3 exactly before trusting them.
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(200):
        o = rng.normal(0, 1, m.observation_space.shape).astype(np.float32)
        ref, _ = m.predict(o, deterministic=True)
        got = forward(d, o)
        worst = max(worst, float(np.abs(ref - got).max()))
    print(f"  {out}")
    print(f"  shapes: {[tuple(w.shape) for w, _ in layers]}")
    print(f"  max |numpy - SB3| over 200 random observations: {worst:.2e}")
    assert worst < 1e-5, "exported weights do not reproduce the policy"
    print("  ok")


def forward(d, obs):
    x = np.asarray(obs, np.float32)
    n = int(d["n_layers"])
    for i in range(n - 1):
        x = np.tanh(d[f"w{i}"] @ x + d[f"b{i}"])
    x = d[f"w{n-1}"] @ x + d[f"b{n-1}"]
    return np.clip(x, d["low"], d["high"])


if __name__ == "__main__":
    main()
