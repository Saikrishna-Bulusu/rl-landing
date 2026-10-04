#!/usr/bin/env python3
"""
clone_teacher.py -- can the policy class represent the answer at all?

PPO reached 72% where an oracle reaches 99.5%. Two very different explanations:

  (a) the answer IS representable from what the drone can observe, and PPO's
      exploration simply never found it -- a fixable optimisation problem;
  (b) the observation does not contain what the oracle uses, so no amount of
      reinforcement learning will ever close the gap.

Supervised imitation separates them. Show a network the teacher's action on
states it actually visits, and fit it directly. No exploration, no credit
assignment, no reward shaping -- if the gap survives that, it is (b).

THE CONTROL MATTERS MORE THAN THE TREATMENT

Cloning the ORACLE alone proves nothing: if it fails we cannot tell a broken
imitation setup from a real information gap. So the rule is cloned too. The
rule uses ONLY the observation, so a faithful clone of it must reach the rule's
own score. If the rule clones and the oracle does not, the difference is
exactly the privileged information -- which is the answer we are after.

DAGGER, NOT PLAIN BEHAVIOUR CLONING

A network trained only on the teacher's own trajectories drifts into states the
teacher never visited and has nothing to say about. Rolling the STUDENT out and
labelling the states IT reaches removes that confound, so a failure here is
about information rather than distribution shift.
"""

import argparse
import numpy as np
import torch
import torch.nn as nn

from approach_env_v5 import ApproachEnvV5
import rule_agent as R


class E(ApproachEnvV5):
    N_OBST = (8, 10); OBST_H = (7.2, 12.0); PAD_CLEAR = 0.5; OBST_R = (2.5, 5.0)


def mlp(n_in, n_out=3):
    return nn.Sequential(nn.Linear(n_in, 128), nn.Tanh(),
                         nn.Linear(128, 64), nn.Tanh(), nn.Linear(64, n_out))


def collect(env, teacher, student, n_ep, seed0, beta):
    """Roll out; act with the student (prob 1-beta) but always record the teacher."""
    X, Y = [], []
    for i in range(n_ep):
        teacher.reset()
        o, _ = env.reset(seed=seed0 + i)
        while True:
            a_t = teacher.act(o)
            X.append(np.asarray(o, np.float32)); Y.append(np.asarray(a_t, np.float32))
            if student is None or np.random.rand() < beta:
                a = a_t
            else:
                with torch.no_grad():
                    a = student(torch.tensor(o, dtype=torch.float32)).numpy()
            o, _, term, trunc, _ = env.step(np.clip(a, -1, 1))
            if term or trunc:
                break
    return np.array(X), np.array(Y)


def evaluate(env, net, n, seed0):
    ok = 0
    for i in range(n):
        o, _ = env.reset(seed=seed0 + i)
        while True:
            with torch.no_grad():
                a = net(torch.tensor(o, dtype=torch.float32)).numpy()
            o, _, term, trunc, info = env.step(np.clip(a, -1, 1))
            if term or trunc:
                ok += bool(info["success"]); break
    return 100.0 * ok / n


def fit(net, X, Y, epochs=60, bs=256):
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    X_t = torch.tensor(X); Y_t = torch.tensor(Y)
    for _ in range(epochs):
        idx = torch.randperm(len(X_t))
        for b in range(0, len(idx), bs):
            j = idx[b:b + bs]
            opt.zero_grad()
            loss = ((net(X_t[j]) - Y_t[j]) ** 2).mean()
            loss.backward(); opt.step()
    return float(loss.item())


def run(env, teacher, label, rounds, ep_per_round, n_eval):
    net = mlp(env.observation_space.shape[0])
    X = Y = None
    print(f"\n{label}")
    for r in range(rounds):
        beta = 1.0 if r == 0 else 0.0
        x, y = collect(env, teacher, None if r == 0 else net,
                       ep_per_round, 3_000_000 + r * 1000, beta)
        X = x if X is None else np.concatenate([X, x])
        Y = y if Y is None else np.concatenate([Y, y])
        loss = fit(net, X, Y)
        acc = evaluate(env, net, n_eval, 4_000_000)
        print(f"  round {r}: {len(X):6d} states  loss {loss:.4f}  "
              f"clone delivers {acc:5.1f}%", flush=True)
    return acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--episodes", type=int, default=120)
    ap.add_argument("--eval", type=int, default=250)
    a = ap.parse_args()
    torch.manual_seed(0); np.random.seed(0)
    env = E(pool="sigma_pool.npz", arm="sigma", seed=0, t_max=35, rho=0.707,
            sigma_scale=3.5)

    rule_score = 100.0 * np.mean([R.rollout(R.SigmaRuleMean(env, 0.5), env,
                                            4_000_000 + i)["success"]
                                  for i in range(a.eval)])
    orac_score = 100.0 * np.mean([R.rollout(R.PadOracle(env), env,
                                            4_000_000 + i)["success"]
                                  for i in range(a.eval)])
    print(f"teachers on the same {a.eval} episodes: "
          f"rule {rule_score:.1f}%   oracle {orac_score:.1f}%")

    c_rule = run(env, R.SigmaRuleMean(env, 0.5),
                 "CONTROL -- cloning the rule (uses only what the drone sees)",
                 a.rounds, a.episodes, a.eval)
    c_orac = run(env, R.PadOracle(env),
                 "TREATMENT -- cloning the oracle (uses the true pad)",
                 a.rounds, a.episodes, a.eval)

    print("\n" + "=" * 62)
    print(f"  rule        {rule_score:5.1f}%      clone of it   {c_rule:5.1f}%")
    print(f"  oracle      {orac_score:5.1f}%      clone of it   {c_orac:5.1f}%")
    print("=" * 62)
    if c_rule < rule_score - 10:
        print("  The CONTROL failed: imitation itself is not working here, so")
        print("  nothing can be concluded about the oracle. Fix this first.")
    elif c_orac > rule_score + 5:
        print("  The oracle's behaviour IS reproducible from the drone's own")
        print("  observations. The gap was PPO's exploration, not information.")
        print("  -> reinforcement learning is salvageable on this task.")
    else:
        print("  The control clones fine but the oracle does not: what the")
        print("  oracle knows is simply NOT IN the observation. No amount of")
        print("  reinforcement learning can close that gap.")
        print("  -> the remaining route is better SENSING, not better learning.")


if __name__ == "__main__":
    main()
