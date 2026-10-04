#!/usr/bin/env python3
"""
clone_check.py -- is PPO's shortfall a learning problem or a seeing problem?

DAgger-clone the same-sensor planner (shield_obs, 95.3%) into the RL's OWN
network: same features, same 14 candidates, same 0.5 s decisions. Then fly the
clone greedily on the ladder's 4096 episodes.

  clone ~ planner      -> the network sees enough; PPO failed to FIND the policy.
                          Fix: start PPO from the clone, not from scratch.
  clone ~ rule or less -> the features cannot carry what the planner uses (its
                          20 s rollouts, the building map). Fix: hand the policy
                          the planner's per-candidate predictions.
"""
import json

import torch

import urban_agents as A
import urban_env as U
from urban_rl import DEV, Learned, Net, copy_planner, inputs


def main():
    out = {}
    for world in ("corridor", "random"):
        torch.manual_seed(0)
        env, g = U.UrbanEnv(8, world, seed=1), A.MPC("obs")
        o = env.reset(); g.reset(env); g.act(o)
        net = Net(inputs(o, g, eyes=False).shape[-1]).to(DEV)
        copy_planner(world, net, eyes=False, tag=world)
        f = lambda x: 100 * x.float().mean().item()
        s = U.run(U.UrbanEnv(4096, world, seed=2026), Learned(net, greedy=True))   # the ladder's episodes
        viol = s["crash"] | s["nmac"] | (s["viol_s"] > 0).any(-1)
        out[world] = dict(clean=f(s["clean"]), violation=f(viol), timeout=f(s["outcome"] == 6))
        print(f"  [{world}] CLONE greedy on the ladder's 4096 episodes: clean {out[world]['clean']:.1f}%  "
              f"violations {out[world]['violation']:.1f}%  timeouts {out[world]['timeout']:.1f}%", flush=True)
        torch.save(net.state_dict(), f"clone_{world}.pt")
    json.dump(out, open("clone_check.json", "w"), indent=1)


if __name__ == "__main__":
    main()
