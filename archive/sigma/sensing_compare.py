#!/usr/bin/env python3
"""
sensing_compare.py -- does stereo-grade obstacle sensing tell the policy
anything that eight range readings do not?

The question that decides whether a new simulator is worth building. The policy
already had eight bearing-binned ranges. If richer sensing carries no extra
information, a better-rendered environment supplies fidelity, not knowledge.

Answered by imitation, so no reinforcement learning is involved: take a teacher
that knows every canopy exactly, and try to reproduce it from each observation.
DAgger, so distribution shift cannot be blamed. The rule is cloned as a control:
if that fails, the setup is broken and nothing can be concluded.

Scored on REWARD. Delivery rate prices flying into a tree the same as a
cautious abort, which inverted every obstacle result measured before this.
"""
import os, sys
import numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "shared", "px4_gazebo"))   # forest_spec
import forest_spec as F
import sensing_headroom as S
import rule_agent as R


class Base(S.ApproachEnvV5):
    # From forest_spec, not retyped: make_trees.py and forest.py build the
    # Gazebo world from the same module, and they run under a python without
    # torch so they cannot import this class. One source or they drift.
    N_OBST = F.N_OBST; OBST_H = F.OBST_H; PAD_CLEAR = F.PAD_CLEAR
    OBST_R = F.OBST_R; CANOPY_R = F.CANOPY_R; OBST_SPAN = F.OBST_SPAN
    TRUNK_R = F.TRUNK_R; CANOPY_FRAC = F.CANOPY_FRAC


def make(mode):
    return Base(pool="sigma_pool.npz", arm="sigma", seed=0, t_max=35,
                rho=0.707, sigma_scale=3.5, sense_mode=mode)


def dagger(env, teacher, rounds=4, eps=100, n_eval=200, seed0=6_000_000):
    net = S.mlp(env.observation_space.shape[0]); X = Y = None
    for r in range(rounds):
        xs, ys = [], []
        for i in range(eps):
            teacher.reset(); o, _ = env.reset(seed=seed0 + r * 997 + i)
            while True:
                at = teacher.act(o)
                xs.append(np.asarray(o, np.float32)); ys.append(np.asarray(at, np.float32))
                if r == 0: a = at
                else:
                    with torch.no_grad():
                        a = net(torch.tensor(o, dtype=torch.float32)).numpy()
                o, _, t, tr, _ = env.step(np.clip(a, -1, 1))
                if t or tr: break
        X = np.array(xs) if X is None else np.concatenate([X, xs])
        Y = np.array(ys) if Y is None else np.concatenate([Y, ys])
        opt = torch.optim.Adam(net.parameters(), 1e-3)
        xt, yt = torch.tensor(X), torch.tensor(Y)
        for _ in range(60):
            idx = torch.randperm(len(xt))
            for b in range(0, len(idx), 256):
                j = idx[b:b+256]; opt.zero_grad()
                loss = ((net(xt[j]) - yt[j]) ** 2).mean(); loss.backward(); opt.step()
    rew, ok, cr = [], [], []
    for i in range(n_eval):
        o, _ = env.reset(seed=7_500_000 + i); tot = 0.0
        while True:
            with torch.no_grad():
                a = net(torch.tensor(o, dtype=torch.float32)).numpy()
            o, rr, t, tr, info = env.step(np.clip(a, -1, 1)); tot += rr
            if t or tr:
                rew.append(tot); ok.append(info["success"])
                cr.append(info.get("outcome") == "crash"); break
    return float(np.mean(rew)), 100*np.mean(ok), 100*np.mean(cr), float(loss.item())


def main():
    torch.manual_seed(0); np.random.seed(0)
    env = make("sectors")
    h = S.validated_headroom(env, lambda: R.SigmaRuleMean(env, 0.5),
                             lambda: S.ObstacleTeacher(env), 200)
    print(f"pad blocked by a canopy: "
          f"{100*np.mean([(env.reset(seed=i), env.descent_blocked(env.pad))[1] for i in range(200)]):.0f}%")
    print(f"  rule     reward {h['rule'][0]:+7.1f}   {h['rule'][1]:5.1f}% delivered   {h['rule'][2]:5.1f}% crash")
    print(f"  teacher  reward {h['teacher'][0]:+7.1f}   {h['teacher'][1]:5.1f}% delivered   {h['teacher'][2]:5.1f}% crash")
    print(f"  VALIDATED HEADROOM {h['headroom']:+.1f} reward\n")

    print("cloning that teacher from each obstacle representation:")
    out = {}
    for mode in ("sectors", "rich"):
        e = make(mode)
        rew, ok, cr, loss = dagger(e, S.ObstacleTeacher(e))
        out[mode] = rew
        print(f"  {mode:<8} ({e.observation_space.shape[0]:2d} ch)  reward {rew:+7.1f}  "
              f"{ok:5.1f}% delivered  {cr:5.1f}% crash   fit loss {loss:.4f}")

    e = make("sectors")
    rew, ok, cr, loss = dagger(e, R.SigmaRuleMean(e, 0.5))
    print(f"\n  CONTROL, cloning the rule (uses no obstacle data):")
    print(f"  sectors  reward {rew:+7.1f}  {ok:5.1f}% delivered  fit loss {loss:.4f}"
          f"   (rule itself {h['rule'][0]:+.1f})")

    gap = out["rich"] - out["sectors"]
    print(f"\n  rich - sectors = {gap:+.1f} reward")
    print(f"  teacher - rule = {h['headroom']:+.1f} reward  (the room available)")
    # NO VERDICT FROM ONE SEED. This printed "worth building" off a single
    # +6.4, and three seeds per arm then put the gap at +6.9 with p=0.22 --
    # the spread across initialisations (+/-4 to +/-7) is the size of the
    # effect. Whatever this run shows, sensing_seeds.py decides it.
    print(f"\n  ONE SEED ONLY -- the across-seed spread here is as large as the")
    print(f"  gap itself. Run sensing_seeds.py before concluding anything.")


if __name__ == "__main__":
    main()
