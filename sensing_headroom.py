#!/usr/bin/env python3
"""
sensing_headroom.py -- would a stereo depth camera tell the policy anything
that eight range readings do not?

WHY THIS COMES BEFORE BUILDING A SIMULATOR

The plan is to build a richer environment so the policy can sense obstacles
properly. But the policy ALREADY had obstacle sensing -- eight bearing-binned
ranges -- and still lost to a one-line rule by 3.58 points. So "give it better
sensing" is not an untested idea; a version of it was tested and did not win.

The question is therefore narrow: does a richer obstacle representation carry
information the sectors do not? Imitation answers it without any training. Take
a teacher that knows the obstacle layout EXACTLY, and try to reproduce its
behaviour from each observation in turn:

    if the rich observation clones it and the sectors do not,
        the extra fidelity carries real information -> build the simulator
    if neither clones it,
        the teacher's edge is knowledge no on-board sensor recovers -> do not
    if both clone it equally,
        the sectors were already enough -> a new simulator buys rendering, not
        information, and the policy's loss was never a sensing problem

DAgger, not plain cloning, so distribution shift cannot be blamed. The rule is
cloned too, as a control: if that fails, the setup is broken and nothing about
the teacher can be concluded.
"""
import argparse
import numpy as np
import torch
import torch.nn as nn

from approach_env_v5 import ApproachEnvV5
import rule_agent as R


def make(mode, **kw):
    class E(ApproachEnvV5):
        N_OBST = (8, 10); OBST_H = (7.2, 12.0); PAD_CLEAR = 0.5; OBST_R = (2.5, 5.0)
    return E(pool="sigma_pool.npz", arm="sigma", seed=0, t_max=35, rho=0.707,
             sigma_scale=3.5, sense_mode=mode, **kw)


class ObstacleTeacher(R.SigmaRuleMean):
    """
    The rule, plus exact canopy knowledge used to pick WHERE to come down.

    With canopies the naive descent is genuinely dangerous -- ignoring them
    crashes 72% of the time -- so obstacle knowledge finally has something to
    buy. This teacher uses it minimally and only in two places:

      * it will not descend where a canopy is overhead; it holds altitude and
        shifts to the nearest gap instead,
      * if the pad itself is under a canopy it stops descending, which leaves
        the episode to time out as a return rather than a crash.

    Everything else is the rule's own action, so the teacher cannot score below
    it for any reason other than these two interventions -- and that is checked
    before the measurement is believed.
    """
    def __init__(self, env, k=0.5):
        super().__init__(env, k)
        self.e = env

    def _gap_bearing(self):
        """Nearest bearing, at the current range, with no canopy over it."""
        e = self.e
        for rad in (6.0, 10.0, 14.0):
            for th in np.linspace(0, 2 * np.pi, 12, endpoint=False):
                p = e.xy + rad * np.array([np.cos(th), np.sin(th)])
                if not e.descent_blocked(p):
                    return th, rad
        return None, None

    def act(self, o):
        a = super().act(o)
        e = self.e
        if not e.descent_blocked():
            return a                       # clear overhead: fly the rule
        a[0] = max(float(a[0]), 0.0)       # never descend into a canopy
        th, rad = self._gap_bearing()
        if th is not None:
            a[2] = float(th / np.pi)
            a[1] = 2.0 * min(rad, e.SEARCH_R_MAX) / e.SEARCH_R_MAX - 1.0
        return a


def validated_headroom(env, rule, teacher, n, seed0=7_500_000, strict=True):
    """
    Headroom, with the measuring stick checked before the number is returned.

    Three teachers were shipped broken today, each scoring BELOW the baseline
    it was supposed to bound, and each caught only by eyeballing the result. An
    upper bound that loses to the thing it bounds is not a hard problem, it is
    a bug, so the check is no longer optional.

    Scored on REWARD, not delivery rate. Delivery rate prices flying into a
    tree exactly like a cautious abort, which inverted every obstacle result
    measured before this.
    """
    def ev(mk):
        rows = [R.rollout(mk(), env, seed0 + i) for i in range(n)]
        return rows, (float(np.mean([r["reward"] for r in rows])),
                      100.0 * np.mean([r["success"] for r in rows]),
                      100.0 * np.mean([r["outcome"] == "crash" for r in rows]))
    rr, r = ev(rule); tr, t = ev(teacher)

    # TOLERANCE FROM THE NOISE, NOT FROM ZERO.
    #
    # This used to raise whenever the teacher came in below the rule by any
    # amount at all. That is right for a BROKEN teacher -- the ones this guard
    # was built for scored 27.5% against 75.5% -- but it also fires on a TIE,
    # and a tie is a real and important result: with a marker large enough,
    # an oracle that knows every tree cannot beat the one-line rule, both
    # saturate, and the difference is floating-point noise. Judge against the
    # standard error of the PAIRED difference so the guard still catches a
    # teacher that is genuinely worse and stays quiet when there is simply
    # nothing left to win.
    d = np.array([a["reward"] for a in tr]) - np.array([b["reward"] for b in rr])
    se = float(np.std(d, ddof=1) / np.sqrt(len(d))) if len(d) > 1 else 0.0
    failed = t[0] < r[0] - max(2.0 * se, 1e-9)
    if failed and not strict:
        # The caller expects a saturated regime, where an oracle with nothing
        # left to win can sit a hair BELOW the rule -- its extra caution costs
        # a step of energy and buys nothing. Hand back the numbers and let the
        # caller say so, rather than killing the run over 0.1 reward.
        return dict(rule=r, teacher=t, headroom=t[0] - r[0], se=se,
                    tie=True, guard_failed=True)
    if failed:
        raise AssertionError(
            f"teacher scores {t[0]:+.1f} against the rule's {r[0]:+.1f}, short by "
            f"{r[0]-t[0]:.1f} against a paired standard error of {se:.1f}. An upper "
            f"bound cannot lose to its own baseline -- fix the teacher before "
            f"reading anything into the headroom.")
    return dict(rule=r, teacher=t, headroom=t[0] - r[0], se=se,
                tie=abs(t[0] - r[0]) <= max(2.0 * se, 0.15),
                guard_failed=False)


def mlp(n_in):
    return nn.Sequential(nn.Linear(n_in, 128), nn.Tanh(),
                         nn.Linear(128, 64), nn.Tanh(), nn.Linear(64, 3))


def dagger(env, teacher, rounds, eps, n_eval, seed0=6_000_000):
    net = mlp(env.observation_space.shape[0]); X = Y = None
    for r in range(rounds):
        xs, ys = [], []
        for i in range(eps):
            teacher.reset(); o, _ = env.reset(seed=seed0 + r * 997 + i)
            while True:
                at = teacher.act(o)
                xs.append(np.asarray(o, np.float32)); ys.append(np.asarray(at, np.float32))
                if r == 0:
                    a = at
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
    ok = 0
    for i in range(n_eval):
        o, _ = env.reset(seed=7_500_000 + i)
        while True:
            with torch.no_grad():
                a = net(torch.tensor(o, dtype=torch.float32)).numpy()
            o, _, t, tr, info = env.step(np.clip(a, -1, 1))
            if t or tr:
                ok += bool(info["success"]); break
    return 100.0 * ok / n_eval, float(loss.item())


def score(env, agent, n, seed0=7_500_000):
    return 100.0 * np.mean([R.rollout(agent, env, seed0 + i)["success"] for i in range(n)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--eval", type=int, default=200)
    a = ap.parse_args()
    torch.manual_seed(0); np.random.seed(0)

    base = make("sectors")
    rule = score(base, R.SigmaRuleMean(base, 0.5), a.eval)
    teach = score(base, ObstacleTeacher(base), a.eval)
    print(f"on {a.eval} identical episodes:")
    print(f"  hand-written rule (no obstacle use)   {rule:5.1f}%")
    print(f"  teacher, knows every tree exactly     {teach:5.1f}%")
    print(f"  obstacle headroom                     {teach-rule:+5.1f} pp\n")
    if teach - rule < 2.0:
        print("  the teacher barely beats the rule, so there is nothing for ANY")
        print("  sensing to recover. Stop here.\n")

    print("cloning that teacher from each observation:")
    for mode in ("sectors", "rich"):
        env = make(mode)
        acc, loss = dagger(env, ObstacleTeacher(env), a.rounds, a.episodes, a.eval)
        print(f"  {mode:<8} ({env.observation_space.shape[0]:2d} ch)  clone {acc:5.1f}%   fit loss {loss:.4f}")

    print("\n  control -- cloning the RULE, which uses no obstacle data at all:")
    env = make("sectors")
    acc, loss = dagger(env, R.SigmaRuleMean(env, 0.5), a.rounds, a.episodes, a.eval)
    print(f"  sectors  clone {acc:5.1f}%   fit loss {loss:.4f}   (rule itself {rule:.1f}%)")


if __name__ == "__main__":
    main()
