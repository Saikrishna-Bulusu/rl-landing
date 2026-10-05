#!/usr/bin/env python3
"""
train.py -- the Tello learner, trained the way the Logan one is (shared/fast_sim/urban_rl.py): copy the cage planner
(DAgger, 5% label smoothing), let the value net warm up, then practise with PPO; keep each read-out's best
round on validation flights; test once on the ladder's 4,096 flights, paired against the rule and planner.
It sees what the planner sees plus the planner's forecast for every candidate ("eyes"). Landing is masked
unless the customer's marker has been read: the identity check stays outside the policy, as a hard layer.

    python train.py --seeds 0 1 2 [--gens 120]     -> cage_rl_s{N}_{greedy|sampled}.pt, cage_rl.json
"""
import argparse
import copy
import json
import os
import sys
import time

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, f"{os.path.dirname(os.path.dirname(HERE))}/shared/fast_sim")
import urban_rl as R            # Net, ppo_update, fit, masked, repeats: one learner for both worlds
import cage as C

DEV = C.DEV
VAL_N, VAL_SEED, TEST_N, TEST_SEED = 2048, 3030, 4096, 2026
READ = {"greedy": True, "sampled": False}


def features(o, pl, env):
    """The planner's view of this decision, in the believed frame, plus its forecast per candidate."""
    b, v, n = o["b"], o["v"], o["b"].shape[0]
    lock = o["seen"][:, 0].float()
    known = (o["pole"][:, 0] < 100).float()
    pole = (o["pole"][:, :2] - b[:, :2]).clamp(-2, 2) * known[:, None]
    xy = torch.cat([env.houses[..., :2], env.trees[..., :2]], 1) - b[:, None, :2]      # the map, nearest four
    size = torch.cat([env.houses[..., 2:4].norm(dim=-1), env.trees[..., 2]], 1)
    top = torch.cat([env.houses[..., 4], env.trees[..., 3]], 1) - b[:, 2:3]
    idx = (xy.norm(dim=-1) - size).argsort(-1)[:, :4]
    near = torch.cat([xy.gather(1, idx[..., None].expand(-1, -1, 2)), size.gather(1, idx)[..., None],
                      top.gather(1, idx)[..., None]], -1).flatten(1)
    walls = torch.stack([b[:, 0], C.L - b[:, 0], b[:, 1], C.B - b[:, 1], C.H - b[:, 2]], -1)
    return torch.cat([b / torch.tensor([C.L, C.B, C.H], device=b.device), v / C.V_XY, o["sig"][:, None] / 0.1,
                      pl.aim - b[:, :2], pl.goal - b[:, :2], lock[:, None], o["seen"][:, 1:].any(-1).float()[:, None],
                      o["rel"][:, 0] * lock[:, None], pole, known[:, None], near, walls,
                      torch.full((n, 1), env.t / C.T_MAX, device=b.device), (pl.leg % 7).float()[:, None] / 7,
                      pl.first.float() / pl.T, pl.safe.float(), pl.gain / 0.75, F.one_hot(pl.choice, 14).float()], -1)


def mask(o, pl):
    m = R.repeats(pl.C)
    m[:, 13] |= ~o["seen"][:, 0]                    # land only on a marker read as the customer's
    return m


class Learned(C.Planner):
    """The policy behind the decider interface: the planner's candidates and forecasts, the net's pick."""

    def __init__(self, net, greedy=True):
        super().__init__("obs")
        self.net, self.greedy = net, greedy

    @torch.no_grad()
    def act(self, o):
        C.Planner.act(self, o)                      # candidates + forecasts; the planner's own pick is a feature
        logits = R.masked(self.net(features(o, self, self.env))[0], mask(o, self))
        a = logits.argmax(-1) if self.greedy else torch.distributions.Categorical(logits=logits).sample()
        return self.C[torch.arange(len(a), device=a.device), a], a == 13


def evaluate(net, greedy, n=VAL_N, seed=VAL_SEED):
    s = C.run(C.Cage(n, seed=seed), Learned(net, greedy))
    return 100 * (s["outcome"] == 1).float().mean().item(), s


def copy_planner(net, betas=(1.0, 0.5, 0.0), n=2048, seed0=20_000, tag=""):
    """DAgger: each round flies a planner/copy mix, labels every decision with the planner's pick, refits."""
    Xs, Ys = [], []
    for r, beta in enumerate(betas):
        env, pl = C.Cage(n, seed=seed0 + r), Learned(net)
        o = env.reset(); pl.reset(env)
        ar, k = torch.arange(n, device=DEV), 0
        cmd, land = torch.zeros(n, 3, device=DEV), torch.zeros(n, dtype=torch.bool, device=DEV)
        while not bool(env.done.all()) and env.t < C.T_MAX - 1e-6:
            if k % C.HOLD == 0:
                C.Planner.act(pl, o)
                x = features(o, pl, env)
                with torch.no_grad():
                    a = R.masked(net(x)[0], mask(o, pl)).argmax(-1)
                a = torch.where(torch.rand(n, device=DEV) < beta, pl.choice, a)
                cmd, land = pl.C[ar, a], a == 13
                live = ~env.done
                Xs.append(x[live].half()); Ys.append(pl.choice[live])
            o = env.step(cmd, land if k % C.HOLD == 0 else None)
            k += 1
        acc, acc_dev, share = R.fit(net, torch.cat(Xs), torch.cat(Ys), smooth=0.05)
        print(f"  [{tag}] copy round {r}  planner flies {beta:4.0%}  flown {100 * (env.outcome == 1).float().mean().item():5.1f}%  "
              f"copies planner {acc:5.1f}% (its overrides of the rule, {share:4.1f}% of decisions: {acc_dev:5.1f}%)", flush=True)


def generation(net, n, seed):
    """One batch of complete flights under the dice policy, with rewards: +1 delivered, -1 crash or net."""
    env, pl = C.Cage(n, seed=seed), Learned(net, greedy=False)
    o = env.reset(); pl.reset(env)
    ar = torch.arange(n, device=DEV)
    X, A, LP, V, Rw, M, D = [], [], [], [], [], [], []
    while not bool(env.done.all()) and env.t < C.T_MAX - 1e-6:
        live = ~env.done
        C.Planner.act(pl, o)
        x, m = features(o, pl, env), mask(o, pl)
        with torch.no_grad():
            logits, v = net(x)
            dist = torch.distributions.Categorical(logits=R.masked(logits, m))
            a = dist.sample()
        for h in range(C.HOLD):
            o = env.step(pl.C[ar, a], (a == 13) if h == 0 else None)
            if env.t >= C.T_MAX - 1e-6:
                break
        ended = live & env.done
        r = ended.float() * ((env.outcome == 1).float() - ((env.outcome == 4) | (env.outcome == 5)).float()) - 0.001 * live.float()
        X.append(x.half()); A.append(a); LP.append(dist.log_prob(a)); V.append(v); Rw.append(r * live); M.append(live); D.append(m)
    return [torch.stack(t) for t in (X, A, LP, V, Rw, M, D)]


def train(seed, gens, n_env=2048, every=10, tag=""):
    torch.manual_seed(seed)
    env, pl = C.Cage(8, seed=1), C.Planner(); o = env.reset(); pl.reset(env); pl.act(o)
    net = R.Net(features(o, pl, env).shape[-1]).to(DEV)
    copy_planner(net, seed0=20_000 + 100 * seed, tag=tag)
    best = {k: (evaluate(net, g)[0], -1, copy.deepcopy(net.state_dict())) for k, g in READ.items()}
    log = [dict(gen=-1, **{k: v[0] for k, v in best.items()})]
    print(f"  [{tag}] the copy: " + "  ".join(f"{k} {v[0]:.1f}%" for k, v in best.items()), flush=True)
    opt, t0 = torch.optim.Adam(net.parameters(), lr=1e-4), time.time()
    for g in range(gens):
        batch = generation(net, n_env, 10_000 * (seed + 1) + g)
        R.ppo_update(net, opt, batch, ent=0.002 + (0.0002 - 0.002) * g / max(gens - 1, 1), pg=float(g >= 5))
        if g % every == 0 or g == gens - 1:
            row = dict(gen=g, practice=100 * (batch[4].sum(0) > 0.5).float().mean().item())
            for k, gr in READ.items():
                row[k] = evaluate(net, gr)[0]
                if row[k] > best[k][0]:
                    best[k] = (row[k], g, copy.deepcopy(net.state_dict()))
            log.append(row)
            print(f"  [{tag}] gen {g:3d}  validation: top {row['greedy']:5.1f}%  dice {row['sampled']:5.1f}%  ({time.time() - t0:4.0f} s)", flush=True)
    for k, v in best.items():
        torch.save(v[2], f"{HERE}/cage_rl_{tag}_{k}.pt")
    log.append({f"kept_{k}": dict(gen=v[1], validation=v[0]) for k, v in best.items()})
    return best, log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--gens", type=int, default=120)
    a = ap.parse_args()
    ref = {k: C.run(C.Cage(TEST_N, seed=TEST_SEED), ag)["outcome"] == 1
           for k, ag in (("rule", C.Rule()), ("planner", C.Planner()), ("all_knowing", C.Planner("full")))}
    out = {"baselines": {k: 100 * v.float().mean().item() for k, v in ref.items()}}
    print("  test baselines:", out["baselines"], flush=True)
    path = f"{HERE}/cage_rl_{'_'.join(map(str, a.seeds))}.json"
    for sd in a.seeds:
        tag = f"s{sd}"
        best, log = train(sd, a.gens, tag=tag)
        ev = {}
        for k, gr in READ.items():
            net = R.Net(best[k][2]["body.0.weight"].shape[1]).to(DEV)
            net.load_state_dict(best[k][2])
            _, s = evaluate(net, gr, TEST_N, TEST_SEED)
            ok = (s["outcome"] == 1).float()
            ev[k] = dict(delivered=100 * ok.mean().item(),
                         outcomes={name: 100 * (s["outcome"] == i).float().mean().item() for i, name in enumerate(C.OUTCOMES) if i})
            for b, r in ref.items():
                d = ok - r.float()
                ev[k][f"vs_{b}"] = [100 * d.mean().item(), 196 * d.std().item() / len(d) ** 0.5]
            print(f"  [{tag}] TEST {k:<8} delivered {ev[k]['delivered']:.1f}%  vs rule {ev[k]['vs_rule'][0]:+.2f}  "
                  f"vs planner {ev[k]['vs_planner'][0]:+.2f} +/-{ev[k]['vs_planner'][1]:.2f}  "
                  f"vs all-knowing {ev[k]['vs_all_knowing'][0]:+.2f}", flush=True)
        out[tag] = dict(log=log, eval=ev)
        json.dump(out, open(path, "w"), indent=1)


if __name__ == "__main__":
    main()
