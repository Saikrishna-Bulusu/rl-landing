#!/usr/bin/env python3
"""
urban_rl.py -- checkpoint 2: a learned decision layer, trained on the GPU.

WHAT IS LEARNED. Every 0.5 s the policy picks one of the SAME 14 candidate
manoeuvres the rule, the planners and the oracle choose from (the rule's own
action, land-anyway, sidesteps, half speed, hover, climb, flee the helipad...).
The candidates are identical across all deciders, so any difference in results
is the decision and nothing else.

WHAT IT SEES. Its own state, the pad (visual fix while tracked, averaged GeoAnchor
fix otherwise), the helipad and its CASA duty, and the tracks of every intruder
its sensors report -- sorted by distance so the network sees "the nearest drone",
not "drone slot 3".
With --eyes it ALSO sees what the same-sensor planner concluded about each
candidate this decision: its 20 s violation forecast, its 5 s progress, its
deadline overrun, and which candidate it would fly. Why: PPO from scratch
(91.8%) lost to the rule, and a copy of the planner in this same network, taught
by imitation, reached 94.1% but no further -- it reproduced only 48% of the
planner's overrides. The raw tracks do not carry the planner's 20 s rollouts, so
the policy is handed them (clone_check.py). Everything the planner sees, the
drone has: no oracle features.

REWARD, priced against the goal (lesson: shaping can outpay the goal)
  +1   clean delivery (touchdown on the pad with no violation all flight)
  -1   crash;  -1 once per near mid-air collision
  -0.25 per violation-second (counted after the env's 3 s grace, as scored)
  -0.001 per decision -- a full 300 s of waiting costs 0.6, less than a clean
        delivery is worth, so waiting out a hazard always beats violating it.

TRAINING. Synchronous generations: N episodes start together and run to the end,
then one PPO update on the whole batch. From scratch the action-0 logit starts at
+3: it begins by mostly doing what the rule does. With --eyes it begins as a copy
of the planner (imitation with the planner flying, then the copy flying), the
value network is fitted for a few generations with the policy frozen, and PPO
then looks for what the planner's 20 s window misses.
"""
import argparse
import copy
import json
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

import urban_agents as A
import urban_env as U

N_ACT, HOLD, GAMMA, LAM = 14, 2, 0.997, 0.95
DEV = U.DEV
MAP = None                           # --map: e.g. logan_map_lidar.npz; None = the env's default map
NAV = "geoanchor"                    # --nav gps: GNSS-grade positioning instead of GeoAnchor's
BIRDS = False                        # --birds: a swooping magpie near half the pads; the net sees it
FOG = False                          # --fog: every flight in fog (patches measured in Gazebo)
UNMAPPED = False                     # --unmapped: structures the survey misses; plan on the voxel map
CAMERAS = False                      # --cameras: the magpie seen only by the drone's own two cameras
HUNTER = False                       # --hunter: the magpie comes from behind and keeps steering
RIG = False                          # --rig: the magpie as a six-camera rig's detector + tracker see it


def clearml_log(task, it, pts):
    """Live curves: pts = [(title, series, value)] at step it (a DAgger round or a PPO generation). Never stops a run."""
    if task is None:
        return
    try:
        lg = task.get_logger()
        for t, sname, v in pts:
            if v == v:                                           # NaN: not measured
                lg.report_scalar(t, sname, float(v), it)
    except Exception as e:
        print("clearml: logging stopped:", e, flush=True)


def world_env(n, world, seed):
    """Every world this script builds goes through here, so none falls back to the old map."""
    return U.UrbanEnv(n, world, seed=seed, nav=NAV, p_swoop=U.P_SWOOP if BIRDS else 0.0,
                      p_fog=U.P_FOG if FOG else 0.0, p_new=U.P_NEW if UNMAPPED else 0.0,
                      bird_sense="rig" if RIG else "cameras" if CAMERAS else "omni",
                      magpie="hunter" if HUNTER else "straight", **({"map_path": f"{U.HERE}/{MAP}"} if MAP else {}))
assert A.MPC.REPLAN == HOLD          # the policy decides on the planner's beat


def features(o, cand, block):
    tgt = torch.where(o["track"][:, None], o["pad_rel_vis"], o["pad_rel_bel"])
    tfrac = torch.full_like(o["pos_z"], o["t"] / U.T_MAX)
    own = torch.stack([o["pos_z"] / 50, *(o["vel"] / 8).unbind(-1), tfrac,
                       o["track"].float(), o["ever_seen"].float()], -1)
    pad = torch.cat([(tgt / 100).clamp(-10, 10), (torch.linalg.norm(tgt, dim=-1, keepdim=True) / 100).clamp(max=10),
                     (o["pad_rel_bel"] / 100).clamp(-10, 10)], -1)
    hls = o["hls_rel_bel"]
    hel = torch.cat([hls / 1400, torch.linalg.norm(hls, dim=-1, keepdim=True) / 1400,
                     o["obliged"].float()[:, None], o["aware"].float()[:, None]], -1)
    hs = o["heli_seen"][:, None]
    heli = torch.cat([hs.float(), torch.where(hs, o["heli_rel"] / 1000, torch.zeros_like(o["heli_rel"])),
                      o["heli_vel"] / 25], -1)

    def nearest_first(rel, vel, seen, sr, sv):
        d = torch.where(seen, torch.linalg.norm(rel, dim=-1), torch.full_like(seen, U.FAR, dtype=torch.float32))
        i = d.argsort(-1)
        rel = rel.gather(1, i[..., None].expand(-1, -1, 3)); vel = vel.gather(1, i[..., None].expand(-1, -1, 3))
        seen = seen.gather(1, i)
        rel = torch.where(seen[..., None], rel / sr, torch.zeros_like(rel))
        return torch.cat([seen.float()[..., None], rel.clamp(-5, 5), vel / sv], -1).flatten(1)

    dr = nearest_first(o["drone_rel"], o["drone_vel"], o["drone_seen"], 200.0, 10.0)
    gr = nearest_first(o["ground_rel"], o["ground_vel"], o["ground_seen"], 50.0, 10.0)
    x = torch.cat([own, pad, hel, heli, dr, gr, (cand / 8).flatten(1), block.float()[:, None]], -1)
    if BIRDS:                         # only then, so nets trained without birds still load
        bs = o["bird_seen"][:, None]
        x = torch.cat([x, bs.float(), torch.where(bs, o["bird_rel"] / U.D_BIRD, torch.zeros_like(o["bird_rel"])),
                       o["bird_vel"] / 15], -1)
    return x


def planner_view(gen):
    """What the same-sensor planner concluded about each candidate on this decision
    (left behind by gen.act): violation forecast, progress, deadline overrun, its pick."""
    late = (gen.base + gen.prog).clamp(min=0.0)          # base = deadline term - progress
    return torch.cat([torch.log1p(gen.risk.clamp(min=0.0)), gen.prog / 10, torch.log1p(late),
                      F.one_hot(gen.choice, N_ACT).float()], -1)


def inputs(o, gen, eyes):
    x = features(o, gen.C, gen.block)
    return torch.cat([x, planner_view(gen)], -1) if eyes else x


def repeats(C):
    """True where a candidate repeats an earlier one. In open sky the rule's action,
    the nominal command and land-anyway are the SAME velocity. A policy choosing
    between identical commands earns nothing either way, so its preference drifts,
    and the drift leaks into the rare states where they differ: corridor seed 0
    picked land-anyway on 18.6% of decisions and people violations went from
    ~0.1% to 2.5%. Masked, a policy only ever chooses between different things.
    EXACT matches only: identical commands cost the planner the same bit for bit,
    so its pick is always the lowest index and never masked. Near-twins are not
    the same choice -- 1 mm/s can cross one of its thresholds (1 m/s away from the
    helipad) -- and masking within 1e-3 masked the planner's own pick."""
    same = (C[:, :, None] == C[:, None]).all(-1)
    earlier = torch.ones(N_ACT, N_ACT, dtype=torch.bool, device=C.device).tril(-1)
    return (same & earlier).any(-1)


def masked(logits, rep):
    return logits.masked_fill(rep, -1e9)        # -1e9, not -inf: 0 * log 0 stays 0 in the KL


class Net(nn.Module):
    """Policy and value on separate bodies: fitting the value never moves the
    policy, which matters once the policy starts as a copy worth keeping."""

    def __init__(self, f, h=256):
        super().__init__()
        body = lambda: nn.Sequential(nn.Linear(f, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU())
        self.body, self.vbody = body(), body()
        self.pi, self.v = nn.Linear(h, N_ACT), nn.Linear(h, 1)
        with torch.no_grad():
            self.pi.weight.mul_(0.01); self.pi.bias.zero_(); self.pi.bias[0] = 3.0   # start near the rule

    def forward(self, x):
        return self.pi(self.body(x)), self.v(self.vbody(x)).squeeze(-1)


class Learned:
    """The trained policy behind the same interface as every other decider.
    greedy=True flies its top choice, False rolls its dice, "mean" flies the
    odds-weighted AVERAGE command: the dice policy's value partly comes from
    blending manoeuvres that are not on the menu (7.9% "climb" among "hold" is a
    gentle climb; corridor, mask-only: timeouts 43 -> 22 of 1024), and the top
    choice turns that into a full climb through the 120 m ceiling (3.3%)."""

    def __init__(self, net, greedy=True, eyes=False):
        self.net, self.greedy, self.eyes, self.gen = net, greedy, eyes, A.MPC("obs")
        self.name = "rl"

    def reset(self, env):
        self.gen.reset(env); self.cmd = torch.zeros(env.n, 3, device=env.dev)
        self.ar = torch.arange(env.n, device=env.dev)

    @torch.no_grad()
    def act(self, o):
        decide = self.gen.k % HOLD == 0
        self.gen.act(o)                 # on the beat: candidates + forecast; off it: guidance only
        if decide:
            logits = masked(self.net(inputs(o, self.gen, self.eyes))[0], repeats(self.gen.C))
            if self.greedy == "mean":
                self.cmd = (torch.softmax(logits, -1)[..., None] * self.gen.C).sum(1)
                return self.cmd
            a = logits.argmax(-1) if self.greedy else torch.distributions.Categorical(logits=logits).sample()
            self.cmd = self.gen.C[self.ar, a]
            return self.cmd
        return self.gen.birds(o, self.cmd)   # between decisions: the same bird reflex as MPC.act


def generation(env, gen, net, eyes):
    """One batch of complete episodes under the current policy, with rewards."""
    o = env.reset(); gen.reset(env)
    ar = torch.arange(env.n, device=env.dev)
    X, Aa, LP, V, R, M, D = [], [], [], [], [], [], []
    viol0, nmac0 = env.viol.sum(-1).clone(), env.nmac.clone()
    while not bool(env.done.all()) and env.t < U.T_MAX - 1e-6:
        live = ~env.done
        gen.act(o)
        x = inputs(o, gen, eyes)
        rep = repeats(gen.C)
        with torch.no_grad():
            logits, v = net(x)
            logits = masked(logits, rep)
            dist = torch.distributions.Categorical(logits=logits)
            a = dist.sample()
        cmd = gen.C[ar, a]
        for h in range(HOLD):
            o = env.step(cmd if h == 0 else gen.birds(o, cmd))   # the bird reflex, as in Learned.act
            if env.t >= U.T_MAX - 1e-6:
                break
            if h < HOLD - 1:
                gen.act(o)      # off the beat: guidance only, same clock as Learned.act
        vn = env.viol.sum(-1)
        r = -0.25 * (vn - viol0) - 1.0 * (env.nmac & ~nmac0).float() - 0.001 * live.float()
        viol0, nmac0 = vn, env.nmac.clone()
        ended = live & env.done
        clean = (env.outcome == 1) & ~env.nmac & (env.viol.sum(-1) == 0)
        crash = U.crashed(env.outcome)
        r = r + ended.float() * (clean.float() - crash.float())
        # fp16 inputs: a generation is 600 x 2048 x 221 numbers, and two runs share 8 GB.
        # Re-scoring them in PPO shifts the logits by ~1e-3, nothing next to the 0.2 clip.
        X.append(x.half()); Aa.append(a); LP.append(dist.log_prob(a)); V.append(v); R.append(r * live); M.append(live)
        D.append(rep)
    return [torch.stack(t) for t in (X, Aa, LP, V, R, M, D)], env.summary()


def gae(V, R, M):
    T = R.shape[0]
    adv = torch.zeros_like(R); last = torch.zeros_like(R[0])
    for t in reversed(range(T)):
        nxt = M[t + 1].float() if t + 1 < T else torch.zeros_like(R[0])
        vn = V[t + 1] if t + 1 < T else torch.zeros_like(R[0])
        delta = R[t] + GAMMA * vn * nxt - V[t]
        last = delta + GAMMA * LAM * nxt * last
        adv[t] = last
    return adv, adv + V


def ppo_update(net, opt, batch, epochs=4, mb=32768, clip=0.2, ent=0.01, vcoef=0.5, pg=1.0, ref=None, kl=0.0, stats=None):
    """pg=0 fits the value alone (policy frozen). ref/kl: charge kl * KL(policy || ref) per decision,
    so leaving ref's pick has to be paid for by advantage. stats: dict filled with the update's mean entropy,
    value loss and (with ref) KL, for ClearML."""
    X, Aa, LP, V, R, M, D = batch
    adv, ret = gae(V, R, M)
    m = M.flatten()
    x, a, lp, adv, ret = X.flatten(0, 1)[m], Aa.flatten()[m], LP.flatten()[m], adv.flatten()[m], ret.flatten()[m]
    rep = D.flatten(0, 1)[m]
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    n = x.shape[0]
    seen = []
    for _ in range(epochs):
        perm = torch.randperm(n, device=x.device)
        for i in range(0, n, mb):
            j = perm[i:i + mb]
            logits, v = net(x[j].float())
            logits = masked(logits, rep[j])
            d = torch.distributions.Categorical(logits=logits)
            ratio = (d.log_prob(a[j]) - lp[j]).exp()
            pl = -torch.min(ratio * adv[j], ratio.clamp(1 - clip, 1 + clip) * adv[j]).mean()
            if ref is not None:
                with torch.no_grad():
                    lq = torch.log_softmax(masked(ref(x[j].float())[0], rep[j]), -1)
                lp_ = torch.log_softmax(logits, -1)
                k = (lp_.exp() * (lp_ - lq)).sum(-1).mean()
                pl = pl + kl * k
            H, vl = d.entropy().mean(), (v - ret[j]).pow(2).mean()
            loss = pg * (pl - ent * H) + vcoef * vl
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 0.5); opt.step()
            seen.append((H.detach(), vl.detach(), k.detach() if ref is not None else None))
    if stats is not None:
        stats["entropy"], stats["value loss"] = (torch.stack([t[i] for t in seen]).mean().item() for i in (0, 1))
        if ref is not None:
            stats["KL to the copy"] = torch.stack([t[2] for t in seen]).mean().item()
    return n


# ---- imitation: a copy of the same-sensor planner ------------------------------------
def fit(net, X, Y, epochs=8, mb=32768, smooth=0.0, stats=None):
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    for _ in range(epochs):
        perm = torch.randperm(len(X), device=DEV)
        for i in range(0, len(X), mb):
            j = perm[i:i + mb]
            loss = F.cross_entropy(net(X[j].float())[0], Y[j], label_smoothing=smooth)
            opt.zero_grad(); loss.backward(); opt.step()
    if stats is not None:
        stats["loss"] = loss.item()                  # the last minibatch of the last epoch
    with torch.no_grad():
        pred = torch.cat([net(X[i:i + mb].float())[0].argmax(-1) for i in range(0, len(X), mb)])
    dev = Y != 0                                     # the decisions where the planner overrides the rule
    return 100 * (pred == Y).float().mean().item(), 100 * (pred[dev] == Y[dev]).float().mean().item(), \
        100 * dev.float().mean().item()


def copy_planner(world, net, eyes, betas=(1.0, 0.5, 0.25, 0.0, 0.0), n=2048, seed0=20_000, smooth=0.0, tag="", task=None):
    """DAgger: each round flies a planner/copy mix (the planner flies a decision with
    chance beta), labels every visited decision with the planner's pick, and refits
    on everything so far -- so the copy also learns to recover from its own slips."""
    Xs, Ys, t0, ar = [], [], time.time(), torch.arange(n, device=DEV)
    for r, beta in enumerate(betas):
        env, gen = world_env(n, world, seed0 + r), A.MPC("obs")
        o = env.reset(); gen.reset(env)
        cmd = torch.zeros(n, 3, device=DEV)
        while not bool(env.done.all()) and env.t < U.T_MAX - 1e-6:
            decide = gen.k % HOLD == 0
            gen.act(o)
            if decide:
                x = inputs(o, gen, eyes)
                with torch.no_grad():
                    a = masked(net(x)[0], repeats(gen.C)).argmax(-1)
                assert not repeats(gen.C)[ar, gen.choice].any(), "planner picked a masked repeat"
                a = torch.where(torch.rand(n, device=DEV) < beta, gen.choice, a)
                cmd = gen.C[ar, a]
                live = ~env.done
                Xs.append(x[live].half()); Ys.append(gen.choice[live])   # fp16: millions of rows fit in 8 GB
            o = env.step(cmd)
        flown = 100 * env.summary()["clean"].float().mean().item()
        del env, gen; torch.cuda.empty_cache()
        st = {}
        acc, acc_dev, share = fit(net, torch.cat(Xs), torch.cat(Ys), smooth=smooth, stats=st)
        clearml_log(task, r, [("DAgger", "loss", st["loss"]), ("DAgger", "copies planner %", acc),
                              ("DAgger", "copies planner on its overrides %", acc_dev), ("DAgger", "flown clean %", flown)])
        print(f"  [{tag}] copy round {r}  planner flies {beta:4.0%}  flown clean {flown:5.1f}%  "
              f"data {sum(len(x) for x in Xs):>9,}  copies planner {acc:5.1f}%  "
              f"on its overrides ({share:4.1f}% of decisions) {acc_dev:5.1f}%  ({time.time() - t0:4.0f} s)",
              flush=True)
    del Xs, Ys; torch.cuda.empty_cache()


EVAL_N, EVAL_SEED = 1024, 2026     # from-scratch learning curve; baselines scored on the same
VAL_N, VAL_SEED = 2048, 3030       # eyes runs: the curve AND the checkpoint choice -- never the ladder's episodes


READOUTS = {"greedy": True, "sampled": False, "mean": "mean"}   # top choice, dice, average command
LOGKEY = {"greedy": "eval_clean", "sampled": "eval_sampled", "mean": "eval_mean"}


def greedy_eval(net, world, eyes, n=EVAL_N, seed=EVAL_SEED, greedy=True):
    s = U.run(world_env(n, world, seed), Learned(net, greedy=greedy, eyes=eyes))
    f = lambda x: 100 * x.float().mean().item()
    return f(s["clean"]), f(s["crash"] | s["nmac"] | (s["viol_s"] > 0).any(-1))


def baselines(world, n=EVAL_N, seed=EVAL_SEED):
    """Rule, best same-sensor planner and oracle on the learning curve's own episodes."""
    f = lambda x: 100 * x.float().mean().item()
    out = {}
    for k, a in {"rule": A.Rule(), "planner": A.MPC("obs"), "oracle": A.MPC("full", "safeplan")}.items():
        s = U.run(world_env(n, world, seed), a)
        out[k] = f(s["clean"])
    return out


def train(world, seed, gens, n_env, tag, eyes=False, eval_every=10, kl=0.0, eval_seed=None, task=None):
    """
    From scratch the exploration bonus anneals 0.01 -> 0.0005. At a constant 0.01
    the first policy stayed spread over near-equivalent manoeuvres: sampled it
    scored 92.4% clean, but taken greedily it picked "climb" every time that was
    the argmax, climbed straight through the 120 m ceiling it had never visited in
    training, and scored 40.5%. PPO had optimised a policy that relied on its own dice.

    With eyes the policy starts as the planner's copy. The copy is taught with 5%
    label smoothing so it still tries the planner's runner-up now and then -- a
    policy certain of every pick explores nothing and PPO could never improve it.
    The value network then learns for WARM generations before the policy may move,
    at a third of the from-scratch learning rate: an untrained value hands PPO
    noise for advantages, and noise is how a good starting policy gets unlearned.

    kl > 0 anchors the policy to the copy instead of to uniform. With only the
    exploration bonus (0.002 -> 0.0002), 200 generations flattened the copy: it
    followed the planner on 95% of decisions at the start and 53% by generation
    100, mostly where the choice hardly matters and no advantage pushes back, so
    near-ties flipped and the top choice fell from 95.3% to 89.8% (corridor).
    The anchor pulls those decisions back to the planner's pick; the bonus goes.
    It also stops the learning: anchored at 0.05 or 0.2 (repeats masked) the top
    choice sat at the copy's 94.2% for 60 generations (validation, corridor).

    So with eyes the run keeps its BEST checkpoint instead: the top choice is
    scored on validation episodes every eval_every generations, the copy counts
    as a checkpoint, and the best is returned. The unanchored random run was
    +1.8 pp over the planner at generation 100 and gave it back by 150; picking
    on validation and reporting on the ladder's episodes keeps what it found
    without letting the pick flatter the report. PPO trains the dice-rolling
    policy and its top-choice version drifts apart from it (mask-only, corridor:
    practice 96.0% while the top choice fell to 90.3%), so each version keeps its
    own best checkpoint -- and so does the average-command read-out (Learned).
    Returns (net, log, states): net is the best top-choice policy; states holds
    each read-out's best checkpoint and "last".
    """
    eval_seed, eval_n = eval_seed or (VAL_SEED if eyes else EVAL_SEED), (VAL_N if eyes else EVAL_N)
    torch.manual_seed(seed)
    env = world_env(n_env, world, 10_000 + seed)
    gen = A.MPC("obs")
    o = env.reset(); gen.reset(env); gen.act(o)
    net = Net(inputs(o, gen, eyes).shape[-1]).to(DEV)
    log = []
    if eyes:
        copy_planner(world, net, eyes, betas=(1.0, 0.5, 0.0), seed0=20_000 + 100 * seed, smooth=0.05, tag=tag, task=task)
        warm, lr, (ent0, ent1) = 5, 1e-4, ((0.0, 0.0) if kl else (0.002, 0.0002))
        ref = copy.deepcopy(net).requires_grad_(False)
        c, _ = greedy_eval(net, world, eyes, n=eval_n, seed=eval_seed)
        log.append(dict(gen=-1, env_steps=0, episodes=0, eval_clean=c))
        best = {"greedy": (c, -1, copy.deepcopy(net.state_dict()))}
        for k in ("sampled", "mean"):
            log[-1][LOGKEY[k]], _ = greedy_eval(net, world, eyes, n=eval_n, seed=eval_seed, greedy=READOUTS[k])
            best[k] = (log[-1][LOGKEY[k]], -1, copy.deepcopy(net.state_dict()))
        print(f"  [{tag}] the copy, greedy, before any PPO: {c:.1f}%", flush=True)
    else:
        warm, lr, ent0, ent1, ref, best = 0, 3e-4, 0.01, 0.0005, None, None
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    t0 = time.time(); steps = 0
    for g in range(gens):
        batch, s = generation(env, gen, net, eyes)
        steps += int(batch[5].sum().item()) * HOLD
        st = {}
        ppo_update(net, opt, batch, ent=ent0 + (ent1 - ent0) * g / max(gens - 1, 1), pg=float(g >= warm),
                   ref=ref if kl else None, kl=kl, stats=st)
        f = lambda x: 100 * x.float().mean().item()
        viol = s["crash"] | s["nmac"] | (s["viol_s"] > 0).any(-1)
        live = batch[5]
        row = dict(gen=g, env_steps=steps, episodes=(g + 1) * n_env, clean=f(s["clean"]),
                   delivered=f(s["delivered"]), violation=f(viol), rule_share=f(batch[1][live] == 0))
        if eyes:                                         # how often it flew what the planner would have
            row["planner_share"] = f(batch[1][live] == batch[0][live][:, -N_ACT:].argmax(-1))
        log.append(row)
        pts = [("clean delivery %", "training", row["clean"]), ("mean reward", "per episode", batch[4].sum(0).mean().item()),
               ("follows the rule %", "rule", row["rule_share"])] + [("PPO", k, v) for k, v in st.items()]
        if eyes:
            pts.append(("follows the planner %", "planner", row["planner_share"]))
        if g % eval_every == 0 or g == gens - 1:
            row["eval_clean"], row["eval_violation"] = greedy_eval(net, world, eyes, n=eval_n, seed=eval_seed)
            if best is not None:
                for k in ("sampled", "mean"):
                    row[LOGKEY[k]], _ = greedy_eval(net, world, eyes, n=eval_n, seed=eval_seed, greedy=READOUTS[k])
                for k in READOUTS:
                    if row[LOGKEY[k]] > best[k][0]:
                        best[k] = (row[LOGKEY[k]], g, copy.deepcopy(net.state_dict()))
            pts += [("validation clean %", k, row[v]) for k, v in LOGKEY.items() if v in row]
            pts.append(("validation violation %", "greedy", row["eval_violation"]))
        clearml_log(task, g, pts)
        if g % 10 == 0 or g == gens - 1:
            share = f"planner {row['planner_share']:4.1f}%" if eyes else f"rule {row['rule_share']:4.1f}%"
            print(f"  [{tag}] gen {g:3d}  episodes {row['episodes']:>7,}  clean {row['clean']:5.1f}%  "
                  f"viol {row['violation']:4.1f}%  picks {share}  "
                  f"| top {row.get('eval_clean', float('nan')):5.1f}%"
                  + (f" dice {row['eval_sampled']:5.1f}% avg {row['eval_mean']:5.1f}%" if 'eval_mean' in row else "")
                  + f"  ({time.time() - t0:5.0f} s)",
                  flush=True)
    states = None
    if best is not None:
        states = {**{k: v[2] for k, v in best.items()}, "last": copy.deepcopy(net.state_dict())}
        net.load_state_dict(best["greedy"][2])
        log.append({f"kept_{k}": dict(gen=v[1], validation=v[0]) for k, v in best.items()})
        print(f"  [{tag}] kept (validation): " + ", ".join(f"{k} gen {v[1]} {v[0]:.1f}%" for k, v in best.items()),
              flush=True)
        for k, st in states.items():
            torch.save(st, f"rl_{tag}_{k}.pt")
    torch.save(net.state_dict(), f"rl_{tag}.pt")
    return net, log, states


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", default="corridor")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--gens", type=int, default=120)
    ap.add_argument("--n-env", type=int, default=2048)
    ap.add_argument("--eyes", action="store_true", help="also see the planner's per-candidate forecast")
    ap.add_argument("--kl", type=float, default=0.0, help="anchor to the copy: price of KL(policy || copy)")
    ap.add_argument("--map", default=None, help="e.g. logan_map_lidar.npz (default: the env's map)")
    ap.add_argument("--nav", default="geoanchor", choices=("geoanchor", "gps"))
    ap.add_argument("--birds", action="store_true", help="a swooping magpie near half the pads")
    ap.add_argument("--fog", action="store_true", help="every flight in fog")
    ap.add_argument("--unmapped", action="store_true", help="structures the survey does not show")
    ap.add_argument("--cameras", action="store_true", help="magpie seen only by the forward and landing cameras")
    ap.add_argument("--hunter", action="store_true", help="the magpie attacks from behind and keeps steering")
    ap.add_argument("--rig", action="store_true", help="magpie seen by the six-camera rig's detector + tracker")
    ap.add_argument("--no-clearml", action="store_true", help="no live logging to ClearML (rl-landing/fast_sim)")
    ap.add_argument("--tag", default="", help="appended to the names, e.g. _v2 with RIG_CALIB=rig_calib_v2.json (rig v2)")
    a = ap.parse_args()
    assert not (a.cameras and a.rig)
    global MAP, NAV, BIRDS, FOG, UNMAPPED, CAMERAS, HUNTER, RIG
    MAP, NAV, BIRDS, FOG, UNMAPPED, CAMERAS, HUNTER, RIG = a.map, a.nav, a.birds, a.fog, a.unmapped, a.cameras, a.hunter, a.rig
    gps = ("_gps" if a.nav == "gps" else "") + ("_birds" if a.birds else "") + ("_fog" if a.fog else "") + ("_unmapped" if a.unmapped else "") + ("_cameras" if a.cameras else "") \
        + ("_hunter" if a.hunter else "") + ("_rig" if a.rig else "") + a.tag
    pre = ("rl_eyes" if a.eyes else "rl") + (f"_kl{a.kl:g}" if a.kl else "") + ("_lidar" if a.map else "") + gps
    f = lambda x: 100 * x.float().mean().item()
    n_c, s_c = (VAL_N, VAL_SEED) if a.eyes else (EVAL_N, EVAL_SEED)
    out = {"map": a.map or "logan_map.npz", "nav": a.nav, "birds": a.birds, "fog": a.fog, "unmapped": a.unmapped, "cameras": a.cameras,
           "hunter": a.hunter, "rig": a.rig, "baselines": baselines(a.world, n=n_c, seed=s_c)}
    print(f"  baselines on the curve's {n_c} episodes (seed {s_c}): {out['baselines']}", flush=True)
    ladder = lambda agent: U.run(world_env(4096, a.world, 2026), agent)   # the headroom ladder's episodes
    ref = {k: ladder(agent)["clean"].float() for k, agent in (("rule", A.Rule()), ("planner", A.MPC("obs")))}
    for sd in a.seeds:
        tag = f"{a.world}{'_eyes' if a.eyes else ''}{f'_kl{a.kl:g}' if a.kl else ''}{'_lidar' if a.map else ''}{gps}_s{sd}"
        task = None
        if not a.no_clearml:
            try:
                from clearml import Task
                task = Task.init(project_name="rl-landing/fast_sim", task_name=f"{pre}_{a.world}_s{sd}", auto_connect_frameworks=False)
                task.connect({**vars(a), "seed": sd})
                print("ClearML:", task.get_output_log_web_page(), flush=True)
            except Exception as e:
                task = None
                print("clearml unavailable, training without it:", e, flush=True)
        net, log, states = train(a.world, sd, a.gens, a.n_env, tag, eyes=a.eyes, kl=a.kl, task=task)
        runs = [("greedy", net, True), ("sampled", net, False)]
        if states is not None:                        # each read-out's own best, plus the last generation
            pn = {k: copy.deepcopy(net) for k in states}
            for k, st in states.items():
                pn[k].load_state_dict(st)
            runs = [(k, pn[k], r) for k, r in READOUTS.items()] + [(f"last_{k}", pn["last"], r) for k, r in READOUTS.items()]
        ev = {}
        for mode, pnet, greedy in runs:
            s = ladder(Learned(pnet, greedy=greedy, eyes=a.eyes))
            viol = s["crash"] | s["nmac"] | (s["viol_s"] > 0).any(-1)
            ev[mode] = dict(clean=f(s["clean"]), delivered=f(s["delivered"]), violation=f(viol),
                            ceiling=f(s["viol_s"][:, 0] > 0), median_t=s["t_done"][s["clean"]].median().item())
            for k, r in ref.items():                  # paired: same 4096 episodes, per-episode difference
                d = s["clean"].float() - r
                ev[mode][f"vs_{k}"] = [100 * d.mean().item(), 196 * d.std().item() / len(d) ** 0.5]
            print(f"  [{tag}] EVAL {mode:<12} on the ladder's 4096 episodes: clean {ev[mode]['clean']:.1f}%  "
                  f"(vs rule {ev[mode]['vs_rule'][0]:+.2f} +/-{ev[mode]['vs_rule'][1]:.2f}, "
                  f"vs planner {ev[mode]['vs_planner'][0]:+.2f} +/-{ev[mode]['vs_planner'][1]:.2f} pp)  "
                  f"violations {ev[mode]['violation']:.1f}% (ceiling {ev[mode]['ceiling']:.1f}%)  "
                  f"median {ev[mode]['median_t']:.0f} s", flush=True)
        out[tag] = dict(log=log, eval=ev)
        if task:
            task.close()                              # one task per seed
        json.dump(out, open(f"{pre}_{a.world}_{'_'.join(map(str, a.seeds))}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
