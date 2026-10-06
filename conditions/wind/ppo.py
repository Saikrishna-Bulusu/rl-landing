#!/usr/bin/env python3
"""
ppo.py -- PPO in plain JAX for the wind landing (land_env.py), with a wind curriculum: the learner starts in light
wind, and the wind rises as it lands reliably (Sai, 26 Sept: RL on gradually increasing wind, flying + deciding).
No RL library (none installed; nothing downloaded): an MLP, Adam and the PPO loss, ~200 lines.

    python conditions/wind/ppo.py --tag both   --seed 0                  flying + deciding
    python conditions/wind/ppo.py --tag fly    --no-decide --seed 0      flying only (the rule keeps its height and gate)
    python conditions/wind/ppo.py --tag decide --no-fly    --seed 0      deciding only (no feed-forward)
Outputs conditions/wind/runs/<tag>_s<seed>.{npz,jsonl}: the checkpoint is numpy (the Gazebo loop runs it without JAX).

Curriculum: each flight's W10 is drawn from [0, W] (half) or the top 3 m/s of it (half); W starts at 4 m/s and
rises 1 m/s whenever 80% of the finished flights in that top band landed (or after 150 rounds stuck), up to 20 m/s.
"""
import argparse
import json
import math
import os
import time

import numpy as np

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import jax
import jax.numpy as jnp

import land_env as E

HERE = os.path.dirname(os.path.abspath(__file__))
KL_TARGET = 0.02                      # 26 Sept: a steady 3e-4 and no KL check let the policy slide after its peak
GAMMA, LAM, CLIP, VF, ENT, LR, MAX_GN = 0.9998, 0.95, 0.2, 0.5, 0.003, 3e-4, 0.5  # a 200 s horizon: waiting 20 s for a
# safe landing costs ~10%, so patience pays as it should (0.997/0.999 made the learner gamble: 22 s, hard landings)
W_START, W_STEP, W_TOP, W_OK, STUCK = 4.0, 1.0, 20.0, 0.8, 150
MIN_ROUNDS, MIN_BAND = 20, 300     # per level: a flight lasts ~15-35 rounds, and the first to finish are the quick landings


# ---- network ------------------------------------------------------------------------------------------------------------------
def mlp_init(key, sizes, last=0.01):
    out = []
    for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:])):
        key, k = jax.random.split(key)
        out.append((jax.nn.initializers.orthogonal(math.sqrt(2) if i < len(sizes) - 2 else last)(k, (a, b)), jnp.zeros(b)))
    return out


def mlp(ps, x):
    for W, b in ps[:-1]:
        x = jnp.tanh(x @ W + b)
    W, b = ps[-1]
    return x @ W + b


def init_params(key):
    k1, k2 = jax.random.split(key)
    pi = mlp_init(k1, [E.OBS_DIM, 256, 256, 7])
    W, b = pi[-1]
    pi[-1] = (W, b.at[2].set(4.0))            # the 'rule' option first (p ~ 0.93): the learner starts as the rule
    return dict(pi=pi, logstd=jnp.full(2, -1.2), v=mlp_init(k2, [E.OBS_DIM, 256, 256, 1], last=1.0))


def heads(p, obs_n):
    out = mlp(p["pi"], obs_n)
    return out[..., :2], p["logstd"], out[..., 2:]


def log_prob(mean, logstd, logits, a_c, a_d, fly, decide):
    lp_c = -0.5 * (((a_c - mean) / jnp.exp(logstd)) ** 2 + 2 * logstd + math.log(2 * math.pi)).sum(-1)
    lp_d = jnp.take_along_axis(jax.nn.log_softmax(logits), a_d[..., None], -1)[..., 0]
    return fly * lp_c + decide * lp_d


def normalize(rms, obs):
    return jnp.clip((obs - rms["mean"]) / jnp.sqrt(rms["var"] + 1e-8), -10, 10)


def rms_update(rms, x):
    bm, bv, bn = x.mean(0), x.var(0), x.shape[0]
    d, tot = bm - rms["mean"], rms["count"] + bn
    mean = rms["mean"] + d * bn / tot
    var = (rms["var"] * rms["count"] + bv * bn + d ** 2 * rms["count"] * bn / tot) / tot
    return dict(mean=mean, var=var, count=tot)


def policy_act(p, rms, obs, key, greedy=False):
    mean, logstd, logits = heads(p, normalize(rms, obs))
    k1, k2 = jax.random.split(key)
    a_c = jnp.where(greedy, mean, mean + jnp.exp(logstd) * jax.random.normal(k1, mean.shape))
    a_d = jnp.where(greedy, jnp.argmax(logits, -1), jax.random.categorical(k2, logits))
    return a_c, a_d


# ---- Adam -------------------------------------------------------------------------------------------------------------------------
def adam_init(p):
    z = jax.tree_util.tree_map(jnp.zeros_like, p)
    return dict(m=z, v=z, t=jnp.array(0))


def adam(p, g, s, lr):
    gn = jnp.sqrt(sum(jnp.sum(x ** 2) for x in jax.tree_util.tree_leaves(g)))
    g = jax.tree_util.tree_map(lambda x: x * jnp.minimum(1.0, MAX_GN / (gn + 1e-6)), g)
    t = s["t"] + 1
    m = jax.tree_util.tree_map(lambda a, b: 0.9 * a + 0.1 * b, s["m"], g)
    v = jax.tree_util.tree_map(lambda a, b: 0.999 * a + 0.001 * b * b, s["v"], g)
    c1, c2 = 1 - 0.9 ** t, 1 - 0.999 ** t
    p = jax.tree_util.tree_map(lambda x, a, b: x - lr * (a / c1) / (jnp.sqrt(b / c2) + 1e-8), p, m, v)
    return p, dict(m=m, v=v, t=t)


# ---- rollout ----------------------------------------------------------------------------------------------------------------------
def draw_w10(key, w_cur):
    r = jax.random.uniform(key, (2,))
    return jnp.where(r[0] < 0.5, r[1] * w_cur, jnp.maximum(w_cur - 3.0, 0.0) + r[1] * jnp.minimum(3.0, w_cur))


def flyer_mean(flyer, obs):
    """A frozen flying learner's feed-forward (stage 2 trains only the deciding on top of it)."""
    fp, frms = flyer
    return heads(fp, normalize(frms, obs))[0]


def make_fns(n_env, t_roll, fly, decide, shield=0.0, flyer=None, hp_tau=0.0):
    fly_f, dec_f = (0.0 if flyer is not None else float(fly)), float(decide)
    vstep = jax.vmap(lambda s, a, k: E.step(s, a, k, fly, decide, shield, hp_tau))
    vreset = jax.vmap(E.reset)
    vobs = jax.vmap(E.obs_of)

    def where_tree(c, a, b):
        return jax.tree_util.tree_map(lambda x, y: jnp.where(c.reshape((-1,) + (1,) * (x.ndim - 1)), x, y), a, b)

    def rollout(p, rms, st, obs, key, w_cur, fl):
        def one(carry, k):
            st, obs = carry
            ks = jax.random.split(k, 5)
            a_c, a_d = policy_act(p, rms, obs, ks[0])
            if flyer is not None:                      # the flyer's weights come in as an argument, not baked-in
                a_c = flyer_mean(fl, obs)              # constants (those broke XLA's GPU fusion, 26 Sept)
            mean, logstd, logits = heads(p, normalize(rms, obs))
            lp = log_prob(mean, logstd, logits, a_c, a_d, fly_f, dec_f)
            val = mlp(p["v"], normalize(rms, obs))[:, 0]
            # built by index, not concatenate: XLA's GPU emitter failed on that fusion with a flyer (26 Sept)
            act = jnp.zeros((n_env, 3)).at[:, :2].set(a_c).at[:, 2].set(a_d.astype(jnp.float32))
            new, r, done = vstep(st, act, jax.random.split(ks[1], n_env))
            w10 = jax.vmap(draw_w10, in_axes=(0, None))(jax.random.split(ks[2], n_env), w_cur)
            fresh = vreset(jax.random.split(ks[3], n_env), w10)
            out = dict(outcome=new["outcome"], w10=new["wind"]["w10"], miss=new["miss"], t=new["t"])
            st2 = where_tree(done, fresh, new)
            obs2 = vobs(st2, jax.random.split(ks[4], n_env))
            return (st2, obs2), dict(obs=obs, a_c=a_c, a_d=a_d, lp=lp, val=val, r=r, done=done, **out)
        (st, obs), tr = jax.lax.scan(one, (st, obs), jax.random.split(key, t_roll))
        last_v = mlp(p["v"], normalize(rms, obs))[:, 0]
        return st, obs, tr, last_v

    def gae(tr, last_v):
        def back(carry, x):
            adv, nv = carry
            r, v, d = x
            delta = r + GAMMA * nv * (1 - d) - v
            adv = delta + GAMMA * LAM * (1 - d) * adv
            return (adv, v), adv
        _, adv = jax.lax.scan(back, (jnp.zeros_like(last_v), last_v), (tr["r"], tr["val"], tr["done"].astype(jnp.float32)),
                              reverse=True)
        return adv, adv + tr["val"]

    def loss(p, rms, b):
        on = normalize(rms, b["obs"])
        mean, logstd, logits = heads(p, on)
        lp = log_prob(mean, logstd, logits, b["a_c"], b["a_d"], fly_f, dec_f)
        ratio = jnp.exp(lp - b["lp"])
        a = (b["adv"] - b["adv"].mean()) / (b["adv"].std() + 1e-8)
        pg = -jnp.minimum(ratio * a, jnp.clip(ratio, 1 - CLIP, 1 + CLIP) * a).mean()
        v = mlp(p["v"], on)[:, 0]
        vc = b["val"] + jnp.clip(v - b["val"], -CLIP * 10, CLIP * 10)
        vl = 0.5 * jnp.maximum((v - b["ret"]) ** 2, (vc - b["ret"]) ** 2).mean()
        ent = -(jax.nn.softmax(logits) * jax.nn.log_softmax(logits)).sum(-1).mean() * dec_f
        kl = ((ratio - 1) - (lp - b["lp"])).mean()
        return pg + VF * vl - ENT * ent, (pg, vl, ent, kl)

    def update(p, opt, rms, tr, adv, ret, key, lr, epochs=4, n_mb=8):
        flat = lambda x: x.reshape((-1,) + x.shape[2:])
        data = dict(obs=flat(tr["obs"]), a_c=flat(tr["a_c"]), a_d=flat(tr["a_d"]), lp=flat(tr["lp"]),
                    val=flat(tr["val"]), adv=flat(adv), ret=flat(ret))
        n = data["obs"].shape[0]
        mb = n // n_mb

        def epoch(carry, k):
            p, opt, stop = carry
            perm = jax.random.permutation(k, n)

            def mbstep(carry, i):
                p, opt, stop = carry
                idx = jax.lax.dynamic_slice(perm, (i * mb,), (mb,))
                b = jax.tree_util.tree_map(lambda x: x[idx], data)
                (l, aux), g = jax.value_and_grad(loss, has_aux=True)(p, rms, b)
                stop = stop | (aux[3] > 1.5 * KL_TARGET)      # moved far enough this round: no more steps
                p2, opt2 = adam(p, g, opt, lr)
                keep = lambda a, b_: jax.tree_util.tree_map(lambda x, y: jnp.where(stop, x, y), a, b_)
                return (keep(p, p2), keep(opt, opt2), stop), jnp.array([l, *aux])
            (p, opt, stop), info = jax.lax.scan(mbstep, (p, opt, stop), jnp.arange(n_mb))
            return (p, opt, stop), info.mean(0)
        (p, opt, stop), info = jax.lax.scan(epoch, (p, opt, jnp.array(False)), jax.random.split(key, epochs))
        return p, opt, jnp.concatenate([info.mean(0), stop[None].astype(jnp.float32)])

    return jax.jit(rollout), jax.jit(gae), jax.jit(update), vreset, vobs


def save(path, p, rms, meta):
    flat = {}
    for i, (W, b) in enumerate(p["pi"]):
        flat[f"pi_W{i}"], flat[f"pi_b{i}"] = np.array(W), np.array(b)
    for i, (W, b) in enumerate(p["v"]):
        flat[f"v_W{i}"], flat[f"v_b{i}"] = np.array(W), np.array(b)
    np.savez(path, logstd=np.array(p["logstd"]), rms_mean=np.array(rms["mean"]), rms_var=np.array(rms["var"]),
             meta=json.dumps(meta), **flat)


def load(path):
    # a stage-2 checkpoint names its flyer by absolute path; the 5 Oct reorganization moved wind/runs to conditions/wind/runs
    z = np.load(path if os.path.exists(path) else os.path.join(HERE, "runs", os.path.basename(path)))
    n = len([k for k in z.files if k.startswith("pi_W")])
    m = len([k for k in z.files if k.startswith("v_W")])
    p = dict(pi=[(jnp.array(z[f"pi_W{i}"]), jnp.array(z[f"pi_b{i}"])) for i in range(n)], logstd=jnp.array(z["logstd"]),
             v=[(jnp.array(z[f"v_W{i}"]), jnp.array(z[f"v_b{i}"])) for i in range(m)])
    return p, dict(mean=jnp.array(z["rms_mean"]), var=jnp.array(z["rms_var"]), count=1.0), json.loads(str(z["meta"]))


def greedy_policy(p, rms, flyer=None):
    def f(obs, st):
        a_c, a_d = policy_act(p, rms, obs, jax.random.PRNGKey(0), greedy=True)
        if flyer is not None:
            a_c = flyer_mean(flyer, obs)
        return jnp.zeros((obs.shape[0], 3)).at[:, :2].set(a_c).at[:, 2].set(a_d.astype(jnp.float32))
    return f


def load_policy(path):
    """A checkpoint as (greedy policy, fly, decide, shield); stage-2 checkpoints bring their flyer."""
    p, rms, meta = load(path)
    flyer = None
    if meta.get("flyer"):
        fp, frms, _ = load(meta["flyer"])
        flyer = (fp, frms)
    return greedy_policy(p, rms, flyer), meta["fly"] or flyer is not None, meta["decide"], meta.get("shield", 0.0)


def hp_of(path):
    """The high-pass time constant a checkpoint was trained with (a stage-2 checkpoint: its flyer's)."""
    meta = load(path)[2]
    return meta.get("hp_tau", 0.0) or (hp_of(meta["flyer"]) if meta.get("flyer") else 0.0)


PPO_NAMES = ["total loss", "policy loss", "value loss", "entropy (deciding)", "approx KL", "stopped early (KL)"]   # update()'s vector


def clearml_log(task, row, n_dec, dt):
    """Live curves for one round (rows are the jsonl rows; speed = decisions/s of rollout+update). Never stops a run."""
    try:
        lg, it = task.get_logger(), row["it"]
        pts = [("outcomes (training flights)", k, row[k]) for k in ("landed", "crashed", "hard", "timeout")]
        pts += [("return", "mean episode return", row["ret"]), ("curriculum", "W (wind m/s)", row["w_cur"]), ("curriculum", "band landed", row["band_landed"]),
                ("speed", "decisions/s", n_dec / dt)]
        pts += [("PPO", n, v) for n, v in zip(PPO_NAMES, row["loss"])]
        pts += [("policy std", n, v) for n, v in zip(("x", "y"), row["std"])]
        if "val" in row:
            pts += [("validation landed", f"W{w}", v) for w, v in zip((9, 12, 15, 18), row["val"])]
        pts += [("test-yard landed", f"W{e['w10']:.0f}", e["landed"]) for e in row.get("eval", [])]
        for t, sname, v in pts:
            if v == v:                                           # NaN: nothing finished yet
                lg.report_scalar(t, sname, float(v), it)
    except Exception as e:
        print("clearml: logging stopped:", e, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="both")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-fly", action="store_true")
    ap.add_argument("--no-decide", action="store_true")
    ap.add_argument("--iters", type=int, default=1500)
    ap.add_argument("--envs", type=int, default=2048)
    ap.add_argument("--steps", type=int, default=64)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--w-start", type=float, default=W_START, help="20 = no curriculum: the full range of wind from the start")
    ap.add_argument("--w-ok", type=float, default=W_OK, help="landed share in the top band before the wind rises")
    ap.add_argument("--shield", type=float, default=0.0, help="k > 0: no descent below 6 m beyond k x the rule's limits")
    ap.add_argument("--flyer", help="stage 2: a flying-only checkpoint, frozen; only the deciding is trained")
    ap.add_argument("--lr", type=float, default=LR, help="starting learning rate (annealed to a tenth)")
    ap.add_argument("--hp-tau", type=float, default=0.0, help="s > 0: the push's slow part removed (PX4's integral cancels it)")
    ap.add_argument("--crash-r", type=float, default=E.CRASH_R, help="reward for a crash (a lost drone)")
    ap.add_argument("--hard-r", type=float, default=E.HARD_R, help="reward for a hard / off-pad touchdown")
    ap.add_argument("--flip-r", type=float, default=E.FLIP_R, help="price of each climb<->descend reversal (land_env.FLIP_R)")
    ap.add_argument("--shape-g", type=float, default=E.SHAPE_G, help="the shaping's discount (land_env.SHAPE_G); GAMMA makes it exact")
    ap.add_argument("--log-test", action="store_true", help="also score the TEST yards at every eval (logging only, ~55%% of eval time)")
    ap.add_argument("--ladder", metavar="CKPT", help="score a checkpoint (or 'rule') on the full wind ladder, 1024 paired flights")
    ap.add_argument("--no-clearml", action="store_true", help="no live logging to ClearML (rl-landing/wind)")
    a = ap.parse_args()
    if a.ladder:
        if a.ladder == "rule":
            pol, fly, decide = (lambda obs, st: jnp.zeros((obs.shape[0], 3))), False, False
        else:
            pol, fly, decide, sh = load_policy(a.ladder)
        sh = 0.0 if a.ladder == "rule" else sh
        hp = 0.0 if a.ladder == "rule" else hp_of(a.ladder)
        res = [E.evaluate(pol, float(w), n=1024, seed=10_000, fly=fly, decide=decide, shield=sh, hp_tau=hp) for w in (0, 3, 6, 9, 12, 15, 18, 20)]
        out = a.ladder[:-4] + "_ladder.json" if a.ladder != "rule" else f"{HERE}/runs/rule_ladder.json"
        json.dump(res, open(out, "w"), indent=1)
        for r in res:
            print(json.dumps(r))
        return
    fly, decide = not a.no_fly, not a.no_decide
    task = None
    if not a.no_clearml:
        try:
            from clearml import Task
            task = Task.init(project_name="rl-landing/wind", task_name=f"{a.tag}_s{a.seed}", auto_connect_frameworks=False,
                             tags=[n for n, on in (("fly", fly), ("decide", decide), ("flyer", bool(a.flyer))) if on])
            task.connect(vars(a))
            print("ClearML:", task.get_output_log_web_page(), flush=True)
        except Exception as e:
            task = None
            print("clearml unavailable, training without it:", e, flush=True)
    E.CRASH_R, E.HARD_R, E.FLIP_R, E.SHAPE_G = a.crash_r, a.hard_r, a.flip_r, a.shape_g    # read by land_env.step when traced
    flyer = None
    if a.flyer:
        fp, frms, _ = load(a.flyer)
        flyer, fly = (fp, frms), True
    os.makedirs(f"{HERE}/runs", exist_ok=True)
    base = f"{HERE}/runs/{a.tag}_s{a.seed}"
    log = open(base + ".jsonl", "w")
    hp = a.hp_tau or (hp_of(a.flyer) if a.flyer else 0.0)
    rollout, gae, update, vreset, vobs = make_fns(a.envs, a.steps, fly, decide, a.shield, flyer, hp)
    key = jax.random.PRNGKey(a.seed)
    key, k1, k2, k3 = jax.random.split(key, 4)
    p = init_params(k1)
    opt = adam_init(p)
    w_cur, stuck, band_oc, best_val = a.w_start, 0, [], -1.0
    w10 = jax.vmap(draw_w10, in_axes=(0, None))(jax.random.split(k2, a.envs), w_cur)
    st = vreset(jax.random.split(k3, a.envs), w10)
    key, k = jax.random.split(key)
    obs = vobs(st, jax.random.split(k, a.envs))
    rms = dict(mean=jnp.zeros(E.OBS_DIM), var=jnp.ones(E.OBS_DIM), count=1e-4)
    rms = rms_update(rms, obs)
    t0 = time.time()
    for it in range(1, a.iters + 1):
        t_round = time.time()
        key, kr, ku = jax.random.split(key, 3)
        st, obs, tr, last_v = rollout(p, rms, st, obs, kr, jnp.float32(w_cur), flyer if flyer is not None else 0.0)
        adv, ret = gae(tr, last_v)
        lr = jnp.float32(a.lr * max(0.1, 1.0 - (it - 1) / a.iters))     # annealed to a tenth
        p, opt, info = update(p, opt, rms, tr, adv, ret, ku, lr)
        rms = rms_update(rms, tr["obs"].reshape(-1, E.OBS_DIM))
        done = np.array(tr["done"]).ravel()
        oc, wv = np.array(tr["outcome"]).ravel()[done], np.array(tr["w10"]).ravel()[done]
        band = wv >= w_cur - 3.0
        band_oc.extend(oc[band].tolist())                     # this level's finished flights in the top band
        succ = float(np.mean(np.array(band_oc) == 1)) if len(band_oc) >= MIN_BAND else float("nan")
        row = dict(it=it, w_cur=w_cur, n_ep=int(done.sum()), band_n=len(band_oc), band_landed=succ,
                   landed=float((oc == 1).mean()) if len(oc) else float("nan"), crashed=float((oc == 2).mean()) if len(oc) else float("nan"),
                   hard=float((oc == 3).mean()) if len(oc) else float("nan"), timeout=float((oc == 4).mean()) if len(oc) else float("nan"),
                   ret=float(np.array(tr["r"]).sum() / max(1, done.sum())), loss=[float(x) for x in info],
                   std=[float(x) for x in np.exp(np.array(p["logstd"]))], wall=round(time.time() - t0, 1))
        t_round = time.time() - t_round
        stuck += 1
        if stuck >= MIN_ROUNDS and ((succ == succ and succ >= a.w_ok) or stuck >= STUCK) and w_cur < W_TOP:
            row["advance"] = "landed" if succ == succ and succ >= a.w_ok else "stuck"
            w_cur, stuck, band_oc = min(W_TOP, w_cur + W_STEP), 0, []
        if it % a.eval_every == 0 or it == a.iters:
            pol = greedy_policy(p, rms, flyer)
            if a.log_test:                                    # the test yards: logged, never used to choose
                row["eval"] = [E.evaluate(pol, float(w), n=512, seed=10_000, fly=fly, decide=decide, shield=a.shield, hp_tau=hp) for w in (0, 9, 12, 15, 18)]
            # choose on VALIDATION yards (seed 20000), never on the test yards above
            row["val"] = [E.evaluate(pol, float(w), n=512, seed=20_000, fly=fly, decide=decide, shield=a.shield, hp_tau=hp)["landed"] for w in (9, 12, 15, 18)]
            meta = dict(tag=a.tag, seed=a.seed, it=it, w_cur=w_cur, fly=fly, decide=decide, shield=a.shield, val=row["val"], hp_tau=a.hp_tau,
                        crash_r=a.crash_r, hard_r=a.hard_r, flip_r=a.flip_r, shape_g=a.shape_g,
                        flyer=os.path.abspath(a.flyer) if a.flyer else None)
            save(base + ".npz", p, rms, meta)
            save(f"{base}_it{it}.npz", p, rms, meta)
            if float(np.mean(row["val"])) > best_val:
                best_val = float(np.mean(row["val"]))
                save(base + "_best.npz", p, rms, meta)
                row["best"] = True
        log.write(json.dumps(row) + "\n"); log.flush()
        if task:
            clearml_log(task, row, a.envs * a.steps, t_round)
        if it % 10 == 0 or "val" in row:
            print(f"it {it} W {w_cur:.0f} band {succ:.2f} landed {row['landed']:.2f} crash {row['crashed']:.3f} "
                  f"timeout {row['timeout']:.2f} ret {row['ret']:.1f} std {row['std'][0]:.2f} {row['wall']:.0f}s"
                  + (" eval " + " ".join(f"{e['w10']:.0f}:{e['landed']:.2f}" for e in row.get("eval", [])) + " val " + " ".join(f"{v:.2f}" for v in row["val"])
                     + (" BEST" if row.get("best") else "") if "val" in row else ""), flush=True)
    if task and os.path.exists(base + "_best.npz"):
        task.upload_artifact("best checkpoint", base + "_best.npz")


if __name__ == "__main__":
    main()
