#!/usr/bin/env python3
"""
render.py -- see the wind environment: the landing rule and a trained learner fly the SAME yard in the SAME gusts, side by
side (left rule, right learner), one MP4 per wind speed, real time (25 frames/s = one frame per 0.04 s decision).

    python wind/render.py                                   W10 9, 15, 18 m/s, the default decider, wind/videos/*.mp4
    python wind/render.py --w10 18 --episode 7 --frames /tmp/f   one clip on yard 7, plus 3 PNG stills per clip to look at
    python wind/render.py --clearml                         also upload the clips to ClearML (rl-landing/wind)

Both fly through land_env.step with identical reset keys and identical per-step keys, so the yard, the start, the gust
sequence and the sensor noise are the same; only the policy differs. The drawing is MuJoCo's offscreen renderer (EGL) over an
MJCF built for display only: yard (pad with its 0.75 m landing circle, house, tree), the drone (scaled up with distance so it
stays visible), its trail, and an arrow for the wind AT the drone (mean + turbulence + gust; length ~ speed, so gusts show).
Episode i is evaluate()'s yard i for the same seed (ppo.py scores seed 10000). --episode -1 picks, among the first 20 yards
with a tree, the one where the rule and the learner end differently at the highest W10 (the rule times out, the learner lands).
"""
import argparse
import math
import os

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import imageio
import jax
import jax.numpy as jnp
import matplotlib
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import land_env as E
import ppo
import x500_mjx as X

HERE = os.path.dirname(os.path.abspath(__file__))
FPS, HOLD = 25, 50                  # one frame per decision; 2 s on the final frame
PW, PH, HEAD, FOOT = 640, 600, 44, 76
OUTCOME = {1: ("LANDED", (60, 200, 90)), 2: ("CRASHED", (235, 70, 60)), 3: ("HARD / OFF PAD", (240, 160, 40)), 4: ("TIMEOUT", (200, 200, 200))}
FONT = os.path.join(matplotlib.get_data_path(), "fonts/ttf/DejaVuSans.ttf")


# ---- fly both policies on the same yards, recording every decision --------------------------------------------------------------
N_ALL = 1024        # evaluate()'s batch: the same keys (so episode i IS its flight i), and the only size XLA's GPU emitter compiles
                    # (any smaller batch fails in a concatenate fusion, 4 Oct)


def fly(pol, fly_, decide, shield, hp, w10, m, seed):
    """evaluate()'s flights for this seed, the first m recorded: per-step records, shape (T, m, ...)."""
    n = N_ALL
    st = jax.vmap(lambda k: E.reset(k, w10))(jax.random.split(jax.random.PRNGKey(seed), n))

    def one(st, k):
        ks = jax.random.split(k, n * 2).reshape(n, 2, 2)
        act = pol(jax.vmap(E.obs_of)(st, ks[:, 0]), st)
        # the wind the drone feels this step: the same call step() makes (deterministic in state + key)
        wind = jax.vmap(lambda s, kk: E.wind_step(s["wind"], s["sim"]["d"].qpos[2], jax.random.split(kk, 3)[1])[1])(st, ks[:, 1])
        new, _, _ = jax.vmap(lambda s, a, kk: E.step(s, a, kk, fly_, decide, shield, hp))(st, act, ks[:, 1])
        st = jax.tree_util.tree_map(lambda a, b: jnp.where(st["done"].reshape((-1,) + (1,) * (a.ndim - 1)), a, b), st, new)
        d = st["sim"]["d"]
        return st, jax.tree_util.tree_map(lambda x: x[:m], dict(p=d.qpos[:, :3], q=d.qpos[:, 3:7], wind=wind, t=st["t"], outcome=st["outcome"], done=st["done"]))

    n_steps = int(E.T_MAX / E.DT_C) + 1
    ks = jax.random.split(jax.random.PRNGKey(seed + 1), n_steps)
    rec = jax.jit(lambda s, k: jax.lax.scan(one, s, k)[1])(st, ks)
    return jax.tree_util.tree_map(np.array, rec)


def yards_of(seed, m):
    """The first m yards (a yard does not depend on W10)."""
    st = jax.vmap(lambda k: E.reset(k, 0.0))(jax.random.split(jax.random.PRNGKey(seed), N_ALL)[:m])
    return jax.tree_util.tree_map(np.array, st["yard"])


def episode(rec, i):
    """Flight i cut at its first done step (the state is frozen after it)."""
    d = rec["done"][:, i]
    L = int(d.argmax()) + 1 if d.any() else len(d)
    return {k: v[:L, i] for k, v in rec.items()}


# ---- the picture ----------------------------------------------------------------------------------------------------------------
def yard_xml(y):
    hx, hy = y["hc"]
    cz = float(y["ttop"] - y["tr"])
    tree = ""
    if y["tree"]:
        tx, ty = y["tc"]
        tree = (f'<geom type="cylinder" fromto="{tx} {ty} 0 {tx} {ty} {cz}" size="0.2" rgba=".42 .28 .15 1"/>'
                f'<geom type="sphere" pos="{tx} {ty} {cz}" size="{y["tr"]}" rgba=".15 .55 .2 .55"/>')
    return f"""<mujoco>
  <visual><global offwidth="{PW}" offheight="{PH}"/><headlight ambient=".5 .5 .5" diffuse=".5 .5 .5"/><map znear="0.02"/></visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1=".42 .62 .9" rgb2=".86 .92 .98" width="256" height="256"/>
    <texture name="g" type="2d" builtin="checker" rgb1=".40 .58 .33" rgb2=".34 .52 .28" width="512" height="512"/>
    <material name="g" texture="g" texuniform="true" texrepeat="0.5 0.5"/>
  </asset>
  <default><geom contype="0" conaffinity="0"/></default>
  <worldbody>
    <light pos="20 -30 60" dir="-.3 .5 -1" diffuse=".6 .6 .6" castshadow="false"/>
    <geom type="plane" size="300 300 .1" material="g"/>
    <geom type="cylinder" pos="0 0 .004" size=".80 .004" rgba=".85 .15 .12 1"/>
    <geom type="cylinder" pos="0 0 .009" size=".70 .004" rgba=".97 .97 .97 1"/>
    <geom type="cylinder" pos="0 0 .014" size=".08 .004" rgba=".85 .15 .12 1"/>
    <geom type="box" pos="{hx} {hy} {y["hh"] / 2}" euler="0 0 {math.degrees(y["hang"])}" size="4 5 {y["hh"] / 2}" rgba=".88 .82 .70 1"/>
    <geom type="box" pos="{hx} {hy} {y["hh"] + .15}" euler="0 0 {math.degrees(y["hang"])}" size="4.4 5.4 .15" rgba=".55 .22 .18 1"/>
    {tree}
  </worldbody>
</mujoco>"""


def add(scn, gtype, size, pos, mat, rgba):
    g = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(g, gtype, np.asarray(size, float), np.asarray(pos, float), np.asarray(mat, float).ravel(), np.asarray(rgba, np.float32))
    scn.ngeom += 1


def conn(scn, gtype, width, a, b, rgba):
    g = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(g, gtype, np.zeros(3), np.zeros(3), np.zeros(9), np.asarray(rgba, np.float32))
    mujoco.mjv_connector(g, gtype, width, np.asarray(a, float), np.asarray(b, float))
    scn.ngeom += 1


def drone(scn, pos, q, s):
    """The x500 drawn s times life size: body, two arms, four propeller discs (front pair red = nose)."""
    R = np.array(X.qrot(jnp.asarray(q)))
    I = np.eye(3).ravel()
    add(scn, mujoco.mjtGeom.mjGEOM_BOX, [.11 * s, .11 * s, .04 * s], pos, R, [.15, .15, .17, 1])
    for a, b in ((0, 1), (2, 3)):
        conn(scn, mujoco.mjtGeom.mjGEOM_CAPSULE, .012 * s, pos + R @ (s * np.append(X.ROTORS[a], X.ROTOR_Z)),
             pos + R @ (s * np.append(X.ROTORS[b], X.ROTOR_Z)), [.2, .2, .22, 1])
    for r in X.ROTORS:
        add(scn, mujoco.mjtGeom.mjGEOM_CYLINDER, [.12 * s, .004 * s, 0], pos + R @ (s * np.append(r, X.ROTOR_Z)), R,
            [.9, .15, .1, .8] if r[0] > 0 else [.9, .9, .9, .7])


def camera(tr):
    """Per-frame (lookat, distance): the midpoint of pad and drone, far enough to hold both, smoothed (~0.5 s)."""
    p = tr["p"]
    tgt = np.c_[p[:, 0] / 2, p[:, 1] / 2, p[:, 2] / 2]
    dist = np.maximum(12.0, 1.6 * np.linalg.norm(p, axis=1) + 4.0)
    la, di = np.empty_like(tgt), np.empty_like(dist)
    la[0], di[0] = tgt[0], dist[0]
    for i in range(1, len(p)):
        la[i], di[i] = la[i - 1] + 0.08 * (tgt[i] - la[i - 1]), di[i - 1] + 0.08 * (dist[i] - di[i - 1])
    return la, di


def view_azimuth(y):
    """Camera side: as far as possible (in angle) from both the house and the tree, so neither hides the pad."""
    cands = np.linspace(0, 2 * math.pi, 72, endpoint=False)
    ad = lambda a, b: np.abs((a - b + math.pi) % (2 * math.pi) - math.pi)
    sc = ad(cands, float(y["hang"]))
    if y["tree"]:
        sc = np.minimum(sc, ad(cands, math.atan2(y["tc"][1], y["tc"][0])))
    return math.degrees(cands[sc.argmax()]) + 180.0        # MuJoCo azimuth: the direction the camera looks along


def text(img, xy, s, size, fill):
    ImageDraw.Draw(img).text(xy, s, font=ImageFont.truetype(FONT, size), fill=fill)


def panel(rend, data, az, tr, cam, i, label, color, trail_color):
    la, di = cam
    p, q, w = tr["p"][i], tr["q"][i], tr["wind"][i]
    c = mujoco.MjvCamera()
    c.type, c.lookat, c.distance, c.azimuth, c.elevation = mujoco.mjtCamera.mjCAMERA_FREE, la[i], di[i], az, -32.0
    rend.update_scene(data, c)
    scn, s, D = rend.scene, max(1.0, di[i] / 12.0), di[i]
    sub = tr["p"][: i + 1 : 5]                                              # the trail, a point every 0.2 s
    for a, b in zip(sub[:-1], sub[1:]):
        conn(scn, mujoco.mjtGeom.mjGEOM_CAPSULE, .004 * D, a, b, trail_color)
    conn(scn, mujoco.mjtGeom.mjGEOM_CAPSULE, .002 * D, p, [p[0], p[1], 0.02], [1, 1, 1, .5])         # drop line to its ground spot
    add(scn, mujoco.mjtGeom.mjGEOM_CYLINDER, [.3 * s, .003, 0], [p[0], p[1], .02], np.eye(3), [0, 0, 0, .35])    # shadow
    drone(scn, p, q, s)
    spd = float(np.linalg.norm(w))
    if spd > 0.1:
        conn(scn, mujoco.mjtGeom.mjGEOM_ARROW, .006 * D, p, p + w / spd * .012 * D * spd, [.1, .6, 1, 1])
    img = Image.fromarray(rend.render().copy())
    h = float(p[2])
    lean = math.degrees(math.acos(np.clip(X.qrot(jnp.asarray(q))[2, 2], -1, 1)))
    text(img, (10, 6), label, 26, color)
    text(img, (10, 40), f"t {tr['t'][i]:5.1f} s    height {h:5.1f} m    miss {np.linalg.norm(p[:2]):5.1f} m", 17, (255, 255, 255))
    text(img, (10, 64), f"wind {spd:4.1f} m/s    lean {lean:4.1f} deg", 17, (120, 205, 255))
    if tr["done"][i]:
        name, col = OUTCOME[int(tr["outcome"][i])]
        text(img, (10, PH - 40), name, 30, col)
    return np.array(img)


def check_yard(m, data, y):
    """The picture agrees with land_env: rays from just outside the pad, house and canopy hit them where clearance() says."""
    gid = np.zeros(1, np.int32)
    ray = lambda p, v: mujoco.mj_ray(m, data, np.asarray(p, float), np.asarray(v, float), None, 1, -1, gid)
    assert abs(ray([.4, 0, 1], [0, 0, -1]) - (1 - .013)) < 1e-3, "pad is not at the origin"
    u = np.array([math.cos(y["hang"]), math.sin(y["hang"])])
    p = np.append(y["hc"] - 4.5 * u, y["hh"] / 2)                       # 0.5 m in front of the house's near wall
    assert abs(ray(p, np.append(u, 0)) - .5) < 1e-3 and abs(float(E.clearance(y, jnp.asarray(p))[0]) - .5) < 1e-3, "house misplaced"
    if y["tree"]:
        p = np.array([y["tc"][0] + y["tr"] + 1, y["tc"][1], y["ttop"] - y["tr"]])        # 1 m outside the canopy's equator
        assert abs(ray(p, [-1, 0, 0]) - 1) < 1e-3 and abs(float(E.clearance(y, jnp.asarray(p))[1]) - 1) < 1e-3, "tree misplaced"


def clip(path, w10, yard, ep, rule_tr, learn_tr, labels, frames_dir=None):
    az = view_azimuth(yard)
    m = mujoco.MjModel.from_xml_string(yard_xml(yard))
    rend = mujoco.Renderer(m, PH, PW)
    data = mujoco.MjData(m)
    mujoco.mj_forward(m, data)
    check_yard(m, data, yard)
    n = max(len(rule_tr["t"]), len(learn_tr["t"])) + HOLD
    pad = lambda tr: {k: np.concatenate([v, np.repeat(v[-1:], n - len(v), 0)]) for k, v in tr.items()}
    a, b = pad(rule_tr), pad(learn_tr)
    ca, cb = camera(a), camera(b)
    head = f"same yard, same gusts: house{' + tree' if yard['tree'] else ''}   W10 = {w10:g} m/s   yard {ep}"
    foot = ["ring = 0.75 m landing circle      ground squares = 1 m      blue arrow = wind at the drone (blows toward the head, length ~ speed)      orange/green line = path",
            "drone drawn larger when far away; one frame = one 0.04 s decision, real time; the path is every 0.2 s"]
    with imageio.get_writer(path, fps=FPS, codec="libx264", quality=7, pixelformat="yuv420p", macro_block_size=None) as wr:
        for i in range(n):
            canvas = Image.new("RGB", (2 * PW, HEAD + PH + FOOT), (24, 24, 28))
            canvas.paste(Image.fromarray(panel(rend, data, az, a, ca, i, labels[0], (255, 190, 90), [1, .6, .1, 1])), (0, HEAD))
            canvas.paste(Image.fromarray(panel(rend, data, az, b, cb, i, labels[1], (130, 235, 140), [.2, .9, .3, 1])), (PW, HEAD))
            text(canvas, (10, 8), head, 22, (255, 255, 255))
            for j, f in enumerate(foot):
                text(canvas, (10, HEAD + PH + 8 + 26 * j), f, 15, (200, 200, 205))
            fr = np.array(canvas)
            wr.append_data(fr)
            if frames_dir and i in (0, n // 2, n - 1):
                os.makedirs(frames_dir, exist_ok=True)
                imageio.imwrite(f"{frames_dir}/{os.path.basename(path)[:-4]}_{i:04d}.png", fr)
    rend.close()
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=f"{HERE}/runs/stage2_c100h100g_s0_best.npz", help="the learner; a stage-2 checkpoint brings its own flyer")
    ap.add_argument("--w10", type=float, nargs="+", default=[9, 15, 18])
    ap.add_argument("--seed", type=int, default=10_000, help="ppo.py scores seed 10000: episode i is its yard i")
    ap.add_argument("--episode", type=int, default=-1, help="-1: the first of 20 yards with a tree where rule and learner differ at the top W10")
    ap.add_argument("--out", default=f"{HERE}/videos")
    ap.add_argument("--frames", help="also write 3 stills per clip (start, middle, end) to this folder")
    ap.add_argument("--clearml", action="store_true", help="upload the clips (project rl-landing/wind)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    pol, fly_l, dec_l, sh = ppo.load_policy(a.ckpt)
    hp = ppo.hp_of(a.ckpt)
    rule = (lambda obs, st: jnp.zeros((obs.shape[0], 3)), False, False, 0.0, 0.0)
    learner = (pol, fly_l, dec_l, sh, hp)
    n = max(20, a.episode + 1)
    runs = {w: [fly(*pl, float(w), n, a.seed) for pl in (rule, learner)] for w in a.w10}
    yards = yards_of(a.seed, n)
    ep = a.episode
    if ep < 0:
        r, l = (x["outcome"][-1] for x in runs[max(a.w10)])     # frozen after done: the last record is the final outcome
        has = yards["tree"][:20]
        pick = [i for i in range(20) if has[i] and r[i] != l[i]]
        pick.sort(key=lambda i: (not (r[i] == 4 and l[i] == 1), not (l[i] == 1), i))
        print("top-W10 outcomes (rule/learner, 1 landed 2 crashed 3 hard 4 timeout), * = tree:",
              " ".join(f"{i}:{r[i]}/{l[i]}{'*' if has[i] else ''}" for i in range(20)))
        ep = pick[0] if pick else int(np.argmax(has))
        print("chosen yard", ep, "(outcomes differ)" if pick else "(NO differing yard with a tree in the first 20: first tree yard)")
    yard = {k: v[ep] for k, v in yards.items()}
    lab = ("RULE", "LEARNER (" + os.path.basename(a.ckpt)[:-4] + ")")
    paths = []
    for w in a.w10:
        tr = [episode(x, ep) for x in runs[w]]
        path = f"{a.out}/wind_w{w:g}_yard{ep}.mp4"
        frames = clip(path, w, yard, ep, tr[0], tr[1], lab, a.frames)
        res = " | ".join(f"{nm}: {OUTCOME[int(t['outcome'][-1])][0]} at {t['t'][-1]:.1f} s, miss {np.linalg.norm(t['p'][-1, :2]):.2f} m" for nm, t in zip(("rule", "learner"), tr))
        print(f"W10 {w:g}: {res}  -> {path} ({frames} frames)", flush=True)
        paths.append((w, path, res))
    if a.clearml:
        from clearml import Task
        task = Task.init(project_name="rl-landing/wind", task_name="environment video: rule vs learner", reuse_last_task_id=False)
        lg = task.get_logger()
        for w, path, res in paths:
            lg.report_media("environment video", f"W10 {w:g} m/s, yard {ep}", iteration=int(w), local_path=path)
            lg.report_text(f"W10 {w:g} m/s, yard {ep}: {res}")
        print("ClearML:", task.get_output_log_web_page())
        task.close()


if __name__ == "__main__":
    main()
