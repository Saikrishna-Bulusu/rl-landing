#!/usr/bin/env python3
"""
render_worlds.py -- one decider, one yard, three worlds side by side (Sai, 7 Oct: show the difference between Gazebo,
the fast simulator's nominal world and its realistic world):
  left    PX4 + Gazebo: the logged ladder flight replayed (path and tilt as logged; Gazebo's gusts were not logged, so
          its arrow is the mean wind and the lean is drawn into it)
  middle  fast simulator, nominal world: the Gazebo x500, exact sensing
  right   fast simulator, realistic world: this flight's own drone (drag, mass, thrust, delay), sensing errors shaped
          like Gazebo's, a turning wind; the faint ring is where the drone THINKS the pad is
Same yard, start and wind speed in all three (yard i = ladder yard i of seed 10000; starts agree within 0.7 m).
    python conditions/wind/render_worlds.py --w10 18      -> outputs/wind/videos/worlds_w18_yard<e>.{mp4,png} (+ --clearml)
"""
import argparse
import csv
import json
import math
import os

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import imageio
import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np
from PIL import Image

import land_env as E
import ppo
import render as R
import x500_mjx as X

HERE = os.path.dirname(os.path.abspath(__file__))
G = os.path.join(HERE, "..", "..", "outputs", "gazebo_wind", "ladder_rule")
OC = {"landed": 1, "crashed": 2, "hard": 3, "timeout": 4}
RAND_ALL = ("sense", "drone", "gust")


def fly(pol, fly_, dec, sh, hp, w10, m, seed, rand):
    """render.fly plus the pad-estimate error (the estimate the decider saw, minus the truth) and each flight's drone."""
    E.RAND = frozenset(rand)
    n = R.N_ALL
    st = jax.vmap(lambda k: E.reset(k, w10))(jax.random.split(jax.random.PRNGKey(seed), n))

    def one(st, k):
        ks = jax.random.split(k, n * 2).reshape(n, 2, 2)
        err = jax.vmap(E.estimates)(st, ks[:, 0])[0][:, :2] - st["sim"]["d"].qpos[:, :2]
        act = pol(jax.vmap(E.obs_of)(st, ks[:, 0]), st)
        wind = jax.vmap(lambda s, kk: E.wind_step(s["wind"], s["sim"]["d"].qpos[2], jax.random.split(kk, 3)[1])[1])(st, ks[:, 1])
        new, _, _ = jax.vmap(lambda s, a, kk: E.step(s, a, kk, fly_, dec, sh, hp))(st, act, ks[:, 1])
        st = jax.tree_util.tree_map(lambda a, b: jnp.where(st["done"].reshape((-1,) + (1,) * (a.ndim - 1)), a, b), st, new)
        d = st["sim"]["d"]
        return st, jax.tree_util.tree_map(lambda x: x[:m], dict(p=d.qpos[:, :3], q=d.qpos[:, 3:7], wind=wind, t=st["t"],
                                                               outcome=st["outcome"], done=st["done"], err=err))
    ks = jax.random.split(jax.random.PRNGKey(seed + 1), int(E.T_MAX / E.DT_C) + 1)
    rec = jax.tree_util.tree_map(np.array, jax.jit(lambda s, k: jax.lax.scan(one, s, k)[1])(st, ks))
    prm = np.array(st["sim"]["prm"][:m]) if "drone" in rand else None
    E.RAND = frozenset()
    return rec, prm


def quat(axis, ang):
    a = np.asarray(axis, float)
    a = a / (np.linalg.norm(a) + 1e-12)
    return np.r_[math.cos(ang / 2), math.sin(ang / 2) * a]


def qmul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def gazebo(e, w10, name, ref):
    """The logged Gazebo flight on the fast simulator's 0.04 s frames; lean drawn into the mean wind of the nominal run."""
    rows = list(csv.DictReader(open(f"{G}/e{e}_w{w10:g}_{name}.csv")))
    t = np.array([float(r["t"]) for r in rows])
    xyz = np.array([[float(r[k]) for k in "xyz"] for r in rows])
    tilt = np.radians([float(r["tilt"]) for r in rows])
    res = next(d for d in map(json.loads, open(f"{G}/flights.jsonl")) if d["episode"] == e and d["w10"] == w10 and d["decider"] == name)
    tt = np.arange(0.0, t[-1] + 1e-9, E.DT_C)
    p = np.stack([np.interp(tt, t, xyz[:, j]) for j in range(3)], 1)
    w = ref["wind"].mean(0)
    w[2] = 0.0
    q0 = ref["q"][0]
    yaw = quat([0, 0, 1], 2 * math.atan2(q0[3], q0[0]))
    axis = [w[1], -w[0], 0.0]                                  # tips the body z axis toward -w: leaning into the wind
    q = np.array([qmul(quat(axis, a), yaw) for a in np.interp(tt, t, tilt)])
    L = len(tt)
    return dict(p=p, q=q, wind=np.repeat(w[None], L, 0), t=tt, outcome=np.r_[np.zeros(L - 1, int), OC[res["outcome"]]],
                done=np.r_[np.zeros(L - 1, bool), True], err=np.zeros((L, 2))), res


def panel(rend, data, az, tr, cam, i, label, color, trail, ghost, lines):
    """render.panel plus a faint ring where the drone thinks the pad is (ghost) and extra text lines."""
    la, di = cam
    p, q, w = tr["p"][i], tr["q"][i], tr["wind"][i]
    c = mujoco.MjvCamera()
    c.type, c.lookat, c.distance, c.azimuth, c.elevation = mujoco.mjtCamera.mjCAMERA_FREE, la[i], di[i], az, -32.0
    rend.update_scene(data, c)
    scn, s, D = rend.scene, max(1.0, di[i] / 12.0), di[i]
    sub = tr["p"][: i + 1: 5]
    for a, b in zip(sub[:-1], sub[1:]):
        R.conn(scn, mujoco.mjtGeom.mjGEOM_CAPSULE, .004 * D, a, b, trail)
    R.conn(scn, mujoco.mjtGeom.mjGEOM_CAPSULE, .002 * D, p, [p[0], p[1], 0.02], [1, 1, 1, .5])
    R.add(scn, mujoco.mjtGeom.mjGEOM_CYLINDER, [.3 * s, .003, 0], [p[0], p[1], .02], np.eye(3), [0, 0, 0, .35])
    if ghost:
        g = -tr["err"][i]                       # the pad where the drone's estimate puts it: true pad (0, 0) minus the error
        R.add(scn, mujoco.mjtGeom.mjGEOM_CYLINDER, [.75, .006, 0], [g[0], g[1], .03], np.eye(3), [1, .85, .1, .45])
    R.drone(scn, p, q, s)
    spd = float(np.linalg.norm(w))
    if spd > 0.1:
        R.conn(scn, mujoco.mjtGeom.mjGEOM_ARROW, .006 * D, p, p + w / spd * .012 * D * spd, [.1, .6, 1, 1])
    img = Image.fromarray(rend.render().copy())
    lean = math.degrees(math.acos(np.clip(X.qrot(jnp.asarray(q))[2, 2], -1, 1)))
    R.text(img, (10, 6), label, 24, color)
    R.text(img, (10, 38), f"t {tr['t'][i]:5.1f} s    height {p[2]:5.1f} m    miss {np.linalg.norm(p[:2]):5.1f} m", 17, (255, 255, 255))
    R.text(img, (10, 62), f"wind {spd:4.1f} m/s    lean {lean:4.1f} deg", 17, (120, 205, 255))
    for j, ln in enumerate(lines(i)):
        R.text(img, (10, 86 + 24 * j), ln, 16, (255, 220, 120))
    if tr["done"][i]:
        nm, col = R.OUTCOME[int(tr["outcome"][i])]
        R.text(img, (10, R.PH - 40), nm, 30, col)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=f"{HERE}/runs/stage2_c100h100g_s0_best.npz", help="a decider flown on Gazebo's ladder (v1, run 1)")
    ap.add_argument("--w10", type=float, default=18.0)
    ap.add_argument("--episode", type=int, default=-1, help="-1: the first yard where Gazebo and the nominal world both land")
    ap.add_argument("--out", default=os.path.join(HERE, "..", "..", "outputs", "wind", "videos"))
    ap.add_argument("--clearml", action="store_true")
    a = ap.parse_args()
    name = os.path.basename(a.ckpt)[:-4]
    pol, fly_, dec, sh = ppo.load_policy(a.ckpt)
    hp = ppo.hp_of(a.ckpt)
    nom, _ = fly(pol, fly_, dec, sh, hp, a.w10, 20, 10_000, ())
    rea, prm = fly(pol, fly_, dec, sh, hp, a.w10, 20, 10_000, RAND_ALL)
    assert np.abs(nom["p"][0] - rea["p"][0]).max() < 0.2, "the two worlds do not start the same flights"
    yards = R.yards_of(10_000, 20)
    print("yard: Gazebo | nominal | realistic   (outcome, seconds)")
    rows = []
    for e in range(20):
        en, er = R.episode(nom, e), R.episode(rea, e)
        try:
            _, g = gazebo(e, a.w10, name, en)
        except (FileNotFoundError, StopIteration):
            continue
        rows.append((e, g, en, er))
        f = lambda tr: f"{R.OUTCOME[int(tr['outcome'][-1])][0]:>14s} {tr['t'][-1]:5.1f}"
        print(f"{e:4d}: {g['outcome']:>14s} {g['t']:5.1f} | {f(en)} | {f(er)}")
    e = a.episode if a.episode >= 0 else next(r[0] for r in rows if r[1]["outcome"] == "landed" and int(r[2]["outcome"][-1]) == 1)
    en, er = R.episode(nom, e), R.episode(rea, e)
    eg, g = gazebo(e, a.w10, name, en)
    y = {k: v[e] for k, v in yards.items()}
    pr = prm[e]
    who = (f"this drone: drag x{pr[0] / X.CDA_H:.2f}  mass x{pr[2]:.2f}  thrust x{pr[3]:.2f}  delay {4 * pr[5]:.0f} ms",)
    sets = [(eg, "PX4 + GAZEBO (logged flight)", (255, 190, 90), [1, .6, .1, 1], False, lambda i: ()),
            (en, "FAST SIM: NOMINAL WORLD", (130, 235, 140), [.2, .9, .3, 1], False, lambda i: ("the Gazebo drone, exact sensing",)),
            (er, "FAST SIM: REALISTIC WORLD", (150, 200, 255), [.3, .6, 1, 1], True,
             lambda i: who + (f"pad estimate off by {np.linalg.norm(er['err'][min(i, len(er['err']) - 1)]):.2f} m (yellow ring)",))]
    m = mujoco.MjModel.from_xml_string(R.yard_xml(y))
    rend, data = mujoco.Renderer(m, R.PH, R.PW), mujoco.MjData(m)
    mujoco.mj_forward(m, data)
    R.check_yard(m, data, y)
    az = R.view_azimuth(y)
    n = max(len(s[0]["t"]) for s in sets) + R.HOLD
    padded = [{k: np.concatenate([v, np.repeat(v[-1:], n - len(v), 0)]) for k, v in s[0].items()} for s in sets]
    cams = [R.camera(tr) for tr in padded]
    os.makedirs(a.out, exist_ok=True)
    path = f"{a.out}/worlds_w{a.w10:g}_yard{e}.mp4"
    head = f"same decider (learned v1, run 1), same yard {e}, same start, wind {a.w10:g} m/s at 10 m height: one Gazebo flight and two fast-simulator worlds"
    foot = ["left: the logged PX4 + Gazebo flight replayed (path and tilt as logged; Gazebo's gusts were not logged, so its arrow is the mean wind)",
            "middle: the Gazebo x500 with exact sensing.   right: its own drone, sensing errors like Gazebo's, a turning wind; yellow ring = where it thinks the pad is"]
    stills = []
    with imageio.get_writer(path, fps=R.FPS, codec="libx264", quality=7, pixelformat="yuv420p", macro_block_size=None) as wr:
        for i in range(n):
            cv = Image.new("RGB", (3 * R.PW, R.HEAD + R.PH + R.FOOT), (24, 24, 28))
            for j, (s, tr, cam) in enumerate(zip(sets, padded, cams)):
                cv.paste(panel(rend, data, az, tr, cam, i, s[1], s[2], s[3], s[4], s[5]), (j * R.PW, R.HEAD))
            R.text(cv, (10, 8), head, 21, (255, 255, 255))
            for j, f in enumerate(foot):
                R.text(cv, (10, R.HEAD + R.PH + 8 + 26 * j), f, 15, (200, 200, 205))
            fr = np.array(cv)
            wr.append_data(fr)
            if i in (n // 4, n // 2, n - 1):
                stills.append(fr)
    rend.close()
    for k, fr in enumerate(stills):
        imageio.imwrite(path[:-4] + f"_still{k + 1}.png", fr)
    # the same three flights as plain graphs: path from above, height over time, the estimate error
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
    for (s, tr), col in zip(zip(sets, (eg, en, er)), ("#e8902a", "#2fa84f", "#3b78d8")):
        ax[0].plot(tr["p"][:, 0], tr["p"][:, 1], color=col, label=s[1].title().replace("Px4", "PX4"))
        ax[1].plot(tr["t"], tr["p"][:, 2], color=col)
    ax[0].add_patch(plt.Circle((0, 0), E.PAD_R, fill=False, color="k"))
    ax[0].set(xlabel="x (m)", ylabel="y (m)")
    ax[0].set_aspect("equal", adjustable="datalim")
    ax[0].legend(fontsize=7)
    ax[1].set(xlabel="time (s)", ylabel="height (m)")
    ax[2].plot(er["t"], np.linalg.norm(er["err"], axis=1), color="#3b78d8")
    ax[2].set(xlabel="time (s)", ylabel="pad estimate error (m)")
    for k, t in enumerate(("(a) path seen from above", "(b) height", "(c) realistic world: how wrong the pad estimate is")):
        ax[k].set_title(t, y=-0.25)
        ax[k].grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(path[:-4] + ".png", dpi=120)
    res = f"Gazebo {g['outcome']} at {g['t']:.1f} s | nominal {R.OUTCOME[int(en['outcome'][-1])][0]} at {en['t'][-1]:.1f} s | realistic {R.OUTCOME[int(er['outcome'][-1])][0]} at {er['t'][-1]:.1f} s"
    print(f"yard {e}: {res} -> {path} ({n} frames), stills and {path[:-4]}.png")
    if a.clearml:
        from clearml import Task
        task = Task.init(project_name="rl-landing/wind", task_name=f"worlds video: Gazebo vs fast sim nominal vs realistic ({a.w10:g} m/s)",
                         reuse_last_task_id=False, auto_connect_frameworks=False, auto_connect_arg_parser=False)
        lg = task.get_logger()
        lg.report_media("worlds video", f"{a.w10:g} m/s, yard {e}", iteration=0, local_path=path)
        lg.report_image("worlds graphs", f"{a.w10:g} m/s, yard {e}", iteration=0, local_path=path[:-4] + ".png")
        for k in range(len(stills)):
            lg.report_image("worlds stills", f"still {k + 1}", iteration=0, local_path=path[:-4] + f"_still{k + 1}.png")
        lg.report_table("every ladder yard: outcome and seconds", f"{a.w10:g} m/s", iteration=0,
                        table_plot=[["yard", "PX4 + Gazebo", "fast sim nominal", "fast sim realistic"]]
                        + [[str(r[0]), f"{r[1]['outcome']} {r[1]['t']:.1f} s"] + [f"{R.OUTCOME[int(x['outcome'][-1])][0].lower()} {x['t'][-1]:.1f} s" for x in r[2:]] for r in rows])
        lg.report_text(res)
        print("ClearML:", task.get_output_log_web_page())
        task.close()


if __name__ == "__main__":
    main()
