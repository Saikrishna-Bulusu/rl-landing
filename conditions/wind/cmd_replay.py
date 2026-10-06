"""Does MJX's PX4 model answer Gazebo's own vertical commands the way PX4 in Gazebo did? Open loop: one Gazebo flight's
logged vz_cmd (and horizontal path, tracked) is fed to x500_mjx with the same wind speed and direction, each command held
as Gazebo held it and delivered `lat` s late. Prints both vertical speeds side by side and, below 6 m, mean and swing.
    python cmd_replay.py ../../outputs/gazebo_wind/obscheck/e4_w9_stage2_c100h100_s1_best.csv --yard 4 --w 9"""
import argparse, json
import numpy as np
import jax, jax.numpy as jnp
import x500_mjx as X, land_env as E

ap = argparse.ArgumentParser(); ap.add_argument("csv"); ap.add_argument("--yard", type=int, required=True)
ap.add_argument("--w", type=float, required=True); ap.add_argument("--lat", type=float, default=0.06)
ap.add_argument("--k-xy", type=float, default=1.0); ap.add_argument("--quiet", action="store_true")
ap.add_argument("--thr-max", type=float, default=1.0, help="x500_mjx.THR_MAX, the collective cap")
ap.add_argument("--show", type=float, nargs=2, help="print this time window (default: below 9 m)")
a = ap.parse_args()
X.THR_MAX = a.thr_max
d = np.genfromtxt(a.csv, delimiter=",", names=True)
y = json.load(open(f"{E.__file__.rsplit('/', 3)[0]}/outputs/gazebo_wind/yards_s10000.json"))[a.yard]
p0 = np.array([d["x"][0], d["y"][0], d["z"][0]])
assert np.hypot(*(p0[:2] - np.array(y["start"][:2]))) < 1.0, f"CSV start {p0} is not the yard's ENU start {y['start']}"
step = jax.jit(lambda s, v, af, w: X.control_step(s, v, af, y["yaw"], w))
wstep = jax.jit(E.wind_step)
s = X.init_state(jnp.array(p0), yaw=y["yaw"])
ws = dict(E.wind_new(jax.random.PRNGKey(0), a.w), th=jnp.array(y["wind_dir"]), gth=jnp.array(y["wind_dir"]))
key, out = jax.random.PRNGKey(1), []
for k in range(int(d["t"][-1] / E.DT_C)):
    t = k * E.DT_C
    j = max(np.searchsorted(d["t"], t - a.lat, side="right") - 1, 0)          # the command PX4 holds at t
    p, v = np.array(s["d"].qpos[:3]), np.array(s["d"].qvel[:3])
    pg = np.array([np.interp(t, d["t"], d[c]) for c in ("x", "y")]); vg = np.array([np.interp(t, d["t"], d[c]) for c in ("vx", "vy")])
    vel_sp = jnp.array([*(vg + a.k_xy * (pg - p[:2])), d["vz_cmd"][j]])
    key, kw = jax.random.split(key)
    ws, wind = wstep(ws, s["d"].qpos[2], kw)
    s = step(s, vel_sp, jnp.array([d["ax"][j], d["ay"][j], 0.0]), wind)
    out.append((t, p[2], v[2], np.interp(t, d["t"], d["z"]), np.interp(t, d["t"], d["vz"]), d["vz_cmd"][j]))
o = np.array(out)
if not a.quiet:
    for r in (o[(o[:, 0] >= a.show[0]) & (o[:, 0] <= a.show[1])] if a.show else o[o[:, 3] < 9][::2]):
        print("t %.2f  cmd %+.1f   gazebo z %.2f vz %+.2f   mjx z %.2f vz %+.2f" % (r[0], r[5], r[3], r[4], r[1], r[2]))
m = (o[:, 3] > 1) & (o[:, 3] < 6)
print(json.dumps(dict(csv=a.csv.split("/")[-1], lat=a.lat, thr_max=a.thr_max, gz_vz_mean=round(o[m, 4].mean(), 2), mjx_vz_mean=round(o[m, 2].mean(), 2),
                      gz_swing=round(o[m, 4].std(), 2), mjx_swing=round(o[m, 2].std(), 2),
                      corr=round(float(np.corrcoef(o[m, 4], o[m, 2])[0, 1]), 2))))
