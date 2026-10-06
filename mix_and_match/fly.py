#!/usr/bin/python3
"""mix_and_match/fly.py -- one Gazebo flight under any combination of conditions the existing scripts can compose.

    WIND=9 CRANE=1 MAGPIE=1 SKY=clear GNSS=denied DECIDER=learner SEED=0 OUT=mix1 python3 mix_and_match/fly.py [--dry-run]

WIND   W10 in m/s, 0 = calm (default 0)       CRANE   0/1  the unmapped 58 m crane on the way in (default 0)
MAGPIE 0/1  hunter magpie, seen by rig v2l    SKY     clear/mist/fog (default clear)
GNSS   on/denied (default on)                 DECIDER rule/planner/learner (default rule)
SEED   n (default 0)                          OUT     output folder name (default: built from the conditions)
Optional: WIND_DIR (deg, default 0 = from the west), INSTANCE (PX4 instance), CKPT (learner checkpoint in shared/fast_sim/).

It picks the world and the existing launcher (geofly.sh, voxfly.sh, rigfly.sh, sitl.sh, run_weather.sh), writes the wind
config, and prints the commands. A combination the code cannot do is REFUSED with the reason, never flown as something
else. --dry-run prints the commands and starts nothing; without it they run, one SITL at a time, never on battery.
"""
import os
import shlex
import subprocess
import sys
from pathlib import Path

sys.dont_write_bytecode = True                      # importing gz_voxel for the crane spot must leave nothing behind

R = Path(__file__).resolve().parents[1]
E = os.environ.get
PY = "/usr/bin/python3"
GEO_W = "shared/px4_gazebo/gz_logan/geo_clean/geo_w9.config"          # WindDrag config; wind speed, direction, seed rewritten
YOLO = "shared/px4_gazebo/gz_logan/rig_v2/yolo/run_geo/weights/best.pt"
CK_BIRDS = "rl_corridor_eyes_lidar_gps_birds_hunter_rig_v2l_s0_mean.pt"   # trained with the magpie (and flies the voxel map)
CK_CRANE = "rl_corridor_eyes_lidar_gps_unmapped_s0_mean.pt"               # trained on the unmapped world
WIND_NPZ = "conditions/wind/runs/stage2_c100h100g_s0_best.npz"             # the paper's wind decider (frozen flyer inside)
FLOWN = {("denied", 0, 0, 0), ("denied", 1, 0, 0), ("denied", 0, 1, 0), ("denied", 0, 0, 1), ("denied", 1, 1, 1),
         ("on", 1, 0, 0), ("on", 0, 1, 0), ("on", 0, 0, 1)}                # (gnss, wind>0, crane, magpie) flown before


def refuse(why):
    sys.exit(f"REFUSED: {why}")


def pick(name, default, allowed):
    v = E(name, default)
    if v not in allowed:
        refuse(f"{name}={v!r} is not one of {'/'.join(allowed)}")
    return v


try:
    WIND, SEED = float(E("WIND", 0)), int(E("SEED", 0))
except ValueError:
    refuse("WIND must be a number (m/s) and SEED an integer")
CRANE, MAGPIE = int(pick("CRANE", "0", ("0", "1"))), int(pick("MAGPIE", "0", ("0", "1")))
SKY, GNSS, DECIDER = pick("SKY", "clear", ("clear", "mist", "fog")), pick("GNSS", "on", ("on", "denied")), pick("DECIDER", "rule", ("rule", "planner", "learner"))
WIND_DIR, CKPT = E("WIND_DIR", "0"), E("CKPT")
tag = f"w{WIND:g}_c{CRANE}_m{MAGPIE}_{SKY}_{GNSS}_{DECIDER}_s{SEED}"
OUT = E("OUT", f"mix_{tag}")
if WIND < 0 or SEED < 0:
    refuse("WIND and SEED cannot be negative")
if "/" in OUT:
    refuse("OUT is a folder name, not a path")
if WIND == 0 and E("WIND_DIR"):
    refuse("WIND_DIR given but WIND=0")
if CKPT and DECIDER != "learner":
    refuse("CKPT only means something to DECIDER=learner")

steps, env, note = [], {}, []                       # steps: shell lines, run in order from the repo root


def sh(*a):
    return shlex.join(str(x) for x in a)


def envrun(e, *cmd):
    return sh("env", *[f"{k}={v}" for k, v in e.items()], *cmd)


def wind_config(d, plugins=True):
    """The WindDrag config in the flight's folder (as stress_30sep.sh does): speed, direction, seed rewritten."""
    steps.append(sh("mkdir", "-p", d))
    steps.append(sh("sed", "-e", f"s|<mean_speed>[^<]*</mean_speed>|<mean_speed>{WIND:g}</mean_speed>|",
                    "-e", f"s|<direction_deg>[^<]*</direction_deg>|<direction_deg>{WIND_DIR}</direction_deg>|",
                    "-e", f"s|<seed>[^<]*</seed>|<seed>{SEED}</seed>|", GEO_W) + f" > {shlex.quote(d + '/wind.config')}")
    env["SERVER_CONFIG"] = f"{R}/{d}/wind.config"
    if plugins:                                     # libWindDrag.so (geofly.sh and sitl.sh set it themselves)
        env["EXTRA_PLUGINS"] = f"{R}/conditions/wind/gz_winddrag/build"


# ---- what the code cannot do -----------------------------------------------------------------------------------------
if SKY != "clear":
    # the skies are their own worlds (gz_weather.py: the plain Logan world + particles), flown by run_weather.sh's px4_mission only
    if GNSS == "denied":
        refuse("fog/mist exist only in the plain-GPS weather worlds; the GNSS-denied worlds (geo_*, photo floor) have no sky variant")
    if WIND or CRANE or MAGPIE:
        refuse("mist/fog worlds are calm and have no crane or magpie; run_weather.sh flies only the plain mission there")
    if DECIDER != "rule" or SEED != 0 or CKPT:
        refuse("run_weather.sh flies px4_mission.py's own sigma-rule landing, seeds 0..N-1 as a queue: only DECIDER=rule SEED=0 (one flight)")
    if E("OUT"):
        refuse(f"run_weather.sh writes outputs/weather_runs/{SKY}/ and takes no OUT name")
    d = f"outputs/weather_runs/{SKY}"
    steps.append(sh("conditions/weather/run_weather.sh", SKY, 1))
    note.append("run_weather.sh pkills every bin/px4 and gz sim first: this launcher refuses to start while a Gazebo is up")
elif GNSS == "denied":
    d = f"outputs/gazebo_geo/{OUT}"
    env.update(OUT=OUT, INSTANCE=E("INSTANCE", 5), WORLD={(0, 0): "geo_clean", (1, 0): "geo_floor", (0, 1): "geo_magpie_v2l", (1, 1): "geo_stress_v2l"}[CRANE, MAGPIE])
    if WIND:
        wind_config(d, plugins=False)
    if not (CRANE or MAGPIE):                       # the wind lander (rule, or the wind learner): geo_calm4 / geo_wind10
        if DECIDER == "planner":
            refuse("DECIDER=planner is gz_learner.py's, which always moves a crane or a magpie; with neither only the wind lander "
                   "(rule, or learner = the paper's wind decider) is wired in")
        cmd = ["conditions/wind/gz_windland.py", "--geo", "--episode", SEED, "--w10", f"{WIND:g}",
               "--decider", "rule" if DECIDER == "rule" else WIND_NPZ, "--out", f"{d}/flights.jsonl", "--video", f"{d}/mix.mp4"]
        if CKPT:
            refuse("CKPT is for gz_learner.py's learners; the wind learner is conditions/wind/runs/stage2_c100h100g_s0_best.npz")
    else:
        if CRANE and MAGPIE and DECIDER != "learner":
            refuse("crane + magpie has flown only with the learner (the stress test): the voxel-map decider takes the magpie inputs "
                   "only through the birds checkpoint, rule/planner on it are untested")
        cmd = ["shared/px4_gazebo/gz_learner.py", "--geo", "--decider", DECIDER, "--seed", SEED, "--magpie", "hunter" if MAGPIE else "none"]
        env["TREES_JSON"] = "shared/px4_gazebo/gz_logan/trees_logan.json"
        if MAGPIE:                                  # the rig's detector and tracker; its landing at 0.1x, for the detector
            cmd += ["--sense", "rig"]
            env.update(RIG="v2l", RIG_YOLO=f"{R}/{YOLO}", RIG_CALIB=f"{R}/shared/fast_sim/rig_calib_v2l.json", GEO_LAND_SPEED="0.1")
        if CRANE:                                   # hand-over 90 m out at 75 m, the crane on the last leg (stress_30sep.sh's spot)
            cmd += ["--voxel"] + (["--local3d"] if DECIDER != "planner" else [])
            env["GEO_GOAL_OUT"] = "90"
            sys.path.insert(0, f"{R}/shared/px4_gazebo")
            import numpy as np
            import gz_voxel as GV
            pad, ref = np.array([120.0, 0.0]), np.array([260.0, -120.0])
            u = (ref - pad) / np.linalg.norm(ref - pad); n = np.array([-u[1], u[0]])
            rng = np.random.default_rng(2000 + SEED)
            while True:
                x, y = pad + rng.uniform(30.0, 60.0) * u + rng.uniform(-1.5, 1.5) * n
                os.environ["CRANE_AT"] = f"{x:.2f},{y:.2f}"
                try:
                    GV.crane_at(0)
                    break
                except AssertionError:
                    pass
            env["CRANE_AT"] = os.environ["CRANE_AT"]
        if DECIDER == "learner":
            cmd += ["--ckpt", CKPT or (CK_BIRDS if MAGPIE else CK_CRANE)]
        cmd += ["--video", f"{d}/mix.mp4"]
    steps.append(envrun(env, "timeout", 3600, "gnss_denied/geofly.sh", PY, "-u", *cmd))
else:                                               # GNSS on: the plain-GPS worlds and their launchers
    if CRANE and MAGPIE:
        refuse("no GPS world holds both a crane and a magpie (voxfly has the crane, rigfly the magpie); only the GNSS-denied "
               "geo_stress_v2l world has both")
    if not (CRANE or MAGPIE):                       # wind_gz ladder's lane: the yard world of episode SEED
        if DECIDER == "planner":
            refuse("DECIDER=planner is gz_learner.py's, which always moves a crane or a magpie; with neither only the wind lander is wired in")
        if CKPT:
            refuse("CKPT is for gz_learner.py's learners; the wind learner is conditions/wind/runs/stage2_c100h100g_s0_best.npz")
        if int(WIND) != WIND or int(WIND) not in (0, 9, 12, 15, 18):
            refuse("GPS wind flights use windyard/e<SEED>/w<W>.config, built only for W = 0, 9, 12, 15, 18")
        if SEED > 19:
            refuse("windyard worlds exist for SEED 0..19")
        if E("WIND_DIR"):
            refuse("the windyard configs fix the wind direction (each yard's own)")
        d = f"outputs/gazebo_wind/{OUT}"
        env.update(OUT=OUT, WORLD=f"windyard/e{SEED}", INSTANCE=E("INSTANCE", 1),
                   SERVER_CONFIG=f"{R}/shared/px4_gazebo/gz_logan/windyard/e{SEED}/w{int(WIND)}.config")
        steps.append(envrun(env, "timeout", 900, "shared/px4_gazebo/sitl.sh", PY, "-u", "conditions/wind/gz_windland.py", "--episode", SEED,
                            "--w10", f"{WIND:g}", "--decider", "rule" if DECIDER == "rule" else WIND_NPZ, "--out", f"{R}/{d}/flights.jsonl"))
    elif CRANE:                                     # conditions/static_obstacles/voxfly.sh, as land_gate.sh flies it
        if CKPT:
            refuse("voxfly.sh hardcodes its learner checkpoint")
        d = f"outputs/gazebo_voxel/{OUT}"
        if WIND:
            wind_config(d)
        env.update(OUT=OUT, SEEDS=SEED, RUNS=f"{DECIDER}:{'voxel' if DECIDER == 'planner' else 'local3d'}", RIG="v2", SPEED="1.0", FAST=1,
                   FULL=1, INSTANCE=E("INSTANCE", 4))
        steps.append(envrun(env, "conditions/static_obstacles/voxfly.sh"))
    else:                                           # conditions/magpie/rigfly.sh: hunter magpie, seen through rig v2l
        d = f"outputs/gazebo_rig/{OUT}"
        if WIND:
            wind_config(d)
        env.update(OUT=OUT, SEEDS=SEED, DECIDERS=DECIDER, SENSE="rig", RIG="v2l", INSTANCE=E("INSTANCE", 1))
        if DECIDER == "learner":
            env["CKPT"] = CKPT or CK_BIRDS
        steps.append(envrun(env, "conditions/magpie/rigfly.sh"))

if (GNSS, WIND > 0, CRANE, MAGPIE) not in FLOWN and SKY == "clear":
    note.append("these parts have never flown together: composed from the existing launchers, unverified")

# ---- show, then (unless --dry-run) fly -------------------------------------------------------------------------------
used = (R / d).exists() and any((R / d).iterdir())
print(f"# WIND={WIND:g} CRANE={CRANE} MAGPIE={MAGPIE} SKY={SKY} GNSS={GNSS} DECIDER={DECIDER} SEED={SEED} -> {d}/")
for n in note:
    print(f"# note: {n}")
print(f"cd {R}")
for s in steps:
    print(s)
if "--dry-run" in sys.argv:
    if used:
        print(f"# note: {d}/ already has files: a real run would refuse (pick another OUT)")
    sys.exit(0)
if used:
    refuse(f"{d}/ already has files; pick another OUT")
if subprocess.run(["pgrep", "-f", "gz si[m]"], capture_output=True).stdout:
    refuse("a Gazebo is already up: one SITL at a time")
if Path("/sys/class/power_supply/ADP0/online").read_text().strip() != "1" and not E("NO_AC_WAIT"):
    refuse("on battery: plug in (NO_AC_WAIT=1 flies anyway)")
for s in steps:
    rc = subprocess.run(["bash", "-c", s], cwd=R).returncode
    if rc:
        sys.exit(f"step failed (rc {rc}): {s}")
