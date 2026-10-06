# mix_and_match

One launcher, `fly.py`, that composes the conditions for a Gazebo flight from the machinery that already exists
(the world builders, the wind plugin, the weather skies, GNSS denial, the deciders). It picks the world and the existing
launcher, writes the wind config, and prints the commands. If the code cannot do a combination it **refuses and says why**;
it never flies something else instead.

It was only tested with `--dry-run` (no Gazebo was started when it was written). Run it from anywhere:

```bash
WIND=9 CRANE=1 MAGPIE=1 GNSS=denied DECIDER=learner SEED=3 OUT=mix1 python3 mix_and_match/fly.py --dry-run   # prints, starts nothing
WIND=9 CRANE=1 MAGPIE=1 GNSS=denied DECIDER=learner SEED=3 OUT=mix1 python3 mix_and_match/fly.py             # flies it
```

| variable | values | default | what it switches |
|---|---|---|---|
| `WIND` | W10 in m/s, `0` = calm | 0 | the WindDrag plugin (same gusty wind and body drag as the MJX model) |
| `CRANE` | 0 / 1 | 0 | the unmapped 58 m crane on the way in (the voxel map and the local 3D planner come with it) |
| `MAGPIE` | 0 / 1 | 0 | the hunter magpie, seen only through rig v2l's cameras (detector + tracker) |
| `SKY` | clear / mist / fog | clear | the weather worlds' particles |
| `GNSS` | on / denied | on | `denied` = GeoAnchor feeds PX4 and GNSS is switched off over the suburb |
| `DECIDER` | rule / planner / learner | rule | who flies the approach and the landing |
| `SEED` | n | 0 | the flight's seed (crane spot, magpie, wind gusts, yard) |
| `OUT` | a folder name | built from the conditions | `outputs/<launcher's folder>/<OUT>/` |
| `WIND_DIR`, `INSTANCE`, `CKPT` | optional | 0 deg (from the west), the launcher's own, see below | wind direction, PX4 instance, learner checkpoint |

`--dry-run` prints `cd <repo>`, then every command (the `mkdir` and `sed` that write the wind config, then the one `env ... launcher`
line). A real run does the same steps, and also refuses to start while a Gazebo is already up (one SITL at a time) or on
battery (`NO_AC_WAIT=1` overrides), and refuses when the output folder already has files.

## What each combination becomes

GNSS denied (`gnss_denied/geofly.sh`: photo-floor world, GeoAnchor, EKF2 fed over AGP):

| CRANE | MAGPIE | world | flight script | deciders |
|---|---|---|---|---|
| 0 | 0 | `geo_clean` | `conditions/wind/gz_windland.py --geo` | rule, or learner = the paper's wind decider (`stage2_c100h100g_s0_best.npz`) |
| 1 | 0 | `geo_floor` | `shared/px4_gazebo/gz_learner.py --geo --magpie none --voxel` | rule (+ `--local3d`), planner, learner (crane checkpoint) |
| 0 | 1 | `geo_magpie_v2l` | `gz_learner.py --geo --magpie hunter --sense rig` | rule, planner, learner (birds checkpoint) |
| 1 | 1 | `geo_stress_v2l` | `gz_learner.py --geo --magpie hunter --sense rig --voxel --local3d` | learner only |

GNSS on:

| CRANE | MAGPIE | launcher | world | notes |
|---|---|---|---|---|
| 0 | 0 | `shared/px4_gazebo/sitl.sh` + `gz_windland.py` | `windyard/e<SEED>` | W is 0, 9, 12, 15 or 18; SEED 0 to 19; the yard's own wind direction |
| 1 | 0 | `conditions/static_obstacles/voxfly.sh` | `voxfast_v2` | fast, no film; the crane spot is `gz_voxel.crane_at(SEED)` |
| 0 | 1 | `conditions/magpie/rigfly.sh` | `rigfly_v2l` | 0.1x speed (the six cameras); films each flight |
| 1 | 1 | refused | | no GPS world holds both |

`SKY=mist` or `fog` is a different world again: `conditions/weather/run_weather.sh <sky> 1` (GNSS on, nothing else).

With wind on, the config is `geo_w9.config` with its speed, direction and seed rewritten (as `all_conditions/queue/stress_30sep.sh`
does), written into the flight's own folder. For crane in a GNSS-denied flight the crane is placed on the last leg, 30 to 60 m
from the pad, by the same rule the stress test uses.

## What it refuses (and why)

* `SKY` mist/fog with anything else: the skies are their own worlds (the plain Logan world plus particle emitters, `gz_weather.py`).
  They have no GNSS-denied version (no photo floor), no crane, no magpie, no WindDrag; `run_weather.sh` flies `px4_mission.py`'s
  own sigma-rule landing as a queue of seeds 0 to N-1, so only `DECIDER=rule SEED=0`, no `OUT`, no `CKPT`.
* `CRANE=1 MAGPIE=1` with GNSS on: `voxfly` worlds hold the crane, `rigfly` worlds the magpie, none both.
* `CRANE=1 MAGPIE=1` denied with `rule`/`planner`: only the learner has flown it (the stress test). The voxel-map decider
  (`decider_server.py --voxel`) hands the magpie inputs on only through a checkpoint whose name says `birds`.
* `DECIDER=planner` with neither crane nor magpie: `gz_learner.py` always moves a crane or a magpie; with neither, the wind
  lander (`gz_windland.py`) is the only flight script, and it takes `rule` or an `.npz` policy.
* GPS wind-only: W outside 0, 9, 12, 15, 18, SEED above 19, or `WIND_DIR`: the 20 `windyard` worlds have only those configs.
* `CKPT` where the launcher cannot take one (`voxfly.sh` hardcodes its checkpoint; the wind lander takes an `.npz`, not a `.pt`).
* Bad values (negative wind or seed, unknown `SKY`/`GNSS`/`DECIDER`, `OUT` containing a slash).

## Which combinations have flown

Flown before: all four denied worlds (calm and wind with the wind lander, crane with the rule, magpie with the learner,
and the stress test: crane + magpie + wind + learner), GPS wind-only (the Gazebo ladders), GPS crane-only (`voxfly`), GPS
magpie-only (`rigfly`), the five skies. Anything else (wind on a crane-only or magpie-only flight, either with GNSS denied or
on, calm GPS wind-lander with `W=0`) is parts stitched together from the existing launchers and is **never flown**: `fly.py`
prints a `# note:` saying so. The decider choices outside the rows above are also unflown. Not carried over from the stress test:
its manifest.json of code hashes, its random wind direction per seed (here the direction is `WIND_DIR`, default 0), and its
wait for the previous queue.

Outputs land where each launcher puts them: `outputs/gazebo_geo/<OUT>/`, `outputs/gazebo_wind/<OUT>/`,
`outputs/gazebo_voxel/<OUT>/`, `outputs/gazebo_rig/<OUT>/`, `outputs/weather_runs/<sky>/`.

## The fast simulator mixes differently

`shared/fast_sim/urban_rl.py` (trains a learner) and `urban_headroom.py` (planner and oracle headroom) take the conditions as
independent flags, so any subset works (run from `shared/fast_sim/`, with the `mujoco_drones` Python):

| condition | flag | note |
|---|---|---|
| GNSS denied / on | `--nav geoanchor` (default) / `--nav gps` | geoanchor replays GeoAnchor's measured error (about 9 to 15 m); gps is GNSS-grade |
| magpie | `--birds` | a swooping magpie near half the pads |
| magpie attacks from behind | `--hunter` | only matters with `--birds` (with no bird there is nothing to hunt with) |
| magpie seen through cameras | `--cameras` or `--rig` (not both: asserted) | default is the truth; again only with `--birds` |
| unmapped structures | `--unmapped` | five random things the survey lacks, one tall: stands in for the crane, not the same object |
| fog | `--fog` | every flight in fog; there is no mist level |
| map | `--map logan_map_lidar.npz` | the LiDAR map |
| decider | none: `urban_headroom.py` scores the rule, the planners and the oracles side by side; `urban_rl.py` trains the learner | |

```bash
cd shared/fast_sim && python urban_headroom.py --nav gps --birds --hunter --rig --unmapped --fog
```

**Wind exists only in MJX and Gazebo.** The fast simulator has no wind at all (`urban_env.py` has none). Wind lives in
`conditions/wind/x500_mjx.py` and `land_env.py` (MJX, the pad, a house and a tree, no Logan city) and in Gazebo's WindDrag plugin.
So a wind-plus-anything-else study can run only in Gazebo, through this launcher; there is no fast-simulator equivalent.
