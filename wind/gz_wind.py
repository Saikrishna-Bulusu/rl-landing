#!/usr/bin/env python3
"""
gz_wind.py -- Gazebo worlds for the wind study: the rig v2 world (voxfast_v2, no video cameras) with a wind entity,
and one server config per wind (PX4's own systems + wind/gz_winddrag's WindDrag: the same gusty wind and body drag
as wind/x500_mjx.py).

    python wind/gz_wind.py                                    -> gz_logan/wind_v2/aruco.sdf
    python wind/gz_wind.py --config w7 --mean 7 --dir 90     -> gz_logan/wind_v2/w7.config
Fly one: WORLD=wind_v2 SERVER_CONFIG=gz_logan/wind_v2/w7.config ./wind/sitl.sh ...
"""
import argparse
import os

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.dirname(HERE)
OUT = f"{R}/gz_logan/wind_v2"
BASE_WORLD = f"{R}/gz_logan/voxfast_v2/aruco.sdf"
BASE_CONFIG = "/home/sai/thesis_2.0/PX4-Autopilot/src/modules/simulation/gz_bridge/server.config"
PLUGINS = f"{HERE}/gz_winddrag/build"                  # libWindDrag.so: sitl.sh puts it on GZ_SIM_SYSTEM_PLUGIN_PATH


def world():
    os.makedirs(OUT, exist_ok=True)
    w = open(BASE_WORLD).read()
    assert w.count("  </world>") == 1 and "<plugin" not in w, "base world changed: plugins in a world disable server.config"
    open(f"{OUT}/aruco.sdf", "w").write(w.replace("  </world>", "    <wind><linear_velocity>0 0 0</linear_velocity></wind>\n  </world>"))
    return f"{OUT}/aruco.sdf"


def config(name, mean, direction_deg=0.0, seed=0, cda_h=0.04, cda_v=0.05, turbulence=1, gust_rate=1 / 15, start_on=1, out=OUT):
    cfg = open(BASE_CONFIG).read()
    add = (f'    <plugin entity_name="*" entity_type="world" filename="libWindDrag.so" name="WindDrag">\n'
           f"      <mean_speed>{mean}</mean_speed><direction_deg>{direction_deg}</direction_deg><seed>{seed}</seed>\n"
           f"      <cda_h>{cda_h}</cda_h><cda_v>{cda_v}</cda_v><turbulence>{turbulence}</turbulence><gust_rate>{gust_rate}</gust_rate>\n"
           f"      <start_on>{start_on}</start_on>\n"
           f"    </plugin>\n")
    assert cfg.count("  </plugins>") == 1
    os.makedirs(out, exist_ok=True)
    p = f"{out}/{name}.config"
    open(p, "w").write(cfg.replace("  </plugins>", add + "  </plugins>"))
    return p


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--mean", type=float, default=0.0)
    ap.add_argument("--dir", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cda-h", type=float, default=0.04)
    ap.add_argument("--cda-v", type=float, default=0.05)
    ap.add_argument("--turbulence", type=int, default=1)
    ap.add_argument("--gust-rate", type=float, default=1 / 15)
    a = ap.parse_args()
    print(world())
    if a.config:
        print(config(a.config, a.mean, a.dir, a.seed, a.cda_h, a.cda_v, a.turbulence, a.gust_rate))
