#!/usr/bin/env python3
"""
gz_weather.py -- the Logan Gazebo world under five skies, for the robustness check.

    python gz_weather.py            write gz_logan/weather/<sky>/{aruco.sdf, server.config}
    python gz_weather.py report     summarise weather_runs/<sky>/ into weather_report.json

  calm    the world as flown so far (the baseline, re-flown the same night, same seeds)
  breeze  3 m/s from the south-east (the SE trades), swelling +/-20%          Beaufort 2
  gusty   7 m/s from the south-east, gusting to ~10 m/s                       Beaufort 4
  mist    calm; thin particle fog over the delivery area -- the camera looks through it
  fog     calm; thick fog: from 60 m up, roofs 60 m away are barely visible
  (rain was tried and dropped: gz 8's particle spray ignored every rotation meant to
   point it down -- drops flew sideways in lines -- and real rain's camera effects,
   water on the lens and lost contrast, are not particles anyway. Fog covers visibility.)

WIND reaches the x500 through its propellers: gz's MulticopterMotorModel applies rotor
drag against the relative air velocity. PX4's rotorDragCoefficient 8.06e-5 x ~760 rad/s
at hover x 4 rotors ~ 0.24 N per m/s of wind, so 7 m/s pushes ~1.7 N on the 2 kg frame
(~5 deg of tilt to hold station). WindEffects only makes the wind vary in time; no x500
link has enable_wind, so it adds no second force. Tilt is measured, not assumed: the
report prints each sky's roll/pitch while hovering.

PLUGINS go in a per-sky server.config, not the world: gz-sim loads the server config's
systems (physics, sensors, PX4's camera and flow plugins) ONLY when the world declares
none, so a single <plugin> in the world would silently drop them all.

FOG and RAIN are what the camera sees -- particles between it and the pad. Not modelled:
water on the lens, rain on the rotors, any GPS or IMU effect.
"""
import glob
import json
import math
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.dirname(HERE)
BASE_WORLD = f"{R}/gz_logan/worlds/aruco.sdf"
BASE_CONFIG = "/home/sai/thesis_2.0/PX4-Autopilot/src/modules/simulation/gz_bridge/server.config"
OUT = f"{R}/gz_logan/weather"
SE = np.array([-1.0, 1.0]) / math.sqrt(2)          # FROM the south-east = blowing toward the north-west (ENU)

# sky: (mean wind m/s, gust swell as a fraction, gust period s, direction wander deg, particles)
SKIES = {"calm": (0.0, 0.0, 9.0, 0.0, None), "breeze": (3.0, 0.2, 12.0, 8.0, None),
         "gusty": (7.0, 0.4, 9.0, 12.0, None), "mist": (0.0, 0.0, 9.0, 0.0, "mist"),
         "fog": (0.0, 0.0, 9.0, 0.0, "fog")}


def textures():
    """A soft white puff (fog) and a thin streak (rain), RGBA, written here -- the
    gz example's fog is a Fuel model that would have to be downloaded."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    y, x = np.mgrid[-1:1:128j, -1:1:128j]
    puff = np.clip(1 - np.hypot(x, y), 0, 1) ** 2
    plt.imsave(f"{OUT}/fog.png", np.dstack([np.ones((128, 128, 3)), puff]))
    streak = np.clip(1 - np.abs(x) * 8, 0, 1) * np.clip(1 - np.abs(y), 0, 1)
    plt.imsave(f"{OUT}/rain.png", np.dstack([np.full((128, 128, 3), 0.85), streak * 0.8]))


def particles(kind):
    """Emitter models over the delivery area: spawn (0,0) to the pad (120,0), +/-60 m."""
    if kind in ("mist", "fog"):   # ~12,000 puffs 12 m across over spawn-to-pad, 0-50 m up, drifting
        a = 0.08 if kind == "mist" else 0.18
        pose, size, v, life, rate, psize, rgba = "100 0 25 0 0 0", "320 220 50", (0.05, 0.2), 40, 300, "12 12 12", f"0.92 0.92 0.92 {a}"
    else:   # rain: a slab at 55 m over the pad emitting DOWN (pitch +90 deg: emitter x -> world -z).
            # Spread over the whole route it was invisible; real rain (~1,000 drops/m^3) cannot be
            # rendered, so this is visual clutter where the camera looks while landing, no more.
        pose, size, v, life, rate, psize, rgba = "120 0 55 0 0 0", "1 90 90", (8.0, 9.0), 6.5, 12000, "0.08 1.0 0.08", "0.85 0.85 0.9 0.7"
    return f"""
    <model name="{kind}">
      <static>true</static>
      <pose>{pose}</pose>
      <link name="link">
        <particle_emitter name="{kind}_emitter" type="box">
          <pose>0 0 0 0 {1.5708 if kind == "rain" else 0} 0</pose>   <!-- on the model it did not turn the spray -->
          <emitting>true</emitting>
          <size>{size}</size>
          <particle_size>{psize}</particle_size>
          <lifetime>{life}</lifetime>
          <rate>{rate}</rate>
          <min_velocity>{v[0]}</min_velocity>
          <max_velocity>{v[1]}</max_velocity>
          <color_start>{rgba}</color_start>
          <color_end>{rgba}</color_end>
          <material>
            <diffuse>1 1 1 1</diffuse>
            <pbr><metal><albedo_map>{OUT}/{'rain' if kind == 'rain' else 'fog'}.png</albedo_map></metal></pbr>
          </material>
        </particle_emitter>
      </link>
    </model>
"""


def write(sky):
    v, swell, period, wander, kind = SKIES[sky]
    d = f"{OUT}/{sky}"
    os.makedirs(d, exist_ok=True)
    world = open(BASE_WORLD).read()
    assert world.count("</world>") == 1 and "<plugin" not in world, "base world changed: re-check the plugin rule"
    extra = ""
    if v > 0:
        wx, wy = v * SE
        extra += f"\n    <wind>\n      <linear_velocity>{wx:.3f} {wy:.3f} 0</linear_velocity>\n    </wind>\n"
    if kind:
        extra += particles(kind)
    open(f"{d}/aruco.sdf", "w").write(world.replace("  </world>", extra + "  </world>"))

    cfg = open(BASE_CONFIG).read()
    add = ""
    if v > 0:   # the gust model: a sine swell on the magnitude, a slow wander in direction, noise on both
        add += f"""    <plugin entity_name="*" entity_type="world" filename="gz-sim-wind-effects-system" name="gz::sim::systems::WindEffects">
      <force_approximation_scaling_factor>1</force_approximation_scaling_factor>
      <horizontal>
        <magnitude>
          <time_for_rise>2</time_for_rise>
          <sin><amplitude_percent>{swell}</amplitude_percent><period>{period}</period></sin>
          <noise type="gaussian"><mean>0</mean><stddev>{0.1 * swell:.3f}</stddev></noise>
        </magnitude>
        <direction>
          <time_for_rise>10</time_for_rise>
          <sin><amplitude>{wander}</amplitude><period>{3 * period}</period></sin>
          <noise type="gaussian"><mean>0</mean><stddev>0.5</stddev></noise>
        </direction>
      </horizontal>
      <vertical><noise type="gaussian"><mean>0</mean><stddev>0.05</stddev></noise></vertical>
    </plugin>
"""
    if kind:
        add += '    <plugin entity_name="*" entity_type="world" filename="gz-sim-particle-emitter-system" name="gz::sim::systems::ParticleEmitter"/>\n'
    assert cfg.count("  </plugins>") == 1
    open(f"{d}/server.config", "w").write(cfg.replace("  </plugins>", add + "  </plugins>"))
    return d


def report():
    """Per sky: landed on the pad, miss, first marker read (altitude), time, and the
    tilt it took to hold station while centring -- the wind's measured effect."""
    import csv
    out = {}
    for sky in SKIES:
        runs = sorted(glob.glob(f"{R}/weather_runs/{sky}/m_*.json"), key=lambda p: int(re.findall(r"(\d+)\.json", p)[0]))
        if not runs:
            continue
        rs = [json.load(open(p))[0] for p in runs]
        tilt, t_sim = [], []
        for p in runs:
            tp = p.replace("m_", "trace_").replace(".json", ".csv")
            if not os.path.exists(tp):
                continue
            rows = list(csv.DictReader(open(tp)))
            c = [r for r in rows if r["phase"] == "centre" and 3.0 < float(r["z"]) < 35.0]
            if c:
                tilt.append(math.degrees(float(np.median([math.hypot(float(r["roll"]), float(r["pitch"])) for r in c]))))
            if rows:
                t_sim.append(float(rows[-1]["tsim"]) - float(rows[0]["tsim"]))
        ok = [r["outcome"] == "landed" and r["miss_m"] is not None and r["miss_m"] < 1.0 for r in rs]
        med = lambda xs: float(np.median(xs)) if xs else None
        out[sky] = {"flights": len(rs), "on_pad": int(sum(ok)),
                    "outcomes": {o: sum(r["outcome"] == o for r in rs) for o in {r["outcome"] for r in rs}},
                    "miss_m_median": med([r["miss_m"] for r in rs if r["miss_m"] is not None]),
                    "first_read_alt_m_median": med([r["acq_alt_m"] for r in rs if r.get("acq_alt_m") is not None]),
                    "tilt_deg_median_centring": med(tilt), "t_sim_s_median": med(t_sim)}
        o = out[sky]
        f = lambda x, fmt: "  n/a" if x is None else format(x, fmt)
        print(f"{sky:<7} on pad {o['on_pad']}/{o['flights']}  miss {f(o['miss_m_median'], '5.2f')} m  "
              f"first read {f(o['first_read_alt_m_median'], '5.1f')} m  tilt {f(o['tilt_deg_median_centring'], '4.1f')} deg  "
              f"time {f(o['t_sim_s_median'], '4.0f')} s  {o['outcomes']}")
    json.dump(out, open(f"{R}/weather_report.json", "w"), indent=1)
    return out


if __name__ == "__main__":
    if sys.argv[1:] == ["report"]:
        report()
    else:
        os.makedirs(OUT, exist_ok=True)
        textures()
        for sky in SKIES:
            print(write(sky))
