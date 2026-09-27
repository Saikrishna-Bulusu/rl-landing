#!/usr/bin/env python3
"""
render_x500.py -- show the x500 controller flying a slow, level landing.

    MUJOCO_GL=glfw python3 render_x500.py

Contrast with the earlier cage GIF, where a 27 g Crazyflie flailed because its
reward paid for fast descent and nothing discouraged wild attitude. Here the
low-level controller is doing the flying: the commanded profile is descend at
0.5 m/s, slow to 0.15 m/s inside the last metre, while tracking laterally to
the pad.
"""
import os, sys
import numpy as np
import mujoco
import imageio.v2 as imageio

import x500, x500_ctrl
sys.path.insert(0, os.path.expanduser("~/RL_projects/MuJoCo-drones-gym"))
from multi_drone_mujoco.envs.base_aviary import BaseAviary, _generate_aviary_xml
from multi_drone_mujoco.utils.enums import Physics, ActionType, ObservationType, DroneModel

PAD = np.array([1.5, 1.0])   # closer: lateral authority is limited


class X500Demo(BaseAviary):
    def __init__(self, spawn_z=8.0, **kw):
        self._spawn_z = spawn_z
        super().__init__(drone_model=x500.X500_MODEL, num_drones=1,
                         physics=Physics.MJC_GND_DRAG_DW, act_type=ActionType.ATTITUDE,
                         obs_type=ObservationType.KIN, ctrl_freq=48, sim_freq=240,
                         initial_xyzs=np.array([[0.0, 0.0, spawn_z]]), **kw)
        self._inject_pad()

    def _inject_pad(self):
        xml = _generate_aviary_xml(1, x500.X500_MODEL,
                                   np.array([[0.0, 0.0, self._spawn_z]]),
                                   np.zeros((1, 3)), False, False, 1.0 / self.SIM_FREQ)
        pad = f"""
  <geom name="pad" type="box" pos="{PAD[0]} {PAD[1]} 0.01" size="0.6 0.6 0.01"
        rgba="0.95 0.95 0.95 1"/>
  <geom name="padx" type="box" pos="{PAD[0]} {PAD[1]} 0.02" size="0.55 0.12 0.005"
        rgba="0.95 0.45 0.05 1"/>
  <geom name="pady" type="box" pos="{PAD[0]} {PAD[1]} 0.02" size="0.12 0.55 0.005"
        rgba="0.95 0.45 0.05 1"/>
  <geom name="padc" type="cylinder" pos="{PAD[0]} {PAD[1]} 0.03" size="0.13 0.004"
        rgba="0.85 0.1 0.1 1"/>
"""
        xml = xml.replace("</worldbody>", pad + "\n  </worldbody>")
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        self._renderer = None
        mujoco.mj_forward(self.model, self.data)

    def _computeObs(self): return np.zeros(1)
    def _computeReward(self): return 0.0
    def _computeTerminated(self): return False
    def _computeTruncated(self): return False
    def _computeInfo(self): return {}


def main():
    x500_ctrl.patch_gains()
    env = X500Demo(spawn_z=8.0)
    env.reset()
    c = x500_ctrl.VelocityCarrot()
    rend = mujoco.Renderer(env.model, height=480, width=640)
    cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE

    frames, hist = [], []
    for k in range(6000):
        p = env.pos[0]; h = float(p[2])
        # A PROPER LANDING PATTERN, not a diagonal dive.
        #
        # Commanding lateral motion and descent at once makes the aircraft tilt
        # to accelerate sideways, which steals vertical thrust and produced an
        # uncommanded 4.96 m/s drop against a commanded 0.5 m/s. Real landings
        # separate the phases: translate at altitude, align, then descend.
        d = PAD - p[:2]
        dist = float(np.linalg.norm(d))
        align = dist < 0.25
        if not align and h > 2.0:
            vxy = np.clip(d, -0.5, 0.5); vz = 0.0          # translate, hold height
        elif not align:
            vxy = np.clip(d, -0.25, 0.25); vz = 0.0        # fine align, hold
        else:
            vxy = np.clip(d, -0.2, 0.2)                    # keep centred
            vz = -0.5 if h > 1.0 else -0.15                # descend
        env.step(np.array([c.waypoint(p, [vxy[0], vxy[1], vz])]))
        hist.append((h, float(env.vel[0][2]), float(np.linalg.norm(PAD - env.pos[0][:2]))))
        if k % 6 == 0:
            mid = 0.5 * (env.pos[0] + np.array([PAD[0], PAD[1], 0.0]))
            cam.lookat[:] = [mid[0], mid[1], max(0.5, mid[2])]
            cam.distance = max(6.0, float(np.linalg.norm(env.pos[0] - np.array([PAD[0], PAD[1], 0.0]))) * 1.6 + 3.0)
            cam.azimuth, cam.elevation = -55, -18
            rend.update_scene(env.data, cam)
            frames.append(rend.render())
        if h <= 0.14:
            break
    os.makedirs("media", exist_ok=True)
    imageio.mimsave("media/x500_landing.gif", frames, duration=0.05, loop=0)
    idx = np.linspace(0, len(frames) - 1, 6).astype(int)
    imageio.imwrite("media/x500_landing_sheet.png",
                    np.concatenate([frames[i] for i in idx], axis=1))
    h, vz, dxy = hist[-1]
    print(f"  frames {len(frames)}   touchdown h={h:.2f} m  vz={abs(vz):.3f} m/s  "
          f"miss={dxy:.2f} m")
    print(f"  max descent speed during flight: {max(abs(v) for _,v,_ in hist):.2f} m/s")
    print("  wrote media/x500_landing.gif and _sheet.png")


if __name__ == "__main__":
    main()
