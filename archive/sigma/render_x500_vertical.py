#!/usr/bin/env python3
"""Render the part of the x500 controller that DOES work: a slow vertical landing."""
import os, numpy as np, mujoco, imageio.v2 as imageio
import x500, x500_ctrl
import render_x500 as R

R.PAD = np.array([0.0, 0.0])          # pad directly below: vertical only
env = R.X500Demo(spawn_z=6.0); env.reset()
c = x500_ctrl.VelocityCarrot()
rend = mujoco.Renderer(env.model, height=480, width=640)
cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE
frames=[]; hist=[]
for k in range(4000):
    h=float(env.pos[0][2])
    vz = -0.5 if h > 1.0 else -0.15
    env.step(np.array([c.waypoint(env.pos[0], [0,0,vz])]))
    hist.append((h, float(env.vel[0][2]), float(np.abs(np.degrees(env.rpy[0][:2])).max())))
    if k % 5 == 0:
        cam.lookat[:] = [0.4, 0.4, max(0.6, h*0.55)]
        cam.distance = max(4.0, h*1.5+2.5); cam.azimuth, cam.elevation = -55, -16
        rend.update_scene(env.data, cam); frames.append(rend.render())
    if h <= 0.14: break
os.makedirs("media", exist_ok=True)
imageio.mimsave("media/x500_vertical_landing.gif", frames, duration=0.05, loop=0)
idx=np.linspace(0,len(frames)-1,6).astype(int)
imageio.imwrite("media/x500_vertical_sheet.png", np.concatenate([frames[i] for i in idx],axis=1))
h,vz,t=hist[-1]
print(f"  touchdown h={h:.2f} m  vz={abs(vz):.3f} m/s")
print(f"  max descent speed {max(abs(v) for _,v,_ in hist):.2f} m/s")
print(f"  max tilt during descent {max(t for _,_,t in hist):.1f} deg")
print(f"  {len(frames)} frames -> media/x500_vertical_landing.gif")
