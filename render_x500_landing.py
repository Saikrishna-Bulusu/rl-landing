#!/usr/bin/env python3
"""Render the x500 flying a full landing pattern with the cascaded velocity controller."""
import os, numpy as np, mujoco, imageio.v2 as imageio
import x500, x500_vel
import render_x500 as R

R.PAD = np.array([2.0, 1.5])
env = R.X500Demo(spawn_z=8.0); env.reset()
ctrl = x500_vel.VelocityController(env.M, env.HOVER_RPM, env.MAX_RPM)
rend = mujoco.Renderer(env.model, height=480, width=640)
cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE
frames=[]; hist=[]; dt=1/48
traj = x500_vel.land(ctrl, env, R.PAD, cruise_z=8.0)   # same routine as the checks
env.reset(); ctrl.reset()
stalled=0
for k in range(6000):
    p=env.pos[0].copy(); h=float(p[2]); d=R.PAD-p[:2]; dist=float(np.linalg.norm(d))
    if dist > 0.20:
        v=np.clip(d*0.8,-1.0,1.0); vz=np.clip((8.0-h)*0.6,-0.3,0.3)
    else:
        v=np.clip(d*0.8,-0.3,0.3); vz=-0.5 if h>1.0 else -0.5
    if h<0.45 and dist<0.20 and abs(float(env.vel[0][2]))<0.20: stalled+=1
    else: stalled=0
    a = np.array([-1.0,0.,0.,0.]) if stalled>=10 else ctrl([v[0],v[1],vz], env.vel[0], env.rpy[0], env.ang_v[0], dt)
    env.step(np.array([a]))
    hist.append((h, float(env.vel[0][2]), float(np.abs(np.degrees(env.rpy[0][:2])).max()), dist))
    if k % 5 == 0:
        mid=0.5*(env.pos[0]+np.array([R.PAD[0],R.PAD[1],0.0]))
        cam.lookat[:]=[mid[0],mid[1],max(0.6,mid[2])]
        cam.distance=max(5.0, float(np.linalg.norm(env.pos[0]-np.array([R.PAD[0],R.PAD[1],0.])))*1.5+4.0)
        cam.azimuth,cam.elevation=-55,-18
        rend.update_scene(env.data,cam); frames.append(rend.render())
    if h <= 0.15 and env.data.ncon > 0: break
os.makedirs("media",exist_ok=True)
imageio.mimsave("media/x500_full_landing.gif", frames[::2], duration=0.07, loop=0)
idx=np.linspace(0,len(frames)-1,6).astype(int)
imageio.imwrite("media/x500_full_landing_sheet.png", np.concatenate([frames[i] for i in idx],axis=1))
h,vz,t,dd=hist[-1]
print(f"  duration {len(hist)/48:.1f} s, {len(frames)} frames")
print(f"  touchdown  h={h:.3f} m   vz={abs(vz):.3f} m/s   miss={dd:.3f} m")
print(f"  max descent {max(abs(v) for _,v,_,_ in hist):.2f} m/s   max tilt {max(x for _,_,x,_ in hist):.1f} deg")
