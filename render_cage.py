#!/usr/bin/env python3
"""
render_cage.py -- render real MuJoCo episodes of a trained cage policy.

    MUJOCO_GL=glfw python3 render_cage.py --arm sigma --episodes 3

Produces an animated GIF plus a contact-sheet PNG of key frames. This exists
because a table of success rates does not show anyone what the drone is doing;
a video does. Frames are real MuJoCo renders, not a schematic.
"""
import argparse, glob, os
import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="sigma")
    ap.add_argument("--run", default=None, help="run dir with ppo_<arm>.zip")
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--out", default="media")
    ap.add_argument("--every", type=int, default=3, help="keep 1 frame in N")
    ap.add_argument("--camera", default="fixed",
                    choices=["fixed", "track", "front", "fpv"])
    a = ap.parse_args()

    from stable_baselines3 import PPO
    from cage_env import CageLandingEnv
    import imageio.v2 as imageio
    import mujoco

    def wide_shot(env, renderer):
        """
        Our own camera. The package's modes are unusable here: "track" sets
        distance = spread*1.2+0.3, which is 0.5 m for a single drone, and
        "fixed" looks at the cage centre from 1.2 m. Both crop out either the
        drone or the pad. This frames BOTH, which is the whole point of a video.
        """
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        d = env.pos[0]
        pad = np.array([env._pad_xy[0], env._pad_xy[1], 0.0])
        mid = 0.5 * (d + pad)
        cam.lookat[:] = [mid[0], mid[1], max(0.4, mid[2])]
        sep = float(np.linalg.norm(d - pad))
        cam.distance = max(3.2, sep * 1.8 + 1.5)
        cam.azimuth = -50
        cam.elevation = -22
        renderer.update_scene(env.data, cam)
        return renderer.render()

    run = a.run or os.path.dirname(sorted(glob.glob(f"cage_final2/{a.arm}_s*/ppo_{a.arm}.zip"))[0])
    model = PPO.load(os.path.join(run, f"ppo_{a.arm}.zip"), device="cpu")
    env = CageLandingEnv(pool="sigma_pool.npz", arm=a.arm, rho=0.707,
                         render_mode="rgb_array")
    os.makedirs(a.out, exist_ok=True)
    rend = mujoco.Renderer(env.model, height=480, width=640)

    for ep in range(a.episodes):
        obs, info = env.reset(seed=1234 + ep)
        frames, k = [], 0
        sigma0 = env.sigma
        while True:
            act, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(act)
            if k % a.every == 0:
                f = wide_shot(env, rend)
                if f is not None:
                    frames.append(np.asarray(f))
            k += 1
            if term or trunc:
                break
        tag = "LANDED" if info["landed"] else "MISSED"
        name = f"{a.out}/cage_{a.arm}_{a.camera}_ep{ep}_{tag}"
        imageio.mimsave(f"{name}.gif", frames, duration=0.06, loop=0)
        # contact sheet: 6 evenly spaced frames
        idx = np.linspace(0, len(frames) - 1, 6).astype(int)
        sheet = np.concatenate([frames[i] for i in idx], axis=1)
        imageio.imwrite(f"{name}_sheet.png", sheet)
        print(f"  ep{ep}: {tag}  sigma={sigma0:.2f} m  miss={info['h_dist']:.2f} m  "
              f"{len(frames)} frames -> {name}.gif")

if __name__ == "__main__":
    main()
