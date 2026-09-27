#!/usr/bin/env python3
"""
decider_server.py -- the fast simulator's planner or learner, unchanged, deciding for ONE drone
flown elsewhere (gz_learner.py: PX4 in Gazebo). Reads one JSON observation per line on stdin,
writes one JSON command per line on stdout, every 0.25 s of simulated time -- the rate it trained
at (it decides every second call, with the bird reflex in between, exactly as in urban_env).

The observation carries what Gazebo can say about this flight: position and velocity, the pad,
the marker lock, the magpie. Everything the Gazebo world does not have -- helicopters, other
drones, people, cars -- is reported absent, as the fast simulator reports an empty sky.

    CUDA_VISIBLE_DEVICES="" python decider_server.py --decider {planner,learner} [--ckpt ...]
Frames: the TRAINING map's (x east, y north, z up), not Gazebo's or PX4's.
"""
import argparse
import json
import sys

import numpy as np
import torch

import urban_agents as A
import urban_env as U
import urban_rl as R

MAP = "logan_map_lidar.npz"


class BridgeEnv:
    """What the deciders ask of an environment besides the observation: the survey map (for
    the planner's and the rule's obstacle checks) and the stage-3 switch. voxel=False: Gazebo's
    Logan world is the survey. voxel=True (gz_voxel.py's world): the voxel map the flight builds
    from the depth camera's images arrives with each observation (the columns whose top rose),
    and known_rel reads it around the drone's own position estimate -- the one it was built with."""
    n, dev, p_new, d_heli, d_drone = 1, "cpu", 0.0, U.D_HELI, U.D_DRONE

    def __init__(self, voxel=False, local3d=False):
        m = np.load(f"{U.HERE}/{MAP}")
        self.H = torch.as_tensor(m["H"].astype(np.float32))
        self.half, self.res = float(m["half"]), float(m["res"])
        self.p_new = 1.0 if voxel else 0.0
        self.local3d = local3d                 # the rule flies local_planner's path on this map
        self.vox, self.me = torch.zeros_like(self.H), torch.zeros(1, 2)

    def tops(self, xy):
        """The map as the tops of 1 m columns at xy (numpy, (..., 2), training map frame): survey + voxel map."""
        ij = ((xy + self.half) / self.res).astype(int).clip(0, self.H.shape[0] - 1)
        return np.maximum(self.H.numpy()[ij[..., 1], ij[..., 0]], self.vox.numpy()[ij[..., 1], ij[..., 0]])

    def see(self, d):
        self.me = torch.tensor([d["bel_xy"]], dtype=torch.float32)
        for x, y, top in d.get("vox", []):              # each column and its 8 neighbours: the fast simulator flies a
            i, j = int((y + self.half) / self.res), int((x + self.half) / self.res)   # point, PX4's x500 is 0.9 m across
            blk = self.vox[i - 1:i + 2, j - 1:j + 2]
            blk.clamp_(min=top)

    def survey_at(self, xy):
        ij = ((xy + self.half) / self.res).long().clamp(0, self.H.shape[0] - 1)
        return self.H[ij[..., 1], ij[..., 0]]

    height_at = survey_at

    def known_rel(self, d):
        if not self.p_new:
            return torch.zeros(d.shape[:-1])
        ij = ((self.me.reshape(1, *([1] * (d.dim() - 2)), 2) + d + self.half) / self.res).long().clamp(0, self.H.shape[0] - 1)
        return self.vox[ij[..., 1], ij[..., 0]]


def obs_from(d):
    t = lambda v, dt=torch.float32: torch.tensor([v], dtype=dt)
    far = lambda *s: torch.full((1, *s), U.FAR)
    G = U.N_PEOPLE + U.N_CAR
    bird_seen = bool(d["bird_seen"])
    return {"t": float(d["t"]), "pos_z": t(d["pos_z"]), "vel": t(d["vel"]), "bel_xy": t(d["bel_xy"]),
            "pad_xy": t(d["pad_xy"]), "pad_rel_bel": t(d["pad_rel_bel"]), "pad_rel_vis": t(d["pad_rel_vis"]),
            "pad_rel_fix": t(d["pad_rel_fix"]), "seen": t(d["seen"], torch.bool),
            "ever_seen": t(d["ever_seen"], torch.bool), "track": t(d["track"], torch.bool),
            "hls_rel_bel": t(d["hls_rel_bel"]),
            "heli_rel": far(3), "heli_vel": torch.zeros(1, 3), "heli_seen": t(False, torch.bool),
            "aware": t(False, torch.bool), "obliged": t(False, torch.bool),
            "drone_rel": far(U.N_DRONE, 3), "drone_vel": torch.zeros(1, U.N_DRONE, 3),
            "drone_seen": torch.zeros(1, U.N_DRONE, dtype=torch.bool),
            "ground_rel": far(G, 3), "ground_seen": torch.zeros(1, G, dtype=torch.bool),
            "ground_vel": torch.zeros(1, G, 3), "done": t(False, torch.bool),
            "bird_rel": t(d["bird_rel"]) if bird_seen else far(3), "bird_vel": t(d["bird_vel"]),
            "bird_seen": t(bird_seen, torch.bool), "fog": t(False, torch.bool)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--decider", choices=("planner", "learner", "rule"), required=True)
    ap.add_argument("--ckpt", default="rl_corridor_eyes_lidar_gps_birds_s1_mean.pt")
    ap.add_argument("--voxel", action="store_true", help="stage 3: plan on the voxel map the flight sends")
    ap.add_argument("--local3d", action="store_true", help="stage 3: the rule flies a local 3D planner's path")
    a = ap.parse_args()
    torch.set_num_threads(2)
    env = BridgeEnv(a.voxel, a.local3d)
    # the learner's inputs carry the magpie only if it trained with one (urban_rl.features); the stage-3
    # learners (--voxel) trained without birds, on the unmapped world
    R.MAP, R.NAV, R.BIRDS = MAP, "gps", not a.voxel
    if a.decider == "rule":                   # stage 3, static obstacles: the rule flying the local planner's path
        agent = A.Rule()
    elif a.decider == "planner":
        agent = A.MPC("obs")
    else:
        probe = U.UrbanEnv(8, "corridor", seed=1, map_path=f"{U.HERE}/{MAP}", nav="gps", device="cpu",
                           **({"p_new": U.P_NEW} if a.voxel else {"p_swoop": U.P_SWOOP}))
        o = probe.reset(); g = A.MPC("obs"); g.reset(probe); g.act(o)
        net = R.Net(R.inputs(o, g, True).shape[-1])
        net.load_state_dict(torch.load(f"{U.HERE}/{a.ckpt}", map_location="cpu", weights_only=True)); net.eval()
        read = "mean" if "_mean" in a.ckpt else ("sampled" not in a.ckpt)
        agent = R.Learned(net, greedy=read, eyes=True)
    agent.reset(env)
    print(json.dumps({"ready": a.decider}), flush=True)
    for line in sys.stdin:
        d = json.loads(line)
        env.see(d)
        cmd = agent.act(obs_from(d))
        lp = getattr(agent, "local", None) or getattr(getattr(agent, "gen", None), "local", None)
        path = [[round(float(v), 2) for v in p] for p in lp.path] if lp is not None and lp.path is not None else None
        print(json.dumps({"cmd": [round(float(c), 4) for c in cmd[0]], "path": path}), flush=True)


if __name__ == "__main__":
    main()
