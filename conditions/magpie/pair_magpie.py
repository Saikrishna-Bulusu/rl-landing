"""Magpie-through-the-rig learners (3 seeds, dice read-out) against the SAFE planner on the ladder's 4,096 test flights,
paired flight by flight. The training run compared each seed only with the planner it copied (shield); this adds the
paired comparison with the best same-sensor planner.   python pair_magpie.py  -> pair_magpie.json"""
import json, torch
import urban_rl as R, urban_agents as A, urban_env as U
R.MAP, R.NAV, R.BIRDS, R.HUNTER, R.RIG = "logan_map_lidar.npz", "gps", True, True, True
torch.manual_seed(2026)
ladder = lambda agent: U.run(R.world_env(4096, "corridor", 2026), agent)
ref = {k: ladder(a)["clean"].float() for k, a in (("shield", A.MPC("obs")), ("safeplan", A.MPC("obs", "safeplan")))}
out = {k: 100 * v.mean().item() for k, v in ref.items()}
print("planners on the ladder:", out, flush=True)
for sd in (0, 1, 2):
    o = R.world_env(8, "corridor", 1).reset(); gen = A.MPC("obs"); env = R.world_env(8, "corridor", 1); o = env.reset(); gen.reset(env); gen.act(o)
    net = R.Net(R.inputs(o, gen, True).shape[-1]).to(R.DEV)
    net.load_state_dict(torch.load(f"rl_corridor_eyes_lidar_gps_birds_hunter_rig_s{sd}_sampled.pt", map_location=R.DEV))
    s = ladder(R.Learned(net, greedy=False, eyes=True))["clean"].float()
    row = {"clean": 100 * s.mean().item()}
    for k, r in ref.items():
        d = s - r; row[f"vs_{k}"] = [100 * d.mean().item(), 196 * d.std().item() / len(d) ** 0.5]
    out[f"s{sd}"] = row; print(sd, row, flush=True)
print("CHECK vs headroom run: shield", out["shield"], "(69.775 expected), safeplan", out["safeplan"], "(73.51 expected)")
json.dump(out, open("pair_magpie.json", "w"), indent=1)
