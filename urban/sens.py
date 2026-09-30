"""Sensitivity of the ladder and the trained learners to the fast simulator's assumptions (no retraining).
One condition per call; the assumption to change comes in through the environment:
    SENS_LEAVE=10,120 | SENS_T_MAX=400 | SENS_DRONE_SEP=50 | HUNTER_T_COMMIT=0.5 | RIG_CALIB=rig_calib_half.json
    python sens.py clear|fog|rig|unmapped LABEL      -> sens/<cond>_<LABEL>.json
    (unmapped: base only, for the paper's gain over the best planner; not part of the sensitivity study)
Deciders on the ladder's 4,096 test flights (corridor, seed 2026): rule, shield (the planner the learners copied),
safe planner, and the three learners (dice read-out). Learner gains are paired, flight by flight."""
import json, os, sys, time, torch
import urban_rl as R, urban_agents as A, urban_env as U
cond, label = sys.argv[1], sys.argv[2]
R.MAP, R.NAV = "logan_map_lidar.npz", "gps"
R.FOG = cond == "fog"
R.UNMAPPED = cond == "unmapped"
R.BIRDS = R.HUNTER = R.RIG = cond == "rig"
ck = {"clear": "rl_corridor_eyes_lidar_gps_s{}_sampled.pt", "fog": "rl_corridor_eyes_lidar_gps_fog_s{}_sampled.pt",
      "rig": "rl_corridor_eyes_lidar_gps_birds_hunter_rig_s{}_sampled.pt",
      "unmapped": "rl_corridor_eyes_lidar_gps_unmapped_s{}_sampled.pt"}[cond]
torch.manual_seed(2026); t0 = time.time()
ladder = lambda agent: U.run(R.world_env(4096, "corridor", 2026), agent)
res = {k: ladder(a) for k, a in (("rule", A.Rule()), ("shield", A.MPC("obs")), ("safeplan", A.MPC("obs", "safeplan")))}
clean = {k: v["clean"].float() for k, v in res.items()}
out = dict(cond=cond, label=label, env={k: os.environ[k] for k in os.environ if k.startswith(("SENS_", "HUNTER_", "RIG_CALIB"))},
           **{k: 100 * v.mean().item() for k, v in clean.items()})
if cond == "rig":
    out["struck_magpie_rule"] = 100 * res["rule"]["struck_magpie"].item() if "struck_magpie" in res["rule"] else None
best = max(("shield", "safeplan"), key=lambda k: out[k]); out["best_planner"] = best
for sd in (0, 1, 2):
    env = R.world_env(8, "corridor", 1); o = env.reset(); gen = A.MPC("obs"); gen.reset(env); gen.act(o)
    net = R.Net(R.inputs(o, gen, True).shape[-1]).to(R.DEV)
    net.load_state_dict(torch.load(ck.format(sd), map_location=R.DEV))
    s = ladder(R.Learned(net, greedy=False, eyes=True))["clean"].float()
    row = {"clean": 100 * s.mean().item()}
    for k in ("shield", best):
        d = s - clean[k]; row[f"vs_{k}"] = [100 * d.mean().item(), 196 * d.std().item() / len(d) ** 0.5]
    out[f"s{sd}"] = row
out["minutes"] = round((time.time() - t0) / 60, 1)
json.dump(out, open(f"sens/{cond}_{label}.json", "w"), indent=1)
print(json.dumps({k: (round(v, 2) if isinstance(v, float) else v) for k, v in out.items() if not k.startswith("s") or k == "safeplan" or k == "shield"}),
      {f"s{i}": [round(out[f's{i}']['clean'], 1), round(out[f's{i}']['vs_shield'][0], 2)] for i in range(3)})
