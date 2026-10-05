import sys; sys.path.insert(0,"/home/sai/RL_projects/rl-landing/papers/research/scripts")
import wind_stats, hold_turb
r=wind_stats.stats(15.0,30.0); print({k:round(v,2) for k,v in r.items()})
r=hold_turb.run(15.0,30.0,0.04); print({k:round(v,3) for k,v in r.items()})
r=hold_turb.run(18.0,30.0,0.04); print({k:round(v,3) for k,v in r.items()})
