import sys, time, json
sys.path.insert(0, "/home/sai/RL_projects/rl-landing/conditions/wind")
import land_env as E
from dob import make
res = {}
print("validation yards (seed 20000), W18 only, choose k:")
best = None
for k in (0.5, 1.0, 1.5, 2.0):
    r = E.evaluate(make(k), 18.0, n=1024, seed=20000, fly=True, decide=False)
    print(" k", k, round(r["landed"], 3), "crash", round(r["crashed"], 3), "timeout", round(r["timeout"], 3), flush=True)
    if best is None or r["landed"] > best[1]: best = (k, r["landed"])
k = best[0]; print("chosen k", k)
out = []
for w in (9.0, 12.0, 15.0, 18.0, 20.0):
    r = E.evaluate(make(k), w, n=1024, seed=10000, fly=True, decide=False)
    out.append(dict(w10=w, landed=r["landed"], crashed=r["crashed"], hard=r["hard"], timeout=r["timeout"], t_mean=r["t_mean"]))
    print(" TEST seed 10000 W", w, {a: round(b, 3) for a, b in out[-1].items() if a != "w10"}, flush=True)
json.dump(dict(k=k, test=out), open("dob_test.json", "w"))
