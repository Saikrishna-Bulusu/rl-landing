import json, numpy as np
from hold_steady import run, summarise
Us = np.arange(0, 40.01, 0.5); res = {}
for cda in (0.04, 0.08, 0.12):
    tr = run(Us, cda, "rule", T=40.0, ramp=10.0)
    res[f"cda{cda}_rule_ramp10"] = summarise(Us, tr)
json.dump(res, open("hold_ramp.json", "w"))
