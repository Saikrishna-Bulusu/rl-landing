import sys; sys.path.insert(0,"/home/sai/RL_projects/rl-landing/papers/research/scripts")
import numpy as np
from hold_steady import run, summarise
Us=np.array([10.,11.,11.5,12.,26.0,26.25,26.5,27.0])
for r in summarise(Us, run(Us,0.04,"rule",30.0,0.0)):
    print(r['U'],"maxdrift %.2f final %.2f tilt_max %.1f"%(r['maxdrift'],r['finaldrift'],r['tilt_max']))
