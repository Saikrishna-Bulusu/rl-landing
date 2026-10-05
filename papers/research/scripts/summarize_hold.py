import json,sys
d=json.load(open(sys.argv[1] if len(sys.argv)>1 else 'hold_steady.json'))
def lim(S, f):
    last=None
    for r in S:
        if f(r): last=r["U"]
        else: break
    return last
for k,S in d.items():
    settled=lambda r: r["finaldrift"]<=1.0 and r["speed_end"]<0.1 and r["tilt_max"]<68
    trans=lambda r: r["maxdrift"]<=1.0 and r["tilt_max"]<68
    ls=lim(S,settled); lt=lim(S,trans)
    rs=next(r for r in S if r["U"]==ls); 
    print(f"{k:28s} settled-hold up to {ls}  (tilt {rs['tilt_max']:.1f} deg, coll_end {rs['coll_end']:.3f}, top motor {rs['wtop_max']:.0f} rad/s)   transient<=1m up to {lt}")
