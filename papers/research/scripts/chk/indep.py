# independent steady solver: unknowns (theta, T); 2D force balance, fresh code
import math
from scipy.optimize import fsolve
m,g,kt,crd,rho=2.5843,9.81,8.54858e-6,8.06428e-5,1.225
Tmax=4*kt*1000**2; mg=m*g
def eq(x,U,cda):
    th,T=x
    w=math.sqrt(T/(4*kt))
    Dp=4*crd*w*U*math.cos(th)      # prop drag along horizontal-ish, per-rotor c*w*v_perp with v_perp=U cos th
    Db=0.5*rho*cda*U*U
    # body axes: thrust along body z tilted into wind; drag along air velocity (horizontal, downwind)
    # prop drag acts along the in-plane relative wind direction (in rotor plane), here direction (cos th, sin th)
    return [-T*math.sin(th)+Dp*math.cos(th)+Db, T*math.cos(th)+Dp*math.sin(th)-mg]
print("Tmax",Tmax,"mg",mg,"acos",math.degrees(math.acos(mg/Tmax)))
for U in (5,18,26,26.6,27,28,30,35):
    th,T=fsolve(eq,[0.3,mg],args=(U,0.04))
    print(U, "tilt %.1f"%math.degrees(th), "T%% %.1f"%(100*T/Tmax))
# tilt = 45 deg crossing
lo,hi=20,35
f=lambda U: math.degrees(fsolve(eq,[0.5,mg*1.2],args=(U,0.04))[0])-45
for _ in range(40):
    mid=(lo+hi)/2
    if f(mid)>0: hi=mid
    else: lo=mid
print("45deg at",mid)
