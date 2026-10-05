"""Steady hover against a steady horizontal wind U (analytic force balance, uniform rotor speed), x500_mjx constants."""
import math, numpy as np
from scipy.optimize import brentq
MASS, G, KT, C_RD, RHO = 2.5843, 9.81, 8.54858e-6, 8.06428e-5, 1.225
W_MIN, W_MAX = 150.0, 1000.0
TMAX = 4 * KT * W_MAX**2
MG = MASS * G
def solve(U, cda):
    """unknown tilt th and total thrust T: T(-sin,cos) + Dp(cos,sin) + (Db,0) + (0,-mg) = 0"""
    def res(th):
        # horizontal: -T sin + Dp cos + Db = 0 ; vertical: T cos + Dp sin = mg  -> T from vertical, residual horizontal
        def T_of(th):
            T = MG / math.cos(th)
            for _ in range(60):                       # Dp depends on w(T): fixed point
                w = math.sqrt(T / (4 * KT)); Dp = 4 * w * C_RD * U * math.cos(th)
                T = (MG - Dp * math.sin(th)) / math.cos(th)
            return T, Dp
        T, Dp = T_of(th)
        return -T * math.sin(th) + Dp * math.cos(th) + 0.5 * RHO * cda * U * U, T, Dp
    if U == 0: return 0.0, MG, 0.0, 0.0
    f = lambda th: res(th)[0]
    th = brentq(f, 0.0, 1.45)
    _, T, Dp = res(th)
    return th, T, Dp, 0.5 * RHO * cda * U * U
def row(U, cda):
    th, T, Dp, Db = solve(U, cda)
    w = math.sqrt(T / (4 * KT)); u = (w - W_MIN) / (W_MAX - W_MIN)
    return math.degrees(th), T, T / TMAX * 100, u, Dp, Db
if __name__ == "__main__":
    print(f"mass {MASS} kg  weight {MG:.2f} N  Tmax {TMAX:.2f} N  T/W {TMAX/MG:.3f}  hover thrust {MG/TMAX*100:.1f}% of max")
    print("pure-hover tilt at full thrust (no drag): acos(mg/Tmax) = %.1f deg" % math.degrees(math.acos(MG / TMAX)))
    for cda in (0.0, 0.04, 0.08, 0.12):
        print(f"\nCdA {cda} m^2")
        print("  U   tilt   T[N]  T%max  u(mean)  propdrag[N] bodydrag[N]")
        for U in (0, 5, 10, 12, 15, 18, 20, 22, 24, 26, 28, 30, 35, 40):
            try:
                th, T, p, u, Dp, Db = row(U, cda)
                print(f"{U:4d} {th:6.1f} {T:6.1f} {p:6.1f} {u:7.3f} {Dp:9.2f} {Db:9.2f}" + ("  >45deg" if th > 45 else "") + ("  >Tmax" if T > TMAX else ""))
            except ValueError:
                print(f"{U:4d} no equilibrium below 83 deg")
        tilt_gap = lambda U: row(U, cda)[0] - 45
        thr_gap = lambda U: row(U, cda)[1] - TMAX
        # first U where tilt reaches 45 deg and where mean thrust reaches Tmax (scan, 0.05 m/s)
        def first(f):
            for U in np.arange(1, 150, 0.05):
                try:
                    if f(U) >= 0: return f"{U:.1f}"
                except ValueError: return f"{U:.1f}"
            return ">150"
        print(f"  -> tilt 45 deg at U={first(tilt_gap)} m/s; mean thrust = Tmax at U={first(thr_gap)} m/s")
