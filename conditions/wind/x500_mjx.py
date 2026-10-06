#!/usr/bin/env python3
"""
x500_mjx.py -- the rig v2 x500 flown by an emulation of PX4's multicopter controller in MuJoCo MJX, in gusty wind,
thousands of drones at once on the GPU: where the wind learners train (Sai, 26 Sept: "target RL on gradually
increasing wind conditions", MuJoCo MJX, everything on the GPU).

The drone is PX4's Gazebo model, not a guess:
  airframe  x500_base + gimbal + rig v2's camera link: 2.584 kg, CoM 1.6 cm below base_link and 3.4 mm ahead of the
            rotor centre (the gimbal hangs forward), composite inertia
  motors    Gazebo's MulticopterMotorModel: thrust 8.54858e-6 w^2, w = 150 + 850 u rad/s (SIM_GZ_EC_MIN/MAX),
            first-order lag 12.5 ms up / 25 ms down, propeller drag 8.06428e-5 |w| v_perp, rolling moment 1e-6 |w| v_perp
  PX4       PositionControl (velocity PID -> thrust vector, tilt limit, thrust margin, anti-windup), bodyzToAttitude,
            AttitudeControl (reduced attitude, yaw weight 0.4), RateControl (PID, integral factor and limit),
            control allocation (pseudo-inverse of the CA_ROTOR effectiveness, PX4's normalisation, sequential
            desaturation with airmode off); default gains (the x500 airframe sets only MPC_THR_HOVER, and the hover
            thrust estimator converges to the true hover throttle, used here)
  wind      mean + Dryden-like turbulence (MIL-F-8785C low altitude) + random 1-cosine gusts; it pushes through the
            propeller drag (as in Gazebo) and body drag 0.5 rho CdA |v| v (Gazebo had none: gz_weather.py), pushed where
            gz_winddrag pushes it, at base_link's origin, 1.6 cm above the CoM
Frames: world ENU, body FLU (MuJoCo's); PX4's NED/FRD laws are written here in ENU/FLU, the allocation in FRD.
Checked against PX4 in Gazebo (shared/px4_gazebo/px4_steps.py, truth + sim time; step responses, then the steady-wind
limit):  python conditions/wind/x500_mjx.py --check
"""
import argparse
import math
import os

import numpy as np

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")     # share the GPU with Gazebo / torch
import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.dirname(os.path.dirname(HERE))
G = 9.81
DT = 0.004                       # physics and PX4's inner loops, 250 Hz
N_SUB = 10                       # physics steps per decision: 25 Hz, what the Gazebo flight loop sustains

# ---- airframe (PX4 Tools/simulation/gz/models: x500_base/model.sdf, gimbal/model.sdf; gz_unified.rig_link) --------------
MASS = 2.5843
INERTIA = (0.0291, 0.0291, 0.0444)          # about the CoM: base + rotors + gimbal (4 x 0.1 kg below) + rig link
ROTORS = np.array([[0.174, -0.174], [-0.174, 0.174], [0.174, 0.174], [-0.174, -0.174]])   # FLU, rotor_0..3
ROTOR_Z = 0.076                             # hubs above the CoM
COM_X = 0.0034                              # m, the CoM ahead of the rotor centre: the SDFs' masses, and a calm Gazebo
                                            # hover's front-back motor split (+0.0099 each, 6 Oct) agree
DRAG_AT = np.array([-COM_X, 0.0, 0.016])    # FLU from the CoM: base_link's origin, where gz_winddrag pushes the body drag
SPIN = np.array([1.0, 1.0, -1.0, -1.0])     # ccw, ccw, cw, cw
KT, KM = 8.54858e-06, 0.016
W_MIN, W_MAX = 150.0, 1000.0
TAU_UP, TAU_DN = 0.0125, 0.025
C_RD, C_RM = 8.06428e-05, 1e-06
RHO = 1.225
CDA_H, CDA_V = 0.04, 0.05                   # m^2, body drag areas: an assumption, the range wind tunnels report for
                                            # 1-3 kg multirotors (Russell et al., NASA 2016); Gazebo's x500 had none
W_HOVER = math.sqrt(MASS * G / 4 / KT)      # 861 rad/s
U_HOVER = (W_HOVER - W_MIN) / (W_MAX - W_MIN)   # 0.836: PX4's hover throttle once its estimator has converged

# ---- PX4 defaults (src/modules/mc_pos_control, mc_att_control, mc_rate_control; airframe 4001_gz_x500) ------------------
VEL_P = jnp.array([1.8, 1.8, 4.0])
VEL_I = jnp.array([0.4, 0.4, 2.0])
VEL_D = jnp.array([0.2, 0.2, 0.0])
TILT_MAX = math.radians(45.0)
THR_MIN, THR_MAX, THR_XY_MARG = 0.12, 1.0, 0.3
VXY_MAX, VZ_UP, VZ_DN = 12.0, 3.0, 1.5
ATT_P = jnp.array([6.5, 6.5, 2.8])
YAW_W = 0.4
RATE_MAX = jnp.radians(jnp.array([220.0, 220.0, 200.0]))
RATE_P = jnp.array([0.15, 0.15, 0.2])
RATE_I = jnp.array([0.2, 0.2, 0.1])
RATE_D = jnp.array([0.003, 0.003, 0.0])
RATE_INT_LIM = jnp.array([0.3, 0.3, 0.3])
LP_VDOT, LP_RACC = 5.0, 30.0                # Hz: MPC_VELD_LP, IMU_DGYRO_CUTOFF


def _allocation():
    """PX4's mixer for the x500: effectiveness from CA_ROTORn_PX/PY/KM (FRD, CA_ROTOR_CT 6.5), pseudo-inverse,
    normalised as ControlAllocationPseudoInverse (roll and pitch one scale, yaw by its column max, thrust to 1)."""
    pos = np.array([[0.13, 0.22], [-0.13, -0.20], [0.13, -0.22], [-0.13, 0.20]])   # 4001_gz_x500
    km = np.array([0.05, 0.05, -0.05, -0.05])
    ct = 6.5
    E = np.zeros((4, 4))                                    # rows roll, pitch, yaw, thrust_z (FRD)
    for i, ((px, py), k) in enumerate(zip(pos, km)):
        E[:, i] = [-ct * py, ct * px, ct * k, -ct]          # ct (p x (0,0,-1)) - ct km (0,0,-1); thrust ct (0,0,-1)
    mix = np.linalg.pinv(E)                                 # actuators x controls
    rp = max(math.sqrt((mix[:, 0] ** 2).sum() / 2), math.sqrt((mix[:, 1] ** 2).sum() / 2))
    scale = np.array([rp, rp, mix[:, 2].max(), np.abs(mix[:, 3]).mean()])
    return jnp.array(mix / scale), jnp.array(E), jnp.array(scale)


MIX, EFF, SCALE = _allocation()


# ---- quaternions (w, x, y, z), body -> world ---------------------------------------------------------------------------------
def qmul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return jnp.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                      w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def qconj(q):
    return q * jnp.array([1.0, -1.0, -1.0, -1.0])


def qrot(q):
    w, x, y, z = q
    return jnp.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                      [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                      [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def qfrom_rot(Rm):
    """Rotation matrix -> quaternion (branch-free enough for jit: Shepperd's method via the largest diagonal)."""
    t = jnp.trace(Rm)
    cands = jnp.array([[1 + t, Rm[2, 1] - Rm[1, 2], Rm[0, 2] - Rm[2, 0], Rm[1, 0] - Rm[0, 1]],
                       [Rm[2, 1] - Rm[1, 2], 1 + Rm[0, 0] - Rm[1, 1] - Rm[2, 2], Rm[0, 1] + Rm[1, 0], Rm[0, 2] + Rm[2, 0]],
                       [Rm[0, 2] - Rm[2, 0], Rm[0, 1] + Rm[1, 0], 1 - Rm[0, 0] + Rm[1, 1] - Rm[2, 2], Rm[1, 2] + Rm[2, 1]],
                       [Rm[1, 0] - Rm[0, 1], Rm[0, 2] + Rm[2, 0], Rm[1, 2] + Rm[2, 1], 1 - Rm[0, 0] - Rm[1, 1] + Rm[2, 2]]])
    k = jnp.argmax(jnp.array([t, Rm[0, 0], Rm[1, 1], Rm[2, 2]]))
    q = cands[k]
    q = q / jnp.linalg.norm(q)
    return q * jnp.sign(q[0] + 1e-12)


def qbetween(u, v):
    """Shortest rotation taking unit u to unit v (PX4 Quatf(u, v))."""
    c = jnp.cross(u, v)
    q = jnp.concatenate([jnp.array([1.0 + jnp.dot(u, v)]), c])
    n = jnp.linalg.norm(q)
    return jnp.where(n > 1e-6, q / jnp.maximum(n, 1e-9), jnp.array([0.0, 1.0, 0.0, 0.0]))


# ---- PX4 ------------------------------------------------------------------------------------------------------------------------
def bodyz_to_q(bz, yaw):
    """ControlMath::bodyzToAttitude in ENU/FLU: body z along the thrust, nose along yaw."""
    y_c = jnp.array([-jnp.sin(yaw), jnp.cos(yaw), 0.0])
    bx = jnp.cross(y_c, bz)
    bx = jnp.where(bz[2] < 0, -bx, bx)
    bx = bx / jnp.maximum(jnp.linalg.norm(bx), 1e-9)
    by = jnp.cross(bz, bx)
    return qfrom_rot(jnp.stack([bx, by, bz], axis=1))


def position_control(v, vel_sp, acc_ff, vel_int, vdot):
    """PositionControl::_velocityControl + _accelerationControl, ENU. Returns the thrust vector (normalised, up),
    the new integral."""
    vel_sp = jnp.concatenate([vel_sp[:2] * jnp.minimum(1.0, VXY_MAX / jnp.maximum(jnp.linalg.norm(vel_sp[:2]), 1e-6)),
                              jnp.clip(vel_sp[2:], -VZ_DN, VZ_UP)])
    verr = vel_sp - v
    acc_sp = acc_ff + VEL_P * verr + vel_int - VEL_D * vdot
    bz = jnp.array([acc_sp[0], acc_sp[1], G + acc_sp[2]])
    bz = bz / jnp.maximum(jnp.linalg.norm(bz), 1e-6)
    tilt = jnp.arccos(jnp.clip(bz[2], -1.0, 1.0))                     # ControlMath::limitTilt
    ax = jnp.where(tilt > TILT_MAX, 1.0, 0.0)
    h = bz[:2] / jnp.maximum(jnp.linalg.norm(bz[:2]), 1e-9)
    bz = ax * jnp.array([h[0] * math.sin(TILT_MAX), h[1] * math.sin(TILT_MAX), math.cos(TILT_MAX)]) + (1 - ax) * bz
    coll = jnp.maximum(U_HOVER * (G + acc_sp[2]) / G / bz[2], THR_MIN)
    thr = bz * coll
    # vertical priority with a horizontal margin
    thr_xy_n = jnp.linalg.norm(thr[:2])
    thr_z = jnp.minimum(thr[2], jnp.sqrt(THR_MAX ** 2 - jnp.minimum(thr_xy_n, THR_XY_MARG) ** 2))
    thr_xy_max = jnp.sqrt(jnp.maximum(THR_MAX ** 2 - thr_z ** 2, 0.0))
    thr_xy = thr[:2] * jnp.minimum(1.0, thr_xy_max / jnp.maximum(thr_xy_n, 1e-9))
    thr = jnp.concatenate([thr_xy, thr_z[None]])
    # anti-windup: vertical at the thrust limits, horizontal by tracking (Rundqwist)
    ez = jnp.where(((thr_z <= THR_MIN) & (verr[2] <= 0)) | ((thr_z >= THR_MAX) & (verr[2] >= 0)), 0.0, verr[2])
    acc_prod = thr_xy * (G / U_HOVER)
    sat = jnp.sum(acc_sp[:2] ** 2) > jnp.sum(acc_prod ** 2)
    exy = jnp.where(sat, verr[:2] - 2.0 / VEL_P[0] * (acc_sp[:2] - acc_prod), verr[:2])
    vel_int = vel_int + VEL_I * jnp.concatenate([exy, ez[None]]) * DT
    vel_int = vel_int.at[2].set(jnp.clip(vel_int[2], -G, G))
    return thr, vel_int


def attitude_control(q, qd):
    """AttitudeControl::update: roll/pitch first (reduced attitude), yaw weighted; rate setpoint in the body frame."""
    e_z, e_z_d = qrot(q)[:, 2], qrot(qd)[:, 2]
    qd_red = qmul(qbetween(e_z, e_z_d), q)
    q_mix = qmul(qconj(qd_red), qd)
    q_mix = q_mix * jnp.sign(q_mix[0] + 1e-12)
    qd = qmul(qd_red, jnp.array([jnp.cos(YAW_W * jnp.arccos(jnp.clip(q_mix[0], -1.0, 1.0))), 0.0, 0.0,
                                 jnp.sin(YAW_W * jnp.arcsin(jnp.clip(q_mix[3], -1.0, 1.0)))]))
    qe = qmul(qconj(q), qd)
    eq = 2.0 * jnp.sign(qe[0] + 1e-12) * qe[1:]
    return jnp.clip(eq * ATT_P, -RATE_MAX, RATE_MAX)


def desat_gain(vec, u, lo, hi):
    """ControlAllocationSequentialDesaturation::computeDesaturationGain."""
    ok = jnp.abs(vec) > 1e-7
    k_lo = jnp.where(ok & (u < lo), (lo - u) / jnp.where(ok, vec, 1.0), 0.0)
    k_hi = jnp.where(ok & (u > hi), (hi - u) / jnp.where(ok, vec, 1.0), 0.0)
    ks = jnp.concatenate([k_lo, k_hi])
    return jnp.minimum(ks.min(), 0.0) + jnp.maximum(ks.max(), 0.0)


def desaturate(u, vec, increase_only=False, lo=0.0, hi=1.0):
    for f in (1.0, 0.5):
        g = f * desat_gain(vec, u, lo, hi)
        u = u + jnp.where(increase_only & (g < 0), 0.0, g) * vec
    return u


def allocate(torque_flu, coll):
    """Sequential desaturation, airmode off (mixAirmodeDisabled): roll/pitch + thrust, thrust only reduced, then roll
    and pitch trimmed, yaw last with a 15% margin. Returns motor throttles and the unallocated torque (FLU)."""
    tq = torque_flu * jnp.array([1.0, -1.0, -1.0])                      # FLU -> FRD
    u = MIX[:, 0] * tq[0] + MIX[:, 1] * tq[1] + MIX[:, 3] * (-coll)
    u = desaturate(u, MIX[:, 3], increase_only=True)
    u = desaturate(u, MIX[:, 0])
    u = desaturate(u, MIX[:, 1])
    u = u + MIX[:, 2] * tq[2]
    u = desaturate(u, MIX[:, 2], hi=1.15)
    u = desaturate(u, MIX[:, 3], increase_only=True)
    u = jnp.clip(u, 0.0, 1.0)
    alloc = (EFF @ u) * SCALE
    return u, (tq - alloc[:3]) * jnp.array([1.0, -1.0, -1.0])


def rate_control(om, rate_sp, rate_int, racc, unalloc):
    """RateControl::update: PID, integral shrunk for large errors, frozen in a saturated direction."""
    err = rate_sp - om
    torque = RATE_P * err + rate_int - RATE_D * racc
    e_i = jnp.where(unalloc > 1e-4, jnp.minimum(err, 0.0), jnp.where(unalloc < -1e-4, jnp.maximum(err, 0.0), err))
    f = jnp.maximum(0.0, 1.0 - (e_i / math.radians(400.0)) ** 2)
    rate_int = jnp.clip(rate_int + f * RATE_I * e_i * DT, -RATE_INT_LIM, RATE_INT_LIM)
    return torque, rate_int


# ---- the rigid body in MJX ---------------------------------------------------------------------------------------------------
XML = f"""<mujoco model="x500_rigv2"><option timestep="{DT}" gravity="0 0 -{G}"/>
<worldbody><body name="x500" pos="0 0 10"><freejoint/>
<inertial pos="0 0 0" mass="{MASS}" diaginertia="{INERTIA[0]} {INERTIA[1]} {INERTIA[2]}"/></body></worldbody></mujoco>"""
MJ = mujoco.MjModel.from_xml_string(XML)
MX = mjx.put_model(MJ)
ROT3 = jnp.array(np.c_[ROTORS - [COM_X, 0.0], np.full(4, ROTOR_Z)])   # hubs from the CoM
SPIN_J = jnp.array(SPIN)


def forces(q, v, om, w, wind, cda):
    """World-frame force and torque on the CoM: thrust, rotor reaction, propeller drag and rolling moment against the
    air (Gazebo's MulticopterMotorModel), body drag."""
    Rm = qrot(q)
    T = KT * w ** 2                                    # gz-sim 8.15's motor model: no thrust loss with speed
    f_b = jnp.array([0.0, 0.0, T.sum()])
    tau_b = jnp.array([(ROT3[:, 1] * T).sum(), -(ROT3[:, 0] * T).sum(), -(SPIN_J * KM * T).sum()])
    v_hub = (v - wind)[None, :] + (Rm @ jnp.cross(om[None, :], ROT3).T).T     # air-relative, world
    vb = v_hub @ Rm                                                          # -> body
    perp = vb.at[:, 2].set(0.0)
    fd = -jnp.abs(w)[:, None] * C_RD * perp
    f_b = f_b + fd.sum(0)
    tau_b = tau_b + jnp.cross(ROT3, fd).sum(0) - (jnp.abs(w)[:, None] * C_RM * perp).sum(0)
    vr = v - wind
    f_body = -0.5 * RHO * jnp.array([cda[0] * jnp.linalg.norm(vr[:2]) * vr[0], cda[0] * jnp.linalg.norm(vr[:2]) * vr[1],
                                     cda[1] * jnp.abs(vr[2]) * vr[2]])
    return Rm @ f_b + f_body, Rm @ tau_b + jnp.cross(Rm @ DRAG_AT, f_body)


def init_state(pos, yaw=0.0, vel=(0.0, 0.0, 0.0)):
    """One drone hovering (motors at hover, integrals at the hover solution) at pos, ENU."""
    d = mjx.make_data(MX)
    q = jnp.array([jnp.cos(yaw / 2), 0.0, 0.0, jnp.sin(yaw / 2)])
    d = d.replace(qpos=jnp.concatenate([jnp.asarray(pos, jnp.float32), q]),
                  qvel=jnp.concatenate([jnp.asarray(vel, jnp.float32), jnp.zeros(3)]))
    z3 = jnp.zeros(3)
    return dict(d=d, w=jnp.full(4, W_HOVER), vel_int=z3, vdot=z3, rate_int=z3, racc=z3)


def physics_step(s, vel_sp, acc_ff, yaw_sp, wind, cda=jnp.array([CDA_H, CDA_V])):
    """PX4's loops, the motors, the forces, one MJX step (DT)."""
    d = s["d"]
    q, v, om = d.qpos[3:7], d.qvel[:3], d.qvel[3:6]
    thr, vel_int = position_control(v, vel_sp, acc_ff, s["vel_int"], s["vdot"])
    coll = jnp.linalg.norm(thr)
    qd = bodyz_to_q(thr / jnp.maximum(coll, 1e-6), yaw_sp)
    rate_sp = attitude_control(q, qd)
    # the allocation of the previous torque tells the rate loop where it saturated (control_allocator_status)
    torque, _ = rate_control(om, rate_sp, s["rate_int"], s["racc"], jnp.zeros(3))
    u, unalloc = allocate(torque, coll)
    _, rate_int = rate_control(om, rate_sp, s["rate_int"], s["racc"], unalloc)
    w_cmd = W_MIN + (W_MAX - W_MIN) * u
    w = s["w"] + (w_cmd - s["w"]) * jnp.where(w_cmd > s["w"], 1 - math.exp(-DT / TAU_UP), 1 - math.exp(-DT / TAU_DN))
    F, Tq = forces(q, v, om, w, wind, cda)
    d = d.replace(xfrc_applied=jnp.zeros((MJ.nbody, 6)).at[1].set(jnp.concatenate([F, Tq])))
    d = mjx.step(MX, d)
    a_v, a_r = 1 - math.exp(-2 * math.pi * LP_VDOT * DT), 1 - math.exp(-2 * math.pi * LP_RACC * DT)
    vdot = s["vdot"] + a_v * ((d.qvel[:3] - v) / DT - s["vdot"])
    racc = s["racc"] + a_r * ((d.qvel[3:6] - om) / DT - s["racc"])
    return dict(d=d, w=w, vel_int=vel_int, vdot=vdot, rate_int=rate_int, racc=racc)


def control_step(s, vel_sp, acc_ff, yaw_sp, wind, cda=jnp.array([CDA_H, CDA_V])):
    """One decision (N_SUB physics steps) with the setpoint held, as PX4 holds an offboard setpoint."""
    def body(c, _):
        return physics_step(c, vel_sp, acc_ff, yaw_sp, wind, cda), None
    return jax.lax.scan(body, s, None, length=N_SUB)[0]


def rpy(q):
    w, x, y, z = q
    return jnp.array([jnp.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)),
                      jnp.arcsin(jnp.clip(2 * (w * y - z * x), -1, 1)), jnp.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))])


# ---- checks -------------------------------------------------------------------------------------------------------------------------
def step_metrics(t, vel, t0, prev, tgt, span=4.8):
    sel = (t >= t0) & (t < t0 + span)
    frac = (vel[sel] - prev) / (tgt - prev)
    tt = t[sel] - t0
    cross = lambda f: float(tt[np.argmax(frac >= f)]) if (frac >= f).any() else float("nan")
    return cross(0.1), cross(0.632), cross(0.9), 100 * (frac.max() - 1)


def check():
    """PX4 in Gazebo (outputs/gazebo_wind/steps/steps.npz: the same commands, truth + sim time) vs this model."""
    import sys
    sys.path.insert(0, f"{R}/shared/px4_gazebo")
    from px4_steps import STEPS
    g = np.load(f"{R}/outputs/gazebo_wind/steps/steps.npz")
    tr, c = g["truth"], g["cmds"]
    tg = tr[:, 0]
    w_, x_, y_, z_ = tr[:, 4:8].T                         # the CoM (the model origin is at the feet, which swing back
    up = np.stack([2 * (x_ * z_ + w_ * y_), 2 * (y_ * z_ - w_ * x_), 1 - 2 * (x_ * x_ + y_ * y_)], axis=1)   # as it pitches)
    vg = np.gradient(tr[:, 1:4] + 0.224 * up, tg, axis=0)
    wq = tr[:, 4:8]
    tilt_g = np.degrees(np.arccos(np.clip(1 - 2 * (wq[:, 1] ** 2 + wq[:, 2] ** 2), -1, 1)))
    chg = np.where(np.any(np.abs(np.diff(c[:, 1:4], axis=0)) > 0.01, axis=1))[0] + 1
    t_chg = list(c[chg, 0])[1:]                     # when each new command was first sent (the first ends the climb)
    # MJX: hover at 30 m, then the same held commands (north, east, up -> ENU east, north, up)
    step = jax.jit(control_step)
    s = init_state((0.0, 0.0, 30.0), yaw=float(np.arctan2(*[0, 1])))
    zero = jnp.zeros(3)
    ts, vs, tilts, t = [], [], [], 0.0
    for secs, vn, ve, vu in STEPS:
        for _ in range(int(round(secs / (DT * N_SUB)))):
            s = step(s, jnp.array([ve, vn, vu], jnp.float32), zero, 0.0, zero)
            t += DT * N_SUB
            ts.append(t); vs.append(np.array(s["d"].qvel[:3])); r = np.array(rpy(s["d"].qpos[3:7]))
            tilts.append(math.degrees(math.acos(max(-1.0, min(1.0, math.cos(r[0]) * math.cos(r[1]))))))
    ts, vs, tilts = np.array(ts), np.array(vs), np.array(tilts)
    print(f"{'step':24s} {'Gazebo 10/63/90%  overshoot  tilt':40s} MJX 10/63/90%  overshoot  tilt")
    tm = 0.0
    prev = (0.0, 0.0, 0.0)
    for secs, vn, ve, vu in STEPS:
        tgt = (ve, vn, vu)
        if tgt != prev:
            ax = int(np.argmax(np.abs(np.subtract(tgt, prev))))
            span = min(secs, 4.8)
            t0g = t_chg.pop(0)
            mg = step_metrics(tg, vg[:, ax], t0g, prev[ax], tgt[ax], span)
            mm = step_metrics(ts, vs[:, ax], tm, prev[ax], tgt[ax], span)
            selg, selm = (tg >= t0g) & (tg < t0g + span), (ts >= tm) & (ts < tm + span)
            print(f"{'ENU'[ax]} {prev[ax]:+.1f}->{tgt[ax]:+.1f} m/s{'':9s} {mg[0]:.2f}/{mg[1]:.2f}/{mg[2]:.2f} s  {mg[3]:4.0f}%  {tilt_g[selg].max():4.1f} deg"
                  f"{'':10s}{mm[0]:.2f}/{mm[1]:.2f}/{mm[2]:.2f} s  {mm[3]:4.0f}%  {tilts[selm].max():4.1f} deg")
        prev = tgt
        tm += secs


def tilt_in_wind(speeds=(3, 7, 10, 12, 15, 18), cda=CDA_H):
    """Steady hover against a steady wind: tilt and throttle. cda=0 is Gazebo's x500 today (propeller drag only)."""
    step = jax.jit(control_step)
    cd = jnp.array([cda, CDA_V if cda else 0.0])
    out = []
    for ws in speeds:
        s = init_state((0.0, 0.0, 30.0))
        wind = jnp.array([float(ws), 0.0, 0.0])
        for _ in range(int(20 / (DT * N_SUB))):
            s = step(s, jnp.zeros(3), jnp.zeros(3), 0.0, wind, cd)
        r = np.array(rpy(s["d"].qpos[3:7]))
        drift = float(np.linalg.norm(np.array(s["d"].qvel[:2])))
        out.append((ws, math.degrees(math.acos(math.cos(r[0]) * math.cos(r[1]))), float(s["w"].max()), drift))
    return out


def check_hold():
    """Steady wind at 30 m, velocity setpoint 0, at the heading PX4 held: PX4 in Gazebo (outputs/gazebo_wind/hold/u*.npz,
    px4_steps.py --hover 60 --alt 30 under gz_winddrag with turbulence and gusts off, 6 Oct; u<U>[_tag].npz, the wind
    direction as logged) vs this model. Near 21-26 m/s, by where the wind comes from, a motor saturates and the drone is
    blown away: lean and drift must match there (6 Oct: MJX held 44 deg and 0 m/s at 26 m/s until the body drag was
    pushed where Gazebo pushes it and the CoM put where it is)."""
    import glob
    step = jax.jit(lambda s, w, yaw: control_step(s, jnp.zeros(3), jnp.zeros(3), yaw, w))
    prof = lambda h: math.log(max(h, 1.0) / 0.3) / math.log(10 / 0.3)
    for f in sorted(glob.glob(f"{R}/outputs/gazebo_wind/hold/u*.npz")):
        U, gz = float(os.path.basename(f)[1:3]), np.load(f)
        g, wl = gz["truth"], gz["wind"][len(gz["wind"]) // 2:]
        th = math.atan2(wl[:, 2].mean(), wl[:, 1].mean())                     # the wind blows toward th, ENU
        t0 = g[np.argmax(g[:, 3] >= 29.5), 0]                                 # the climb ends, the 60 s hover starts
        late = g[(g[:, 0] > t0 + 30) & (g[:, 0] < t0 + 60)]
        w_, x_, y_, z_ = late[:, 4:8].T
        yaw = float(np.arctan2(2 * (w_ * z_ + x_ * y_), 1 - 2 * (y_ * y_ + z_ * z_)).mean())
        tilt_g = float(np.degrees(np.arccos(np.clip(1 - 2 * (x_ ** 2 + y_ ** 2), -1, 1))).mean())
        spd_g = float(np.hypot(*(late[-1, 1:3] - late[0, 1:3])) / (late[-1, 0] - late[0, 0]))
        s, rows = init_state(jnp.array([0.0, 0.0, 30.0]), yaw=yaw), []
        for k in range(int(60 / (DT * N_SUB))):
            s = step(s, U / prof(30.0) * prof(float(s["d"].qpos[2])) * jnp.array([math.cos(th), math.sin(th), 0.0]), yaw)
            rows.append((*np.array(s["d"].qpos[:2]), float(qrot(s["d"].qpos[3:7])[2, 2])))
        r = np.array(rows[len(rows) // 2:])
        tilt_m, spd_m = float(np.degrees(np.arccos(np.clip(r[:, 2], -1, 1))).mean()), float(np.hypot(*(r[-1, :2] - r[0, :2])) / 30)
        print(f"steady {U:.0f} m/s at 30 m from {(math.degrees(th - yaw) + 180) % 360:3.0f} deg off the nose (left +): Gazebo tilt {tilt_g:4.1f} deg, drift {spd_g:4.2f} m/s"
              f"   MJX tilt {tilt_m:4.1f} deg, drift {spd_m:4.2f} m/s")
        assert abs(tilt_m - tilt_g) < 1.5 and abs(spd_m - spd_g) < 0.75, "the twin's wind limit no longer matches Gazebo's"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    if a.check:
        check()
        check_hold()
        for cda in (0.0, CDA_H):
            print(f"steady wind, body drag CdA {cda} m^2:",
                  "  ".join(f"{ws} m/s: tilt {tl:.1f} deg, top motor {wm:.0f} rad/s, drift {dr:.2f} m/s" for ws, tl, wm, dr in tilt_in_wind(cda=cda)))
