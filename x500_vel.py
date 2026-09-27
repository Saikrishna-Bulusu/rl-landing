#!/usr/bin/env python3
"""
x500_vel.py -- a proper cascaded velocity controller for the x500.

WHY THIS REPLACES THE CARROT-WAYPOINT APPROACH

Feeding the aviary's PID a waypoint that moves with the drone makes a
position-seeking loop chase a target that runs away from it. Measured: it
overshoots the pad, swings back past it, and the oscillation grows until the
aircraft tumbles at 116 degrees and 24 m off target.

The bug that hid this: every calibration commanded a CONSTANT velocity in one
direction, which never requires the aircraft to stop or turn around. An
under-damped loop only diverges when it has to arrest. Constant-command testing
cannot see that, and it produced a confident "2% tracking error" result for a
controller that cannot fly to a point.

STRUCTURE (what PX4 does, so the eventual Gazebo transfer is honest)

  outer, velocity -> acceleration:
      a_des = KP_V * (v_target - v)  -  KD_V * a_measured
      F_des = m * (a_des + g)                          gravity feedforward
  attitude extraction:
      thrust  = F_des . z_body                         project onto body axis
      roll/pitch desired from the direction of F_des, clipped to MAX_TILT
  inner, attitude -> torque:
      rate_cmd = KP_ATT * (att_des - att) - KD_ATT * omega

The inner loop is necessary because ActionType.ATTITUDE is a raw MIXER, not a
controller: its roll/pitch inputs set motor differentials directly, so they
command torque. Nothing stabilises attitude unless we do it.

TUNING RULE, learned the hard way: every gain change must be validated with a
GO-TO-POINT-AND-STOP manoeuvre. A velocity measurement cannot distinguish
flying there from falling there.
"""
import numpy as np

G = 9.81


class VelocityController:

    def __init__(self, mass, hover_rpm, max_rpm,
                 kp_v=(1.2, 1.2, 2.0), kd_v=(0.15, 0.15, 0.1),
                 kp_att=0.9, kd_att=0.22, max_tilt_deg=20.0):
        self.m = float(mass)
        self.hover_rpm = float(hover_rpm)
        self.max_rpm = float(max_rpm)
        self.kp_v = np.asarray(kp_v, float)
        self.kd_v = np.asarray(kd_v, float)
        self.kp_att = float(kp_att)
        self.kd_att = float(kd_att)
        self.max_tilt = np.radians(max_tilt_deg)
        self._v_prev = np.zeros(3)

    def reset(self):
        self._v_prev = np.zeros(3)

    def __call__(self, v_target, vel, rpy, ang_vel, dt):
        v = np.asarray(vel, float)
        a_meas = (v - self._v_prev) / max(dt, 1e-4)
        self._v_prev = v.copy()

        # ---- outer loop: velocity error -> desired acceleration ----------
        a_des = self.kp_v * (np.asarray(v_target, float) - v) - self.kd_v * a_meas

        # ---- desired force, with gravity feedforward ---------------------
        f_des = self.m * (a_des + np.array([0.0, 0.0, G]))
        f_z = max(f_des[2], 0.25 * self.m * G)          # never let thrust vanish

        # ---- attitude that points body-z along f_des ---------------------
        roll_des = np.clip(np.arctan2(-f_des[1], f_z), -self.max_tilt, self.max_tilt)
        pitch_des = np.clip(np.arctan2(f_des[0], f_z), -self.max_tilt, self.max_tilt)

        # thrust needed along the (tilted) body z axis
        tilt = max(np.cos(roll_des) * np.cos(pitch_des), 0.5)
        thrust_n = f_z / tilt

        # collective RPM such that 4*KF*rpm^2 = thrust_n, expressed via hover
        rpm = self.hover_rpm * np.sqrt(max(thrust_n / (self.m * G), 0.0))
        thrust_cmd = (rpm - self.hover_rpm) / max(self.max_rpm - self.hover_rpm, 1e-6)

        # ---- inner loop: attitude error -> torque command ----------------
        #
        # THE MIXER'S CHANNELS ARE SWAPPED AND INVERTED. Measured directly by
        # applying a small constant command for 0.5 s:
        #
        #     cmd            roll      pitch       dx       dy
        #     "roll"  +0.02   0.00    -12.05    -0.042    0.000
        #     "pitch" +0.02 -12.05      0.00     0.000   +0.042
        #
        # So channel 1 (named roll_cmd) drives PITCH, negated, and channel 2
        # (named pitch_cmd) drives ROLL, negated. Commanding them by their names
        # is positive feedback on both axes, which flipped the aircraft to 179
        # degrees within a second of any tilt being requested.
        roll, pitch = float(rpy[0]), float(rpy[1])
        p, q = float(ang_vel[0]), float(ang_vel[1])
        pitch_effort = self.kp_att * (pitch_des - pitch) - self.kd_att * q
        roll_effort = self.kp_att * (roll_des - roll) - self.kd_att * p
        ch1 = -pitch_effort      # channel 1 -> pitch, inverted
        ch2 = -roll_effort       # channel 2 -> roll,  inverted

        return np.clip([thrust_cmd, ch1, ch2, 0.0], -1.0, 1.0)


def goto(ctrl, env, target_xyz, max_speed=1.0, steps=2000, land_at=None):
    """
    Fly to a point and STOP. This is the manoeuvre that exposes an under-damped
    loop; constant-velocity commands cannot.
    """
    traj = []
    dt = 1.0 / env.CTRL_FREQ
    for _ in range(steps):
        p = env.pos[0].copy()
        d = np.asarray(target_xyz, float) - p
        dist = np.linalg.norm(d)
        # proportional approach with a speed cap -> naturally decelerates
        v_t = np.clip(d * 0.8, -max_speed, max_speed)
        if land_at is not None and p[2] <= land_at:
            break
        a = ctrl(v_t, env.vel[0], env.rpy[0], env.ang_v[0], dt)
        env.step(np.array([a]))
        traj.append((p.copy(), env.vel[0].copy(),
                     float(np.abs(np.degrees(env.rpy[0][:2])).max())))
        if dist < 0.08 and np.linalg.norm(env.vel[0]) < 0.12:
            break
    return traj


def land(ctrl, env, pad_xy, cruise_z=None, touch_z=0.15,
         v_lat=1.0, v_desc=0.5, v_flare=0.5, align_r=0.20,
         cut_h=0.45, cut_vz=0.20, cut_hold=10, steps=6000):
    """
    A real landing pattern: translate at altitude, align, descend, flare, CUT.

    Descending and translating at once makes the aircraft tilt to accelerate
    sideways, which steals vertical thrust -- measured as an uncommanded 4.96
    m/s drop against a commanded 0.5 m/s. Separating the phases avoids it.

    THE THRUST CUT IS NOT OPTIONAL. Ground effect holds the aircraft up: it
    settles at 0.274 m with ZERO ground contacts and refuses to descend further
    even when commanded -0.8 m/s, because the controller will not drop thrust
    below its floor. Resting altitude is 0.094 m with 4 contacts. So the last
    0.18 m has to be a deliberate thrust cut once the aircraft is low and slow
    -- which is exactly what a real land-detector does before disarming.
    """
    dt = 1.0 / env.CTRL_FREQ
    pad = np.asarray(pad_xy, float)
    traj = []
    stalled = 0
    for _ in range(steps):
        p = env.pos[0].copy()
        h = float(p[2])
        d = pad - p[:2]
        dist = float(np.linalg.norm(d))
        if dist > align_r:                       # translate / align, hold height
            v_xy = np.clip(d * 0.8, -v_lat, v_lat)
            vz = 0.0 if cruise_z is None else np.clip((cruise_z - h) * 0.5, -0.3, 0.3)
        else:                                    # centred: descend, then flare
            v_xy = np.clip(d * 0.8, -0.3, 0.3)
            vz = -v_desc if h > 1.0 else -v_flare
        # Land detector: trigger on a STALLED descent, not a fixed altitude.
        # Ground effect holds the aircraft at whatever height balances the
        # commanded descent (measured: 0.60 m at -0.15 m/s, 0.42 at -0.3,
        # 0.32 at -0.5, 0.24 at -0.8), so any fixed threshold is wrong for some
        # flare setting. "Low, centred, and no longer descending" is not.
        if h < cut_h and dist < align_r and abs(float(env.vel[0][2])) < cut_vz:
            stalled += 1
        else:
            stalled = 0
        if stalled >= cut_hold:
            a = np.array([-1.0, 0.0, 0.0, 0.0])          # cut thrust, settle
        else:
            a = ctrl([v_xy[0], v_xy[1], vz], env.vel[0], env.rpy[0], env.ang_v[0], dt)
        env.step(np.array([a]))
        traj.append((env.pos[0].copy(), env.vel[0].copy(),
                     float(np.abs(np.degrees(env.rpy[0][:2])).max())))
        if h <= touch_z and env.data.ncon > 0:
            break
    return traj
