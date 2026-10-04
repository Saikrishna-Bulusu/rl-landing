#!/usr/bin/env python3
"""
x500_ctrl.py -- make the aviary's PID usable on a 1.6 kg airframe.

Two defects had to be corrected before an x500 can be flown at all, and both
are silent: the drone still flies, just far slower than commanded, so a policy
trained on top would learn to compensate for a mis-tuned controller and would
not transfer to a correctly-tuned one.

1. THE ATTITUDE GAINS MUST BE LEFT ALONE. WITHDRAWN CLAIM, RECORDED BECAUSE IT
   WAS NEARLY SHIPPED. The source comments them as sized for the Crazyflie's
   ~0.004 Nm torque envelope, so they were scaled by the x500's 381x larger
   envelope. That LOOKED like an improvement -- lateral tracking of a 2 m/s
   command rose from 0.76 to 1.46 m/s -- but the drone was TUMBLING, and moving
   fast because it was out of control:

       gains unscaled, 0.5 m/s lateral: max pitch  6.6 deg, altitude 10.0 -> 10.0 m
       gains scaled,   0.5 m/s lateral: max roll 180.0 deg, altitude 10.0 ->  0.7 m

   A velocity measurement cannot distinguish "flying there" from "falling there".
   The original gains are correct; `patch_gains()` is retained as a no-op only
   so callers do not break, and under-tracking is corrected with the carrot gain
   instead, which does not touch the attitude loop.

2. ActionType.VEL IS NOT A VELOCITY CONTROLLER. It sets a waypoint ONE control
   step ahead (target = pos + v*dt, dt = 1/48 s), so a 0.3 m/s command produces
   a 6 mm position error -- far too small to drive the loop. Measured: 1% of
   commanded velocity achieved. Replaced with a carrot waypoint LOOKAHEAD
   seconds ahead, with a calibration gain fitted below.

The calibration gain is deliberately a knob rather than a constant: a real
airframe never matches the model, and this is exactly the place that needs
tuning against hardware.
"""
import os, sys
import numpy as np

import x500  # registers the airframe
sys.path.insert(0, os.path.expanduser("~/RL_projects/MuJoCo-drones-gym"))
from multi_drone_mujoco.control import pid_control

_CF_TORQUE = 0.0397 * 3.16e-10 * (np.sqrt(0.027 * 9.81 / (4 * 3.16e-10)) * 1.5) ** 2
_X5_TORQUE = 0.25 * x500.KF * (np.sqrt(2.3 * 1.6 * 9.81 / (4 * x500.KF))) ** 2
TORQUE_RATIO = _X5_TORQUE / _CF_TORQUE

_patched = False


def patch_gains():
    """
    NO-OP. Kept so existing callers do not break.

    Scaling the attitude gains to the x500 torque envelope destabilised the
    attitude loop and flipped the aircraft (see module docstring). Do not
    reinstate without an attitude-stability check, not a velocity measurement.
    """
    return 1.0


class VelocityCarrot:
    """
    Turn a velocity command into a waypoint the aviary's PID can actually track.

        waypoint = position + v_cmd * LOOKAHEAD * GAIN

    LOOKAHEAD sets how far ahead the carrot sits; GAIN corrects the steady-state
    under-tracking measured on this airframe. Both are exposed for hardware
    calibration -- see calibrate() for how GAIN was obtained.
    """

    def __init__(self, lookahead=1.0, gain_xy=3.0, gain_z=2.0):
        """
        Per-axis gains, because the axes behave differently:

          vertical  gain 2.0 -> 2-7% tracking error, touchdown 0.136 m/s
          lateral   gain 3.0 -> 0.21 m/s achieved for 0.5 commanded, tilt 9.8 deg

        Lateral authority is WEAK and must be kept slow: measured tilt and
        altitude loss against commanded lateral speed at gain 3.0,

            0.5 m/s ->  9.8 deg tilt, -0.04 m altitude   OK
            1.5 m/s -> 27.6 deg tilt, -0.33 m altitude   marginal
          and at gain 4.0+, 1.5 m/s -> 35-51 deg tilt and up to -1.6 m: UNSTABLE.

        A real X500 manages several m/s laterally, so this loop is
        under-performing and wants proper tuning. Until then, keep lateral
        commands at or below ~0.5 m/s and translate BEFORE descending.
        """
        self.lookahead = float(lookahead)
        self.gain_xy = float(gain_xy)
        self.gain_z = float(gain_z)

    @property
    def gain(self):          # backwards compatibility for the vertical checks
        return self.gain_z

    def waypoint(self, pos, v_cmd, yaw=0.0):
        v = np.asarray(v_cmd, float)
        g = np.array([self.gain_xy, self.gain_xy, self.gain_z])
        wp = np.asarray(pos, float) + v * self.lookahead * g
        return np.array([wp[0], wp[1], wp[2], yaw])


def calibrate():
    """
    Runnable calibration + check. GAIN=2.0 was fitted here:

        gain 1.0 -> -0.148 m/s     gain 2.0 -> -0.294 m/s
        gain 1.5 -> -0.221 m/s     gain 2.5 -> -0.367 m/s
        (all for a commanded -0.30 m/s; response is linear in the gain)
    """
    import os, sys
    sys.path.insert(0, os.path.expanduser("~/RL_projects/MuJoCo-drones-gym"))
    from multi_drone_mujoco.envs.base_aviary import BaseAviary
    from multi_drone_mujoco.utils.enums import Physics, ActionType, ObservationType
    patch_gains()

    class _T(BaseAviary):
        def _computeObs(self): return np.zeros(1)
        def _computeReward(self): return 0.0
        def _computeTerminated(self): return False
        def _computeTruncated(self): return False
        def _computeInfo(self): return {}

    e = _T(drone_model=x500.X500_MODEL, num_drones=1,
           physics=Physics.MJC_GND_DRAG_DW, act_type=ActionType.PID,
           obs_type=ObservationType.KIN, ctrl_freq=48, sim_freq=240,
           initial_xyzs=np.array([[0, 0, 40.0]]))
    c = VelocityCarrot()
    # NOTE: assigning env.pos does NOT move the drone -- `pos` is a cached read
    # from MuJoCo, refreshed each step. Spawn altitude must be set through
    # initial_xyzs before construction. An earlier version of this check set
    # env.pos directly, silently measured a free-fall from 40 m, and reported a
    # 3.09 m/s touchdown that had nothing to do with the controller.

    print("velocity tracking (gain %.1f):" % c.gain)
    worst = 0.0
    for tgt in (-0.2, -0.3, -0.5, -1.0, -2.0):
        e.reset(); vs = []
        for k in range(400):
            e.step(np.array([c.waypoint(e.pos[0], [0, 0, tgt])]))
            if k > 150:
                vs.append(float(e.vel[0][2]))
        got = float(np.mean(vs)); err = abs(got - tgt) / abs(tgt)
        worst = max(worst, err)
        print(f"  {tgt:+.2f} m/s -> {got:+.3f} m/s  ({100*err:.0f}% error)")
    assert worst < 0.25, f"velocity tracking off by {100*worst:.0f}%"

    # A full slow descent to touchdown -- the thing that was 2.00 m/s before.
    e2 = _T(drone_model=x500.X500_MODEL, num_drones=1,
            physics=Physics.MJC_GND_DRAG_DW, act_type=ActionType.PID,
            obs_type=ObservationType.KIN, ctrl_freq=48, sim_freq=240,
            initial_xyzs=np.array([[0, 0, 8.0]]))
    e2.reset()
    e = e2
    vz_hist = []
    for _ in range(4000):
        h = float(e.pos[0][2])
        if h <= 0.15:
            break
        v = -0.3 if h > 1.0 else -0.15          # slow further in the last metre
        e.step(np.array([c.waypoint(e.pos[0], [0, 0, v])]))
        vz_hist.append(float(e.vel[0][2]))
    td = abs(vz_hist[-1])
    print(f"\nfull descent from 8 m -> touchdown at {abs(e.pos[0][2]):.2f} m")
    print(f"  touchdown speed {td:.3f} m/s   (the old point-mass policy: 2.00 m/s)")
    assert td < 0.5, f"touchdown too fast: {td:.2f} m/s"
    print("\nok  x500 controller is usable for slow, stable landings")


if __name__ == "__main__":
    calibrate()
