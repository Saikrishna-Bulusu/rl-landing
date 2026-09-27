#!/usr/bin/env python3
"""
measure_tau.py -- how fast does a velocity COMMAND become actual velocity?

WHY

Every environment in this project lets the policy set a velocity and then moves
the point mass at that velocity immediately. A real multirotor cannot do that:
it has to tilt to accelerate, and tilting takes time. If the cheap environment
responds instantly and the real aircraft takes a second, a policy tuned in the
cheap one arrives with the wrong reflexes.

So measure the one number that captures it -- the first-order lag tau, the time
to reach 63.2% of a commanded step -- and put that lag in the point-mass env.
Not a reimplementation of mc_pos_control: one number, measured from the real
firmware, that makes the toy agree with the aircraft where it matters.

IT ALSO DIAGNOSES THE WOBBLE

The same log answers a separate question: when the aircraft oscillates, is that
PX4's inner attitude loop being badly tuned, or my outer guidance loop feeding
it a jittery setpoint? A clean step response with small attitude overshoot says
the firmware is fine and the fault is mine. That distinction decides what to
tune, and guessing it wrong means tuning the wrong loop.
"""

import argparse
import json
import math
import time

import numpy as np

from px4_bridge import PX4Link

ALT = 20.0


def log_segment(link, cmd, secs, hz=50):
    """Hold a velocity command, recording state as fast as it arrives."""
    rows, t0, dt = [], time.time(), 1.0 / hz
    while time.time() - t0 < secs:
        link.send_velocity(*cmd)
        s = link.state(timeout=0.3)
        if s is not None:
            rows.append((time.time() - t0, s["vx"], s["vy"], s["vz"],
                         s["roll"], s["pitch"], s["yaw"], s["alt"]))
        time.sleep(dt)
    return np.array(rows)


def tau_of(t, v, target):
    """
    Time to 63.2% of the settled value -- the textbook first-order lag.

    Deliberately not a curve fit: the 63.2% crossing needs no solver, no
    initial guess, and cannot silently converge on nonsense.
    """
    if len(t) < 10:
        return None, None
    settled = float(np.mean(v[t > t.max() - 3.0]))
    if abs(settled) < 0.2 * abs(target):
        return None, settled            # never got there; lag is meaningless
    thresh = 0.632 * settled
    idx = np.argmax(np.abs(v) >= abs(thresh))
    return (float(t[idx]) if idx > 0 else None), settled


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="px4_response.json")
    a = ap.parse_args()

    link = PX4Link()
    assert link.start_offboard(), "could not arm"
    print("  climbing...")
    log_segment(link, (0, 0, 2.0), 14)
    log_segment(link, (0, 0, 0.0), 4)

    print("  hover baseline (no command changes)...")
    hov = log_segment(link, (0, 0, 0.0), 8)

    print("  lateral step: 2.0 m/s north...")
    lat = log_segment(link, (2.0, 0, 0.0), 12)
    log_segment(link, (0, 0, 0.0), 6)

    print("  vertical step: 1.5 m/s down...")
    ver = log_segment(link, (0, 0, -1.5), 8)
    log_segment(link, (0, 0, 0.0), 4)
    link.send_velocity(0, 0, -1.5)

    t_lat, v_lat = tau_of(lat[:, 0], lat[:, 1], 2.0)
    t_ver, v_ver = tau_of(ver[:, 0], -ver[:, 3], 1.5)

    def tilt_stats(seg, label):
        roll = np.degrees(seg[:, 4]); pitch = np.degrees(seg[:, 5])
        yaw = np.degrees(np.unwrap(seg[:, 6]))
        print(f"    {label:18} roll {roll.mean():+6.2f} +/- {roll.std():4.2f} deg   "
              f"pitch {pitch.mean():+6.2f} +/- {pitch.std():4.2f} deg   "
              f"yaw drift {yaw[-1] - yaw[0]:+6.2f} deg")
        return dict(roll_sd=float(roll.std()), pitch_sd=float(pitch.std()),
                    yaw_drift=float(yaw[-1] - yaw[0]))

    print(f"\n  LATERAL  tau = {t_lat}  settled at {v_lat:.2f} m/s (commanded 2.00)")
    print(f"  VERTICAL tau = {t_ver}  settled at {v_ver:.2f} m/s (commanded 1.50)")
    print("\n  attitude, PX4 alone -- no camera, no guidance loop:")
    h = tilt_stats(hov, "hover"); l = tilt_stats(lat, "lateral step")
    v = tilt_stats(ver, "vertical step")

    out = dict(tau_lat_s=t_lat, tau_ver_s=t_ver, v_lat=v_lat, v_ver=v_ver,
               hover=h, lateral=l, vertical=v)
    json.dump(out, open(a.out, "w"), indent=2)
    print(f"\n  saved {a.out}")


if __name__ == "__main__":
    main()
