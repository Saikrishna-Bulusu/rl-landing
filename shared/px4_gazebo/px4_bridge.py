#!/usr/bin/env python3
"""
px4_bridge.py -- fly the landing policy on real PX4 firmware in Gazebo.

    python3 px4_bridge.py --check            arm, climb, translate, land
    python3 px4_bridge.py --policy PATH      let the trained policy fly it

WHY THIS EXISTS

Everything before this ran on physics I wrote myself. This runs on PX4's own
firmware, its own EKF, and its own position controller -- the same code that
flies the Pixhawk. The policy does not fly the aircraft; it emits two numbers
and PX4 does the flying:

    a[0]  vertical rate      ->  velocity setpoint, down-positive in NED
    a[1]  lateral speed cap  ->  how hard to close on the believed pad

That is exactly the OffboardControlMode velocity interface, which is why the
abstraction survives moving off the simplified simulator. A controller written
here would be a third reimplementation of mc_pos_control, which already exists,
is flight tested, and runs on the real vehicle.

THREE THINGS OFFBOARD MODE FAILS SILENTLY ON

  1. Setpoints must already be streaming BEFORE the mode switch. PX4 rejects
     OFFBOARD if it has not seen a recent setpoint, and the rejection is a
     COMMAND_ACK you will not see unless you read it.
  2. They must keep arriving faster than 2 Hz or PX4 drops out of OFFBOARD
     mid-flight. This sends at 20 Hz.
  3. PX4 is NED: z is DOWN. A descent is a POSITIVE vz. The policy, like every
     other environment in this project, uses up-positive. The sign is converted
     in one place, `_vz_to_ned`, rather than scattered around.

AND A FOURTH, WHICH COST THE MOST TIME

PX4 REFUSES TO ARM WITHOUT A GROUND-STATION HEARTBEAT. The arm command comes
back MAV_RESULT_TEMPORARILY_REJECTED (1) and the console says

    Preflight Fail: No connection to the ground control station
    Arming denied: Resolve system health failures first

That is a safety feature, not a bug: a vehicle with no operator link should not
arm. The fix is to actually be a ground station -- connect as
source_system=255, source_component=190 and send a MAV_TYPE_GCS heartbeat at
1 Hz on a background thread. With that in place the same arm command returns
ACCEPTED immediately.
"""

import threading

import argparse
import math
import time

import numpy as np
from pymavlink import mavutil

# SET_POSITION_TARGET_LOCAL_NED type_mask: ignore everything except velocity.
# bits 0-2 position, 3-5 velocity, 6-8 accel, 9 force, 10 yaw, 11 yaw_rate
IGNORE_ALL_BUT_VEL = 0b0000_11_111_000_111


class PX4Link:
    def __init__(self, endpoint="udpin:0.0.0.0:14540", timeout=30):
        # Identify as a ground station, or PX4 will not arm. See module docstring.
        self.m = mavutil.mavlink_connection(endpoint, source_system=255,
                                            source_component=190)
        hb = self.m.wait_heartbeat(timeout=timeout)
        if hb is None:
            raise RuntimeError(f"no heartbeat on {endpoint}")
        self.sys, self.comp = self.m.target_system, self.m.target_component
        self._req_stream("LOCAL_POSITION_NED", 20)
        self._req_stream("ATTITUDE", 20)
        self._beating = True
        threading.Thread(target=self._heartbeat_loop, daemon=True).start()
        time.sleep(3.0)          # let PX4 register the GCS before arming

    def _heartbeat_loop(self):
        """A GCS heartbeat at 1 Hz. Without this PX4 refuses to arm."""
        while self._beating:
            try:
                self.m.mav.heartbeat_send(
                    mavutil.mavlink.MAV_TYPE_GCS,
                    mavutil.mavlink.MAV_AUTOPILOT_INVALID, 0, 0, 0)
            except Exception:
                pass
            time.sleep(1.0)

    def close(self):
        self._beating = False

    def _req_stream(self, name, hz):
        mid = getattr(mavutil.mavlink, f"MAVLINK_MSG_ID_{name}", None)
        if mid is None:
            return
        self.m.mav.command_long_send(
            self.sys, self.comp, mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
            0, mid, int(1e6 / hz), 0, 0, 0, 0, 0)

    # ---- state ----------------------------------------------------------
    def state(self, timeout=2.0):
        """Local NED position/velocity and attitude. Returns up-positive alt."""
        pos = self.m.recv_match(type="LOCAL_POSITION_NED", blocking=True,
                                timeout=timeout)
        att = self.m.recv_match(type="ATTITUDE", blocking=False)
        if pos is None:
            return None
        s = {"x": pos.x, "y": pos.y, "z": pos.z,          # NED, z DOWN
             "vx": pos.vx, "vy": pos.vy, "vz": pos.vz,
             "alt": -pos.z, "climb": -pos.vz}             # up-positive
        if att is not None:
            s["roll"], s["pitch"], s["yaw"] = att.roll, att.pitch, att.yaw
            self._last_att = (att.roll, att.pitch, att.yaw)
        else:
            r, p, y = getattr(self, "_last_att", (0.0, 0.0, 0.0))
            s["roll"], s["pitch"], s["yaw"] = r, p, y
        return s

    # ---- commanding ------------------------------------------------------
    @staticmethod
    def _vz_to_ned(vz_up):
        """Policy speaks up-positive; PX4 speaks down-positive."""
        return -float(vz_up)

    def send_velocity(self, vx, vy, vz_up, yaw_rate=0.0):
        self.m.mav.set_position_target_local_ned_send(
            int(time.time() * 1e3) & 0xFFFFFFFF, self.sys, self.comp,
            mavutil.mavlink.MAV_FRAME_LOCAL_NED, IGNORE_ALL_BUT_VEL,
            0, 0, 0,
            float(vx), float(vy), self._vz_to_ned(vz_up),
            0, 0, 0, 0, float(yaw_rate))

    def set_mode_offboard(self):
        self.m.mav.command_long_send(
            self.sys, self.comp, mavutil.mavlink.MAV_CMD_DO_SET_MODE, 0,
            mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED, 6, 0, 0, 0, 0, 0)

    def arm(self, on=True):
        self.m.mav.command_long_send(
            self.sys, self.comp, mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0, 1 if on else 0, 0, 0, 0, 0, 0, 0)

    def ack(self, timeout=3.0):
        a = self.m.recv_match(type="COMMAND_ACK", blocking=True, timeout=timeout)
        return None if a is None else (a.command, a.result)

    def start_offboard(self, settle=2.0):
        """
        Stream setpoints first, THEN switch mode, THEN arm. Doing it in any
        other order is the usual reason offboard silently refuses.
        """
        t0 = time.time()
        while time.time() - t0 < settle:
            self.send_velocity(0, 0, 0)
            time.sleep(0.05)
        self.set_mode_offboard()
        time.sleep(0.5)
        self.arm(True)
        t0 = time.time()
        while time.time() - t0 < 6.0:
            self.send_velocity(0, 0, 0)
            hb = self.m.recv_match(type="HEARTBEAT", blocking=False)
            if hb is not None:
                armed = bool(hb.base_mode &
                             mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
                if armed:
                    return True
            time.sleep(0.05)
        return False


def fly_check(link, hz=20):
    """Arm, climb to 20 m, translate 15 m, descend, land. Proves the loop."""
    dt = 1.0 / hz
    print("  starting offboard...")
    if not link.start_offboard():
        print("  FAILED to arm / enter offboard"); return False
    print("  armed and in offboard")

    def hold(vx, vy, vz_up, secs, label, stop=None):
        t0 = time.time(); last = 0
        while time.time() - t0 < secs:
            link.send_velocity(vx, vy, vz_up)
            s = link.state(timeout=0.5)
            if s and time.time() - last > 2.0:
                print(f"    {label}: alt {s['alt']:6.2f} m  "
                      f"x {s['x']:6.2f}  y {s['y']:6.2f}  climb {s['climb']:+5.2f}")
                last = time.time()
            if s and stop and stop(s):
                return s
            time.sleep(dt)
        return link.state(timeout=0.5)

    hold(0, 0, 2.0, 25, "climb", stop=lambda s: s["alt"] > 20.0)
    hold(2.0, 0, 0.0, 12, "translate")
    hold(0, 0, 0.0, 3, "settle")
    s = hold(0, 0, -1.5, 25, "descend", stop=lambda s: s["alt"] < 3.0)
    s = hold(0, 0, -0.4, 15, "flare", stop=lambda s: s["alt"] < 0.4)
    link.arm(False)
    print(f"  final altitude {s['alt']:.2f} m" if s else "  no final state")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", default="udpin:0.0.0.0:14540")
    ap.add_argument("--check", action="store_true",
                    help="fly a scripted profile to prove the link works")
    a = ap.parse_args()
    link = PX4Link(a.endpoint)
    print(f"connected: system {link.sys}")
    s = link.state()
    print(f"  altitude {s['alt']:.2f} m, position ({s['x']:.2f}, {s['y']:.2f})")
    if a.check:
        fly_check(link)


if __name__ == "__main__":
    main()
