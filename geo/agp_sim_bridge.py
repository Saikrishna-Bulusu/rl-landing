#!/usr/bin/env python3
"""
agp_sim_bridge.py -- GeoAnchor's accepted fixes into PX4's EKF2 as Aux Global Position, in SITL.

The mapping is geoanchor-rt/scripts/agp_bridge.py's (accepted fixes only, finite values only, eph = max(sigma, 0.1 m),
no altitude claimed), with ONE difference: the stamps go out as 0, so PX4 stamps each fix on arrival in its own clock
and EKF2_AGP_DELAY carries the pipeline's latency. agp_bridge.py stamps the capture in wall-clock time and relies on
the DDS client's time sync to convert it -- right on an aircraft, but in lockstep SITL PX4's clock is Gazebo's sim
time, which drifts from the wall clock whenever the sim runs slower than real time (the magpie flights, with the rig's
detector, do), so a converted stamp would be off by however far the sim has fallen behind.

Below GeoAnchor's envelope (50 m) the pad marker takes over the same way: the flight script publishes the drone's
position from the marker and the pad's surveyed place (--marker), so the landing is GNSS-free too.

It also logs, for the flight script and the film: every fix it sent or skipped, and PX4's own estimator flags
(cs_gnss_pos, cs_aux_gpos -- is the EKF fusing GPS, GeoAnchor, both?) once a second.

    (ROS 2 + ~/GeoAnchor/ros2_ws sourced) python3 geo/agp_sim_bridge.py --config <run>/config.yaml --ns px4_5 --log <run>/agp.jsonl
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

GA = Path("/home/sai/GeoAnchor/geoanchor-rt")
sys.path.insert(0, str(GA))
for _sp in sorted(GA.glob(".venv/lib/python3.*/site-packages")):       # zmq + geoanchor's deps: APPEND, ROS wins
    if str(_sp) not in sys.path:
        sys.path.append(str(_sp))

from geoanchor import config as cfgmod                                  # noqa: E402
from geoanchor import contracts as K                                    # noqa: E402
from geoanchor.bus import Subscriber                                    # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ns", default="")
    ap.add_argument("--log", required=True)
    ap.add_argument("--marker", default=None, help="a second fix stream: the pad marker's (geo_cruise.MarkerFix)")
    ap.add_argument("--min-sigma-m", type=float, default=0.1)
    a = ap.parse_args()
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy
    from px4_msgs.msg import VehicleGlobalPosition, EstimatorStatusFlags
    pre = f"/{a.ns}" if a.ns else ""
    rclpy.init()
    node = Node("geoanchor_agp_sim_bridge")
    qos = QoSProfile(reliability=QoSReliabilityPolicy.BEST_EFFORT, history=QoSHistoryPolicy.KEEP_LAST, depth=1,
                     durability=QoSDurabilityPolicy.VOLATILE)
    pub = node.create_publisher(VehicleGlobalPosition, f"{pre}/fmu/in/aux_global_position", qos)
    log = open(a.log, "a", buffering=1)
    flags = {}

    def on_flags(m):
        flags.update(gnss=bool(m.cs_gnss_pos), agp=bool(m.cs_aux_gpos), t_px4_us=int(m.timestamp))
    node.create_subscription(EstimatorStatusFlags, f"{pre}/fmu/out/estimator_status_flags", on_flags, qos)
    sub = Subscriber([cfgmod.load(a.config).get("bus.processing_pub")] + ([a.marker] if a.marker else []), [K.T_FIX])
    print(f"{pre}/fmu/in/aux_global_position <- {a.config}", flush=True)
    t_flags = 0.0
    while rclpy.ok():
        rclpy.spin_once(node, timeout_sec=0.0)
        if time.time() - t_flags > 1.0 and flags:
            t_flags = time.time()
            log.write(json.dumps(dict(kind="flags", t_wall=round(t_flags, 3), **flags)) + "\n")
        got = sub.recv(timeout_ms=50)
        if not got:
            continue
        _topic, f, _payload = got
        lat, lon, sigma = f.get("lat"), f.get("lon"), f.get("sigma_m")
        ok = bool(f.get("accepted")) and all(isinstance(v, (int, float)) and math.isfinite(v) for v in (lat, lon, sigma))
        row = dict(kind="fix", source=f.get("source", "geoanchor"), t_wall=round(time.time(), 3), seq=f.get("seq"),
                   accepted=bool(f.get("accepted")), sent=ok,
                   lat=lat, lon=lon, sigma_m=sigma, inliers=f.get("inliers"),
                   latency_ms=round((time.time() - float(f.get("t_capture_unix") or time.time())) * 1e3, 1))
        if ok:
            m = VehicleGlobalPosition()
            m.timestamp = 0                                      # PX4: 0 -> hrt_absolute_time() on arrival
            m.timestamp_sample = 0
            m.lat, m.lon, m.lat_lon_valid = float(lat), float(lon), True
            m.alt, m.alt_valid = 0.0, False                      # no altitude claimed (EKF2_AGP_CTRL bit 1 clear)
            m.eph, m.epv = float(max(sigma, a.min_sigma_m)), float("nan")
            pub.publish(m)
        log.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
