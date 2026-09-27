#!/usr/bin/env python3
"""
flight_server.py -- the forked dashboard's own backend (the GeoAnchor dashboard and its API are untouched).

  /                      the built dashboard (dashboard/out: GeoAnchor's page, Flight log, Mission planner)
  /fl/flights            the flights under gazebo_geo/ (a GeoAnchor record, PX4's .ulg, or both)
  /fl/flight/<name>      one flight as time series, QGC-style: attitude and its setpoints from PX4's own log, heights
                         and speeds, PX4's estimate against Gazebo's truth, GNSS/AGP fusion, phases, fixes
  /fl/world              the pads (Gazebo ENU, marker ids), the map extent, the default route
  /fl/basemap.jpg        the floor the sim flies over (Esri Wayback 2023: internal use only -- never publish)
  /fl/missions           GET the saved missions, POST one (-> geo/missions/<name>.json)
  /fl/fly                POST {"mission": name}: fly it in Gazebo (geo/fly_mission.sh), if nothing else is flying
  /fl/live               Server-Sent Events: the flight in the air, row by row (its live.jsonl)

    python3 dashboard/flight_server.py            # then http://localhost:8010  (system python: pyulog, numpy, cv2)
"""
import argparse
import functools
import glob
import json
import math
import os
import re
import subprocess
import sys
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote, urlparse

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.dirname(HERE)
sys.path[:0] = [f"{R}/geo"]
from geo_world import EXTENT, ll_to_gz                     # noqa: E402  (math and numpy only)

FLIGHTS, MISSIONS = f"{R}/gazebo_geo", f"{R}/geo/missions"
NAME = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")
HZ = 10.0                                                 # series rate sent to the browser


def _nan(a):
    return [None if not np.isfinite(v) else round(float(v), 3) for v in a]


def _ulog(path):
    """PX4's own log at HZ: attitude and setpoints, local position and its setpoints, fusion flags."""
    from pyulog import ULog
    topics = ["vehicle_attitude", "vehicle_attitude_setpoint", "vehicle_local_position",
              "vehicle_local_position_setpoint", "estimator_status_flags"]
    u = ULog(path, topics)
    d = {}
    for m in u.data_list:
        if m.multi_id == 0:
            d[m.name] = m.data
    lp = d["vehicle_local_position"]
    t = np.arange(lp["timestamp"][0], lp["timestamp"][-1], 1e6 / HZ) / 1e6
    at = lambda top, key: np.interp(t, d[top]["timestamp"] / 1e6, d[top][key]) if top in d else np.full(len(t), np.nan)
    step = lambda top, key: (d[top][key][np.clip(np.searchsorted(d[top]["timestamp"] / 1e6, t, "right") - 1, 0, None)]
                             if top in d else np.full(len(t), np.nan))
    q = np.stack([at("vehicle_attitude", f"q[{i}]") for i in range(4)], 1)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    w, x, y, z = q.T
    roll = np.degrees(np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)))
    pitch = np.degrees(np.arcsin(np.clip(2 * (w * y - z * x), -1, 1)))
    yaw = np.degrees(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))) % 360
    s = dict(t=t, roll=roll, pitch=pitch, yaw=yaw,
             tilt=np.degrees(np.arccos(np.clip(1 - 2 * (x * x + y * y), -1, 1))),
             x_est=at("vehicle_local_position", "y"), y_est=at("vehicle_local_position", "x"),   # NED -> Gazebo ENU
             h_est=-at("vehicle_local_position", "z"),
             v_h=np.hypot(at("vehicle_local_position", "vx"), at("vehicle_local_position", "vy")),
             v_down=at("vehicle_local_position", "vz"),
             v_h_sp=np.hypot(step("vehicle_local_position_setpoint", "vx"), step("vehicle_local_position_setpoint", "vy")),
             v_down_sp=step("vehicle_local_position_setpoint", "vz"),
             yaw_sp=np.degrees(step("vehicle_local_position_setpoint", "yaw")) % 360)
    for k, f in (("gnss", "cs_gnss_pos"), ("agp", "cs_aux_gpos"), ("rng", "cs_rng_hgt"), ("dr", "cs_inertial_dead_reckoning")):
        if f in d.get("estimator_status_flags", {}):         # what EKF2 fuses: GNSS, GeoAnchor (AGP), the rangefinder
            s[k] = step("estimator_status_flags", f).astype(float)
    if "vehicle_attitude_setpoint" in d:                   # the setpoint's own quaternion (roll_body went in v1.14)
        a = d["vehicle_attitude_setpoint"]
        if "q_d[0]" in a:
            qd = np.stack([step("vehicle_attitude_setpoint", f"q_d[{i}]") for i in range(4)], 1)
            w, x, y, z = qd.T
            s["roll_sp"] = np.degrees(np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)))
            s["pitch_sp"] = np.degrees(np.arcsin(np.clip(2 * (w * y - z * x), -1, 1)))
    return {k: v for k, v in s.items() if v is not None}


def flight(name):
    D = f"{FLIGHTS}/{name}"
    return _flight(name, max(os.path.getmtime(p) for p in glob.glob(f"{D}/*")))


@functools.lru_cache(maxsize=16)
def _flight(name, _mtime):
    """One flight's record, merged on PX4's clock: the .ulg (if kept) and the GeoAnchor record (truth, phases)."""
    D = f"{FLIGHTS}/{name}"
    rec = sorted(glob.glob(f"{D}/geo_*.json") + glob.glob(f"{D}/*_geo.json") + glob.glob(f"{D}/cruise.json"))
    doc = json.load(open(rec[0])) if rec else {}
    rows = [r for r in doc.get("rows", []) if r.get("est") is not None]
    ulg = sorted(glob.glob(f"{D}/*.ulg"))
    s = _ulog(ulg[-1]) if ulg else None
    rt = np.array([r["t"] for r in rows]) if rows else np.zeros(0)
    if s is None:                                          # no PX4 log kept (flights before 27 Sept): the record only
        if not len(rt):
            raise FileNotFoundError(name)
        t = np.arange(rt[0], rt[-1], 1.0 / HZ)
        est = np.array([r["est"] for r in rows])
        s = dict(t=t, x_est=np.interp(t, rt, est[:, 0]), y_est=np.interp(t, rt, est[:, 1]), h_est=np.interp(t, rt, est[:, 2]),
                 yaw=np.interp(t, rt, np.unwrap(np.radians([r["hdg_px4"] for r in rows]))) * 180 / math.pi % 360)
        v = np.gradient(np.stack([s["x_est"], s["y_est"], s["h_est"]], 1), t, axis=0)
        s.update(v_h=np.hypot(v[:, 0], v[:, 1]), v_down=-v[:, 2])
        s["gnss"] = np.interp(t, rt, [float(bool(r["gnss"])) for r in rows]).round()
        s["agp"] = np.interp(t, rt, [float(bool(r["agp"])) for r in rows]).round()
    t = s["t"]
    if len(rt):                                            # Gazebo's truth, where the record has it
        g = [(r["t"], *r["gz"]) for r in rows if r.get("gz")]
        g = np.array(g) if g else np.zeros((0, 4))
        inside = (t >= g[0, 0]) & (t <= g[-1, 0]) if len(g) else np.zeros(len(t), bool)
        for k, c in (("x_true", 1), ("y_true", 2), ("h_true", 3)):
            s[k] = np.where(inside, np.interp(t, g[:, 0], g[:, c]), np.nan) if len(g) else np.full(len(t), np.nan)
        s["err"] = np.hypot(s["x_est"] - s["x_true"], s["y_est"] - s["y_true"])
    phases, events = [], []
    for r in rows:
        if not phases or phases[-1][2] != r["phase"]:
            phases.append([r["t"], r["t"], r["phase"]])
        phases[-1][1] = r["t"]
    g0 = rows[0]["gnss"] if rows else None
    for a, b in zip(rows, rows[1:]):
        if a["gnss"] and not b["gnss"]:
            events.append([b["t"], "GNSS denied"])
        elif b["gnss"] and not a["gnss"] and g0 is not None:
            events.append([b["t"], "GNSS back"])
    fixes = []
    if rows:
        wall = np.array([r["wall"] for r in rows])
        for f in doc.get("fixes", []):
            if f.get("sent") and f.get("lat") is not None:
                x, y = ll_to_gz(f["lat"], f["lon"])
                fixes.append([round(float(np.interp(f["t_wall"], wall, rt)), 2), round(x, 2), round(y, 2),
                              "marker" if f.get("source") == "marker" else "geoanchor"])
    land = None
    for p in glob.glob(f"{D}/flights.jsonl"):
        land = [json.loads(l) for l in open(p) if l.strip()][-1:]
    run = open(f"{D}/run.log").read().splitlines()[-6:] if os.path.exists(f"{D}/run.log") else []
    return dict(name=name, ulog=bool(ulg), series={k: _nan(v) for k, v in s.items()}, phases=phases, events=events,
                fixes=fixes, report=doc.get("report"), landing=land[0] if land else None, log_tail=run)


def flights():
    out = []
    for D in sorted(glob.glob(f"{FLIGHTS}/*/"), key=os.path.getmtime, reverse=True):
        n = os.path.basename(D.rstrip("/"))
        rec = glob.glob(f"{D}geo_*.json") + glob.glob(f"{D}*_geo.json") + glob.glob(f"{D}cruise.json")
        ulg = glob.glob(f"{D}*.ulg")
        if rec or ulg:
            out.append(dict(name=n, ulog=bool(ulg), record=bool(rec), when=time.strftime("%d %b %H:%M", time.localtime(os.path.getmtime(D))),
                            video=bool(glob.glob(f"{D}*.mp4"))))
    return out


def world():
    pads = json.load(open(f"{R}/gz_logan/pads_logan.json"))
    import geo_cruise as GC                                  # noqa: E402  the default route
    return dict(extent=EXTENT, pads=pads, route=[list(p) for p in GC.ROUTE], alt=GC.ALT, speed=GC.V, depot=[0.0, 0.0])


_BASEMAP = None


def basemap():
    global _BASEMAP
    if _BASEMAP is None:
        import cv2
        im = cv2.imread(f"{R}/gz_logan/geo_floor/materials/textures/logan_2023.png")
        _BASEMAP = cv2.imencode(".jpg", cv2.resize(im, (1300, 1300), interpolation=cv2.INTER_AREA),
                                [cv2.IMWRITE_JPEG_QUALITY, 82])[1].tobytes()
    return _BASEMAP


def check_mission(m):
    """A mission the simulator will fly: a name, 0-12 waypoints inside the map, a pad from pads_logan.json."""
    assert NAME.match(str(m.get("name", ""))), "name: letters, digits, _ . - only"
    x0, x1, y0, y1 = EXTENT
    route = [[float(a), float(b)] for a, b in m.get("route", [])]
    assert len(route) <= 12, "at most 12 waypoints"
    assert all(x0 + 20 <= a <= x1 - 20 and y0 + 20 <= b <= y1 - 20 for a, b in route), "waypoints must be on the map"
    pads = {p["name"]: p for p in json.load(open(f"{R}/gz_logan/pads_logan.json"))}
    pad = pads.get(m.get("pad", {}).get("name") if isinstance(m.get("pad"), dict) else m.get("pad"))
    assert pad is not None, "pick a pad"
    alt, v = float(m.get("alt", 75)), float(m.get("speed", 7))
    assert 60 <= alt <= 100 and 2 <= v <= 10, "cruise 60-100 m (GeoAnchor's envelope), 2-10 m/s"
    lander = m.get("lander", "rule")
    assert lander in ("rule", "flying learner", "none"), "lander: rule, flying learner or none"
    return dict(name=m["name"], created=time.strftime("%Y-%m-%dT%H:%M:%S"), route=route,
                pad=dict(name=pad["name"], enu=pad["enu"], ids=pad["ids"], ours=pad["ours"]),
                alt=alt, speed=v, lander=lander, wind=float(np.clip(float(m.get("wind", 0)), 0, 12)),
                deny_gnss=bool(m.get("deny_gnss", True)))


def busy():
    """Is a Gazebo flight up? One SITL at a time: two at once broke PX4's estimate (27 Sept)."""
    return subprocess.run(["pgrep", "-f", "gz si[m].*gz_logan"], capture_output=True).returncode == 0


class H(SimpleHTTPRequestHandler):
    def __init__(self, *a, **k):
        super().__init__(*a, directory=f"{HERE}/out", **k)

    def log_message(self, *a):
        pass

    def _json(self, obj, code=200):
        b = json.dumps(obj, allow_nan=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        p = unquote(urlparse(self.path).path)
        try:
            if p == "/fl/flights":
                return self._json(flights())
            if p.startswith("/fl/flight/"):
                n = p.split("/")[-1]
                assert NAME.match(n) and os.path.isdir(f"{FLIGHTS}/{n}"), "no such flight"
                return self._json(flight(n))
            if p == "/fl/world":
                return self._json(world())
            if p == "/fl/basemap.jpg":
                b = basemap()
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(b)))
                self.end_headers()
                return self.wfile.write(b)
            if p == "/fl/missions":
                return self._json([json.load(open(f)) for f in sorted(glob.glob(f"{MISSIONS}/*.json"))])
            if p == "/fl/live":
                return self._live()
        except (AssertionError, FileNotFoundError) as e:
            return self._json(dict(error=str(e)), 404)
        if p.startswith("/fl/"):
            return self._json(dict(error="unknown"), 404)
        return super().do_GET()

    def do_POST(self):
        p = urlparse(self.path).path
        n = int(self.headers.get("Content-Length", 0))
        if n > 100_000:
            return self._json(dict(error="too big"), 413)
        try:
            body = json.loads(self.rfile.read(n) or b"{}")
            if p == "/fl/missions":
                m = check_mission(body)
                os.makedirs(MISSIONS, exist_ok=True)
                json.dump(m, open(f"{MISSIONS}/{m['name']}.json", "w"), indent=1)
                return self._json(m)
            if p == "/fl/fly":
                name = str(body.get("mission", ""))
                assert NAME.match(name) and os.path.exists(f"{MISSIONS}/{name}.json"), "save the mission first"
                if busy():
                    return self._json(dict(error="a Gazebo flight is already up: one at a time"), 409)
                out = f"mission_{name}_{time.strftime('%H%M%S')}"
                subprocess.Popen(["setsid", f"{R}/geo/fly_mission.sh", f"{MISSIONS}/{name}.json", out],
                                 stdout=open(f"/tmp/fly_mission_{out}.log", "w"), stderr=subprocess.STDOUT, cwd=R)
                return self._json(dict(started=out))
        except (AssertionError, ValueError, TypeError) as e:
            return self._json(dict(error=str(e)), 400)
        return self._json(dict(error="unknown"), 404)

    def _live(self):
        """The newest flight's live.jsonl, streamed as it grows (Server-Sent Events), from its start."""
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        path, pos, idle = None, 0, 0.0
        try:
            while idle < 600:
                live = sorted(glob.glob(f"{FLIGHTS}/*/live.jsonl"), key=os.path.getmtime)
                if live and live[-1] != path and time.time() - os.path.getmtime(live[-1]) < 30:
                    path, pos = live[-1], 0
                    self.wfile.write(f"event: flight\ndata: {json.dumps(os.path.basename(os.path.dirname(path)))}\n\n".encode())
                sent = False
                if path:
                    with open(path) as f:
                        f.seek(pos)
                        for line in f:
                            if not line.endswith("\n"):
                                break
                            pos += len(line.encode())
                            self.wfile.write(f"data: {line.strip()}\n\n".encode())
                            sent = True
                if not sent:
                    self.wfile.write(b": idle\n\n")
                self.wfile.flush()
                idle = 0.0 if sent else idle + 0.5
                time.sleep(0.5)
        except (BrokenPipeError, ConnectionResetError):
            pass


def check():
    """The QGC-style series from a kept PX4 log, against what the record says (calm4, 27 Sept)."""
    n = next((f["name"] for f in flights() if f["ulog"] and f["record"]), None)
    assert n, "no flight with both a PX4 log and a record"
    f = flight(n)
    s = f["series"]
    assert len(s["t"]) > 100 and all(len(v) == len(s["t"]) for v in s.values()), "series misaligned"
    h = [v for v in s["h_est"] if v is not None]
    assert max(h) > 60, f"never cruised? max height {max(h)}"
    e = [v for v in s["err"] if v is not None]
    assert e and np.median(e) < 10, f"PX4 vs truth median {np.median(e)} m: frames disagree"
    assert f["events"] and f["events"][0][1] == "GNSS denied", f["events"]
    print(f"{n}: {len(s['t'])} samples at {HZ:g} Hz, max height {max(h):.0f} m, PX4 vs truth median {np.median(e):.1f} m, "
          f"{len(f['fixes'])} fixes, events {f['events']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8010)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    if a.check:
        check()
    else:
        print(f"http://localhost:{a.port}", flush=True)
        ThreadingHTTPServer(("127.0.0.1", a.port), H).serve_forever()
