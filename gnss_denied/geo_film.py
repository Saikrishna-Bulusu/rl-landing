#!/usr/bin/env python3
"""
geo_film.py -- a GeoAnchor flight on film. One frame per chase-camera image (Gazebo's camera thread frames it, as
gz_learner.Film does -- the flight loop stalls, the camera doesn't), 1600 x 900:

  left    the chase camera, placed where Gazebo has the drone (not where PX4 thinks it is): behind at cruise, steeply
          above near the pad. Drawn into it, as in the stage-3 films (gz_learner.Film's own code): the voxel map from
          the depth camera (cubes coloured by height) and the local 3D planner's path (the blue ribbon). Insets: the
          landing camera's marker check -- every marker it decodes boxed, ours green, a neighbour's pad red -- and,
          on magpie flights, the rig camera that sees the magpie (as in the rig films)
  right   GeoAnchor's own view (the gimbal camera) with its latest fix: accepted and how far off, or rejected;
          the map: PX4's estimate as a Google-Maps arrow along PX4's heading, Gazebo's truth (white ring), GeoAnchor's
          accepted fixes (green), the pad marker's (yellow), the planner's path (blue), the voxels, the pad (red)
  bottom  phase, GNSS, what it navigates on, height, cruise (horizontal) and descent speeds, PX4's error against the
          truth, the hazard; on wind flights the wind at the drone (WindDrag's /wind_truth) as an arrow on the map --
          for the viewer only, the drone is never told

The floor and the map are the 2023 Esri capture: the film is for internal use (Esri's terms), not the paper or a deck.
"""
import math
import os
import queue
import sys
import threading

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.dirname(HERE)
sys.path[:0] = [f"{R}/shared/px4_gazebo", HERE]
from geo_world import EXTENT
from gz_learner import Film, Mover

W, H, FPS = 1600, 900, 20
GIMBAL = "/world/aruco/model/x500_mono_cam_down_0/link/camera_link/sensor/camera/image"
MAP_PX = 534
BLUE, WHITE, GREEN, RED = (230, 120, 20), (255, 255, 255), (70, 210, 70), (40, 40, 230)


class GeoFilm:
    # the chase-view overlays are the stage-3 films' own (gz_learner.Film): same camera, same drawing
    CUBE, FACES = Film.CUBE, Film.FACES
    px_chase, px_chase_many = staticmethod(Film.px_chase), staticmethod(Film.px_chase_many)
    _cubes, _ribbon = Film._cubes, Film._ribbon

    def __init__(self, path, title, pad, route, eye=None, rig=None, wind=None):
        """eye: px4_perception.ArucoEye (the landing camera's marker check); rig: gz_learner.RigEye (magpie flights);
        wind: (mean m/s at 10 m, WindDrag's direction_deg: ENU, 0 = blowing east)."""
        import cv2
        from gz.transport13 import Node
        from gz.msgs10.image_pb2 import Image
        self.cv2, self.title, self.pad, self.route = cv2, title, np.asarray(pad, float), [np.asarray(p, float) for p in route]
        self.eye, self.rig, self.wind, self.wind_now = eye, rig, wind, None
        self.vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"avc1"), FPS, (W, H))
        assert self.vw.isOpened(), "no H.264 writer"
        tex = cv2.imread(f"{R}/shared/px4_gazebo/gz_logan/geo_floor/materials/textures/logan_2023.png")
        self.map_bg = (cv2.resize(tex, (MAP_PX, MAP_PX), interpolation=cv2.INTER_AREA) * 0.75).astype(np.uint8)
        self.mover, self.yaw, self.placed = Mover(asyn=True), None, {}
        self.state, self.gimbal, self.track, self.est_track = None, None, [], []
        self.fixes, self.mfixes, self.last_fix = [], [], None
        self.vox = self.path = None                          # set by the stage-3 flight loop (Gazebo frame)
        self.n, self.s_last, self.errors, self.closed, self.t_marker = 0, None, 0, False, -1e9
        self.q, self._done = queue.Queue(maxsize=200), threading.Event()
        threading.Thread(target=self._writer, daemon=True).start()
        self.node = Node()
        self.node.subscribe(Image, "/chase/image", self._on_chase)
        self.node.subscribe(Image, GIMBAL, self._on_gimbal)
        if wind is not None:
            from gz.msgs10.vector3d_pb2 import Vector3d
            self.node.subscribe(Vector3d, "/wind_truth", lambda m: setattr(self, "wind_now", (m.x, m.y)))

    # ---- the flight loop's side -------------------------------------------------------------------------------------
    def tick(self, t_abs, gz, est, v, phase, gnss, agp, info=None, banner=None, new_fixes=(), heading=None):
        """Latest state: t_abs PX4 time (= Gazebo's), gz Gazebo's truth (x, y, z), est PX4's estimate (ENU), v the
        truth's velocity, heading PX4's compass heading (deg); new_fixes: the bridge's rows since the last tick."""
        if gz is None:
            return
        for f in new_fixes:
            if f.get("source") == "marker":
                self.mfixes.append(np.asarray(f["gz"], float))
                self.t_marker = t_abs
            else:
                if f["sent"]:
                    self.fixes.append(np.asarray(f["gz"], float))
                self.last_fix = (t_abs, f, np.asarray(gz[:2], float))
        self.track.append(np.asarray(gz, float)); self.est_track.append(np.asarray(est, float))
        mk = None
        e = self.eye
        if e is not None and getattr(e, "look_gray", None) is not None:           # what the marker check last looked at
            mk = dict(img=e.look_gray, stamp=e.look_stamp, id=getattr(e, "last_id", None),
                      found=dict(getattr(e, "found", {})), expect=list(e.expect_ids))
        self.state = dict(t_abs=t_abs, gz=np.asarray(gz, float), est=np.asarray(est, float), v=np.asarray(v, float),
                          phase=phase, gnss=gnss, agp=agp, info=dict(info or {}), banner=banner, heading=heading,
                          marker=t_abs - self.t_marker < 2.0, mk=mk, ins=None if self.rig is None else self.rig.inset,
                          vox=self.vox, path=self.path, wind=self.wind_now)

    def close(self):
        self.closed = True                                   # the chase callback stops moving the camera first:
        self.q.put(None)                                     # it wrote to the closed helper pipe (calm flight, 26 Sept)
        self._done.wait(60)
        self.vw.release()
        if self.mover.p is not None:
            self.mover.p.stdin.close()

    # ---- Gazebo's side ------------------------------------------------------------------------------------------------
    def _on_gimbal(self, m):
        self.gimbal = np.frombuffer(m.data, np.uint8).reshape(m.height, m.width, -1)[..., :3]

    def _on_chase(self, m):
        st = self.state
        if st is None or self.closed:
            return
        s = m.header.stamp.sec + m.header.stamp.nsec * 1e-9
        img = np.frombuffer(m.data, np.uint8).reshape(m.height, m.width, -1)[..., :3].copy()
        pose = self.placed.pop(round(s * FPS), None)           # where the camera was put for this picture
        try:
            self.q.put_nowait((s, img, pose, self.gimbal, st, list(self.track[-3000:]), list(self.est_track[-3000:]),
                               list(self.fixes), list(self.mfixes), self.last_fix))
        except queue.Full:
            pass
        self.placed[round((s + 1.0 / FPS) * FPS)] = self._follow(s + 1.0 / FPS, st)

    def _follow(self, t_next, st):
        """The chase camera where the drone will be at the next picture: behind it along its smoothed course --
        10 m back and 3.5 up at cruise (at 14 and 6 the drone was a speck), 8 and 4 low but away from the pad -- near
        5 back and 12 up, looking steeply down (at 7 back and 3 up it sat in tree crowns and behind walls)."""
        me = st["gz"] + st["v"] * (t_next - st["t_abs"])
        v = st["v"][:2]
        if np.linalg.norm(v) > 1.0:
            want = math.atan2(v[1], v[0])
            self.yaw = want if self.yaw is None else self.yaw + 0.1 * ((want - self.yaw + math.pi) % (2 * math.pi) - math.pi)
        elif self.yaw is None:
            d = self.pad - me[:2]
            self.yaw = math.atan2(d[1], d[0])
        near = np.linalg.norm(me[:2] - self.pad) < 15.0
        back, up = (10.0, 3.5) if me[2] > 35.0 and not near else (5.0, 12.0) if near else (8.0, 4.0)
        cam = [me[0] - back * math.cos(self.yaw), me[1] - back * math.sin(self.yaw), me[2] + up]
        pitch = math.atan2(up, back)
        self.mover.put("chase_cam", cam, yaw=self.yaw, pitch=pitch)
        return (cam, self.yaw, pitch)

    # ---- composing ----------------------------------------------------------------------------------------------------
    def _mpx(self, xy):
        x0, x1, y0, y1 = EXTENT
        return int((xy[0] - x0) / (x1 - x0) * MAP_PX), int((y1 - xy[1]) / (y1 - y0) * MAP_PX)

    def _writer(self):
        while (item := self.q.get()) is not None:
            s = item[0]
            reps = 1 if self.s_last is None else min(round((s - self.s_last) * FPS), 10)
            if reps < 1:
                continue
            try:
                canvas = self._compose(*item)
            except Exception:
                self.errors += 1
                if self.errors == 1:
                    import traceback; traceback.print_exc()
                continue
            self.s_last = s
            for _ in range(reps):
                self.vw.write(canvas); self.n += 1
        self._done.set()

    def _put(self, img, txt, xy, sc=0.6, col=(235, 235, 235), th=1):
        self.cv2.putText(img, txt, xy, self.cv2.FONT_HERSHEY_SIMPLEX, sc, (0, 0, 0), th + 2, self.cv2.LINE_AA)
        self.cv2.putText(img, txt, xy, self.cv2.FONT_HERSHEY_SIMPLEX, sc, col, th, self.cv2.LINE_AA)

    def _marker_box(self, mk, s):
        """The landing camera's last check: the picture around what it decoded, every marker boxed and named."""
        cv2 = self.cv2
        if mk is None or mk["stamp"] is None or s - mk["stamp"] > 1.0:
            return None
        g = mk["img"]
        found = [(i, c, i in mk["expect"]) for i, c in mk["found"].items()]      # ours green, a neighbour's red
        if found:
            pts = np.concatenate([np.asarray(c, float).reshape(-1, 2) for _, c, _ in found])
            (x0, y0), (x1, y1) = pts.min(0), pts.max(0)
            half = max(x1 - x0, y1 - y0) / 2 + 40
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        else:                                                   # nothing decoded: the whole picture, small
            half, cx, cy = g.shape[0] / 2, g.shape[1] / 2, g.shape[0] / 2
        half = max(half, 60.0)
        xa, ya = int(max(cx - half, 0)), int(max(cy - half, 0))
        xb, yb = int(min(cx + half, g.shape[1])), int(min(cy + half, g.shape[0]))
        crop = cv2.cvtColor(np.ascontiguousarray(g[ya:yb, xa:xb]), cv2.COLOR_GRAY2BGR)
        k = 240.0 / max(crop.shape[:2])
        crop = cv2.resize(crop, (max(1, int(crop.shape[1] * k)), max(1, int(crop.shape[0] * k))), interpolation=cv2.INTER_NEAREST)
        box = np.full((240, 240, 3), 20, np.uint8)
        box[:crop.shape[0], :crop.shape[1]] = crop
        labels = []
        for mid, c, ours in found:
            q = ((np.asarray(c, float).reshape(-1, 2) - [xa, ya]) * k).astype(np.int32)
            cv2.polylines(box, [q], True, GREEN if ours else RED, 3 if mid == mk["id"] else 2, cv2.LINE_AA)
            self._put(box, str(mid), (int(q[:, 0].min()), max(12, int(q[:, 1].min()) - 4)), 0.45, GREEN if ours else RED, 1)
            labels.append((mid, ours))
        return box, labels

    def _arrow(self, img, c, heading_deg, col):
        """A Google-Maps arrow at c pointing along a compass heading (0 north = up, 90 east = right)."""
        h = math.radians(heading_deg)
        d = np.array([math.sin(h), -math.cos(h)])
        p = np.array([-d[1], d[0]])
        c = np.asarray(c, float)
        poly = np.array([c + 12 * d, c - 8 * d + 8 * p, c - 3 * d, c - 8 * d - 8 * p], np.int32)
        self.cv2.fillPoly(img, [poly], col)
        self.cv2.polylines(img, [poly], True, WHITE, 2, self.cv2.LINE_AA)

    def _wind(self, m, w, h):
        """The wind at the drone, bottom left of the map: an arrow the way it blows (north up), how hard, where from."""
        w10, deg = self.wind
        mean = w10 * math.log(max(h, 1.0) / 0.3) / math.log(10.0 / 0.3)      # WindDrag's log profile (z0 0.3 m)
        sp = float(np.hypot(*w))
        c = np.array([48.0, MAP_PX - 50.0])
        self.cv2.circle(m, (48, MAP_PX - 50), 38, (40, 40, 40), -1)
        self.cv2.circle(m, (48, MAP_PX - 50), 38, WHITE, 1, self.cv2.LINE_AA)
        if sp > 0.2:
            d = np.array([w[0], -w[1]]) / sp * min(34.0, 8.0 + 2.0 * sp)
            self.cv2.arrowedLine(m, tuple((c - d).astype(int)), tuple((c + d).astype(int)), (120, 220, 255), 3,
                                 self.cv2.LINE_AA, tipLength=0.35)
        frm = "N NE E SE S SW W NW".split()[round(((90.0 - deg + 180.0) % 360.0) / 45.0) % 8]
        self._put(m, f"wind {sp:4.1f} m/s here", (92, MAP_PX - 66), 0.5, (120, 220, 255))
        self._put(m, f"mean {mean:.0f} m/s at {h:.0f} m, {w10:g} at 10 m", (92, MAP_PX - 46), 0.42)
        self._put(m, f"from the {frm}; the drone is not told", (92, MAP_PX - 28), 0.42)

    def _compose(self, s, chase, pose, gimbal, st, track, est_track, fixes, mfixes, last_fix):
        cv2 = self.cv2
        c = np.full((H, W, 3), 28, np.uint8)
        # the chase view, at its own 960 x 540 (the projections are the camera's): cubes, ribbon, insets
        ch = cv2.cvtColor(chase, cv2.COLOR_RGB2BGR)
        # the planner's world (voxels, path) is in PX4's estimate: shifted by PX4's error, so it sits around the drone
        # as the planner saw it (drawn where the estimate put it, the path ran metres beside the drone: 27 Sept)
        shift = st["gz"] - st["est"]
        if pose is not None and st["vox"] is not None and len(st["vox"]):
            self._cubes(ch, np.asarray(st["vox"], float) + shift, pose)
        if pose is not None and st["path"] is not None and len(st["path"]) > 1:
            self._ribbon(ch, np.asarray(st["path"], float) + shift, pose, st["gz"])
        ins = st["ins"]
        if ins is not None and 0 <= s - ins[0] < 0.5:           # the rig camera that saw the magpie, last 0.5 s
            tt, cam, crop, score, dist, (cu, cv_, sz, span) = ins
            k = 200 / span
            im = cv2.cvtColor(cv2.resize(crop, (200, 200), interpolation=cv2.INTER_NEAREST), cv2.COLOR_RGB2BGR)
            h2 = max(sz * k / 2, 6)
            cv2.rectangle(im, (int(cu * k - h2), int(cv_ * k - h2)), (int(cu * k + h2), int(cv_ * k + h2)), RED, 2)
            ch[8:208, 8:208] = im
            cv2.rectangle(ch, (7, 7), (208, 208), WHITE, 2)
            self._put(ch, f"{cam} camera: magpie?", (8, 228), 0.5, WHITE, 1)
            self._put(ch, f"score {score:.2f}, ~{dist:.0f} m", (8, 248), 0.5, WHITE, 1)
        mb = self._marker_box(st["mk"], s)
        if mb is not None:
            box, labels = mb
            ch[8:248, 960 - 248:960 - 8] = box
            cv2.rectangle(ch, (960 - 249, 7), (960 - 7, 248), WHITE, 2)
            self._put(ch, "landing camera: marker check", (960 - 248, 266), 0.45, WHITE, 1)
            ours = [m for m, o in labels if o]
            self._put(ch, f"ids {', '.join(map(str, ours))}: OUR PAD" if ours else ("not our pad" if labels else "no marker in view"),
                      (960 - 248, 286), 0.5, GREEN if ours else RED if labels else WHITE, 1)
            if [m for m, o in labels if not o]:
                self._put(ch, "red: a neighbour's pad, ignored", (960 - 248, 304), 0.42, RED, 1)
        c[50:650, 0:1066] = cv2.resize(ch, (1066, 600))
        self._put(c, "chase camera" + ("  |  cubes: voxel map from the depth camera, band: planned path"
                                       if st["vox"] is not None else ""), (10, 640), 0.5)
        # GeoAnchor's view
        if gimbal is not None:
            c[50:350, 1066:1600] = cv2.resize(cv2.cvtColor(gimbal, cv2.COLOR_RGB2BGR), (534, 300))
        self._put(c, "GeoAnchor's camera (straight down)", (1074, 342), 0.5)
        if last_fix is not None and st["t_abs"] - last_fix[0] < 1.0:
            f = last_fix[1]
            if f["sent"]:
                self._put(c, f"fix: {float(np.hypot(*(np.asarray(f['gz']) - last_fix[2]))):.1f} m off", (1074, 76), 0.7, GREEN, 2)
            else:
                self._put(c, "no fix (rejected)", (1074, 76), 0.7, (60, 160, 240), 2)
        # the map
        m = self.map_bg.copy()
        if len(self.route) > 1:
            cv2.polylines(m, [np.array([self._mpx(p) for p in self.route], np.int32)], False, (150, 150, 150), 1, cv2.LINE_AA)
        if st["vox"] is not None and len(st["vox"]):
            for p in np.asarray(st["vox"])[::3]:
                u, w_ = self._mpx(p)
                cv2.rectangle(m, (u - 1, w_ - 1), (u + 1, w_ + 1), (60, 200, 255), -1)
        for p in fixes[-400:]:
            cv2.circle(m, self._mpx(p), 2, GREEN, -1)
        for p in mfixes[-200:]:
            cv2.circle(m, self._mpx(p), 2, (40, 220, 240), -1)
        if len(track) > 1:
            cv2.polylines(m, [np.array([self._mpx(p) for p in track[::5]], np.int32)], False, WHITE, 2, cv2.LINE_AA)
        if st["path"] is not None and len(st["path"]) > 1:
            cv2.polylines(m, [np.array([self._mpx(p) for p in st["path"]], np.int32)], False, (255, 190, 40), 2, cv2.LINE_AA)
        px, py = self._mpx(self.pad)
        cv2.rectangle(m, (px - 4, py - 4), (px + 4, py + 4), RED, -1)
        bird = st["info"].get("bird")
        if bird is not None:
            cv2.circle(m, self._mpx(bird), 5, (20, 20, 20), -1)
        cv2.circle(m, self._mpx(st["gz"]), 7, WHITE, 2)
        if st["heading"] is not None:
            self._arrow(m, self._mpx(st["est"]), st["heading"], BLUE)
        if self.wind is not None and st["wind"] is not None:
            self._wind(m, st["wind"], st["gz"][2])
        c[356:890, 1066:1600] = m
        self._put(c, "blue arrow: where PX4 thinks it is, pointing its way;", (1074, 372), 0.42)
        self._put(c, "white ring: where it really is; green: GeoAnchor fixes,", (1074, 390), 0.42)
        self._put(c, "yellow: pad-marker fixes, red: the pad", (1074, 408), 0.42)
        # the words
        self._put(c, self.title, (12, 34), 0.85, WHITE, 2)
        err = float(np.hypot(*(st["est"][:2] - st["gz"][:2])))
        spd, vz = float(np.linalg.norm(st["v"][:2])), -float(st["v"][2])
        gn = "GNSS: ON" if st["gnss"] else "GNSS: DENIED"
        nav = ("GPS" if st["gnss"] else "the pad marker" if st.get("marker") else "GeoAnchor" if st["agp"]
               else "dead reckoning")
        self._put(c, f"{st['phase']}", (12, 690), 0.9, WHITE, 2)
        self._put(c, gn, (12, 730), 0.8, GREEN if not st["gnss"] else (200, 200, 200), 2)
        self._put(c, f"navigating on: {nav}", (260, 730), 0.8, WHITE, 2)
        self._put(c, f"t {s:6.1f} s    height {st['gz'][2]:5.1f} m    PX4's error {err:5.1f} m", (12, 766), 0.75)
        self._put(c, f"cruise speed {spd:4.1f} m/s    {'descent' if vz >= 0 else 'climb'} speed {abs(vz):4.1f} m/s",
                  (12, 800), 0.75, (120, 220, 255))
        y = 838
        for k, val in st["info"].items():
            if k != "bird":
                self._put(c, f"{k}: {val}", (12, y), 0.7); y += 34
        if st["banner"]:
            b = st["banner"]
            (w, _), _ = cv2.getTextSize(b, cv2.FONT_HERSHEY_SIMPLEX, 1.4, 3)
            cv2.rectangle(c, (533 - w // 2 - 12, 320), (533 + w // 2 + 12, 376), WHITE, -1)
            cv2.putText(c, b, (533 - w // 2, 362), cv2.FONT_HERSHEY_SIMPLEX, 1.4,
                        (40, 150, 40) if b == "DELIVERED" else (40, 40, 220), 3, cv2.LINE_AA)
        return c
