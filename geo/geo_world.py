#!/usr/bin/env python3
"""
geo_world.py -- Logan in Gazebo with a real aerial capture laid on the floor, for GeoAnchor (Sai, 26 Sept: "download
the map of Logan exactly of the square plot ... check with both overlapping the map onto the gazebo floor and don't
overlap and see if the matcher works in both cases").

    <GeoAnchor venv python> geo/geo_world.py        -> gz_logan/geo_floor/ (aruco.sdf + the floor texture), geo_clean/,
                                                       geo_magpie/ -- all with the films' chase camera

The floor is the 2023-06-29 Esri Wayback capture (the year of the LiDAR survey the houses and trees come from); the
map GeoAnchor matches against is the 2021-07-21 capture -- never the floor's own image (GeoAnchor's hard rule).
Esri's terms: measurement use; the pixels are not to be redistributed (renders of this floor stay internal).

The capture is warped into Gazebo's frame through the SAME chain the LiDAR houses went through: Gazebo ENU =
Logan map (x, y) - pad + (120, 0), Logan map = logan_map.to_xy(lat, lon) (equirectangular about the helipad), so
floor and houses agree to the survey's own accuracy. The world is voxfast_v2's (rig v2 drone, no video cameras)
with the ground plane's visual replaced by the capture and the flat road strips removed (the capture has the roads),
and the roofs and tree crowns draped with the same capture projected straight down (26 Sept: as flat-coloured blocks
they hid ~40% of every suburban frame, and GeoAnchor accepted 11% of frames; the walls and trunks keep their colours).
"""
import math
import os

import numpy as np                                # cv2 / pyproj / rasterio only where the floor is built:
                                                  # the frame conversions below must import anywhere

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.dirname(HERE)
LAT0, LON0 = -27.67039, 153.14105                 # logan_map.HELIPAD_LL
PAD = (135.5, 1135.5)                             # gz_logan.PAD: the pad in Logan map metres
SITE = (120.0, 0.0)                               # where Gazebo puts it
FLOOR = f"{R}/gz_logan/imagery/ref_tile_2023-06-29.tif"
MAP = f"{R}/gz_logan/imagery/ref_tile_2021-07-21.tif"
RES = 0.13                                        # m per floor-texture pixel (the capture's own ~0.132)
EXTENT = (-140.0, 380.0, -260.0, 260.0)           # Gazebo x0, x1, y0, y1: the 480 m crop and a 20 m margin


def gz_to_ll(xg, yg):
    """Gazebo ENU -> lat, lon, the inverse of gz_logan's placement and logan_map.to_xy."""
    xm, ym = xg - SITE[0] + PAD[0], yg - SITE[1] + PAD[1]
    return LAT0 + ym / 110_574.0, LON0 + xm / (math.cos(math.radians(LAT0)) * 111_320.0)


def ll_to_gz(lat, lon):
    xm = (lon - LON0) * math.cos(math.radians(LAT0)) * 111_320.0
    ym = (lat - LAT0) * 110_574.0
    return xm - PAD[0] + SITE[0], ym - PAD[1] + SITE[1]


def floor_texture(src=FLOOR):
    """The capture resampled onto a north-up grid in Gazebo metres (row 0 = north edge)."""
    import cv2, pyproj, rasterio
    x0, x1, y0, y1 = EXTENT
    w, h = int(round((x1 - x0) / RES)), int(round((y1 - y0) / RES))
    xs = x0 + (np.arange(w) + 0.5) * RES
    ys = y1 - (np.arange(h) + 0.5) * RES
    XG, YG = np.meshgrid(xs, ys)
    lat, lon = gz_to_ll(XG, YG)
    with rasterio.open(src) as ds:
        img = np.dstack([ds.read(b) for b in (1, 2, 3)])
        to_utm = pyproj.Transformer.from_crs("EPSG:4326", ds.crs, always_xy=True)
        E, N = to_utm.transform(lon, lat)
        col, row = ~ds.transform * (E, N)
    tex = cv2.remap(img, col.astype(np.float32), row.astype(np.float32), cv2.INTER_LINEAR, borderValue=(90, 110, 70))
    covered = (col >= 0) & (row >= 0) & (col < img.shape[1]) & (row < img.shape[0])
    return tex, float(covered.mean())


def drape(src, dst):
    """The mesh with texture coordinates projected straight down onto the floor capture, so a roof or a crown seen
    from above shows the photo's own roof or canopy at that spot (OBJ: v = 0 at the image's bottom = south)."""
    x0, x1, y0, y1 = EXTENT
    with open(src) as fi, open(dst, "w") as fo:
        for line in fi:
            if line.startswith("v "):
                _, x, y, _z = line.split()
                fo.write(line)
                fo.write(f"vt {(float(x) - x0) / (x1 - x0):.6f} {(float(y) - y0) / (y1 - y0):.6f}\n")
            elif line.startswith("f "):
                fo.write("f " + " ".join(f"{c.split('//')[0]}/{c.split('//')[0]}/{c.split('//')[1]}" for c in line.split()[1:]) + "\n")
            else:
                fo.write(line)


def draped_model(name, textured, colour):
    """logan_buildings / logan_trees inline, the `textured` visual's mesh draped with the capture, the rest as it was."""
    src = open(f"{R}/gz_logan/models/{name}/model.sdf").read()
    m = src[src.index("<model"):src.index("</model>") + len("</model>")]
    m = m.replace(f"model://{name}/meshes/{textured}.obj", f"meshes/{textured}_uv.obj")
    old = (f"<material><ambient>{colour}</ambient><diffuse>{colour}</diffuse><double_sided>true</double_sided></material>")
    assert m.count(old) == 1, (name, textured)
    m = m.replace(old, "<material><ambient>1 1 1 1</ambient><diffuse>1 1 1 1</diffuse><specular>0 0 0 1</specular>"
                       "<double_sided>true</double_sided><pbr><metal><albedo_map>materials/textures/logan_2023.png"
                       "</albedo_map></metal></pbr></material>")
    os.makedirs(f"{R}/gz_logan/geo_floor/meshes", exist_ok=True)
    drape(f"{R}/gz_logan/models/{name}/meshes/{textured}.obj", f"{R}/gz_logan/geo_floor/meshes/{textured}_uv.obj")
    return "    " + m


# PX4's x500_lidar_down sensor, as a link on the drone: the rangefinder GNSS-free landings need (see world())
LIDAR = ("<link name='lidar_sensor_link'><pose relative_to='base_link'>0 0 -0.05 0 1.57 0</pose><inertial><mass>0.001</mass>"
         "<inertia><ixx>0.00001</ixx><iyy>0.00001</iyy><izz>0.00001</izz><ixy>0</ixy><ixz>0</ixz><iyz>0</iyz></inertia>"
         "</inertial><sensor name='lidar' type='gpu_lidar'><pose>0 0 0 3.14 0 0</pose><update_rate>50</update_rate><ray><scan>"
         "<horizontal><samples>1</samples><resolution>1</resolution><min_angle>0</min_angle><max_angle>0</max_angle>"
         "</horizontal><vertical><samples>1</samples><resolution>1</resolution><min_angle>0</min_angle><max_angle>0</max_angle>"
         "</vertical></scan><range><min>0.1</min><max>100.0</max><resolution>0.01</resolution></range></ray>"
         "<always_on>1</always_on><visualize>false</visualize></sensor></link><joint name='lidar_sensor_joint' type='fixed'>"
         "<parent>base_link</parent><child>lidar_sensor_link</child></joint>")


def variant(var, src="geo_magpie"):
    """A GeoAnchor world flown by another rig (gz_unified.VARIANTS, e.g. v2l: the up camera): that drone plus the
    rangefinder -> gz_logan/models/<name>_rng, and src with its drone swapped -> gz_logan/<src>_<var>/ (floor linked)."""
    import sys
    sys.path.insert(0, R)
    import gz_unified as GU
    name = GU.model_name(var)
    m = open(f"{R}/gz_logan/models/{name}/model.sdf").read()
    assert m.count("</model>") == 1 and "lidar_sensor_link" not in m
    d = f"{R}/gz_logan/models/{name}_rng"
    os.makedirs(d, exist_ok=True)
    open(f"{d}/model.sdf", "w").write(m.replace(f"<model name='{name}'>", f"<model name='{name}_rng'>").replace("</model>", LIDAR + "</model>"))
    open(f"{d}/model.config", "w").write(open(f"{R}/gz_logan/models/{name}/model.config").read().replace(name, f"{name}_rng"))
    w = open(f"{R}/gz_logan/{src}/aruco.sdf").read()
    assert w.count("<uri>model://x500_unified_rng</uri>") == 1
    g = f"{R}/gz_logan/{src}_{var}"
    os.makedirs(g, exist_ok=True)
    for sub in ("materials", "meshes"):
        if not os.path.islink(f"{g}/{sub}"):
            os.symlink(f"../{src}/{sub}", f"{g}/{sub}")
    open(f"{g}/aruco.sdf", "w").write(w.replace("<uri>model://x500_unified_rng</uri>", f"<uri>model://{name}_rng</uri>"))
    print(f"{d}/model.sdf, {g}/aruco.sdf")


def world():
    import cv2
    tex, cov = floor_texture()
    d = f"{R}/gz_logan/geo_floor"
    os.makedirs(f"{d}/materials/textures", exist_ok=True)
    cv2.imwrite(f"{d}/materials/textures/logan_2023.png", cv2.cvtColor(tex, cv2.COLOR_RGB2BGR))
    x0, x1, y0, y1 = EXTENT
    src = open(f"{R}/gz_logan/voxfast_v2/aruco.sdf").read()
    # the ground plane keeps its collision; its green visual goes, the capture takes its place
    a = src.index('    <model name="ground_plane">')
    b = src.index("    </model>", a) + len("    </model>")
    ground = src[a:b]
    v0, v1 = ground.index("<visual"), ground.index("</visual>") + len("</visual>")
    ground = ground[:v0] + ground[v1:]
    floor = (f"""    <model name="logan_floor"><static>true</static><pose>{(x0 + x1) / 2:.2f} {(y0 + y1) / 2:.2f} 0.0005 0 0 0</pose>
      <link name="link"><visual name="capture_2023"><geometry><plane><normal>0 0 1</normal><size>{x1 - x0:.2f} {y1 - y0:.2f}</size></plane></geometry>
        <material><ambient>1 1 1 1</ambient><diffuse>1 1 1 1</diffuse><specular>0 0 0 1</specular>
          <pbr><metal><albedo_map>materials/textures/logan_2023.png</albedo_map></metal></pbr></material></visual></link></model>""")
    out = src[:a] + ground + "\n" + floor + src[b:]
    # the drone with a downward rangefinder (PX4's x500_lidar_down sensor): without GNSS, PX4's height came from the
    # barometer alone and ran 2-5 m high in every descent (26 Sept: the calm flight touched down believing 2.8 m; the
    # obstacle flight's depth camera then mapped the ground 4.5 m up under the pad and the planner would not land)
    assert out.count("<uri>model://x500_unified</uri>") == 1
    out = out.replace("<uri>model://x500_unified</uri>", "<uri>model://x500_unified_rng</uri>")
    # Gazebo's GPS in Logan, not PX4's default Zurich: the depot (Gazebo 0, 0) at its own latitude and longitude, so
    # PX4's global position and GeoAnchor's fixes are on the same Earth (26 Sept: with Zurich, EKF2 gated out every fix)
    la, lo = gz_to_ll(0.0, 0.0)
    old = out[out.index("<latitude_deg>"):out.index("</longitude_deg>") + len("</longitude_deg>")]
    out = out.replace(old, f"<latitude_deg>{la:.9f}</latitude_deg>\n      <longitude_deg>{lo:.9f}</longitude_deg>")
    out = out.replace("    <include><uri>model://logan_roads</uri><pose>0 0 0 0 0 0</pose></include>\n", "")
    # roofs and crowns draped with the capture: flat-coloured blocks hid 40% of every suburban frame from the matcher
    for name, textured, colour in (("logan_buildings", "roofs", "0.55 0.33 0.27 1"), ("logan_trees", "crowns", "0.16 0.36 0.13 1")):
        inc = f"    <include><uri>model://{name}</uri><pose>0 0 0 0 0 0</pose></include>"
        assert out.count(inc) == 1, name
        out = out.replace(inc, draped_model(name, textured, colour))
    assert "logan_roads" not in out and "capture_2023" in out and "roofs_uv.obj" in out and "crowns_uv.obj" in out
    # the films' chase camera (gz_learner's, moved by set_pose from a helper process), in every GeoAnchor world
    import sys
    sys.path.insert(0, R)
    from gz_learner import chase_sdf, bird_sdf
    assert out.count("  </world>") == 1
    out = out.replace("  </world>", f"    {chase_sdf()}\n  </world>")
    open(f"{d}/aruco.sdf", "w").write(out)
    print(f"{d}/aruco.sdf  floor {tex.shape[1]}x{tex.shape[0]} px at {RES} m, capture covers {100 * cov:.1f}% of it")
    # geo_clean: the same without stage 3's unmapped obstacles (gz_voxel.py's crane, lights, traffic light, trees, shed
    # round the pad) -- for the flights whose hazard is wind or the magpie, so each flight tests one thing
    import re
    clean, n = re.subn(r"    <model name='(crane|light_\d|traffic_light|tree_\d|shed)'>.*?</model>\n", "", out, flags=re.S)
    assert n == 10 and "name='crane'" not in clean, n            # crane, 4 lights, traffic light, 3 trees, shed
    c = f"{R}/gz_logan/geo_clean"
    os.makedirs(c, exist_ok=True)
    for sub in ("materials", "meshes"):
        if not os.path.islink(f"{c}/{sub}"):
            os.symlink(f"{d}/{sub}", f"{c}/{sub}")
    open(f"{c}/aruco.sdf", "w").write(clean)
    print(f"{c}/aruco.sdf  without the {n} stage-3 obstacles")
    # geo_magpie: geo_clean with gz_learner's magpie and its nest (moved by set_pose during the flight)
    g = f"{R}/gz_logan/geo_magpie"
    os.makedirs(g, exist_ok=True)
    for sub in ("materials", "meshes"):
        if not os.path.islink(f"{g}/{sub}"):
            os.symlink(f"{d}/{sub}", f"{g}/{sub}")
    magpie, nest = bird_sdf()
    open(f"{g}/aruco.sdf", "w").write(clean.replace("  </world>", f"    {magpie}\n    {nest}\n  </world>"))
    print(f"{g}/aruco.sdf  geo_clean + the magpie and its nest")


if __name__ == "__main__":
    # the conversion must round-trip, or floor and houses disagree
    for xg, yg in ((0.0, 0.0), (120.0, 0.0), (-100.0, 200.0), (350.0, -230.0)):
        la, lo = gz_to_ll(xg, yg)
        bx, by = ll_to_gz(la, lo)
        assert abs(bx - xg) < 1e-6 and abs(by - yg) < 1e-6
    la, lo = gz_to_ll(120.0, 0.0)
    assert abs(la - -27.660121) < 2e-6 and abs(lo - 153.142424) < 2e-6, (la, lo)   # the pad, as computed before
    import sys
    if "--variant" in sys.argv:                 # geo/geo_world.py --variant v2l: the magpie world with the up camera
        variant(sys.argv[sys.argv.index("--variant") + 1])
    else:
        world()
