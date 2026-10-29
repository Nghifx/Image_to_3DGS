"""trainer python: render a trained splat from the angles the video never filmed (her back half), for Fixer.

  ml/venv.gsplat.localscheduler/Scripts/python gs_novel_views.py <run> <splat.ply> <person.json> <out_dir>

Cameras ride at the training cameras' median distance from her centre, aimed at it, at azimuths beyond the
filmed side (0 = the side the cameras filmed). Angles from NV_AZ / NV_EL (comma lists, degrees), numbering from
NV_START (default 2001). Writes <out_dir>/raw/NNNN.png (RGBA,
) and <out_dir>/views.json (same layout as cameras.json: R, t world->camera, COLMAP axes).
"""
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gs_render_check as rc  # noqa: E402

AZ = tuple(int(v) for v in os.environ.get("NV_AZ", "100,120,140,160,180,200,220,240,260").split(","))
EL = tuple(int(v) for v in os.environ.get("NV_EL", "0,25,50").split(","))

run, ply, person_json, out = sys.argv[1:5]
root = os.environ.get("GSPLAT_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))).replace("\\", "/")
cams = json.load(open(f"{root}/colmap/_work/{run}/cameras.json"))
person = json.load(open(person_json))
up = np.array(person["R"][1])
centre = np.array(person["foot"]) + up * person["height_m"] / person["scale"] / 2
C = np.array([-np.array(v["R"]).T @ np.array(v["t"]) for v in cams["views"]])
dist = float(np.median(np.linalg.norm(C - centre, axis=1)))
front = (C - centre).mean(0)
front -= up * (front @ up)
front /= np.linalg.norm(front)
side = np.cross(up, front)
print(f"centre {centre.round(3)}, distance {dist:.2f}, front {front.round(2)}")

device = "cuda"
splat = rc.load_splat(ply, device)
fix = rc.pick_world_fix(splat, cams, f"{root}/colmap/{run}/images", device)
os.makedirs(f"{out}/raw", exist_ok=True)
views, k = [], int(os.environ.get("NV_START", 2001))   # numbers clear of views already in the run
for el in EL:
    for az in AZ:
        a, e = np.radians(az), np.radians(el)
        pos = centre + dist * (np.cos(e) * (np.cos(a) * front + np.sin(a) * side) + np.sin(e) * up)
        z = centre - pos
        z /= np.linalg.norm(z)
        x = np.cross(z, up)           # COLMAP camera: x right, y down, z forward
        x /= np.linalg.norm(x)
        y = np.cross(z, x)
        R = np.stack([x, y, z])
        view = {"name": f"{k:04d}.png", "R": R.tolist(), "t": (-R @ pos).tolist(), "az": az, "el": el}
        rgb, alpha = rc.render(splat, cams, view, fix, device)
        img = np.dstack([rgb[..., ::-1], alpha]) * 255
        cv2.imwrite(f"{out}/raw/{view['name']}", img.astype(np.uint8))
        views.append(view)
        k += 1
json.dump({**{c: cams[c] for c in ("width", "height", "fx", "fy", "cx", "cy")}, "views": views},
          open(f"{out}/views.json", "w"), indent=1)
print(len(views), "views ->", out)
