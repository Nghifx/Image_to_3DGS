"""hython shot_views.py <crowd.hip> <shot_cam.abc> <person name> [frame step=2]

Step 1 of the shot fill: which views of one GS person does the shot really need, and which of them did her
training video never see?

The crowd hip (spiral_tracked_v0XX.py) places copies of each carved person: <name>_stand_upright turns her ply
upright and scales her, <name>_slots holds one point per copy (P, orient). The shot camera is the plate track
(Alembic). Its world is our floor track's world up to a similarity (the tracker's scene is ~1.2x ours), so it is
fitted onto /obj/plate_cam: the rotation from the camera orientations (the boom-up centres are nearly a straight
line and cannot fix a turn about it), then scale and offset from the centres.
For every shot frame and every copy of her in view, the shot camera is carried back into her own ply space:
  P_world = Q (A P + t) + pos,  A = s R  ->  camera pose in ply space R' = Rcw Q R,  T' = Rcw (Q t + pos - C) / s
(a uniform scale does not change a picture). That virtual camera sees the splat exactly as the shot does.

Each needed view is scored against the training cameras (colmap/_work/<run>/cameras.json, turned into ply space
with the world turn the trainer applies, picked as the turn that puts the training orbit above her):
  angle  degrees to the nearest training view direction (from her centre): > GAP_DEG = a gap the splat guesses at
  zoom   her height in the shot in pixels / her height in a training frame: > 1 = the shot wants more detail
Writes GSplats/<run>/shot/<name>_views.json and houdini/shot_views_<name>.hip (unit sphere round her: training
directions green, the shot's red to yellow by angle).
"""
import json
import math
import os
import sys

import numpy as np

import hou

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GAP_DEG = 15.0          # a needed view this far from every training view is a gap
SHOT_W, SHOT_H = 1920, 1080

hip, abc, name = sys.argv[1], sys.argv[2], sys.argv[3]
step = int(sys.argv[4]) if len(sys.argv) > 4 else 2
hou.hipFile.load(hip, suppress_save_prompt=True, ignore_load_warnings=True)
crowd = hou.node("/obj/GS_SPIRAL_CROWD")
ply_node, stand, slots = (crowd.node(f"{name}_{s}") for s in ("ply", "stand_upright", "slots"))
ply_path = ply_node.parm("file").eval()
run = os.path.basename(ply_path).replace("_person.ply", "")
f0, f1 = (int(v) for v in hou.playbar.frameRange())


def pts(node) -> np.ndarray:
    return np.array(node.geometry().pointFloatAttribValues("P")).reshape(-1, 3)


def axes_at(node, f: int) -> tuple[np.ndarray, np.ndarray]:
    """Columns = the node's x, y, z axes in world; and its centre."""
    m = np.array(node.worldTransformAtTime(hou.frameToTime(f)).asTupleOfTuples())
    return m[:3, :3].T, m[3, :3]


# her ply -> upright & scaled (the bake/rotate/stand chain), fitted exactly as an affine map on her own points
P_ply, P_up = pts(ply_node), pts(stand)
X = np.c_[P_ply, np.ones(len(P_ply))]
At = np.linalg.lstsq(X, P_up, rcond=None)[0]           # P_up = P_ply @ A^T + t
A, t = At[:3].T, At[3]
s = float(np.cbrt(np.linalg.det(A)))
Rot = A / s
print(f"{name} ({run}): {len(P_ply)} splats, upright scale {s:.3f}, fit error {np.abs(X @ At - P_up).max():.2e}")

copies = []
for p in slots.geometry().points():
    M = np.array(hou.Quaternion(p.attribValue("orient")).extractRotationMatrix3().asTupleOfTuples())
    copies.append((M.T, np.array(p.position())))      # Houdini row-vector matrix -> column form
print("copies in the crowd:", len(copies))

# the shot camera (Alembic) mapped onto the crowd's world through the floor-track plate_cam
plate = hou.node("/obj/plate_cam")
ar = hou.node("/obj").createNode("alembicarchive", "shot_abc")
ar.parm("fileName").set(abc)
ar.parm("buildHierarchy").pressButton()
acam = [n for n in ar.allSubChildren() if n.type().name() == "cam"][0]
frames = list(range(f0, f1 + 1, step))
pa = [axes_at(acam, f) for f in frames]
pp = [axes_at(plate, f) for f in frames]
U, _, Vt = np.linalg.svd(sum(Mp @ Ma.T for (Ma, _), (Mp, _) in zip(pa, pp)))
Rs = U @ np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))]) @ Vt
Ca, Cp = np.array([c for _, c in pa]), np.array([c for _, c in pp])
ma, mp = Ca.mean(0), Cp.mean(0)
k = float(((Cp - mp) * ((Rs @ (Ca - ma).T).T)).sum() / ((Ca - ma) ** 2).sum())
off = mp - k * Rs @ ma
turn_err = max(math.degrees(math.acos(np.clip((np.trace(Rs @ Ma @ Mp.T) - 1) / 2, -1, 1))) for (Ma, _), (Mp, _) in zip(pa, pp))
print(f"Alembic -> crowd world: scale {k:.4f}, centre error {np.abs((k * (Rs @ Ca.T)).T + off - Cp).max():.3f} m, "
      f"orientation error {turn_err:.2f} deg")
fpx = acam.parm("focal").eval() * SHOT_W / acam.parm("aperture").eval()

# training cameras in her ply space: cameras.json is COLMAP; the trainer turns the world about X
cams = json.load(open(f"{ROOT}/colmap/_work/{run}/cameras.json"))
Rt = np.array([v["R"] for v in cams["views"]])
Tt = np.array([v["t"] for v in cams["views"]])
Ct_col = np.einsum("nji,nj->ni", Rt, -Tt)              # centres = -R^T t


def rot_x(deg: float) -> np.ndarray:
    a = math.radians(deg)
    return np.array([[1, 0, 0], [0, math.cos(a), -math.sin(a)], [0, math.sin(a), math.cos(a)]])


centre = np.median(P_ply, 0)


def median_elev(deg: float) -> float:
    """Median elevation of the training cameras over her if ply = rot_x(deg) @ COLMAP world: the right turn puts
    the orbit above her (the video looks down at her), a wrong one below or beside her."""
    d = (rot_x(deg) @ Ct_col.T).T - centre
    return float(np.median((d / np.linalg.norm(d, axis=1, keepdims=True)) @ Rot[1]))


turn = max((0, 90, -90, 180), key=median_elev)
Ct = (rot_x(turn) @ Ct_col.T).T
dir_t = Ct - centre
dist_t = np.linalg.norm(dir_t, axis=1)
dir_t /= dist_t[:, None]
h_ply = float(np.ptp(P_ply @ Rot[1]))                    # her height along her own up axis, ply units
h_train_px = cams["fy"] * h_ply / float(np.median(dist_t))
print(f"world turn {turn} deg, {len(Ct)} training views, she is ~{h_train_px:.0f} px tall in a training frame")
el_t = np.degrees(np.arcsin(np.clip(dir_t @ Rot[1], -1, 1)))
print(f"training views: elevation over her {el_t.min():.0f} .. {el_t.max():.0f} deg (her up = upright Y in ply space)")

views = []
for f in frames:
    Ma, Ca_f = axes_at(acam, f)
    C = k * Rs @ Ca_f + off
    Rcw = np.diag([1, -1, -1]) @ (Rs @ Ma).T              # world -> camera, COLMAP axes (x right, y down, z forward)
    for i, (Q, pos) in enumerate(copies):
        R_ = Rcw @ Q @ Rot
        T_ = Rcw @ (Q @ t + pos - C) / s
        xc = R_ @ centre + T_
        if xc[2] <= 0:
            continue
        u, v = fpx * xc[0] / xc[2] + SHOT_W / 2, fpx * xc[1] / xc[2] + SHOT_H / 2
        if not (-0.1 * SHOT_W < u < 1.1 * SHOT_W and -0.1 * SHOT_H < v < 1.1 * SHOT_H):
            continue
        d = -R_.T @ T_ - centre
        dist = float(np.linalg.norm(d))
        cosang = dir_t @ (d / dist)
        j = int(np.argmax(cosang))
        views.append({"frame": f, "copy": i, "R": R_.tolist(), "t": T_.tolist(), "px": [round(u), round(v)],
                      "elev": round(math.degrees(math.asin(float(np.clip(d @ Rot[1] / dist, -1, 1)))), 1),
                      "az": round(math.degrees(math.atan2(float(d @ Rot[0]), float(d @ Rot[2]))), 1),   # 0 = her front, + = her left
                      "angle": round(math.degrees(math.acos(min(1.0, float(cosang[j])))), 2),
                      "nearest": cams["views"][j]["name"], "zoom": round(fpx * h_ply / dist / h_train_px, 3)})

out_dir = f"{ROOT}/GSplats/{run}/shot"
os.makedirs(out_dir, exist_ok=True)
json.dump({"name": name, "run": run, "ply": ply_path, "world_turn_x": turn, "fpx": fpx, "width": SHOT_W,
           "height": SHOT_H, "centre": centre.tolist(), "views": views}, open(f"{out_dir}/{name}_views.json", "w"))
ang = np.array([v["angle"] for v in views])
zoom = np.array([v["zoom"] for v in views])
print(f"{len(views)} needed views (frames x copies in shot): angle to the nearest training view median "
      f"{np.median(ang):.1f}, max {ang.max():.1f} deg; {int((ang > GAP_DEG).sum())} over {GAP_DEG:.0f} deg; "
      f"zoom median {np.median(zoom):.2f}, max {zoom.max():.2f}; shot elevation over her "
      f"{min(v['elev'] for v in views):.0f} .. {max(v['elev'] for v in views):.0f} deg")

# viewport look: unit sphere round her, training directions green, needed ones red (on a training view) to yellow
need = [(-np.array(v["R"]).T @ np.array(v["t"]) - centre, v["angle"]) for v in views]
geo = hou.node("/obj").createNode("geo", f"SHOT_VIEWS_{name}")
py = geo.createNode("python", "directions")
py.parm("python").set(f"""geo = hou.pwd().geometry()
geo.addAttrib(hou.attribType.Point, "Cd", (1.0, 1.0, 1.0))
for d in {[tuple(map(float, x)) for x in dir_t]!r}:
    p = geo.createPoint(); p.setPosition(d); p.setAttribValue("Cd", (0.1, 0.9, 0.2))
for d, a in {[(tuple(map(float, c / np.linalg.norm(c))), float(a)) for c, a in need]!r}:
    p = geo.createPoint(); p.setPosition(d); p.setAttribValue("Cd", (1.0, min(a / 45.0, 1.0), 0.0))
""")
py.setDisplayFlag(True)
hou.hipFile.save(f"{ROOT}/houdini/shot_views_{name}.hip")
print("wrote", f"{out_dir}/{name}_views.json")
