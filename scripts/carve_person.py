"""venv: keep only the splats of the person, using the rembg masks of the training frames, and stand her upright.

  python carve_person.py <splat.ply> <cameras.json> <mask_dataset_dir> <out.ply> <out.json> <top_frame | orbit:first-last | noroll | mean | plane> [height_m] [front_frame]

A splat survives if its centre lands inside the person mask in >= KEEP of the views that
see it (space carving), so the checker room, the floor under her feet and floaters go.
Big blobs (> MAX_SIZE of her height) are dropped too. Up = from where all cameras aim to the
camera of <top_frame> (the planned straight-down view; PCA is fooled by her arm span), or, for a
path that circles her (orbit:first-last frames), the normal of the circle the cameras ride on,
feet at the lowest 0.3 %, front = towards camera 1.
out.json: {"R": rows x/up/front, "foot": world point, "scale": to height_m} for Houdini:
P' = scale * R (P - foot).
"""
import json
import os
import sys

import cv2
import numpy as np

KEEP = 0.85        # fraction of seeing views where the centre must be inside the mask
MIN_VIEWS = 12     # views that must see a splat before it is trusted
VIEW_STEP = 3      # use every 3rd view
MAX_SIZE = 0.15    # largest splat axis, as a fraction of her height (0.04 dropped the big dark splats of a black skirt)
CYL_R = 0.5        # metres round her axis once upright (A-pose hands reach ~0.45)
MIN_ALPHA = 0.0    # was 0.05: a dark velvet skirt is many faint layers, dropping them punched holes (c003h3, 2026-10-27);
#                    training with alpha supervision leaves no room behind her, so faint splats are hers
# a splat whose blob pokes past the matte edge is the smeared rim (white/blue fuzz on 006b2's sleeves and cap,
# CEO 2026-10-28): its centre must sit EDGE_K x its on-screen radius inside the edge; 0 = centre test only
# CEO 2026-10-29: off by default - at 1.0 it ate c011's hands (thin, always at the matte edge); CARVE_EDGE_K=1 to use
EDGE_K = float(os.environ.get("CARVE_EDGE_K", 0.0))
EDGE_CAP = float(os.environ.get("CARVE_EDGE_CAP", 12))   # px: the big flat splats of her dark shirt reach the edge in
#   side views without being rim; uncapped, k=1 punched holes in her chest. Rim fuzz sits within a few px of the edge

ply_in, cams_json, mask_dir, ply_out, json_out = sys.argv[1:6]
TOP = sys.argv[6]
H = float(sys.argv[7]) if len(sys.argv) > 7 else 1.6
FRONT = int(sys.argv[8]) if len(sys.argv) > 8 else 0   # a frame that sees her face; 0 = front towards camera 1

with open(ply_in, "rb") as f:
    header, names, n = [], [], 0
    while True:
        line = f.readline()
        header.append(line)
        s = line.decode().strip()
        if s.startswith("element vertex"):
            n = int(s.split()[-1])
        elif s.startswith("property"):
            names.append(s.split()[-1])
        elif s == "end_header":
            break
    data = np.frombuffer(f.read(n * 4 * len(names)), dtype="<f4").reshape(n, len(names))
col = {k: i for i, k in enumerate(names)}
xyz = data[:, [col["x"], col["y"], col["z"]]].astype(float)
# the trainer writes the .ply turned -90 deg about X from the COLMAP world the cameras live in
# (Houdini preview turns it back): world = (x, -z, y)
P = np.stack([xyz[:, 0], -xyz[:, 2], xyz[:, 1]], 1)

cams = json.load(open(cams_json))
meta = json.load(open(os.path.join(mask_dir, "meta.json")))
W, Hh = meta["full_size"]
ox, oy = meta["crop_origin"]
sx, sy = cams["width"] / W, cams["height"] / Hh          # masks were cut at the video size
inside = np.zeros(n, int)
sigma = np.exp(data[:, [col["scale_0"], col["scale_1"], col["scale_2"]]]).max(1)   # largest axis, world units
seen = np.zeros(n, int)
masks = sorted(os.listdir(os.path.join(mask_dir, "masks")))
for i in range(0, len(cams["views"]), VIEW_STEP):
    v = cams["views"][i]
    crop = cv2.imread(os.path.join(mask_dir, "masks", masks[i]), cv2.IMREAD_GRAYSCALE)
    full = np.zeros((Hh, W), np.uint8)
    full[oy:oy + crop.shape[0], ox:ox + crop.shape[1]] = crop
    full = cv2.resize(full, (cams["width"], cams["height"]), interpolation=cv2.INTER_NEAREST)
    full = cv2.dilate(full, np.ones((5, 5), np.uint8))     # a little slack for the matte edge
    dist = cv2.distanceTransform((full > 127).astype(np.uint8), cv2.DIST_L2, 5)   # px to the nearest outside pixel
    X = P @ np.array(v["R"]).T + np.array(v["t"])
    z = X[:, 2]
    u = cams["fx"] * X[:, 0] / np.maximum(z, 1e-6) + cams["cx"]
    w = cams["fy"] * X[:, 1] / np.maximum(z, 1e-6) + cams["cy"]
    ok = (z > 0) & (u >= 0) & (u < cams["width"] - 0.5) & (w >= 0) & (w < cams["height"] - 0.5)
    seen += ok
    rad = EDGE_K * cams["fx"] * sigma[ok] / z[ok]
    inside[ok] += dist[np.round(w[ok]).astype(int), np.round(u[ok]).astype(int)] > np.clip(rad, 0.5, EDGE_CAP)
keep = (seen >= MIN_VIEWS) & (inside >= KEEP * np.maximum(seen, 1))
print(f"carving: {keep.sum()} of {n} splats inside the person ({sx:.2f}x mask scale)")

Q = P[keep]
c = Q.mean(0)
A, b = np.zeros((3, 3)), np.zeros(3)
for v in cams["views"]:   # least-squares point all camera axes pass closest to
    Rv = np.array(v["R"]); M = np.eye(3) - np.outer(Rv[2], Rv[2]); A += M; b += M @ (-Rv.T @ np.array(v["t"]))
aim = np.linalg.solve(A, b)
centres = np.array([-np.array(v["R"]).T @ np.array(v["t"]) for v in cams["views"]])
if TOP.startswith("orbit:"):   # the cameras of a level circle round her lie in a horizontal plane
    a, b = (int(x) - 1 for x in TOP[6:].split("-"))
    ring = centres[a:b + 1]
    up = np.linalg.svd(ring - ring.mean(0))[2][2]
    up *= np.sign(up @ (ring.mean(0) - aim))   # the circle is above where they aim
elif TOP == "noroll":   # the camera never rolls, so up is square to every camera's right axis; works for any path
    # (H3 videos circle on a tilted ring, so orbit: leaned those people ~15 deg; on Seedance runs it agrees within 1 deg)
    up = np.linalg.svd(np.array([np.array(v["R"])[0] for v in cams["views"]]))[2][2]
    up *= np.sign(up @ -np.array(cams["views"][0]["R"])[1])   # image up of the first frame
elif TOP == "mean":   # a path with no top view and no level ring but roughly symmetric round her (flower petals)
    up = centres.mean(0) - aim
elif TOP == "plane":   # normal of the plane all camera centres lie closest to (a path that rises and falls evenly)
    up = np.linalg.svd(centres - centres.mean(0))[2][2]
    up *= np.sign(up @ (centres.mean(0) - aim))
else:
    up = centres[int(TOP) - 1] - aim
up /= np.linalg.norm(up)
h = (Q - c) @ up
lo, hi = np.percentile(h, [0.3, 99.7])
scale = H / (hi - lo)
size = np.exp(data[keep][:, [col["scale_0"], col["scale_1"], col["scale_2"]]]).max(1) * scale
small = size < MAX_SIZE * H
print(f"dropped {(~small).sum()} oversized splats; up {up.round(3)}")
idx = np.flatnonzero(keep)[small]
foot = c + lo * up
o0 = -np.array(cams["views"][0]["R"]).T @ np.array(cams["views"][0]["t"])
fwd = (o0 - foot) - ((o0 - foot) @ up) * up
fwd /= np.linalg.norm(fwd)
foot = foot + (((Q[small] - foot) - np.outer((Q[small] - foot) @ up, up)).mean(0))   # centre her on the axis
if FRONT:   # camera 1 may see her side: front = square to her arm span (widest at arm height), towards camera FRONT
    rel = Q[small] - foot
    hh = (rel @ up) / (hi - lo)
    band = rel[(hh > 0.5) & (hh < 0.82)]
    band = band - np.outer(band @ up, up)
    arms = np.linalg.svd(band - band.mean(0), full_matrices=False)[2][0]
    of = -np.array(cams["views"][FRONT - 1]["R"]).T @ np.array(cams["views"][FRONT - 1]["t"]) - foot
    new = np.cross(arms, up)
    new *= np.sign(new @ of)
    print(f"front turned {np.degrees(np.arccos(np.clip(new @ fwd, -1, 1))):.0f} deg from camera 1 to square her arms")
    fwd = new / np.linalg.norm(new)
R =np.stack([np.cross(up, fwd), up, fwd])
# the back of the room behind her is never seen from behind at eye level, so carving keeps it:
# also keep only a cylinder round her in the upright frame, and drop near-transparent splats
U = scale * (P[idx] - foot) @ R.T
alpha = 1 / (1 + np.exp(-data[idx, col["opacity"]]))
inbox = (np.hypot(U[:, 0], U[:, 2]) < CYL_R) & (U[:, 1] > 0.0) & (U[:, 1] < H * 1.08) & (alpha > MIN_ALPHA)
print(f"cylinder/opacity trim: {len(idx)} -> {inbox.sum()}")
idx = idx[inbox]

with open(ply_out, "wb") as f:
    for line in header:
        f.write(line.replace(f"element vertex {n}".encode(), f"element vertex {len(idx)}".encode()))
    f.write(data[idx].astype("<f4").tobytes())
json.dump({"R": R.tolist(), "foot": foot.tolist(), "scale": scale, "height_m": H, "splats": int(len(idx))},
          open(json_out, "w"), indent=1)
print(f"kept {len(idx)} splats, scale x{scale:.3f}, saved {ply_out}")
