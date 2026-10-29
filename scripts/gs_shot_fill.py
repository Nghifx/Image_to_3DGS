"""Step 2 of the shot fill: build pictures of a GS person from the exact angles the shot needs, out of the real
pixels of her own training video (image-based rendering), to train on next to the video frames.

  ml/venv.gsplat.localscheduler/Scripts/python scripts/gs_shot_fill.py <run> <name> [views=40] [k=4]

Runs in the Houdini trainer's own Python (torch + gsplat). Input: GSplats/<run>/shot/<name>_views.json
(shot_views.py: the shot camera carried into her ply space for every frame and copy), her trained splat
(GSplats/<run>/*/gsplats/*.49999.ply, same space), the training cameras (colmap/_work/<run>/cameras.json, turned
by world_turn_x) and photos (colmap/<run>/images, graded RGBA).

1. pick up to <views> of the shot's needed views, spread over their directions (farthest point sampling)
2. each becomes a training-sized frame: the shot's pose, the lens zoomed so she is as tall as in a training
   frame, the image centred on her (a uniform zoom and a lens shift leave the view direction alone)
3. the splat gives the geometry: its depth there; every pixel of her is sent into her <k> nearest training
   photos, kept where the photo's own splat depth agrees (not hidden there), blended by how close that photo's
   direction is; pixels no photo sees keep the splat's render
4. writes GSplats/<run>/shot/fill/<name>_shot_NNN.png (RGBA, alpha = the splat's), fill.json (poses, lens) and
   sheet.jpg (splat render | filled | nearest photo) for a look
"""
import glob
import json
import os
import sys

import cv2
import numpy as np
import torch
from gsplat import rasterization

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gs_render_check import load_photo, load_splat, rot_x  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEV = "cuda"
VIS_TOL = 0.03       # depth agreement, share of the distance, for a photo pixel to count as seen
SHARP = 16           # weight = cos(angle between the view and the photo's direction) ** SHARP


def render(splat: dict, R: np.ndarray, t: np.ndarray, K: np.ndarray, w: int, h: int) -> tuple:
    vm = np.eye(4)
    vm[:3, :3], vm[:3, 3] = R, t
    out, alpha, _ = rasterization(
        splat["means"], splat["quats"], splat["scales"], splat["opacities"], splat["colors"],
        torch.tensor(vm, dtype=torch.float32, device=DEV)[None], torch.tensor(K, dtype=torch.float32, device=DEV)[None],
        w, h, sh_degree=splat["sh_degree"], render_mode="RGB+ED")
    out = out[0].cpu().numpy()
    return np.clip(out[..., :3], 0, 1), out[..., 3], alpha[0, ..., 0].cpu().numpy()


def spread(dirs: np.ndarray, n: int) -> list:
    """Farthest point sampling on the view directions."""
    pick = [0]
    d = 1 - dirs @ dirs[0]
    while len(pick) < min(n, len(dirs)):
        i = int(np.argmax(d))
        pick.append(i)
        d = np.minimum(d, 1 - dirs @ dirs[i])
    return pick


def main() -> None:
    run, name = sys.argv[1], sys.argv[2]
    n_views = int(sys.argv[3]) if len(sys.argv) > 3 else 40
    k_near = int(sys.argv[4]) if len(sys.argv) > 4 else 4
    shot = json.load(open(f"{ROOT}/GSplats/{run}/shot/{name}_views.json"))
    cams = json.load(open(f"{ROOT}/colmap/_work/{run}/cameras.json"))
    splat = load_splat(sorted(glob.glob(f"{ROOT}/GSplats/{run}/*/gsplats/*49999*.ply"))[0], DEV)
    fix = rot_x(shot["world_turn_x"])[:3, :3]
    W, H = cams["width"], cams["height"]
    Kt = np.array([[cams["fx"], 0, cams["cx"]], [0, cams["fy"], cams["cy"]], [0, 0, 1]])
    train = [(v["name"], np.array(v["R"]) @ fix.T, np.array(v["t"])) for v in cams["views"]]
    centre = np.array(shot["centre"])
    tdir = np.array([-R.T @ t - centre for _, R, t in train])
    tdist = float(np.median(np.linalg.norm(tdir, axis=1)))
    tdir /= np.linalg.norm(tdir, axis=1, keepdims=True)
    views = shot["views"]
    vdir = np.array([-np.array(v["R"]).T @ np.array(v["t"]) - centre for v in views])
    vdist = np.linalg.norm(vdir, axis=1)
    vdir /= vdist[:, None]
    out_dir = f"{ROOT}/GSplats/{run}/shot/fill"
    os.makedirs(out_dir, exist_ok=True)
    depth_cache, photo_cache, records, tiles = {}, {}, [], []
    for n, i in enumerate(spread(vdir, n_views)):
        v = views[i]
        R, t = np.array(v["R"]), np.array(v["t"])
        f = cams["fx"] * vdist[i] / tdist                       # zoom: as tall as in a training frame
        xc = R @ centre + t
        K = np.array([[f, 0, W / 2 - f * xc[0] / xc[2]], [0, f, H / 2 - f * xc[1] / xc[2]], [0, 0, 1]])
        rgb_g, dep_g, a_g = render(splat, R, t, K, W, H)
        ys, xs = np.nonzero(a_g > 0.5)
        X = np.stack([(xs - K[0, 2]) / f, (ys - K[1, 2]) / f, np.ones(len(xs))], 1) * dep_g[ys, xs, None]
        Xw = (X - t) @ R                                       # camera -> ply space: R^T (X - t)
        acc, wsum = np.zeros((len(xs), 3)), np.zeros(len(xs))
        for j in np.argsort(-(tdir @ vdir[i]))[:k_near]:
            nm, Rj, tj = train[j]
            if nm not in depth_cache:
                depth_cache[nm] = render(splat, Rj, tj, Kt, W, H)[1].astype(np.float32)
                photo_cache[nm] = load_photo(f"{ROOT}/colmap/{run}/images/{nm}")
            Xj = Xw @ Rj.T + tj
            # remap wants maps under 32767 rows: lay the pixel list out 1024 wide
            npx = len(xs)
            pad = (-npx) % 1024
            uj = np.pad(Kt[0, 0] * Xj[:, 0] / Xj[:, 2] + Kt[0, 2], (0, pad), constant_values=-1).astype(np.float32).reshape(-1, 1024)
            vj = np.pad(Kt[1, 1] * Xj[:, 1] / Xj[:, 2] + Kt[1, 2], (0, pad), constant_values=-1).astype(np.float32).reshape(-1, 1024)
            ph, pa = photo_cache[nm]

            def samp(img: np.ndarray) -> np.ndarray:
                s_ = cv2.remap(img, uj, vj, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
                return s_.reshape(-1, *img.shape[2:])[:npx]
            seen = (np.abs(samp(depth_cache[nm]) - Xj[:, 2]) < VIS_TOL * Xj[:, 2]) & (samp(pa) > 0.5) & (Xj[:, 2] > 0)
            wgt = seen * max(float(tdir[j] @ vdir[i]), 0.0) ** SHARP
            acc += wgt[:, None] * samp(ph)
            wsum += wgt
        fill = rgb_g.copy()
        ok = wsum > 1e-4
        fill[ys[ok], xs[ok]] = acc[ok] / wsum[ok, None]
        cover = float(ok.mean()) if len(ok) else 0.0
        fname = f"{name}_shot_{n:03d}.png"
        rgba = (np.dstack([fill, a_g]) * 255).astype(np.uint8)
        cv2.imwrite(f"{out_dir}/{fname}", cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA))
        records.append({"name": fname, "R": R.tolist(), "t": t.tolist(), "fx": f, "fy": f, "cx": K[0, 2], "cy": K[1, 2],
                        "frame": v["frame"], "copy": v["copy"], "angle": v["angle"], "photo_cover": round(cover, 3)})
        if n < 8:
            near = photo_cache[train[int(np.argmax(tdir @ vdir[i]))][0]][0]
            tiles.append(np.hstack([cv2.resize(x, (240, int(240 * H / W))) for x in (rgb_g, fill, near)]))
        print(f"view {n}: shot frame {v['frame']} copy {v['copy']}, {v['angle']} deg off, photo pixels {cover:.0%}", flush=True)
    json.dump({"run": run, "width": W, "height": H, "world_turn_x": shot["world_turn_x"], "views": records},
              open(f"{out_dir}/fill.json", "w"), indent=1)
    if tiles:
        cv2.imwrite(f"{out_dir}/sheet.jpg", cv2.cvtColor((np.vstack(tiles) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
    print("fills:", len(records), "->", out_dir)


if __name__ == "__main__":
    main()
