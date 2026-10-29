"""Render a trained splat from every solved camera and compare it with the photo.

  ml/venv.gsplat.localscheduler/Scripts/python scripts/gs_render_check.py <name> <splat.ply>

Runs in the Houdini trainer's own Python (torch + gsplat on the GPU). A camera
that sits at the wrong angle trains a splat that cannot match that photo, so
its render comes out soft or doubled and its score drops - a signal that a
COLMAP reprojection check misses, because COLMAP builds its points to fit the
cameras it was given.

Input:  colmap/_work/<name>/cameras.json (orbit_solve), colmap/<name>/images/*.png (RGBA)
Output: colmap/_work/<name>/render_check/
          scores.json        per frame: psnr (inside the character), alpha_iou, sharpness
          scores.png         the scores along the turn
          sheet_1.jpg ...    photo | render | difference, for frames round the turn
"""
import argparse
import json
import math
import os

import cv2
import numpy as np
import torch
from gsplat import rasterization
from plyfile import PlyData

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GREY = 0.5            # both photo and render are composited on this before comparing
SHEET_FRAMES = 24     # frames shown to the reviewer
TILES_PER_SHEET = 12
TILE_W = 260


def load_splat(path: str, device: str) -> dict:
    v = PlyData.read(path)["vertex"].data
    names = v.dtype.names
    col = lambda n: np.asarray(v[n], np.float32)  # noqa: E731
    rest = sorted((n for n in names if n.startswith("f_rest_")), key=lambda n: int(n[7:]))
    k = 1 + len(rest) // 3
    dc = np.stack([col(f"f_dc_{i}") for i in range(3)], 1)[:, None, :]
    sh = np.stack([col(n) for n in rest], 1).reshape(-1, 3, k - 1).transpose(0, 2, 1) if rest else None
    t = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(device)  # noqa: E731
    quats = np.stack([col(f"rot_{i}") for i in range(4)], 1)
    return {
        "means": t(np.stack([col(c) for c in "xyz"], 1)),
        "quats": t(quats / np.linalg.norm(quats, axis=1, keepdims=True)),
        "scales": t(np.exp(np.stack([col(f"scale_{i}") for i in range(3)], 1))),
        "opacities": t(1 / (1 + np.exp(-col("opacity")))),
        "colors": t(np.concatenate([dc, sh], 1) if sh is not None else dc),
        "sh_degree": int(math.sqrt(k)) - 1,
    }


def rot_x(deg: float) -> np.ndarray:
    a = math.radians(deg)
    m = np.eye(4)
    m[1:3, 1:3] = [[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]]
    return m


def render(splat: dict, cams: dict, view: dict, world_fix: np.ndarray, device: str) -> tuple[np.ndarray, np.ndarray]:
    view_m = np.eye(4)
    view_m[:3, :3] = view["R"]
    view_m[:3, 3] = view["t"]
    view_m = view_m @ world_fix
    k = np.array([[cams["fx"], 0, cams["cx"]], [0, cams["fy"], cams["cy"]], [0, 0, 1]])
    rgb, alpha, _ = rasterization(
        splat["means"], splat["quats"], splat["scales"], splat["opacities"], splat["colors"],
        torch.tensor(view_m, dtype=torch.float32, device=device)[None],
        torch.tensor(k, dtype=torch.float32, device=device)[None],
        cams["width"], cams["height"], sh_degree=splat["sh_degree"])
    return rgb[0].clamp(0, 1).cpu().numpy(), alpha[0, ..., 0].cpu().numpy()


def load_photo(path: str) -> tuple[np.ndarray, np.ndarray]:
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    rgb = cv2.cvtColor(img[..., :3], cv2.COLOR_BGR2RGB).astype(np.float32) / 255
    alpha = img[..., 3].astype(np.float32) / 255 if img.shape[2] == 4 else np.ones(img.shape[:2], np.float32)
    return rgb, alpha


def composite(rgb: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    return rgb * alpha[..., None] + GREY * (1 - alpha[..., None])


def score(photo: np.ndarray, p_alpha: np.ndarray, rend: np.ndarray, r_alpha: np.ndarray) -> dict:
    a, b = composite(photo, p_alpha), composite(rend, r_alpha)
    inside = cv2.dilate(((p_alpha > 0.1) | (r_alpha > 0.1)).astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    mse = float(np.mean((a[inside] - b[inside]) ** 2)) if inside.any() else 1.0
    pm, rm = p_alpha > 0.5, r_alpha > 0.5
    iou = float((pm & rm).sum() / max((pm | rm).sum(), 1))
    lap = lambda img: float(cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), cv2.CV_32F)[pm].var())  # noqa: E731
    sharp = lap(b) / max(lap(a), 1e-9) if pm.any() else 0.0
    return {"psnr": round(10 * math.log10(1 / max(mse, 1e-10)), 2), "alpha_iou": round(iou, 4),
            "sharpness": round(sharp, 3)}


def pick_world_fix(splat: dict, cams: dict, image_dir: str, device: str) -> np.ndarray:
    """The trainer may turn COLMAP worlds about X on load; take whichever turn fits frame 1."""
    view = cams["views"][0]
    photo, p_alpha = load_photo(os.path.join(image_dir, view["name"]))
    best = max((score(photo, p_alpha, *render(splat, cams, view, rot_x(d), device))["psnr"], d) for d in (0, 90, -90))
    print(f"world turn about X: {best[1]} deg (frame 1 psnr {best[0]})")
    return rot_x(best[1])


def tile(photo: np.ndarray, p_alpha: np.ndarray, rend: np.ndarray, r_alpha: np.ndarray, label: str) -> np.ndarray:
    a, b = composite(photo, p_alpha), composite(rend, r_alpha)
    diff = np.clip(np.abs(a - b).mean(2) * 4, 0, 1)
    heat = cv2.applyColorMap((diff * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    h = int(TILE_W * a.shape[0] / a.shape[1])
    parts = [cv2.resize(cv2.cvtColor((x * 255).astype(np.uint8), cv2.COLOR_RGB2BGR), (TILE_W, h)) for x in (a, b)]
    row = np.hstack(parts + [cv2.resize(heat, (TILE_W, h))])
    cv2.putText(row, label, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    return row


def plot(scores: list[dict], path: str) -> None:
    w, h, pad = 1200, 320, 30
    img = np.full((h, w, 3), 255, np.uint8)
    ps = np.array([s["psnr"] for s in scores])
    lo, hi = float(ps.min()) - 0.5, float(ps.max()) + 0.5
    pts = [(pad + int((w - 2 * pad) * i / max(len(ps) - 1, 1)), h - pad - int((h - 2 * pad) * (v - lo) / (hi - lo)))
           for i, v in enumerate(ps)]
    cv2.polylines(img, [np.array(pts)], False, (180, 60, 0), 2)
    for f in range(0, len(ps), 20):
        x = pts[f][0]
        cv2.line(img, (x, h - pad), (x, h - pad + 5), (0, 0, 0), 1)
        cv2.putText(img, str(f + 1), (x - 10, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    cv2.putText(img, f"PSNR inside the character per frame ({lo:.1f}..{hi:.1f} dB)", (pad, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
    cv2.imwrite(path, img)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("ply")
    args = ap.parse_args()
    device = "cuda"
    work = os.path.join(ROOT, "colmap", "_work", args.name)
    image_dir = os.path.join(ROOT, "colmap", args.name, "images")
    out = os.path.join(work, "render_check")
    os.makedirs(out, exist_ok=True)
    cams = json.load(open(os.path.join(work, "cameras.json")))
    splat = load_splat(args.ply, device)
    print(f"{len(splat['means'])} splats, SH degree {splat['sh_degree']}, {len(cams['views'])} cameras")
    fix = pick_world_fix(splat, cams, image_dir, device)

    n = len(cams["views"])
    show = set(np.linspace(0, n, SHEET_FRAMES, endpoint=False).astype(int).tolist())
    scores, tiles = [], []
    with torch.no_grad():
        for i, view in enumerate(cams["views"]):
            photo, p_alpha = load_photo(os.path.join(image_dir, view["name"]))
            rend, r_alpha = render(splat, cams, view, fix, device)
            s = dict(score(photo, p_alpha, rend, r_alpha), frame=i + 1)
            scores.append(s)
            if i in show:
                tiles.append(tile(photo, p_alpha, rend, r_alpha, f"{i + 1}  psnr {s['psnr']:.1f}"))
    for k in range(0, len(tiles), TILES_PER_SHEET):
        group = tiles[k:k + TILES_PER_SHEET]
        rows = [np.hstack(group[j:j + 2]) if j + 1 < len(group) else np.hstack([group[j], np.zeros_like(group[j])])
                for j in range(0, len(group), 2)]
        cv2.imwrite(os.path.join(out, f"sheet_{k // TILES_PER_SHEET + 1}.jpg"), np.vstack(rows))
    plot(scores, os.path.join(out, "scores.png"))
    json.dump({"ply": args.ply, "scores": scores}, open(os.path.join(out, "scores.json"), "w"), indent=1)
    ps = np.array([s["psnr"] for s in scores])
    worst = np.argsort(ps)[:8]
    print(f"psnr median {np.median(ps):.2f} dB, worst frames: " + ", ".join(f"{w + 1} ({ps[w]:.1f})" for w in worst))
    print("render check:", out)


if __name__ == "__main__":
    main()
