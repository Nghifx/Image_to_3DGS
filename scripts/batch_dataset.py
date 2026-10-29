"""venv: the two step-2 datasets of one chiLinh54 character (see batch_char.sh).

  tools/venv/Scripts/python scripts/batch_dataset.py raw <id>     before COLMAP
  tools/venv/Scripts/python scripts/batch_dataset.py guide|screen <id>   blue-screen route, before the grade
  tools/venv/Scripts/python scripts/batch_dataset.py masked <id>  blue-screen route, after the grade (COLMAP on her)
  tools/venv/Scripts/python scripts/batch_dataset.py final <id>   LTX route, after the grade

raw:   data/c<id>_raw  frames = the Seedance frames, masks all white, images = the frames.
       COLMAP runs on it with --keep-bg: the checker room is rigid and gives the truest cameras.
final: data/c<id>      frames = the same, images = graded RGBA (chars/<id>/graded), masks = LTX matte > 50 %.
       colmap/c<id>    sparse/0 copied from the raw solve, images = hard links to the graded frames,
       colmap/_work/c<id>/cameras.json copied too (carve_person.py reads it).
"""
import json
import os
import shutil
import sys

import cv2
import numpy as np

ROOT = os.environ.get("GSPLAT_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def link_dir(src: str, dst: str) -> None:
    os.makedirs(dst, exist_ok=True)
    for n in sorted(os.listdir(src)):
        if not os.path.exists(os.path.join(dst, n)):
            os.link(os.path.join(src, n), os.path.join(dst, n))


def write_meta(data: str, frames: str, size: tuple) -> None:
    meta = {"source": frames, "step": 1, "count": len(os.listdir(frames)), "full_size": list(size),
            "crop_origin": [0, 0], "picked": None}
    with open(os.path.join(data, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)


def raw(cid: str) -> None:
    # COLMAP reads data/<run>/frames (orbit_solve "rgb"), so frames, images, masks and meta all stay at the frames'
    # own size. 2026-10-29: images were shrunk to 720 while COLMAP still read the 1080 frames; meta said 720, so the
    # optical centre was fixed at (360, 640) on 1080x1920 frames - every 1080 run (0751, 0751b, 0751c) trained on
    # cameras off by 180 x 320 px and came out ghosted
    char, data = os.path.join(ROOT, "chars", cid), os.path.join(ROOT, "data", f"c{cid}_raw")
    frames = os.path.join(char, "frames")
    link_dir(frames, os.path.join(data, "frames"))
    link_dir(frames, os.path.join(data, "images"))
    os.makedirs(os.path.join(data, "masks"), exist_ok=True)
    h, w = cv2.imread(os.path.join(frames, sorted(os.listdir(frames))[0])).shape[:2]
    white = np.full((h, w), 255, np.uint8)
    for n in os.listdir(frames):
        cv2.imwrite(os.path.join(data, "masks", n + ".png"), white)
    write_meta(data, frames, (w, h))
    print("raw dataset", data, f"{w}x{h}")


GUIDE_MIN = 32   # rembg confidence (0-255) that counts as her in the guide


def guide(cid: str) -> None:
    """Blue-screen route, before batch_grade.py key: a coarse AI cut-out (rembg, local) per frame in
    chars/<id>/guide. The key alone cannot tell her blue embroidery from the screen (006b: same hue, saturation and
    brightness); the Houdini net keeps the key's edge and takes her inside and the drop of the red crosses from this."""
    from rembg import new_session, remove
    char = os.path.join(ROOT, "chars", cid)
    frames, out = os.path.join(char, "frames"), os.path.join(char, "guide")
    os.makedirs(out, exist_ok=True)
    session = new_session("isnet-general-use")
    for n in sorted(os.listdir(frames)):
        img = cv2.imread(os.path.join(frames, n))
        m = np.asarray(remove(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), session=session, only_mask=True))
        # hard, generous: a half-sure guide (her shoes in top views, 006b) would fade the key there
        cv2.imwrite(os.path.join(out, n), ((m >= GUIDE_MIN) * 255).astype(np.uint8))
    print("guide", out)


def screen(cid: str) -> None:
    """Blue-screen route, after guide: the screen's median sRGB colour (pixels well clear of the guide, every 20th
    frame) -> chars/<id>/screen_rgb.json, read by batch_grade.py key for the key and the despill. Each Genjutsu
    video gets its own blue (011: 0.31 0.51 0.72 vs 006b's 0.19 0.49 0.63, which left a pale blue rim)."""
    char = os.path.join(ROOT, "chars", cid)
    px = []
    for n in sorted(os.listdir(os.path.join(char, "frames")))[::20]:
        img = cv2.imread(os.path.join(char, "frames", n))[:, :, ::-1] / 255.0
        g = cv2.imread(os.path.join(char, "guide", n), cv2.IMREAD_GRAYSCALE)
        px.append(img[cv2.dilate((g > 127).astype(np.uint8), np.ones((41, 41), np.uint8)) == 0])
    rgb = [round(float(v), 3) for v in np.median(np.concatenate(px), 0)]
    json.dump(rgb, open(os.path.join(char, "screen_rgb.json"), "w"))
    print("screen", rgb)


def clean_matte(a: np.ndarray, frame: np.ndarray, guide: np.ndarray) -> np.ndarray:
    """Blue-screen route, the clean-up the CEO asked for on 0751 (2026-10-29), done by hand then folded in here:
    1. deep-blue wall keyed in past her outline (hue past the key's band, dark top of frame): cut outside the
       AI guide (+4 px);
    2. thin necks to floor crosses touching her toes: open 13 px, keep the biggest blob, give back 9 px of her
       own fine edges (fingers, hair);
    3. biggest blob only (stray crosses, specks).
    Crosses fused with her black hem are left to carve_person.py (dark splats on the floor)."""
    near = cv2.dilate((guide > 127).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))) > 0
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(float)
    wall = (a > 0) & ~near & (hsv[..., 0] * 2 > 190) & (hsv[..., 0] * 2 < 250) & (hsv[..., 1] > 0.35 * 255)
    a = np.where(wall, 0, a).astype(a.dtype)
    m = (a > 0).astype(np.uint8)
    op = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)))
    n, lab, st, _ = cv2.connectedComponentsWithStats(op, 8)
    if n > 2:
        body = (lab == 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)
        keep = cv2.dilate(body, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19))) > 0
        a = np.where(keep, a, 0).astype(a.dtype)
    return largest_island(a)


def largest_island(a: np.ndarray) -> np.ndarray:
    """Only the biggest connected blob of the matte (her); stray floor crosses and specks keyed in with her are
    cut (CEO 2026-10-29). Anything touching her - a cross under her feet - stays; chars/<id>/mask_sheet.jpg shows it."""
    n, lab, stats, _ = cv2.connectedComponentsWithStats((a > 0).astype(np.uint8), 8)
    if n <= 2:
        return a
    keep = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return np.where(lab == keep, a, 0).astype(a.dtype)


def masked(cid: str) -> tuple:
    """Blue-screen route: data/c<id> (frames, graded RGBA images, key masks) - COLMAP then solves on her alone
    (gs_colmap_ai.py c<id>, no --keep-bg). 2026-10-29, 0751: the AI-drawn blue room drifts against her (its cloth
    folds are redrawn frame to frame); the room solve gave a 48 mm lens and ghosted splats (21 dB), the solve on
    her alone 61 mm and a clean, sharp splat (27.7 dB)."""
    char, data = os.path.join(ROOT, "chars", cid), os.path.join(ROOT, "data", f"c{cid}")
    frames, graded, alpha = (os.path.join(char, d) for d in ("frames", "graded", "alpha"))
    if not os.path.isdir(alpha):   # blue-screen route: the matte is the graded frames' own alpha
        os.makedirs(alpha)
        for n in sorted(os.listdir(graded)):
            img = cv2.imread(os.path.join(graded, n), cv2.IMREAD_UNCHANGED)
            gpath = os.path.join(char, "guide", n)
            if os.path.exists(gpath):
                img[..., 3] = clean_matte(img[..., 3], cv2.imread(os.path.join(frames, n)), cv2.imread(gpath, 0))
            else:
                img[..., 3] = largest_island(img[..., 3])
            cv2.imwrite(os.path.join(graded, n), img)   # the training images lose the islands too
            cv2.imwrite(os.path.join(alpha, n), img[..., 3])
    link_dir(frames, os.path.join(data, "frames"))
    link_dir(graded, os.path.join(data, "images"))
    os.makedirs(os.path.join(data, "masks"), exist_ok=True)
    for n in sorted(os.listdir(alpha)):
        m = cv2.imread(os.path.join(alpha, n), cv2.IMREAD_GRAYSCALE)
        cv2.imwrite(os.path.join(data, "masks", n + ".png"), ((m > 127) * 255).astype(np.uint8))
    h, w = m.shape
    write_meta(data, frames, (w, h))
    print("masked dataset", data)
    return data, w, h


def final(cid: str) -> None:
    """LTX route: the masked dataset plus the cameras of the raw (room) solve."""
    data, w, h = masked(cid)
    graded = os.path.join(ROOT, "chars", cid, "graded")
    src, dst = os.path.join(ROOT, "colmap", f"c{cid}_raw"), os.path.join(ROOT, "colmap", f"c{cid}")
    shutil.rmtree(dst, ignore_errors=True)
    import pycolmap
    rec = pycolmap.Reconstruction(os.path.join(src, "sparse", "0"))
    for cam in rec.cameras.values():   # in case the graded frames differ in size from the solve
        if cam.width != w:
            cam.rescale(w / cam.width)
    os.makedirs(os.path.join(dst, "sparse", "0"))
    rec.write(os.path.join(dst, "sparse", "0"))
    link_dir(graded, os.path.join(dst, "images"))
    work = os.path.join(ROOT, "colmap", "_work", f"c{cid}")
    os.makedirs(work, exist_ok=True)
    cams = json.load(open(os.path.join(ROOT, "colmap", "_work", f"c{cid}_raw", "cameras.json")))
    k = w / cams["width"]
    cams.update(width=w, height=h, **{c: cams[c] * k for c in ("fx", "fy", "cx", "cy")})
    json.dump(cams, open(os.path.join(work, "cameras.json"), "w"), indent=1)
    print("final dataset", data, "+", dst)


if __name__ == "__main__":
    {"raw": raw, "guide": guide, "screen": screen, "masked": masked, "final": final}[sys.argv[1]](sys.argv[2])
