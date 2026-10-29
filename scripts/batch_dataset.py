"""venv: the two step-2 datasets of one chiLinh54 character (see batch_char.sh).

  tools/venv/Scripts/python scripts/batch_dataset.py raw <id>     before COLMAP
  tools/venv/Scripts/python scripts/batch_dataset.py guide|screen <id>   blue-screen route, before the grade
  tools/venv/Scripts/python scripts/batch_dataset.py final <id>   after the grade

raw:   data/c<id>_raw  frames = the Seedance frames, masks all white, images = the frames at RAW_W (720) wide.
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


RAW_W = 720   # CEO 2026-10-29: COLMAP solves the room at 720 wide (a 1080 Genjutsu video ran 18+ min in matching);
              # final() scales the cameras back up to the graded frames' size


def raw(cid: str) -> None:
    char, data = os.path.join(ROOT, "chars", cid), os.path.join(ROOT, "data", f"c{cid}_raw")
    frames = os.path.join(char, "frames")
    link_dir(frames, os.path.join(data, "frames"))
    images = os.path.join(data, "images")
    os.makedirs(images, exist_ok=True)
    for n in sorted(os.listdir(frames)):
        img = cv2.imread(os.path.join(frames, n))
        if img.shape[1] > RAW_W:
            img = cv2.resize(img, (RAW_W, round(img.shape[0] * RAW_W / img.shape[1])), interpolation=cv2.INTER_AREA)
        cv2.imwrite(os.path.join(images, n), img)
    os.makedirs(os.path.join(data, "masks"), exist_ok=True)
    h, w = img.shape[:2]
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


def final(cid: str) -> None:
    char, data = os.path.join(ROOT, "chars", cid), os.path.join(ROOT, "data", f"c{cid}")
    frames, graded, alpha = (os.path.join(char, d) for d in ("frames", "graded", "alpha"))
    if not os.path.isdir(alpha):   # blue-screen route: the matte is the graded frames' own alpha
        os.makedirs(alpha)
        for n in sorted(os.listdir(graded)):
            cv2.imwrite(os.path.join(alpha, n), cv2.imread(os.path.join(graded, n), cv2.IMREAD_UNCHANGED)[..., 3])
    link_dir(frames, os.path.join(data, "frames"))
    link_dir(graded, os.path.join(data, "images"))
    os.makedirs(os.path.join(data, "masks"), exist_ok=True)
    for n in sorted(os.listdir(alpha)):
        m = cv2.imread(os.path.join(alpha, n), cv2.IMREAD_GRAYSCALE)
        cv2.imwrite(os.path.join(data, "masks", n + ".png"), ((m > 127) * 255).astype(np.uint8))
    h, w = m.shape
    write_meta(data, frames, (w, h))
    src, dst = os.path.join(ROOT, "colmap", f"c{cid}_raw"), os.path.join(ROOT, "colmap", f"c{cid}")
    shutil.rmtree(dst, ignore_errors=True)
    import pycolmap
    rec = pycolmap.Reconstruction(os.path.join(src, "sparse", "0"))
    for cam in rec.cameras.values():   # the raw solve ran at RAW_W: back to the graded frames' size
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
    {"raw": raw, "guide": guide, "screen": screen, "final": final}[sys.argv[1]](sys.argv[2])
