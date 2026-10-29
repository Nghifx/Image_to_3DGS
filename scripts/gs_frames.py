"""Step 2 of GSplat Studio: video -> lossless frames -> background cut out.

  tools/venv/Scripts/python scripts/gs_frames.py <video | frames folder> <name> [--step 2] [--max 120]

--step works like the Increment of a ROP Image Output: 1 keeps every frame,
2 keeps one of every two, and so on - the sharpest of each group, so frames the
AI video blurred while the camera moved fast are left out. --max keeps only the
first N picked frames (an orbit video that ends on its first frame has one frame
too many).

Output (data/<name>):
  frames/0001.png ...   lossless RGB, cropped to the subject
  images/0001.png ...   same frames, background transparent (RGBA) - what training sees
  masks/0001.png.png    subject masks (COLMAP naming)
  meta.json             {"source", "step", "count", "full_size": [w, h], "crop_origin": [x, y],
                         "picked": source frame numbers kept (null when every frame is kept)}
"""
import argparse
import json
import os
import shutil
import subprocess
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gs_probe import IMAGE_EXTS, probe, read_image  # noqa: E402
from video_to_dataset import FFMPEG, crop_to_subject, cut_background, progress  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXPORT_SPAN = (0, 25)   # progress range of the frame export; background cut takes 25..95
# Accurate chroma upsampling and rounding: the default bilinear scaler softens
# fine detail (claws, hair strands) when 4:2:0 video becomes RGB.
SWS_FLAGS = "spline+accurate_rnd+full_chroma_int"


def export_video(video: str, out_dir: str, step: int, expected: int) -> None:
    cmd = [FFMPEG, "-hide_banner", "-nostdin", "-loglevel", "error", "-nostats", "-progress", "pipe:1", "-y",
           "-i", video, "-vf", f"select='not(mod(n\\,{step}))'", "-vsync", "vfr", "-frames:v", str(expected),
           "-sws_flags", SWS_FLAGS, "-pix_fmt", "rgb24", os.path.join(out_dir, "%04d.png")]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True)
    for line in proc.stdout:
        if line.startswith("frame="):
            done = int(line.split("=", 1)[1] or 0)
            progress(EXPORT_SPAN[0] + (EXPORT_SPAN[1] - EXPORT_SPAN[0]) * min(1.0, done / expected),
                     f"Xuat anh {done}/{expected}")
    if proc.wait() != 0:
        sys.exit("ffmpeg failed")


def export_images(folder: str, out_dir: str, step: int, expected: int) -> None:
    names = sorted(f for f in os.listdir(folder) if f.lower().endswith(IMAGE_EXTS))[::step][:expected]
    for i, name in enumerate(names, 1):
        img = read_image(os.path.join(folder, name), cv2.IMREAD_COLOR)
        cv2.imwrite(os.path.join(out_dir, f"{i:04d}.png"), img)
        progress(EXPORT_SPAN[0] + (EXPORT_SPAN[1] - EXPORT_SPAN[0]) * i / len(names),
                 f"Xuat anh {i}/{len(names)}")


def sharpness(path: str) -> float:
    """Variance of the Laplacian: drops when a frame is motion-blurred."""
    gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return float(cv2.Laplacian(gray, cv2.CV_32F).var())


def keep_sharpest(frames: str, step: int, limit: int) -> list[int]:
    """Out of every `step` frames in a row keep the sharpest one, renumbered from 0001.

    AI video blurs some frames where the camera moves fast; taking the blind
    every-n-th frame lets those into COLMAP. Returns the kept source frame numbers.
    """
    names = sorted(os.listdir(frames))
    kept = []
    for i in range(0, len(names), step):
        group = names[i:i + step]
        kept.append(max(group, key=lambda n: sharpness(os.path.join(frames, n))))
    kept = kept[:limit] if limit > 0 else kept
    for name in names:
        if name not in kept:
            os.remove(os.path.join(frames, name))
    for i, name in enumerate(kept, 1):  # kept frame i came from frame >= i: no name clash
        if name != f"{i:04d}.png":
            os.rename(os.path.join(frames, name), os.path.join(frames, f"{i:04d}.png"))
    return [int(n[:4]) for n in kept]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("source")
    ap.add_argument("name")
    ap.add_argument("--step", type=int, default=1)
    ap.add_argument("--max", type=int, default=0, help="keep at most this many frames (0 = all)")
    ap.add_argument("--keep-bg", action="store_true",
                    help="keep the full frame (no crop): a textured set (checker floor) helps COLMAP")
    args = ap.parse_args()
    step = max(1, args.step)

    info = probe(args.source)
    root = os.path.join(ROOT, "data", args.name)
    frames, images, masks = (os.path.join(root, d) for d in ("frames", "images", "masks"))
    for d in (frames, images, masks):
        shutil.rmtree(d, ignore_errors=True)
        os.makedirs(d)

    expected = -(-info["frames"] // step)
    if args.max > 0:
        expected = min(expected, args.max)
    progress(0, f"Xuat {expected} anh tu {info['frames']} khung")
    # every frame comes out, then the sharpest of each `step` in a row is kept
    if info["kind"] == "video":
        export_video(args.source, frames, 1, info["frames"])
    else:
        export_images(args.source, frames, 1, info["frames"])
    picked = keep_sharpest(frames, step, args.max) if step > 1 or args.max > 0 else None
    count = len(os.listdir(frames))
    print(f"{count} frames exported at {info['width']}x{info['height']}")

    if args.keep_bg:
        # the set stays in the training images too; masks cover the whole frame
        for name in sorted(os.listdir(frames)):
            shutil.copy2(os.path.join(frames, name), os.path.join(images, name))
            h, w = cv2.imread(os.path.join(frames, name), cv2.IMREAD_GRAYSCALE).shape
            cv2.imwrite(os.path.join(masks, name + ".png"), np.full((h, w), 255, np.uint8))
    else:
        cut_background(frames, images, masks, span=(EXPORT_SPAN[1], 95))
    x0 = y0 = 0
    if not args.keep_bg:
        progress(96, "Cat khung quanh nhan vat")
        x0, y0 = crop_to_subject(frames, images, masks)
    with open(os.path.join(root, "meta.json"), "w") as f:
        json.dump({"source": os.path.abspath(args.source), "step": step, "count": count,
                   "full_size": [info["width"], info["height"]], "crop_origin": [x0, y0],
                   "picked": picked}, f, indent=1)
    progress(100, f"Xong {count} anh")
    print("frames ready:", root)


if __name__ == "__main__":
    main()
