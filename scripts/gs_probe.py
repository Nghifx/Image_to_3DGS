"""Print what a video (or a folder of frames) holds, as one JSON line.

  tools/venv/Scripts/python scripts/gs_probe.py <video | frames folder>
  -> {"kind": "video", "frames": 240, "fps": 24.0, "width": 1920, "height": 1080}

Used by GSplat Studio the moment a source is picked (Houdini's Python has no cv2).
"""
import json
import os
import sys

import cv2
import numpy as np

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")


def read_image(path: str, flags: int = cv2.IMREAD_UNCHANGED) -> np.ndarray:
    """cv2.imread that also copes with non-ASCII Windows paths; exits on unreadable files."""
    img = cv2.imdecode(np.fromfile(path, np.uint8), flags)
    if img is None:
        raise SystemExit(f"cannot read image: {path}")
    return img


def probe(source: str) -> dict:
    if os.path.isdir(source):
        names = sorted(f for f in os.listdir(source) if f.lower().endswith(IMAGE_EXTS))
        if not names:
            raise SystemExit(f"no images in {source}")
        h, w = read_image(os.path.join(source, names[0])).shape[:2]
        return {"kind": "images", "frames": len(names), "fps": 0.0, "width": w, "height": h}
    cap = cv2.VideoCapture(source)
    try:
        info = {"kind": "video", "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                "fps": round(float(cap.get(cv2.CAP_PROP_FPS)), 3),
                "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}
    finally:
        cap.release()
    if info["frames"] <= 0 or info["width"] <= 0:
        raise SystemExit(f"cannot read video: {source}")
    return info


if __name__ == "__main__":
    print(json.dumps(probe(sys.argv[1])))
