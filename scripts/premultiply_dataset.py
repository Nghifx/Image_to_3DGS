"""Copy a COLMAP dataset with its RGBA images premultiplied (background -> black).

LichtFeld's alpha_consistent mode compares color over the whole frame, so the RGB
hidden under transparent pixels must already be black (what a splat over a black
background renders), or the background color leaks into the silhouette.

  python premultiply_dataset.py <src_dataset> <dst_dataset>
"""
import os
import shutil
import sys

import cv2
import numpy as np


def main(src: str, dst: str) -> None:
    os.makedirs(os.path.join(dst, "images"), exist_ok=True)
    shutil.copytree(os.path.join(src, "sparse"), os.path.join(dst, "sparse"), dirs_exist_ok=True)
    names = sorted(os.listdir(os.path.join(src, "images")))
    for name in names:
        img = cv2.imread(os.path.join(src, "images", name), cv2.IMREAD_UNCHANGED).astype(np.float32)
        if img.ndim != 3 or img.shape[2] != 4:
            sys.exit(f"{name} has no alpha channel")
        img[..., :3] *= img[..., 3:4] / 255.0
        cv2.imwrite(os.path.join(dst, "images", name), np.clip(img + 0.5, 0, 255).astype(np.uint8))
    print(f"premultiplied {len(names)} images -> {dst}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
