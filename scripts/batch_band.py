"""venv: a training set with no matte on her edge: the real frame in a band round her, nothing further out.

  tools/venv/Scripts/python scripts/batch_band.py <run> <out_run> [band_px]     e.g. c006b2 c006bn 40

CEO (2026-10-29): the whole room trains ~5 h (c006bf: 11k of 50k steps in 72 min); the keyed matte trains in 16 min
but its soft edge bakes screen blue into her outline. Cropping round her saves nothing on these videos (she fills
the frame top to bottom). So: alpha = the AI guide grown by band_px, hard edged. Inside it the trainer sees the real
pixels, her outline included, and builds the screen just behind her from its own splats; outside it (random
background, alphasupervision) nothing needs a splat, so training stays near the matted cost. carve_person.py with
the keyed masks then drops the screen splats in 3D, which never blends blue into her colour.

In:   colmap/<run>/sparse/0, chars/<id>/frames (unkeyed), chars/<id>/guide (rembg, 255 = her)
Out:  colmap/<out_run>/images/*.png (RGBA) + sparse/0 (copied); colmap/_work/<out_run>/cameras.json (copied)
"""
import os
import shutil
import sys

import cv2
import numpy as np

ROOT = os.environ.get("GSPLAT_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

run, out_run = sys.argv[1], sys.argv[2]
band = int(sys.argv[3]) if len(sys.argv) > 3 else 40   # px at 1080 wide
char = f"{ROOT}/chars/{run[1:]}"
out = f"{ROOT}/colmap/{out_run}"
shutil.rmtree(out, ignore_errors=True)
shutil.copytree(f"{ROOT}/colmap/{run}/sparse", f"{out}/sparse")
os.makedirs(f"{out}/images")
os.makedirs(f"{ROOT}/colmap/_work/{out_run}", exist_ok=True)
shutil.copy2(f"{ROOT}/colmap/_work/{run}/cameras.json", f"{ROOT}/colmap/_work/{out_run}/")
disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * band + 1, 2 * band + 1))
share = []
for n in sorted(os.listdir(f"{ROOT}/colmap/{run}/images")):
    rgb = cv2.imread(f"{char}/frames/{n}")
    a = cv2.dilate((cv2.imread(f"{char}/guide/{n}", cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8) * 255, disk)
    cv2.imwrite(f"{out}/images/{n}", np.dstack([rgb, a]))
    share.append((a > 0).mean())
print(f"{len(share)} frames, band {band} px, trained area {100 * np.mean(share):.0f} % of the frame -> {out}")
