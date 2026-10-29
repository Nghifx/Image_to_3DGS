"""Step 3, classic COLMAP: features -> exhaustive matching -> incremental mapping.

  tools/venv/Scripts/python scripts/gs_colmap_free.py <name>

No masks and no orbit model: every pixel of the frame (the checker set included)
is used, and COLMAP solves each camera freely. Meant for orbit videos shot on a
textured set (gs_frames.py --keep-bg), where the rigid floor and walls give a
true perspective that an orbit model would only get in the way of - a spiral
camera, say.

Input: data/<name>/frames.  Output: colmap/<name>/images + sparse/0.
"""
import argparse
import os
import shutil
import sys

import pycolmap

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import orbit_solve as orbit  # noqa: E402
from video_to_dataset import MIN_SOLVED, progress  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    args = ap.parse_args()
    p = orbit.paths(args.name)
    if not os.path.isdir(p["rgb"]):
        sys.exit(f"run step 2 first: {p['rgb']} is missing")
    os.makedirs(p["work"], exist_ok=True)
    if os.path.exists(p["db"]):
        os.remove(p["db"])

    progress(5, "Tim diem dac trung")
    extract = pycolmap.FeatureExtractionOptions()
    extract.sift.max_num_features = 16384
    pycolmap.extract_features(p["db"], p["rgb"], camera_mode=pycolmap.CameraMode.SINGLE,
                              reader_options=pycolmap.ImageReaderOptions(camera_model="OPENCV"),
                              extraction_options=extract)
    progress(25, "Ghep cap anh")
    pycolmap.match_exhaustive(p["db"])
    progress(50, "Giai vi tri camera")
    shutil.rmtree(p["free"], ignore_errors=True)
    os.makedirs(p["free"])
    maps = pycolmap.incremental_mapping(p["db"], p["rgb"], p["free"])
    if not maps:
        sys.exit("COLMAP could not solve the cameras")
    rec = max(maps.values(), key=lambda r: r.num_reg_images())
    total = len(os.listdir(p["rgb"]))
    cam = next(iter(rec.cameras.values()))
    print(f"cameras solved: {rec.num_reg_images()} / {total} frames, {rec.num_points3D()} points, "
          f"reprojection {rec.compute_mean_reprojection_error():.2f}px, focal {cam.params[0]:.0f}px")
    if rec.num_reg_images() < MIN_SOLVED * total:
        sys.exit(f"only {rec.num_reg_images()} of {total} cameras solved")

    shutil.rmtree(p["out"], ignore_errors=True)
    os.makedirs(p["sparse"])
    rec.write(p["sparse"])
    orbit.link_images(p)
    progress(100, "COLMAP xong")
    print("colmap ready:", p["out"])


if __name__ == "__main__":
    main()
