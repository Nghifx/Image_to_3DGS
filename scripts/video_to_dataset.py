"""Turn an orbit video of a character into a Brush-ready COLMAP dataset.

  tools/venv/Scripts/python scripts/video_to_dataset.py <video.mp4> <name> [--frames 160]

Steps: extract evenly spaced frames, cut the background out (rembg), crop every
frame to one box around the subject, solve camera poses with COLMAP on the
masked frames.
Output: data/<name>/images/*.png (RGBA, background transparent) + data/<name>/sparse/0

Prints "GSS_PROGRESS <percent> <message>" lines for the GSplat Studio progress bar.
"""
import argparse
import os
import shutil
import subprocess
import sys

import cv2
import numpy as np
import pycolmap
from rembg import new_session, remove

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gs_config as cfg  # noqa: E402

from gs_paths import find_ffmpeg  # noqa: E402

FFMPEG = find_ffmpeg()
CROP_PAD = 0.04  # margin around the subject box, as a fraction of the frame size
CLAHE_CLIP = 4.0  # local contrast boost for the copies COLMAP looks at
MASK_GROW = 15    # px the COLMAP mask is grown so thin edges keep their features
MIN_SOLVED = 0.8  # fraction of frames that must get a camera before training is worth it


def progress(percent: float, message: str) -> None:
    print(f"GSS_PROGRESS {percent:.1f} {message}", flush=True)


def frame_count(video: str) -> tuple[int, int, int]:
    cap = cv2.VideoCapture(video)
    try:
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        cap.release()
    if n <= 0 or w <= 0:
        sys.exit(f"cannot read video: {video}")
    return n, w, h


def extract_frames(video: str, out_dir: str, wanted: int) -> None:
    """Pull exactly `wanted` evenly spaced full frames (or every frame if the video is shorter).

    Frame n is kept when n * wanted crosses a multiple of the frame count, which
    spreads the picks evenly instead of rounding the spacing down to an integer.
    """
    n, w, h = frame_count(video)
    wanted = min(wanted, n)
    vf = f"select='lt(mod(n*{wanted}\\,{n})\\,{wanted})'"
    subprocess.check_call([FFMPEG, "-hide_banner", "-loglevel", "error", "-y", "-i", video,
                           "-vf", vf, "-vsync", "vfr", os.path.join(out_dir, "%04d.png")])
    print(f"{n} video frames -> {len(os.listdir(out_dir))} frames ({w}x{h})")


def crop_to_subject(rgb_dir: str, rgba_dir: str, mask_dir: str) -> tuple[int, int]:
    """Crop every frame to one shared box around the subject.

    A centred square cut off outstretched arms in a wide frame; cropping to where
    the subject actually is keeps all of it and drops empty background, so more
    of the training resolution lands on the character. The box is the same for
    every frame, so all frames still share one camera.
    """
    names = sorted(os.listdir(rgb_dir))
    union = None
    for name in names:
        mask = cv2.imread(os.path.join(mask_dir, name + ".png"), cv2.IMREAD_GRAYSCALE)
        union = mask > 0 if union is None else union | (mask > 0)
    ys, xs = np.nonzero(union)
    if xs.size == 0:
        sys.exit("no subject found in any frame")
    h, w = union.shape
    pad = int(CROP_PAD * max(w, h))
    x0, x1 = max(0, xs.min() - pad), min(w, xs.max() + 1 + pad)
    y0, y1 = max(0, ys.min() - pad), min(h, ys.max() + 1 + pad)
    # even sizes keep video-style tools and image pyramids happy
    x1 -= (x1 - x0) % 2
    y1 -= (y1 - y0) % 2
    for name in names:
        for path in (os.path.join(rgb_dir, name), os.path.join(rgba_dir, name),
                     os.path.join(mask_dir, name + ".png")):
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            cv2.imwrite(path, img[y0:y1, x0:x1])
    print(f"cropped to subject: {x1 - x0}x{y1 - y0} at ({x0},{y0}) of {w}x{h}")
    return int(x0), int(y0)


def cut_background(rgb_dir: str, rgba_dir: str, mask_dir: str, span: tuple[float, float] = (10, 45)) -> None:
    session = new_session("isnet-general-use")
    names = sorted(os.listdir(rgb_dir))
    for i, name in enumerate(names, 1):
        progress(span[0] + (span[1] - span[0]) * i / len(names), f"Tach nen {i}/{len(names)}")
        img = cv2.imread(os.path.join(rgb_dir, name))
        alpha = np.asarray(remove(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), session=session, only_mask=True))
        # keep only the largest blob (the character), drop stray specks
        _, binary = cv2.threshold(alpha, 127, 255, cv2.THRESH_BINARY)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
        if count > 1:
            keep = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            alpha = np.where(labels == keep, alpha, 0).astype(np.uint8)
        cv2.imwrite(os.path.join(rgba_dir, name), np.dstack([img, alpha]))
        # COLMAP mask: features only on the character (name.png.png convention)
        cv2.imwrite(os.path.join(mask_dir, name + ".png"), (alpha > 127).astype(np.uint8) * 255)
    print("background removed")


def feature_images(rgb_dir: str, mask_dir: str, work: str) -> tuple[str, str]:
    """Contrast-boosted grey copies of the frames, plus slightly grown masks, for COLMAP only.

    A dark subject on a light backdrop (a black creature, say) gives SIFT almost
    nothing to hold on to: on the plain frames only 17 of 160 cameras solved.
    Local contrast equalisation pulls the surface texture out of the shadows,
    and growing the mask keeps the features on thin edges such as fingers and
    strands. The training images themselves are left untouched.
    """
    feat_dir, feat_masks = os.path.join(work, "feat"), os.path.join(work, "feat_masks")
    for d in (feat_dir, feat_masks):
        shutil.rmtree(d, ignore_errors=True)
        os.makedirs(d)
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=(8, 8))
    grow = np.ones((MASK_GROW, MASK_GROW), np.uint8)
    for name in sorted(os.listdir(rgb_dir)):
        gray = cv2.imread(os.path.join(rgb_dir, name), cv2.IMREAD_GRAYSCALE)
        cv2.imwrite(os.path.join(feat_dir, name), clahe.apply(gray))
        mask = cv2.imread(os.path.join(mask_dir, name + ".png"), cv2.IMREAD_GRAYSCALE)
        cv2.imwrite(os.path.join(feat_masks, name + ".png"), cv2.dilate(mask, grow))
    return feat_dir, feat_masks


def solve_cameras(rgb_dir: str, mask_dir: str, work: str, sparse_out: str) -> None:
    db = os.path.join(work, "database.db")
    if os.path.exists(db):
        os.remove(db)
    feat_dir, feat_masks = feature_images(rgb_dir, mask_dir, work)
    extract = pycolmap.FeatureExtractionOptions()
    extract.sift.max_num_features = 16384
    extract.sift.peak_threshold = 0.004   # default 0.0067 misses faint texture
    progress(50, "Tim diem dac trung")
    pycolmap.extract_features(
        db, feat_dir,
        camera_mode=pycolmap.CameraMode.SINGLE,
        # OPENCV solves for lens distortion as well as focal length. Phone
        # cameras bend straight lines near the frame edge, and a model without
        # distortion pays for it with a blurrier, fatter splat.
        reader_options=pycolmap.ImageReaderOptions(camera_model="OPENCV", mask_path=feat_masks),
        extraction_options=extract,
    )
    progress(60, "Ghep cap anh")
    pycolmap.match_exhaustive(db)
    progress(75, "Giai vi tri camera")
    # Same file names as the feature copies, so points take their colour from the real frames.
    maps = pycolmap.incremental_mapping(db, rgb_dir, work)
    if not maps:
        sys.exit("COLMAP could not solve the cameras")
    best = max(maps.values(), key=lambda r: r.num_reg_images())
    total = len(os.listdir(rgb_dir))
    print(f"cameras solved: {best.num_reg_images()} / {total} frames, {best.num_points3D()} points, "
          f"mean reprojection error {best.compute_mean_reprojection_error():.2f}px")
    if best.num_reg_images() < MIN_SOLVED * total:
        # Training on a handful of views gives a broken splat; better to stop here.
        sys.exit(f"only {best.num_reg_images()} of {total} cameras solved - "
                 "video needs more texture or a slower, steadier orbit")
    os.makedirs(sparse_out, exist_ok=True)
    best.write(sparse_out)


def _rotation_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Rotation matrix taking unit vector a onto unit vector b."""
    v, c = np.cross(a, b), float(np.dot(a, b))
    if np.linalg.norm(v) < 1e-9:
        return np.eye(3) if c > 0 else -np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * (1.0 / (1.0 + c))


def align_upright(sparse: str, height_m: float) -> None:
    """Make the solve Houdini-friendly: Y up, feet on y=0, centered, facing +Z, real size.

    Up = the direction square to every camera's sideways axis (sign from the cameras'
    own up vectors). The orbit videos keep the horizon level, so each camera's x axis
    is horizontal whatever path it flies; the normal of the camera plane only worked
    for a level circle and tilted zigzag / boom paths that climb over the head.
    Front = direction to the first frame's camera. Size = character height.
    """
    rec = pycolmap.Reconstruction(sparse)
    images = sorted(rec.images.values(), key=lambda im: im.name)
    centers = np.array([im.projection_center() for im in images])
    rots = [im.cam_from_world().rotation.matrix() for im in images]
    cam_up = -np.mean([r.T[:, 1] for r in rots], axis=0)

    normal = np.linalg.svd(np.array([r[0] for r in rots]))[2][2]  # camera x axes in world
    up = normal if np.dot(normal, cam_up) > 0 else -normal
    rot = _rotation_between(up / np.linalg.norm(up), np.array([0.0, 1.0, 0.0]))

    pts = np.array([p.xyz for p in rec.points3D.values()]) @ rot.T
    lo, hi = np.percentile(pts, 0.5, axis=0), np.percentile(pts, 99.5, axis=0)
    center = np.median(pts, axis=0)
    front = centers[0] @ rot.T - center
    front[1] = 0.0
    yaw = _rotation_between(front / np.linalg.norm(front), np.array([0.0, 0.0, 1.0]))
    rot = yaw @ rot

    scale = height_m / float(hi[1] - lo[1])
    shift = -scale * (yaw @ np.array([center[0], lo[1], center[2]]))
    sim = pycolmap.Sim3d(scale, pycolmap.Rotation3d(rot), shift)
    rec.transform(sim)
    rec.write(sparse)
    print(f"aligned: Y up, feet at 0, height {height_m} m (scale x{scale:.3f})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("name")
    ap.add_argument("--frames", type=int, default=160)
    ap.add_argument("--cameras-only", action="store_true", help="reuse extracted frames and masks")
    ap.add_argument("--align-only", action="store_true", help="only re-align an existing solve")
    ap.add_argument("--height", type=float, default=1.65, help="character height in meters")
    args = ap.parse_args()

    root = os.path.join(cfg.DATA_DIR, args.name)
    # Scratch files must live OUTSIDE the dataset folder: Brush matches images and
    # cameras.bin by file name anywhere under it and would pick the wrong copies.
    work = os.path.join(cfg.DATA_DIR, "_work", args.name)
    rgb_dir, mask_dir = os.path.join(work, "rgb"), os.path.join(work, "masks")
    rgba_dir = os.path.join(root, "images")
    sparse = os.path.join(root, "sparse", "0")
    if args.align_only:
        align_upright(sparse, args.height)
        return
    if not args.cameras_only:
        for d in (rgb_dir, mask_dir, rgba_dir):
            shutil.rmtree(d, ignore_errors=True)
            os.makedirs(d)
        progress(2, "Tach anh tu video")
        extract_frames(args.video, rgb_dir, args.frames)
        cut_background(rgb_dir, rgba_dir, mask_dir)
        crop_to_subject(rgb_dir, rgba_dir, mask_dir)
    solve_cameras(rgb_dir, mask_dir, work, sparse)
    progress(95, "Dung thang nhan vat")
    align_upright(sparse, args.height)
    progress(100, "Dataset xong")
    print("dataset ready:", root)


if __name__ == "__main__":
    main()
