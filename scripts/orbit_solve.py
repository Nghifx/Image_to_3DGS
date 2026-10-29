"""Solve cameras for a 360-degree orbit render (AI video) with the orbit as the backbone.

  tools/venv/Scripts/python scripts/orbit_solve.py <name> --stage angles|solve|review

Input is step 2's output (gs_frames.py): data/<name>/frames, masks, images, meta.json.
gs_colmap_ai.py runs these stages with a Claude check in between.

AI orbit videos keep the camera on one level circle but repaint surface detail a
little every frame. A free COLMAP solve (video_to_dataset.py) then drifts, or
trades the drift for nonsense lens values. Here the circle is fixed and only the
position along it is measured, so a person (or Claude) can check and correct it:

  angles  free COLMAP solve with a simple locked-shape lens, read off where each
          frame sits on the circle -> _work/<name>/orbit_keys.json (frame -> degrees).
          Edit that table by hand to correct any frame before solving.
  solve   one camera per frame on a perfect circle at those angles, all looking at
          the centre; points are triangulated with the poses held fixed
  review  per-frame report: reprojection error, and how far each frame's own
          features would pull it along the circle; overlays in colmap/_work/<name>/review

Output: colmap/<name>/images (hard links to data/<name>/images) + colmap/<name>/sparse/0.
Scratch lives in colmap/_work/<name>: trainers pick up images and cameras.bin by
file name anywhere under a dataset folder, so nothing extra may sit inside it.
"""
import argparse
import json
import os
import shutil
import sys

import cv2
import numpy as np
import pycolmap

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from video_to_dataset import MIN_SOLVED, align_upright, feature_images, progress  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RADIUS = 4.0          # orbit radius in solve units; only its ratio to the focal length matters
SENSOR_MM = 36.0      # 35 mm equivalent sensor width for the focal length guess
OVERLAY_EVERY = 30    # review overlay for every Nth frame
REPORT_EVERY = 10     # review table row for every Nth frame


def paths(name: str) -> dict:
    data = os.path.join(ROOT, "data", name)
    out = os.path.join(ROOT, "colmap", name)
    work = os.path.join(ROOT, "colmap", "_work", name)
    return {
        "data": data, "rgb": os.path.join(data, "frames"), "masks": os.path.join(data, "masks"),
        "images": os.path.join(data, "images"), "meta": os.path.join(data, "meta.json"),
        "out": out, "sparse": os.path.join(out, "sparse", "0"),
        "work": work, "db": os.path.join(work, "database.db"), "free": os.path.join(work, "free"),
        "keys": os.path.join(work, "orbit_keys.json"), "review": os.path.join(work, "review"),
    }


def save_json(path: str, data: dict) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=1)


def load_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def lens_guess(p: dict, focal_mm: float) -> list[float]:
    """SIMPLE_PINHOLE [f, cx, cy] for the cropped frames: optical centre = centre of the
    original frame (AI video has no lens shift), focal length from a 35 mm guess."""
    meta = load_json(p["meta"])
    (w, h), (x0, y0) = meta["full_size"], meta["crop_origin"]
    return [w * focal_mm / SENSOR_MM, w / 2 - x0, h / 2 - y0]


# ---------------------------------------------------------------- angles

def orbit_frame(centers: np.ndarray, cam_up: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Centre, up axis and the zero direction of the circle the cameras lie on."""
    mid = centers.mean(0)
    up = np.linalg.svd(centers - mid)[2][2]
    up = up if np.dot(up, cam_up) > 0 else -up
    return mid, up, centers[0] - mid


def azimuths(centers: np.ndarray, mid: np.ndarray, up: np.ndarray, zero: np.ndarray) -> np.ndarray:
    """Degrees around `up` from `zero`, same handedness as place_cameras (+Z towards +X, Y up)."""
    e1 = zero - up * np.dot(zero, up)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(up, e1)
    rel = centers - mid
    return np.degrees(np.arctan2(rel @ e2, rel @ e1))


def solved_angles(rec: pycolmap.Reconstruction) -> tuple[list[str], np.ndarray]:
    images = sorted(rec.images.values(), key=lambda im: im.name)
    centers = np.array([im.projection_center() for im in images])
    cam_up = -np.mean([im.cam_from_world().rotation.matrix()[1] for im in images], axis=0)
    mid, up, zero = orbit_frame(centers, cam_up)
    return [im.name for im in images], np.unwrap(azimuths(centers, mid, up, zero), period=360.0)


def measure_angles(p: dict, focal_mm: float, sweep: float, use_masks: bool = True) -> None:
    """Free solve with the lens kept to one focal length and a fixed optical centre.

    The earlier free solve of this kind of video was allowed separate x/y focal
    lengths plus distortion and settled on fx 3420 / fy 159 - the camera path was
    bent to match. One focal length, centred, removes that escape.
    """
    os.makedirs(p["work"], exist_ok=True)
    if os.path.exists(p["db"]):
        os.remove(p["db"])
    progress(2, "Chuan bi anh cho COLMAP")
    feat_dir, feat_masks = feature_images(p["rgb"], p["masks"], p["work"])
    if not use_masks:
        # Features on the set too: a checker floor/wall is rigid and gives true
        # perspective, where an AI-repainted body alone lets the solve drift.
        feat_masks = ""
    extract = pycolmap.FeatureExtractionOptions()
    extract.sift.max_num_features = 16384
    extract.sift.peak_threshold = 0.004
    lens = lens_guess(p, focal_mm)
    progress(5, "Tim diem dac trung")
    pycolmap.extract_features(
        p["db"], feat_dir, camera_mode=pycolmap.CameraMode.SINGLE,
        reader_options=pycolmap.ImageReaderOptions(
            camera_model="SIMPLE_PINHOLE", mask_path=feat_masks, camera_params=",".join(map(str, lens))),
        extraction_options=extract,
    )
    progress(12, "Ghep cap anh")
    pycolmap.match_exhaustive(p["db"])
    progress(25, "Do goc tung anh (giai tu do)")
    opts = pycolmap.IncrementalPipelineOptions()
    opts.ba_refine_principal_point = False
    opts.ba_refine_extra_params = False
    shutil.rmtree(p["free"], ignore_errors=True)
    os.makedirs(p["free"])
    maps = pycolmap.incremental_mapping(p["db"], p["rgb"], p["free"], options=opts)
    total = len(os.listdir(p["rgb"]))
    if not maps:
        # nothing to measure: start from an even turn and let the review fix it
        print("free solve found nothing - starting from an even turn")
        save_json(p["keys"], {"sweep_deg": sweep, "focal_px": round(lens[0], 1), "keys": [[1, 0.0]],
                              "note": "frame -> degrees on the circle; edit, then --stage solve"})
        return
    rec = max(maps.values(), key=lambda r: r.num_reg_images())
    rec.write(p["free"])
    names, ang = solved_angles(rec)
    if ang[-1] < 0:  # report the turn as positive degrees; the sign is kept in the keys
        print("orbit runs clockwise seen from above")
    focal = float(next(iter(rec.cameras.values())).params[0])
    keys = [[int(n[:4]), round(float(a), 2)] for n, a in zip(names, ang)]
    save_json(p["keys"], {"sweep_deg": float(np.sign(ang[-1]) * sweep), "focal_px": round(focal, 1),
                          "keys": keys, "note": "frame -> degrees on the circle; edit, then --stage solve"})
    print(f"free solve: {len(names)}/{total} frames, focal {focal:.0f}px "
          f"({focal * SENSOR_MM / load_json(p['meta'])['full_size'][0]:.0f} mm), "
          f"reprojection {rec.compute_mean_reprojection_error():.2f}px")
    s = np.sign(ang[-1])
    print("frame  measured  uniform  diff")
    for (f, a) in keys[::REPORT_EVERY]:
        u = (f - 1) * sweep / total
        print(f"{f:5d}  {s * a:8.1f}  {u:7.1f}  {s * a - u:+5.1f}")
    print("keys written:", p["keys"])


# ---------------------------------------------------------------- solve

def frame_angles(keys: dict, n: int) -> np.ndarray:
    """Angle for every frame 1..n, linear between keys, closing the loop at frame n+1."""
    k = sorted(keys["keys"])
    f = [a for a, _ in k]
    a = [b for _, b in k]
    first, last = f[0], f[-1]
    # wrap so frames before the first / after the last key interpolate across the seam
    f = [last - n] + f + [first + n]
    a = [a[-1] - keys["sweep_deg"]] + a + [a[0] + keys["sweep_deg"]]
    return np.interp(np.arange(1, n + 1), f, a)


def look_at_pose(angle_deg: float) -> pycolmap.Rigid3d:
    t = np.radians(angle_deg)
    center = RADIUS * np.array([np.sin(t), 0.0, np.cos(t)])
    fwd = -center / RADIUS
    right = np.cross(fwd, [0.0, 1.0, 0.0])
    right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    rot = np.array([right, down, fwd])
    return pycolmap.Rigid3d(pycolmap.Rotation3d(rot), -rot @ center)


def write_circle_model(p: dict, out: str, keys: dict) -> None:
    """COLMAP text model: one camera, one image per frame on the circle, no points yet."""
    db = pycolmap.Database.open(p["db"])
    images = sorted(db.read_all_images(), key=lambda im: im.name)
    cam = next(iter(db.read_all_cameras()))
    db.close()
    lens = [keys["focal_px"], *cam.params[1:3]]
    ang = frame_angles(keys, len(images))
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "cameras.txt"), "w") as f:
        f.write(f"{cam.camera_id} SIMPLE_PINHOLE {cam.width} {cam.height} {' '.join(map(str, lens))}\n")
    with open(os.path.join(out, "images.txt"), "w") as f:
        for im, a in zip(images, ang):
            pose = look_at_pose(a)
            qx, qy, qz, qw = pose.rotation.quat
            tx, ty, tz = pose.translation
            f.write(f"{im.image_id} {qw} {qx} {qy} {qz} {tx} {ty} {tz} {cam.camera_id} {im.name}\n\n")
    open(os.path.join(out, "points3D.txt"), "w").close()


def solve(p: dict, focal_mm: float, height: float) -> None:
    keys = load_json(p["keys"])
    keys.setdefault("focal_px", lens_guess(p, focal_mm)[0])
    circle = os.path.join(p["work"], "circle")
    shutil.rmtree(circle, ignore_errors=True)
    write_circle_model(p, circle, keys)
    rec = pycolmap.Reconstruction(circle)
    tri = os.path.join(p["work"], "tri")
    os.makedirs(tri, exist_ok=True)
    opts = pycolmap.IncrementalPipelineOptions()
    opts.ba_refine_principal_point = False
    opts.ba_refine_extra_params = False
    # poses stay on the circle; only the focal length and the points are adjusted
    rec = pycolmap.triangulate_points(rec, p["db"], p["rgb"], tri, options=opts, refine_intrinsics=True)
    focal = float(next(iter(rec.cameras.values())).params[0])
    print(f"triangulated {rec.num_points3D()} points on {rec.num_reg_images()} cameras, focal {focal:.0f}px, "
          f"reprojection {rec.compute_mean_reprojection_error():.2f}px")
    shutil.rmtree(p["out"], ignore_errors=True)
    os.makedirs(p["sparse"])
    rec.write(p["sparse"])
    align_upright(p["sparse"], height)
    link_images(p)
    export_preview(p)
    export_cameras(p)


def use_free(p: dict, height: float) -> bool:
    """Ship the free solve from the angles stage as is, if it caught enough frames.

    Side by side on khmer_01 (2026-10-23, same frames and trainer) the free cameras
    trained a splat that matched the photos at 28.1 dB against 21.7 dB for the
    circle, which smeared arms and cloth: the AI orbit is not a perfect circle.
    """
    if not os.path.exists(os.path.join(p["free"], "images.bin")):
        return False
    rec = pycolmap.Reconstruction(p["free"])
    total = len(os.listdir(p["rgb"]))
    if rec.num_reg_images() < MIN_SOLVED * total:
        print(f"free solve caught only {rec.num_reg_images()} / {total} frames - using the circle")
        return False
    shutil.rmtree(p["out"], ignore_errors=True)
    os.makedirs(p["sparse"])
    rec.write(p["sparse"])
    align_upright(p["sparse"], height)
    link_images(p)
    export_preview(p)
    export_cameras(p)
    print(f"free solve kept: {rec.num_reg_images()} / {total} frames, {rec.num_points3D()} points")
    return True


def export_cameras(p: dict) -> None:
    """cameras.json for gs_render_check.py, which runs in the trainer's Python (no pycolmap)."""
    rec = pycolmap.Reconstruction(p["sparse"])
    cam = next(iter(rec.cameras.values()))
    f, cx, cy = (float(v) for v in cam.params[:3])
    views = []
    for im in sorted(rec.images.values(), key=lambda im: im.name):
        pose = im.cam_from_world()
        views.append({"name": im.name, "R": pose.rotation.matrix().tolist(), "t": pose.translation.tolist()})
    save_json(os.path.join(p["work"], "cameras.json"),
              {"width": cam.width, "height": cam.height, "fx": f, "fy": f, "cx": cx, "cy": cy, "views": views})


def link_images(p: dict) -> None:
    """colmap/<name>/images = hard links to data/<name>/images: no second copy on disk."""
    dst = os.path.join(p["out"], "images")
    os.makedirs(dst, exist_ok=True)
    for name in sorted(os.listdir(p["images"])):
        src, target = os.path.join(p["images"], name), os.path.join(dst, name)
        try:
            os.link(src, target)
        except OSError:  # other drive or no hard-link support
            shutil.copy2(src, target)


def export_preview(p: dict) -> None:
    """Points (.ply, with colour) and camera frustums (.obj lines) for the Houdini viewport."""
    rec = pycolmap.Reconstruction(p["sparse"])
    pts = list(rec.points3D.values())
    header = ["ply", "format ascii 1.0", f"element vertex {len(pts)}", "property float x", "property float y",
              "property float z", "property uchar red", "property uchar green", "property uchar blue", "end_header"]
    with open(os.path.join(p["work"], "preview_points.ply"), "w") as f:
        f.write("\n".join(header) + "\n")
        for pt in pts:
            x, y, z = pt.xyz
            r, g, b = pt.color
            f.write(f"{x:.5f} {y:.5f} {z:.5f} {r} {g} {b}\n")
    images = sorted(rec.images.values(), key=lambda im: im.name)
    centers = np.array([im.projection_center() for im in images])
    size = 0.08 * float(np.median(np.linalg.norm(centers - centers.mean(0), axis=1)))
    cam = next(iter(rec.cameras.values()))
    hw, hh = 0.5 * cam.width / cam.params[0], 0.5 * cam.height / cam.params[0]
    corners = np.array([[-hw, -hh, 1], [hw, -hh, 1], [hw, hh, 1], [-hw, hh, 1]]) * size
    with open(os.path.join(p["work"], "preview_cams.obj"), "w") as f:
        for k, im in enumerate(images):
            world_from_cam = im.cam_from_world().inverse()
            f.write("v %f %f %f\n" % tuple(world_from_cam.translation))
            for c in corners:
                f.write("v %f %f %f\n" % tuple(world_from_cam * c))
            o = 5 * k + 1
            f.write(f"l {o} {o + 1}\nl {o} {o + 2}\nl {o} {o + 3}\nl {o} {o + 4}\n"
                    f"l {o + 1} {o + 2} {o + 3} {o + 4} {o + 1}\n")


# ---------------------------------------------------------------- review

def review(p: dict) -> None:
    """How well each held camera agrees with its own features.

    For every frame the pose is re-estimated from that frame's 2D points and the
    shared 3D points alone; the difference in angle along the circle is how far
    the frame's own evidence wants it moved. Positive = later along the turn.
    """
    rec = pycolmap.Reconstruction(p["sparse"])
    images = sorted(rec.images.values(), key=lambda im: im.name)
    cam = next(iter(rec.cameras.values()))
    centers = np.array([im.projection_center() for im in images])
    cam_up = -np.mean([im.cam_from_world().rotation.matrix()[1] for im in images], axis=0)
    mid, up, zero = orbit_frame(centers, cam_up)
    held = np.unwrap(azimuths(centers, mid, up, zero), period=360.0)
    sign = np.sign(held[-1])
    os.makedirs(p["review"], exist_ok=True)
    rows = []
    for k, im in enumerate(images):
        obs = [(pt.xy, rec.points3D[pt.point3D_id].xyz) for pt in im.points2D if pt.has_point3D()]
        err = float(np.mean([np.linalg.norm(cam.img_from_cam(im.cam_from_world() * xyz) - xy) for xy, xyz in obs])) \
            if obs else float("nan")
        shift = float("nan")
        if len(obs) >= 12:
            est = pycolmap.estimate_and_refine_absolute_pose(
                np.array([o[0] for o in obs]), np.array([o[1] for o in obs]), cam)
            if est:
                c = est["cam_from_world"].inverse().translation
                d = azimuths(c[None], mid, up, zero)[0] - azimuths(centers[k][None], mid, up, zero)[0]
                shift = sign * ((d + 180) % 360 - 180)
        rows.append((im.name, held[k] * sign, len(obs), err, shift))
        if k % OVERLAY_EVERY == 0:
            draw_overlay(p, im, cam, obs)
    print("frame  angle  points  error_px  wants_move_deg")
    for name, a, n, e, s in rows[::REPORT_EVERY]:
        print(f"{name[:4]}  {a:6.1f}  {n:6d}  {e:7.2f}  {s:+7.2f}")
    shifts = np.array([r[4] for r in rows])
    ok = ~np.isnan(shifts)
    print(f"median |move| {np.median(np.abs(shifts[ok])):.2f} deg, worst:",
          ", ".join(f"{rows[i][0][:4]} {shifts[i]:+.1f}" for i in np.argsort(-np.abs(np.nan_to_num(shifts)))[:8]))
    frames_sheet(p, rows)
    overlay_sheet(p)
    save_json(os.path.join(p["review"], "review.json"),
              {"rows": [{"frame": int(r[0][:4]), "angle": round(r[1], 2), "points": r[2],
                         "error_px": round(r[3], 3), "wants_move_deg": round(r[4], 2)} for r in rows]})


def frames_sheet(p: dict, rows: list, count: int = 24, cols: int = 6) -> None:
    """Frames evenly round the turn, each labelled with the angle it is held at."""
    pick = np.linspace(0, len(rows), count, endpoint=False).astype(int)
    tiles = []
    for i in pick:
        name, angle = rows[i][0], rows[i][1]
        img = cv2.imread(os.path.join(p["rgb"], name))
        h, w = img.shape[:2]
        img = cv2.resize(img, (320, int(320 * h / w)))
        cv2.putText(img, f"{name[:4]}  {angle:.1f}", (6, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        tiles.append(img)
    rows_img = [np.hstack(tiles[r:r + cols]) for r in range(0, len(tiles), cols)]
    cv2.imwrite(os.path.join(p["review"], "frames_sheet.jpg"), np.vstack(rows_img))


def overlay_sheet(p: dict, cols: int = 4) -> None:
    names = sorted(f for f in os.listdir(p["review"]) if f.startswith("overlay_0"))
    tiles = []
    for name in names:
        img = cv2.imread(os.path.join(p["review"], name))
        h, w = img.shape[:2]
        img = cv2.resize(img, (480, int(480 * h / w)))
        cv2.putText(img, name[8:12], (6, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
        tiles.append(img)
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    rows_img = [np.hstack(tiles[r:r + cols]) for r in range(0, len(tiles), cols)]
    cv2.imwrite(os.path.join(p["review"], "overlay_sheet.jpg"), np.vstack(rows_img))


def draw_overlay(p: dict, im: pycolmap.Image, cam: pycolmap.Camera, obs: list) -> None:
    img = cv2.imread(os.path.join(p["rgb"], im.name))
    for xy, xyz in obs:
        proj = cam.img_from_cam(im.cam_from_world() * xyz)
        e = np.linalg.norm(proj - xy)
        color = (0, 255, 0) if e < 1.5 else (0, 200, 255) if e < 4 else (0, 0, 255)
        cv2.line(img, tuple(map(int, xy)), tuple(map(int, proj)), color, 1)
        cv2.circle(img, tuple(map(int, proj)), 2, color, -1)
    cv2.imwrite(os.path.join(p["review"], "overlay_" + im.name.replace(".png", ".jpg")), img)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--stage", choices=["angles", "solve", "review"], required=True)
    ap.add_argument("--sweep", type=float, default=360.0, help="degrees covered by all frames")
    ap.add_argument("--focal-mm", type=float, default=50.0, help="first guess, 35 mm equivalent")
    ap.add_argument("--height", type=float, default=1.65, help="character height in meters")
    args = ap.parse_args()
    p = paths(args.name)
    if args.stage == "angles":
        measure_angles(p, args.focal_mm, args.sweep)
    elif args.stage == "solve":
        solve(p, args.focal_mm, args.height)
    else:
        review(p)


if __name__ == "__main__":
    main()
