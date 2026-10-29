"""Step 2: render a posed image dataset of one character part with Karma.

Writes  data/<part>/images/####.png  (RGBA, sRGB)
        data/<part>/transforms.json  (Nerfstudio format, read by Brush)
        data/<part>/init.ply         (surface samples to start training from)

Run with Houdini 21 hython:
  hython gs_render_dataset.py body            # full render
  hython gs_render_dataset.py body --test 3   # render only the first 3 views
  hython gs_render_dataset.py body --no-render
"""
import argparse
import json
import math
import os
import sys
import time

import hou
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gs_config as cfg  # noqa: E402

MODE = {"body": 1, "cloth": 2}
GOLDEN_ANGLE = math.pi * (3.0 - math.sqrt(5.0))
INIT_POINTS = 200_000


def fibonacci_dirs(n: int, offset: float) -> np.ndarray:
    i = np.arange(n) + offset
    y = 1.0 - 2.0 * i / n
    r = np.sqrt(np.clip(1.0 - y * y, 0.0, 1.0))
    theta = GOLDEN_ANGLE * i
    return np.stack([np.cos(theta) * r, y, np.sin(theta) * r], axis=1)


def look_at(eye: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Camera-to-world 4x4 (row = axis, Houdini/OpenGL convention: looks down -Z, +Y up)."""
    z = eye - target
    z /= np.linalg.norm(z)
    up = np.array([0.0, 1.0, 0.0]) if abs(z[1]) < 0.98 else np.array([0.0, 0.0, 1.0])
    x = np.cross(up, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    m = np.identity(4)
    m[0, :3], m[1, :3], m[2, :3], m[3, :3] = x, y, z, eye
    return m


def build_cameras(points: np.ndarray, seed: int = 7) -> list[np.ndarray]:
    lo, hi = points.min(axis=0), points.max(axis=0)
    center = (lo + hi) * 0.5
    radius = float(np.linalg.norm(hi - lo)) * 0.5
    half_fov = math.atan(cfg.APERTURE_MM * 0.5 / cfg.FOCAL_MM)
    dist = radius / math.sin(half_fov) * 1.05

    cams = [look_at(center + d * dist, center) for d in fibonacci_dirs(cfg.ORBIT_VIEWS, 0.5)]

    rng = np.random.default_rng(seed)
    extent = (hi - lo) * 0.35
    for d in fibonacci_dirs(cfg.CLOSE_VIEWS, 0.25):
        target = center + rng.uniform(-1.0, 1.0, 3) * extent
        cams.append(look_at(target + d * dist * cfg.CLOSE_DISTANCE, target))
    print(f"bbox {lo} .. {hi}  radius {radius:.3f}  orbit distance {dist:.3f}")
    return cams


def key_camera(cam: hou.Node, mats: list[np.ndarray]) -> None:
    names = ("tx", "ty", "tz", "rx", "ry", "rz")
    for n in names:
        cam.parm(n).deleteAllKeyframes()
    for i, m in enumerate(mats):
        frame = cfg.CAM_START_FRAME + i
        hm = hou.Matrix4([list(r) for r in m])
        t = hm.extractTranslates("srt")
        r = hm.extractRotates("srt", "xyz")
        for n, v in zip(names, (*t, *r)):
            k = hou.Keyframe()
            k.setFrame(frame)
            k.setValue(v)
            k.setExpression("constant()")
            cam.parm(n).setKeyframe(k)
    cam.parm("xOrd").set("srt")
    cam.parm("rOrd").set("xyz")


def write_transforms(part_dir: str, mats: list[np.ndarray]) -> None:
    w = h = cfg.RESOLUTION
    focal_px = cfg.FOCAL_MM / cfg.APERTURE_MM * w
    frames = [
        {
            "file_path": f"images/{cfg.CAM_START_FRAME + i:04d}.png",
            # Nerfstudio wants camera-to-world with axes as columns
            "transform_matrix": m.T.tolist(),
        }
        for i, m in enumerate(mats)
    ]
    data = {
        "camera_model": "OPENCV",
        "fl_x": focal_px, "fl_y": focal_px,
        "cx": w / 2.0, "cy": h / 2.0,
        "w": w, "h": h,
        "ply_file_path": "init.ply",
        "frames": frames,
    }
    with open(os.path.join(part_dir, "transforms.json"), "w") as f:
        json.dump(data, f, indent=1)


def sample_surface(out_model: hou.Node) -> np.ndarray:
    """Uniform surface samples of the displayed part, via a temporary Scatter SOP."""
    geo = hou.node("/obj").createNode("geo", "GS_tmp_scatter")
    try:
        merge = geo.createNode("object_merge")
        merge.parm("objpath1").set(out_model.path())
        scatter = geo.createNode("scatter")
        scatter.setInput(0, merge)
        scatter.parm("npts").set(INIT_POINTS)
        pts = scatter.geometry().pointFloatAttribValues("P")
        return np.array(pts, dtype=np.float32).reshape(-1, 3)
    finally:
        geo.destroy()


def write_init_ply(path: str, pts: np.ndarray) -> None:
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {len(pts)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    )
    rec = np.zeros(len(pts), dtype=[("p", "<f4", 3), ("c", "u1", 3)])
    rec["p"] = pts
    rec["c"] = 128
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(rec.tobytes())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("part", choices=sorted(MODE))
    ap.add_argument("--test", type=int, default=0, help="render only the first N views")
    ap.add_argument("--no-render", action="store_true")
    args = ap.parse_args()

    hou.hipFile.load(cfg.GS_HIP, suppress_save_prompt=True, ignore_load_warnings=True)
    create = hou.node(cfg.CREATE)
    create.node("GS_MODE").parm("input").set(MODE[args.part])
    out_model = create.node("OUT_MODEL")
    hou.setFrame(cfg.CAM_START_FRAME)

    pts = np.array(out_model.geometry().pointFloatAttribValues("P"), dtype=np.float32).reshape(-1, 3)
    mats = build_cameras(pts)

    part_dir = os.path.join(cfg.DATA_DIR, args.part)
    os.makedirs(os.path.join(part_dir, "images"), exist_ok=True)
    write_transforms(part_dir, mats)
    write_init_ply(os.path.join(part_dir, "init.ply"), sample_surface(out_model))

    stage = hou.node("/stage")
    key_camera(stage.node("GS_CAM"), mats)
    karma = stage.node("GS_KARMA")
    karma.parm("picture").set(f"{part_dir}/images/$F4.png".replace("\\", "/"))

    n = args.test or len(mats)
    f1, f2 = cfg.CAM_START_FRAME, cfg.CAM_START_FRAME + n - 1
    rop = stage.node("GS_ROP")
    rop.parm("f1").set(f1)
    rop.parm("f2").set(f2)
    hou.hipFile.save(os.path.join(os.path.dirname(cfg.GS_HIP), f"Wukong_gsplat_render_{args.part}.hip"))

    print(f"{len(mats)} cameras, rendering frames {f1}-{f2} -> {part_dir}")
    if args.no_render:
        return
    t0 = time.time()
    rop.render(frame_range=(f1, f2), verbose=True, output_progress=True)
    dt = time.time() - t0
    print(f"render done: {dt:.1f}s total, {dt / n:.1f}s per view")


if __name__ == "__main__":
    main()
