"""Render a 360-degree image dataset of any object in the scene, lit by an HDRI.

  hython gs_render_360.py --hip scene.hip --object /obj/character --name hero
                          --hdri studio.exr --views 160 --samples 64

Writes data/<name>/ as a Nerfstudio dataset (images, transforms.json, init.ply).
Turn it into the COLMAP layout the trainer reads with a separate step, because
that needs pycolmap, which lives in tools/venv rather than in Houdini's Python:

  tools/venv/Scripts/python scripts/nerfstudio_to_colmap.py data/<name> data/<name>_colmap

Progress goes to stdout as "GSS_PROGRESS <percent> <message>" lines, which the
GSplat Studio panel turns into its progress bar.
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
from gs_render_dataset import fibonacci_dirs, look_at, write_init_ply  # noqa: E402

APERTURE_MM = 36.0
FOCAL_MM = 35.0
START_FRAME = 1001
INIT_POINTS = 200_000


def progress(percent: float, message: str = "") -> None:
    print(f"GSS_PROGRESS {percent:.1f} {message}", flush=True)


def resolve_geo(path: str) -> tuple[hou.Node, hou.Node, np.ndarray]:
    """The object to capture, its display SOP, and its points in world space."""
    node = hou.node(path)
    if node is None:
        sys.exit(f"object not found: {path}")
    obj = node if node.type().category() == hou.objNodeTypeCategory() else node.parent()
    sop = node if node.type().category() == hou.sopNodeTypeCategory() else obj.displayNode()
    if sop is None:
        sys.exit(f"{path} has no geometry to render")
    pts = np.array(sop.geometry().pointFloatAttribValues("P"), dtype=np.float64).reshape(-1, 3)
    if not len(pts):
        sys.exit(f"{path} is empty")
    xform = np.array(obj.worldTransform().asTuple(), dtype=np.float64).reshape(4, 4)
    world = np.c_[pts, np.ones(len(pts))] @ xform
    return obj, sop, world[:, :3]


def build_cameras(points: np.ndarray, views: int, close_fraction: float, seed: int = 7) -> list[np.ndarray]:
    """A sphere of cameras framing the whole object, plus closer ones for detail."""
    lo, hi = np.percentile(points, 0.5, axis=0), np.percentile(points, 99.5, axis=0)
    center = (lo + hi) * 0.5
    radius = float(np.linalg.norm(hi - lo)) * 0.5
    half_fov = math.atan(APERTURE_MM * 0.5 / FOCAL_MM)
    dist = radius / math.sin(half_fov) * 1.05

    cams = [look_at(center + d * dist, center) for d in fibonacci_dirs(views, 0.5)]
    close = int(views * close_fraction)
    if close:
        rng = np.random.default_rng(seed)
        extent = (hi - lo) * 0.35
        for d in fibonacci_dirs(close, 0.25):
            target = center + rng.uniform(-1.0, 1.0, 3) * extent
            cams.append(look_at(target + d * dist * 0.5, target))
    print(f"object radius {radius:.3f}, camera distance {dist:.3f}, {len(cams)} views", flush=True)
    return cams


def write_transforms(folder: str, mats: list[np.ndarray], res: tuple[int, int]) -> None:
    w, h = res
    focal_px = FOCAL_MM / APERTURE_MM * w
    data = {
        "camera_model": "OPENCV",
        "fl_x": focal_px, "fl_y": focal_px,
        "cx": w / 2.0, "cy": h / 2.0,
        "w": w, "h": h,
        "ply_file_path": "init.ply",
        "frames": [{"file_path": f"images/{START_FRAME + i:04d}.png",
                    "transform_matrix": m.T.tolist()} for i, m in enumerate(mats)],
    }
    with open(os.path.join(folder, "transforms.json"), "w") as f:
        json.dump(data, f, indent=1)


def sample_surface(sop: hou.Node, count: int) -> np.ndarray:
    """Uniform surface samples, used as the starting point cloud for training."""
    geo = hou.node("/obj").createNode("geo", "GS_tmp_scatter")
    try:
        merge = geo.createNode("object_merge")
        merge.parm("objpath1").set(sop.path())
        merge.parm("xformtype").set("local")  # "Into This Object" -> world space
        scatter = geo.createNode("scatter")
        scatter.setInput(0, merge)
        scatter.parm("npts").set(count)
        return np.array(scatter.geometry().pointFloatAttribValues("P"), dtype=np.float32).reshape(-1, 3)
    finally:
        geo.destroy()


def build_stage(obj: hou.Node, hdri: str, picture: str, res: tuple[int, int],
                samples: int) -> tuple[hou.Node, hou.Node]:
    """Import the object, light it with the HDRI, and set up Karma XPU."""
    stage = hou.node("/stage")
    imp = stage.createNode("sceneimport", "GS360_import")
    imp.parm("objects").set(obj.path())
    imp.parm("importmats").set(True)

    last = imp
    if hdri:
        dome = stage.createNode("domelight", "GS360_hdri")
        dome.setInput(0, imp)
        dome.parm("xn__texturefile_0ta").set(hdri.replace("\\", "/"))
        last = dome

    cam = stage.createNode("camera", "GS360_cam")
    cam.setInput(0, last)
    cam.parm("primpath").set("/cameras/GS360_cam")
    cam.parm("focalLength").set(FOCAL_MM)
    cam.parm("horizontalAperture").set(APERTURE_MM)

    karma = stage.createNode("karmarendersettings", "GS360_karma")
    karma.setInput(0, cam)
    karma.parm("camera").set("/cameras/GS360_cam")
    karma.parm("engine").set("xpu")
    karma.parm("res_mode").set("manual")
    for parm, value in zip(("resolutionx", "resolutiony"), res):
        karma.parm(parm).lock(False)
        karma.parm(parm).set(value)
    karma.parm("pathtracedsamples").set(samples)
    karma.parm("denoiser").set("off")
    karma.parm("picture").set(picture.replace("\\", "/"))

    rop = stage.createNode("usdrender_rop", "GS360_rop")
    rop.setInput(0, karma)
    rop.parm("renderer").set("BRAY_HdKarmaXPU")
    rop.parm("husk_mplay").set(False)
    karma.setDisplayFlag(True)
    stage.layoutChildren()
    return cam, rop


def key_camera(cam: hou.Node, mats: list[np.ndarray]) -> None:
    """One constant keyframe per view, so frame N renders camera N."""
    names = ("tx", "ty", "tz", "rx", "ry", "rz")
    for name in names:
        cam.parm(name).deleteAllKeyframes()
    for i, m in enumerate(mats):
        hm = hou.Matrix4([list(r) for r in m])
        values = (*hm.extractTranslates("srt"), *hm.extractRotates("srt", "xyz"))
        for name, value in zip(names, values):
            key = hou.Keyframe()
            key.setFrame(START_FRAME + i)
            key.setValue(value)
            key.setExpression("constant()")
            cam.parm(name).setKeyframe(key)
    cam.parm("xOrd").set("srt")
    cam.parm("rOrd").set("xyz")


def blank_watermark(folder: str, box: tuple[int, int, int, int]) -> None:
    """Clear the Apprentice watermark alpha so training reads that patch as background."""
    from PIL import Image

    count = 0
    for name in sorted(os.listdir(folder)):
        if not name.endswith(".png"):
            continue
        path = os.path.join(folder, name)
        with Image.open(path) as img:
            rgba = img.convert("RGBA")
        rgba.paste((0, 0, 0, 0), box)
        rgba.save(path)
        count += 1
    print(f"watermark area cleared on {count} images", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hip", required=True)
    ap.add_argument("--object", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--hdri", default="")
    ap.add_argument("--views", type=int, default=160)
    ap.add_argument("--close-fraction", type=float, default=0.5)
    ap.add_argument("--samples", type=int, default=64)
    ap.add_argument("--res", default="1280x720")
    ap.add_argument("--mask-watermark", default="", help="x0,y0,x1,y1 cleared in every image")
    ap.add_argument("--data-dir", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"))
    args = ap.parse_args()

    res = tuple(int(v) for v in args.res.lower().split("x"))
    progress(1, "Mo canh")
    hou.hipFile.load(args.hip, suppress_save_prompt=True, ignore_load_warnings=True)

    obj, sop, world_points = resolve_geo(args.object)
    mats = build_cameras(world_points, args.views, args.close_fraction)

    folder = os.path.join(args.data_dir, args.name)
    images = os.path.join(folder, "images")
    os.makedirs(images, exist_ok=True)
    write_transforms(folder, mats, res)
    write_init_ply(os.path.join(folder, "init.ply"), sample_surface(sop, INIT_POINTS))

    cam, rop = build_stage(obj, args.hdri, os.path.join(images, "$F4.png"), res, args.samples)
    key_camera(cam, mats)
    progress(3, f"Render {len(mats)} goc nhin")

    t0 = time.time()
    for i in range(len(mats)):
        frame = START_FRAME + i
        rop.render(frame_range=(frame, frame), verbose=False)
        done = (i + 1) / len(mats)
        eta = (time.time() - t0) / (i + 1) * (len(mats) - i - 1)
        progress(3 + 82 * done, f"Render {i + 1}/{len(mats)} - con {eta / 60:.0f} phut")

    if args.mask_watermark:
        progress(90, "Xoa watermark")
        blank_watermark(images, tuple(int(v) for v in args.mask_watermark.split(",")))

    progress(100, "Render xong")
    print("dataset:", folder, flush=True)


if __name__ == "__main__":
    main()
