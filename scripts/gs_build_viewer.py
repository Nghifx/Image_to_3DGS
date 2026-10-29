"""Make a small Houdini 21 scene that shows one trained splat .ply in Karma XPU.

  hython gs_build_viewer.py <splat.ply> <out.hip> [--render frame.png]

Scene: /obj/SPLAT (File -> Bake GSplat -> OUT), /stage with SOP import, a front
camera framing the splat, Karma XPU render settings (2x supersampled, no denoiser).
"""
import argparse
import os

import hou
import numpy as np

SUPERSAMPLE = 2
RES = (1920, 1080)


def build(ply: str, rotate_x: float = 0.0) -> hou.Node:
    geo = hou.node("/obj").createNode("geo", "SPLAT")
    src = geo.createNode("file", "ply")
    src.parm("file").set(ply.replace("\\", "/"))
    # Some trainers (LichtFeld) export nx/ny/nz; the File SOP turns them into N,
    # which Karma then uses on the splats. Splats carry no normal, so drop it.
    drop_n = geo.createNode("attribdelete", "drop_normals")
    drop_n.setInput(0, src)
    drop_n.parm("ptdel").set("N nx ny nz")
    bake = geo.createNode("bakegsplat")
    bake.setInput(0, drop_n)
    bake.parm("sphcoeff").set(True)
    last = bake
    if rotate_x:
        # Houdini 22's COLMAP loader rotates the world -90 deg about X. Undo it on
        # positions and splat orientation; restorient stays as trained so Karma
        # still evaluates the view-dependent color in the original frame.
        fix = geo.createNode("attribwrangle", "undo_colmap_rotation")
        fix.setInput(0, bake)
        fix.parm("snippet").set(
            f"matrix3 r = ident(); rotate(r, radians({rotate_x}), {{1, 0, 0}});\n"
            "@P *= r;\n"
            "p@orient = qmultiply(quaternion(r), p@orient);\n")
        last = fix
    out = geo.createNode("null", "OUT")
    out.setInput(0, last)
    out.setDisplayFlag(True)
    out.setRenderFlag(True)
    geo.layoutChildren()
    return out


def frame_camera(cam: hou.Node, out: hou.Node) -> None:
    # robust extent: stray floater splats must not drive the framing
    pts = np.array(out.geometry().pointFloatAttribValues("P")).reshape(-1, 3)
    lo, hi = np.percentile(pts, 0.5, axis=0), np.percentile(pts, 99.5, axis=0)
    center, size = (lo + hi) * 0.5, hi - lo
    focal, aperture = 50.0, 36.0
    # fit the larger of width/height with 15% margin
    span = max(size[0], size[1] * RES[0] / RES[1]) * 1.15
    dist = span * 0.5 * focal / (aperture * 0.5) + size[2] * 0.5
    cam.parm("tx").set(center[0])
    cam.parm("ty").set(center[1])
    cam.parm("tz").set(center[2] + dist)
    cam.parm("focalLength").set(focal)
    cam.parm("horizontalAperture").set(aperture)


def build_stage(out: hou.Node, picture: str, res: tuple[int, int], supersample: int) -> hou.Node:
    stage = hou.node("/stage")
    imp = stage.createNode("sopimport", "splat")
    imp.parm("soppath").set(out.path())
    imp.parm("primpath").set("/splat")
    imp.parm("pathprefix").set("/splat")
    cam = stage.createNode("camera", "front_cam")
    cam.setInput(0, imp)
    cam.parm("primpath").set("/cameras/front_cam")
    frame_camera(cam, out)
    karma = stage.createNode("karmarendersettings", "karma_xpu")
    karma.setInput(0, cam)
    karma.parm("camera").set("/cameras/front_cam")
    karma.parm("engine").set("xpu")
    karma.parm("res_mode").set("manual")
    for parm, value in zip(("resolutionx", "resolutiony"), res):
        karma.parm(parm).lock(False)
        karma.parm(parm).set(value * supersample)
    karma.parm("denoiser").set("off")
    karma.parm("pathtracedsamples").set(64)
    karma.parm("picture").set(picture.replace("\\", "/"))
    rop = stage.createNode("usdrender_rop", "render")
    rop.setInput(0, karma)
    rop.parm("renderer").set("BRAY_HdKarmaXPU")
    rop.parm("husk_mplay").set(False)
    karma.setDisplayFlag(True)
    stage.layoutChildren()
    return rop


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ply")
    ap.add_argument("hip")
    ap.add_argument("--render", default="")
    ap.add_argument("--res", default=f"{RES[0]}x{RES[1]}", help="WxH (Apprentice caps at 1280x720)")
    ap.add_argument("--supersample", type=int, default=SUPERSAMPLE)
    ap.add_argument("--rotate-x", type=float, default=0.0, help="degrees; 90 for Houdini 22 COLMAP training")
    args = ap.parse_args()

    res = tuple(int(v) for v in args.res.lower().split("x"))
    out = build(args.ply, args.rotate_x)
    picture = args.render or os.path.splitext(args.hip)[0] + ".$F4.png"
    rop = build_stage(out, picture, res, args.supersample)
    os.makedirs(os.path.dirname(args.hip), exist_ok=True)
    hou.hipFile.save(args.hip)
    print("saved", args.hip, "splats:", len(out.geometry().points()))
    if args.render:
        rop.render(frame_range=(1, 1))
        print("rendered", args.render)


if __name__ == "__main__":
    main()
