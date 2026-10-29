"""Step 5: render animation frames with splats and with the original mesh, side by side.

  hython gs_render_compare.py 1 200 400
  hython gs_render_compare.py 1 200 400 --hip houdini/Wukong_gsplat_h22.hip --res 1280x720

Splat frames go where GS_SPLAT_KARMA points; mesh frames are written next to them
as wukong_mesh.####.png. Both end up at --res (default: the scene's render size).
"""
import argparse
import os
import subprocess
import sys
import time

import hou

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gs_config as cfg  # noqa: E402


def set_resolution(karma: hou.Node, res: tuple[int, int]) -> None:
    karma.parm("res_mode").set("manual")
    for axis, value in zip(("resolutionx", "resolutiony"), res):
        karma.parm(axis).lock(False)
        karma.parm(axis).set(value)


def mesh_rop(stage: hou.Node, out_dir: str, res: tuple[int, int]) -> hou.Node:
    """Original look-dev render, redirected so the source render folder is untouched."""
    karma = hou.copyNodesTo([stage.node("karmarendersettings")], stage)[0]
    karma.setName("GS_MESH_KARMA", unique_name=True)
    karma.setInput(0, stage.node("prune1"))
    karma.parm("primpath").set("/Render/GS_mesh_settings")
    karma.parm("renderproductsparentprimpath").set("/Render/GS_mesh_Products")
    karma.parm("rendervarsparentprimpath").set("/Render/GS_mesh_Products/Vars")
    karma.parm("setlayerrendersettings").set(False)
    karma.parm("picture").set(f"{out_dir}/wukong_mesh.$F4.png")
    set_resolution(karma, res)
    rop = stage.createNode("usdrender_rop", "GS_MESH_ROP")
    rop.setInput(0, karma)
    rop.parm("rendersettings").set("/Render/GS_mesh_settings")
    rop.parm("husk_mplay").set(False)
    return rop


def render(rop: hou.Node, frame: int) -> float:
    t0 = time.time()
    rop.render(frame_range=(frame, frame))
    return time.time() - t0


def downscale(path: str, width: int, height: int) -> None:
    """A supersampled splat pass is brought back to delivery size (no-op if already there)."""
    tool = os.path.join(os.environ["HFS"], "bin", "hoiiotool.exe")
    subprocess.check_call([tool, path, "--resize", f"{width}x{height}", "-o", path])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("frames", nargs="*", type=int, default=[1, 200, 400])
    ap.add_argument("--hip", default=cfg.GS_HIP)
    ap.add_argument("--res", default="", help="WxH; default = scene render size")
    args = ap.parse_args()

    hou.hipFile.load(args.hip, suppress_save_prompt=True, ignore_load_warnings=True)
    stage = hou.node("/stage")
    splat_karma = stage.node("GS_SPLAT_KARMA")
    out_dir = os.path.dirname(splat_karma.parm("picture").eval()).replace("\\", "/")
    os.makedirs(out_dir, exist_ok=True)
    if args.res:
        res = tuple(int(v) for v in args.res.lower().split("x"))
    else:
        karma = stage.node("karmarendersettings")
        res = (karma.parm("resolutionx").eval(), karma.parm("resolutiony").eval())

    splat = stage.node("GS_SPLAT_ROP")
    mesh = stage.node("GS_MESH_ROP") or mesh_rop(stage, out_dir, res)
    for f in args.frames:
        t_splat = render(splat, f)
        downscale(f"{out_dir}/wukong_gsplat.{f:04d}.png", *res)
        print(f"frame {f}: splats {t_splat:.0f}s, mesh {render(mesh, f):.0f}s", flush=True)


if __name__ == "__main__":
    main()
