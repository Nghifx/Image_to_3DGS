"""Step 4: bring trained splats into Wukong_gsplat.hip and make them follow the rig.

How it binds: every splat is pinned to the closest spot on its part's rest mesh
(position offset + local surface frame). Each frame the same spot is looked up on
the animated mesh, so splats follow bones (body) and the cloth sim (cloth) alike,
and rotate with the surface so joints stay sharp.

Adds:
  /obj/GS_SPLATS                    SOP network, OUT_SPLATS = animated splats
  /stage GS_SPLAT_* branch          original lights + camera, mesh hidden,
                                    splats rendered with Karma XPU

  hython gs_build_rig.py                 # uses latest export of each part
  hython gs_build_rig.py --iter 30000
"""
import argparse
import glob
import os
import re
import sys

import hou

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gs_config as cfg  # noqa: E402

# rest mesh / animated mesh for each part, inside the character network
MESHES = {
    "body": ("GS_body_color", "color2"),
    "cloth": ("GS_cloth_rest", "cleanup1"),
}
MAX_BIND_DIST = 0.15   # scene units (character ~13 tall); farther splats are floaters
MIN_ALPHA = 0.01
SPLAT_SUPERSAMPLE = 2  # render size multiplier; downscale after render (gs_render_compare)

FRAME_VEX = r"""
// Orthonormal surface frame (rows: tangent, bitangent, normal) at a prim/uv
matrix3 gs_frame(int geo; int prim; vector uv) {
    vector n = normalize(prim_normal(geo, prim, uv.x, uv.y));
    int pts[] = primpoints(geo, prim);
    vector t = point(geo, "P", pts[1]) - point(geo, "P", pts[0]);
    t -= n * dot(t, n);
    if (length2(t) < 1e-20) {
        t = abs(n.x) < 0.9 ? set(1, 0, 0) : set(0, 1, 0);
        t -= n * dot(t, n);
    }
    t = normalize(t);
    return set(t, cross(n, t), n);
}
"""

CLEAN_VEX = f"""
if (f@GS_Alpha < {MIN_ALPHA}) removepoint(0, @ptnum);
"""

CAPTURE_VEX = FRAME_VEX + f"""
int prim; vector uv;
float d = xyzdist(1, @P, prim, uv);
if (prim < 0 || d > {MAX_BIND_DIST}) {{ removepoint(0, @ptnum); return; }}
matrix3 R0 = gs_frame(1, prim, uv);
i@gs_prim = prim;
v@gs_uv = uv;
v@gs_off = (@P - primuv(1, "P", prim, uv)) * transpose(R0);
3@gs_R0 = R0;
p@gs_orient0 = p@orient;
"""

DEFORM_VEX = FRAME_VEX + """
matrix3 R1 = gs_frame(1, i@gs_prim, v@gs_uv);
@P = primuv(1, "P", i@gs_prim, v@gs_uv) + v@gs_off * R1;
p@orient = qmultiply(quaternion(transpose(3@gs_R0) * R1), p@gs_orient0);
"""


def latest_export(part: str, iteration: int | None) -> str:
    files = glob.glob(os.path.join(cfg.OUTPUT_DIR, part, f"{part}_*.ply"))
    if iteration:
        files = [f for f in files if f.endswith(f"_{iteration}.ply")]
    if not files:
        sys.exit(f"no trained ply for '{part}' in {cfg.OUTPUT_DIR}")
    return max(files, key=lambda f: int(re.search(r"_(\d+)\.ply$", f).group(1)))


def _wrangle(parent: hou.Node, name: str, snippet: str, inputs: list[hou.Node]) -> hou.Node:
    w = parent.createNode("attribwrangle", name)
    for i, n in enumerate(inputs):
        w.setInput(i, n)
    w.parm("snippet").set(snippet.strip() + "\n")
    return w


def _merge_from(parent: hou.Node, name: str, sop_path: str) -> hou.Node:
    m = parent.createNode("object_merge", name)
    m.parm("objpath1").set(sop_path)
    return m


def build_part(net: hou.Node, part: str, ply: str) -> hou.Node:
    rest_name, anim_name = MESHES[part]
    rest = _merge_from(net, f"{part}_rest_mesh", f"{cfg.CREATE}/{rest_name}")
    anim = _merge_from(net, f"{part}_anim_mesh", f"{cfg.CREATE}/{anim_name}")

    src = net.createNode("file", f"{part}_ply")
    src.parm("file").set(ply.replace("\\", "/"))
    bake = net.createNode("bakegsplat", f"{part}_bake")
    bake.setInput(0, src)
    bake.parm("sphcoeff").set(True)   # keep view-dependent shine from training
    clean = _wrangle(net, f"{part}_drop_faint", CLEAN_VEX, [bake])
    capture = _wrangle(net, f"{part}_bind", CAPTURE_VEX, [clean, rest])

    # Binding is expensive and static: compute once, read from disk afterwards
    cache_path = os.path.join(cfg.OUTPUT_DIR, part, f"{part}_bound.bgeo.sc")
    hou.setFrame(cfg.REST_FRAME)
    rest.cook(force=True)  # the LOP-side character network must cook before binding
    geo = capture.geometry()
    if geo is None:
        sys.exit(f"{part}: binding failed to cook: {capture.errors()} {rest.errors()} {bake.errors()}")
    print(f"{part}: {len(bake.geometry().points())} splats trained, {len(geo.points())} bound to mesh")
    geo.saveToFile(cache_path)
    bound = net.createNode("file", f"{part}_bound_cache")
    bound.parm("file").set(cache_path.replace("\\", "/"))

    return _wrangle(net, f"{part}_follow_rig", DEFORM_VEX, [bound, anim])


def build_render_branch(stage: hou.Node, splat_sop: str) -> None:
    hide = stage.createNode("prune", "GS_SPLAT_HIDE_MESH")
    hide.setInput(0, stage.node("prune1"))
    hide.parm("primpattern1").set("/Chr_Jiao_model_master_v0001")

    imp = stage.createNode("sopimport", "GS_SPLAT_IMPORT")
    imp.setInput(0, hide)
    imp.parm("soppath").set(splat_sop)
    imp.parm("primpath").set("/Chr_Jiao_gsplat")
    imp.parm("pathprefix").set("/Chr_Jiao_gsplat")

    karma = hou.copyNodesTo([stage.node("karmarendersettings")], stage)[0]
    karma.setName("GS_SPLAT_KARMA", unique_name=True)
    karma.setInput(0, imp)
    karma.parm("primpath").set("/Render/GS_splat_settings")
    karma.parm("renderproductsparentprimpath").set("/Render/GS_splat_Products")
    karma.parm("rendervarsparentprimpath").set("/Render/GS_splat_Products/Vars")
    karma.parm("engine").set("xpu")   # Karma XPU is the delegate that draws splats
    karma.parm("setlayerrendersettings").set(False)
    # H21's splat renderer leaves a fixed per-pixel grain that more samples don't
    # remove; rendering at SPLAT_SUPERSAMPLE x size and downscaling does.
    karma.parm("res_mode").set("manual")
    for axis in ("resolutionx", "resolutiony"):
        karma.parm(axis).lock(False)
        karma.parm(axis).set(karma.parm(axis).eval() * SPLAT_SUPERSAMPLE)
    karma.parm("denoiser").set("off")   # denoiser smears and over-saturates splats
    karma.parm("pathtracedsamples").set(64)
    karma.parm("picture").set(os.path.join(cfg.OUTPUT_DIR, "render", "wukong_gsplat.$F4.png").replace("\\", "/"))

    rop = stage.createNode("usdrender_rop", "GS_SPLAT_ROP")
    rop.setInput(0, karma)
    rop.parm("renderer").set("BRAY_HdKarmaXPU")
    rop.parm("rendersettings").set("/Render/GS_splat_settings")
    rop.parm("husk_mplay").set(False)
    stage.layoutChildren()
    karma.setDisplayFlag(True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iter", type=int, default=None)
    args = ap.parse_args()

    hou.hipFile.load(cfg.GS_HIP, suppress_save_prompt=True, ignore_load_warnings=True)
    hou.node(cfg.CREATE).node("GS_MODE").parm("input").set(0)  # original animated output

    old = hou.node("/obj/GS_SPLATS")
    if old:
        old.destroy()
    for n in hou.node("/stage").children():
        if n.name().startswith("GS_SPLAT_"):
            n.destroy()

    net = hou.node("/obj").createNode("geo", "GS_SPLATS")
    merge = net.createNode("merge", "all_parts")
    for i, part in enumerate(cfg.PARTS):
        merge.setInput(i, build_part(net, part, latest_export(part, args.iter)))
    out = net.createNode("null", "OUT_SPLATS")
    out.setInput(0, merge)
    out.setDisplayFlag(True)
    out.setRenderFlag(True)
    net.layoutChildren()

    build_render_branch(hou.node("/stage"), out.path())
    hou.hipFile.save(cfg.GS_HIP)
    print("saved", cfg.GS_HIP)


if __name__ == "__main__":
    main()
