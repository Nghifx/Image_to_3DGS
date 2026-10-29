"""Houdini 22 version of the Wukong splat rig, using native nodes.

Splats trained by Houdini 22 (ml/wukong_<part>/...) are imported with Bake GSplats,
turned back upright (H22's COLMAP loader rotates the world -90 deg about X), then
bound to their part's mesh with the native Surface Deform SOP: captured once on
the rest mesh (cached to disk), deformed every frame by the animated mesh.
`orient` is transformed, `restorient` is left as trained so Karma keeps the
view-dependent colour in the right frame.

  hython gs_build_rig_h22.py            # builds houdini/Wukong_gsplat_h22.hip
"""
import glob
import os
import re
import sys

import hou

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gs_config as cfg  # noqa: E402
from gs_build_rig import CLEAN_VEX, MAX_BIND_DIST, MESHES, _merge_from, _wrangle  # noqa: E402

H22_HIP = os.path.join(os.path.dirname(cfg.GS_HIP), "Wukong_gsplat_h22.hip")
ML_DIR = os.path.join(cfg.ROOT, "ml")
RENDER_RES = (1280, 720)  # Apprentice licence cap

UPRIGHT_VEX = """
matrix3 r = ident(); rotate(r, radians(90), {1, 0, 0});
@P *= r;
p@orient = qmultiply(quaternion(r), p@orient);
"""

DROP_FAR_VEX = f"""
int prim; vector uv;
if (xyzdist(1, @P, prim, uv) > {MAX_BIND_DIST}) removepoint(0, @ptnum);
"""


def trained_ply(part: str) -> str:
    files = glob.glob(os.path.join(ML_DIR, f"wukong_{part}", "*", "gsplats", "*.ply"))
    if not files:
        sys.exit(f"no Houdini 22 training output for '{part}' under {ML_DIR}")
    return max(files, key=lambda f: int(re.search(r"\.(\d+)\.ply$", f).group(1)))


def build_part(net: hou.Node, part: str) -> hou.Node:
    rest_name, anim_name = MESHES[part]
    rest = _merge_from(net, f"{part}_rest_mesh", f"{cfg.CREATE}/{rest_name}")
    anim = _merge_from(net, f"{part}_anim_mesh", f"{cfg.CREATE}/{anim_name}")

    src = net.createNode("file", f"{part}_ply")
    src.parm("file").set(trained_ply(part).replace("\\", "/"))
    bake = net.createNode("bakegsplat", f"{part}_bake")
    bake.setInput(0, src)
    if bake.parm("sphcoeff"):
        bake.parm("sphcoeff").set(True)
    clean = _wrangle(net, f"{part}_drop_faint", CLEAN_VEX, [bake])
    upright = _wrangle(net, f"{part}_upright", UPRIGHT_VEX, [clean])
    near = _wrangle(net, f"{part}_drop_floaters", DROP_FAR_VEX, [upright, rest])

    capture = net.createNode("surfacedeform", f"{part}_capture")
    capture.setInput(0, near)
    capture.setInput(1, rest)
    capture.setInput(2, rest)  # required even in capture-only mode; capture reads the rest surface
    capture.parm("mode").set("capture")

    # Capture is the expensive closest-point search: do it once at the rest frame.
    cache_path = os.path.join(ML_DIR, f"wukong_{part}", f"{part}_captured.bgeo.sc")
    hou.setFrame(cfg.REST_FRAME)
    rest.cook(force=True)
    geo = capture.geometry()
    if geo is None:
        sys.exit(f"{part}: capture failed: {capture.errors()}")
    print(f"{part}: {len(bake.geometry().points())} splats trained, {len(geo.points())} bound to mesh")
    geo.saveToFile(cache_path)
    cached = net.createNode("file", f"{part}_captured_cache")
    cached.parm("file").set(cache_path.replace("\\", "/"))

    deform = net.createNode("surfacedeform", f"{part}_follow_rig")
    deform.setInput(0, cached)
    deform.setInput(1, rest)
    deform.setInput(2, anim)
    deform.parm("mode").set("deform")
    deform.parm("attribs").set("* ^restorient")
    return deform


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
    karma.parm("engine").set("xpu")
    karma.parm("setlayerrendersettings").set(False)
    karma.parm("res_mode").set("manual")
    for axis, value in zip(("resolutionx", "resolutiony"), RENDER_RES):
        karma.parm(axis).lock(False)
        karma.parm(axis).set(value)
    karma.parm("denoiser").set("off")
    karma.parm("picture").set(os.path.join(cfg.OUTPUT_DIR, "render_h22", "wukong_gsplat.$F4.png").replace("\\", "/"))

    rop = stage.createNode("usdrender_rop", "GS_SPLAT_ROP")
    rop.setInput(0, karma)
    rop.parm("renderer").set("BRAY_HdKarmaXPU")
    rop.parm("rendersettings").set("/Render/GS_splat_settings")
    rop.parm("husk_mplay").set(False)
    stage.layoutChildren()
    karma.setDisplayFlag(True)


def main() -> None:
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
        merge.setInput(i, build_part(net, part))
    out = net.createNode("null", "OUT_SPLATS")
    out.setInput(0, merge)
    out.setDisplayFlag(True)
    out.setRenderFlag(True)
    net.layoutChildren()

    build_render_branch(hou.node("/stage"), out.path())
    hou.hipFile.save(H22_HIP)
    print("saved", H22_HIP)


if __name__ == "__main__":
    main()
