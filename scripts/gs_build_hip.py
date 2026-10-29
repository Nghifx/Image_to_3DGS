"""Step 1: build Wukong_gsplat.hip from the source hip (source stays untouched).

Adds to a copy of the scene:
  * GS_MODE switch inside the character network
      0 = original animated output, 1 = body in bind pose, 2 = cloth at REST_FRAME
  * GS_CAM / GS_KARMA / GS_ROP LOP branch used to render the training views

Run with Houdini 21 hython:
  hython gs_build_hip.py
"""
import os
import sys

import hou

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gs_config as cfg  # noqa: E402

MODE_ORIGINAL, MODE_BODY, MODE_CLOTH = 0, 1, 2


def _copy(node: hou.Node, name: str) -> hou.Node:
    new = hou.copyNodesTo([node], node.parent())[0]
    new.setName(name, unique_name=True)
    return new


def build_mode_switch(create: hou.Node) -> hou.Node:
    """Body bind pose and cloth rest frame, selectable without touching the original chain."""
    static = create.node("STATIC_MODEL")
    out_model = create.node("OUT_MODEL")
    original = out_model.input(0)  # merge1

    # Body in bind pose: same cleanup / scale / blast / color as the animated body chain
    body_cleanup = _copy(create.node("cleanup"), "GS_body_cleanup")
    body_cleanup.setInput(0, static)
    # The FBX animation carries a x100 root scale, so the bind pose needs
    # 100x the animated chain's 0.01 to come out at the same size.
    body_xform = _copy(create.node("transform1"), "GS_body_xform")
    body_xform.setInput(0, body_cleanup)
    body_xform.parm("scale").set(1.0)
    body_blast = _copy(create.node("blast1"), "GS_body_blast")
    body_blast.setInput(0, body_xform)
    body_color = _copy(create.node("color2"), "GS_body_color")
    body_color.setInput(0, body_blast)

    # Cloth frozen at REST_FRAME (sim cache is per-frame)
    cloth_rest = create.createNode("timeshift", "GS_cloth_rest")
    cloth_rest.setInput(0, create.node("cleanup1"))
    cloth_rest.parm("frame").deleteAllKeyframes()
    cloth_rest.parm("frame").set(cfg.REST_FRAME)

    switch = create.createNode("switch", "GS_MODE")
    switch.setInput(MODE_ORIGINAL, original)
    switch.setInput(MODE_BODY, body_color)
    switch.setInput(MODE_CLOTH, cloth_rest)
    switch.parm("input").set(MODE_ORIGINAL)
    out_model.setInput(0, switch)
    create.layoutChildren()
    return switch


def build_render_branch(stage: hou.Node) -> None:
    # The lookdev template brings a floor grid; keep its lights, drop the floor
    no_floor = _copy(stage.node("prune1"), "GS_PRUNE_FLOOR")
    no_floor.setInput(0, stage.node("prune1"))
    no_floor.parm("primpattern1").set("/World/Grid /World/ShaderBall")

    cam = stage.createNode("camera", "GS_CAM")
    cam.setInput(0, no_floor)
    cam.parm("primpath").set("/cameras/GS_CAM")
    cam.parm("focalLength").set(cfg.FOCAL_MM)
    cam.parm("aperture").set("set")  # author horizontal + vertical aperture explicitly
    cam.parm("horizontalAperture").set(cfg.APERTURE_MM)
    cam.parm("verticalAperture").set(cfg.APERTURE_MM)
    cam.parm("clippingRange1").set(0.01)
    cam.parm("clippingRange2").set(100000)

    karma = _copy(stage.node("karmarendersettings"), "GS_KARMA")
    karma.setInput(0, cam)
    karma.parm("primpath").set("/Render/GS_settings")
    karma.parm("renderproductsparentprimpath").set("/Render/GS_Products")
    karma.parm("rendervarsparentprimpath").set("/Render/GS_Products/Vars")
    karma.parm("camera").set("/cameras/GS_CAM")
    karma.parm("setlayerrendersettings").set(False)
    karma.parm("res_mode").set("manual")
    for axis in ("resolutionx", "resolutiony"):
        karma.parm(axis).lock(False)
        karma.parm(axis).set(cfg.RESOLUTION)
    karma.parm("engine").set(cfg.ENGINE)
    karma.parm("samplesperpixel").set(cfg.SAMPLES)
    karma.parm("enabledof").set(False)
    karma.parm("enablemblur").set(False)
    karma.parm("denoiser").set("oidn")
    karma.parm("picture").set(os.path.join(cfg.DATA_DIR, "body", "images", "$F4.png").replace("\\", "/"))

    rop = stage.createNode("usdrender_rop", "GS_ROP")
    rop.setInput(0, karma)
    rop.parm("rendersettings").set("/Render/GS_settings")
    rop.parm("renderer").set("BRAY_HdKarmaXPU" if cfg.ENGINE == "xpu" else "BRAY_HdKarma")
    rop.parm("trange").set("normal")
    rop.parm("f1").deleteAllKeyframes()
    rop.parm("f2").deleteAllKeyframes()
    rop.parm("husk_mplay").set(False)
    rop.parm("husk_instantshutter").set(True)
    stage.layoutChildren()


def main() -> None:
    hou.hipFile.load(cfg.SOURCE_HIP, suppress_save_prompt=True, ignore_load_warnings=True)
    create = hou.node(cfg.CREATE)
    if create.node("GS_MODE") is None:
        build_mode_switch(create)
    stage = hou.node("/stage")
    if stage.node("GS_ROP") is None:
        build_render_branch(stage)

    xf = hou.node("/stage/imp_Chr_Jiao_model_master_v0001/xform")
    nondefault = {p.name(): p.evalAsString() for p in xf.parms() if not p.isAtDefault()}
    print("LOP xform inside sopcreate (non-default):", nondefault)

    os.makedirs(os.path.dirname(cfg.GS_HIP), exist_ok=True)
    hou.hipFile.save(cfg.GS_HIP)
    print("saved", cfg.GS_HIP)


if __name__ == "__main__":
    main()
