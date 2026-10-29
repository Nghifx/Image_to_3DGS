"""Show the splats being trained, live, in Houdini's own 3D viewport.

The trainer drops a .ply checkpoint every few thousand steps. We point a File
SOP at the newest one and reload it on a timer; Houdini 22 rasterises splats on
the GPU, so the artist gets a real, navigable 3D view of the training as it
runs, next to the original geometry.
"""
import os

import hou

NODE_NAME = "GS_PREVIEW"
COLMAP_NODE = "GS_COLMAP"
# Houdini 22 turns COLMAP datasets -90 degrees about X on load. Both of our
# paths go through COLMAP, so the preview turns them back.
COLMAP_ROTATE_X = 90.0


def _build() -> hou.Node:
    geo = hou.node("/obj").createNode("geo", NODE_NAME)
    src = geo.createNode("file", "splat_ply")
    drop = geo.createNode("attribdelete", "drop_normals")
    drop.setInput(0, src)
    drop.parm("ptdel").set("N nx ny nz")
    bake = geo.createNode("bakegsplat", "bake")
    bake.setInput(0, drop)
    bake.parm("sphcoeff").set(True)
    fix = geo.createNode("attribwrangle", "undo_colmap_rotation")
    fix.setInput(0, bake)
    fix.parm("snippet").set(
        f"matrix3 r = ident(); rotate(r, radians({COLMAP_ROTATE_X}), {{1, 0, 0}});\n"
        "@P *= r;\n"
        "p@orient = qmultiply(quaternion(r), p@orient);\n")
    out = geo.createNode("null", "OUT")
    out.setInput(0, fix)
    out.setDisplayFlag(True)
    out.setRenderFlag(True)
    geo.layoutChildren()
    return geo


def node(create: bool = True) -> hou.Node | None:
    found = hou.node(f"/obj/{NODE_NAME}")
    if found is None and create:
        found = _build()
    return found


def remove() -> None:
    found = hou.node(f"/obj/{NODE_NAME}")
    if found is not None:
        found.destroy()


def latest_ply(run_dir: str) -> str:
    """Newest checkpoint written by the trainer, or '' before the first one."""
    candidates = []
    for sub in ("checkpoints", "gsplats"):
        folder = os.path.join(run_dir, sub)
        if not os.path.isdir(folder):
            continue
        for name in os.listdir(folder):
            if name.endswith(".ply"):
                path = os.path.join(folder, name)
                candidates.append((os.path.getmtime(path), path))
    if not candidates:
        return ""
    return max(candidates)[1]


def show(ply: str) -> int:
    """Point the preview at this .ply and reload it. Returns the splat count."""
    geo = node()
    src = geo.node("splat_ply")
    if src.parm("file").eval() != ply.replace("\\", "/"):
        src.parm("file").set(ply.replace("\\", "/"))
    else:
        src.parm("reload").pressButton()
    out = geo.node("OUT")
    try:
        return len(out.geometry().points())
    except hou.Error:
        return 0


def show_colmap(points_ply: str, cams_obj: str) -> int:
    """Solved points (in their own colours) and the camera ring, in /obj/GS_COLMAP.
    Returns the point count. Lives in the dataset's world, like the splats."""
    geo = hou.node(f"/obj/{COLMAP_NODE}")
    if geo is None:
        geo = hou.node("/obj").createNode("geo", COLMAP_NODE)
        pts = geo.createNode("file", "points")
        color = geo.createNode("attribwrangle", "ply_color")
        color.setInput(0, pts)
        # the ply reader keeps red/green/blue as plain 0-255 attributes
        color.parm("snippet").set(
            'if (haspointattrib(0, "red")) v@Cd = set(f@red, f@green, f@blue) / 255.0;\n')
        cams = geo.createNode("file", "cameras")
        merge = geo.createNode("merge", "merge")
        merge.setInput(0, color)
        merge.setInput(1, cams)
        out = geo.createNode("null", "OUT")
        out.setInput(0, merge)
        out.setDisplayFlag(True)
        out.setRenderFlag(True)
        geo.layoutChildren()
    for sop, path in (("points", points_ply), ("cameras", cams_obj)):
        parm = geo.node(sop).parm("file")
        if parm.eval() != path.replace("\\", "/"):
            parm.set(path.replace("\\", "/"))
        else:
            geo.node(sop).parm("reload").pressButton()
    try:
        return len(geo.node("points").geometry().points())
    except hou.Error:
        return 0


def frame_in_viewport(node_name: str = NODE_NAME) -> None:
    viewer = hou.ui.paneTabOfType(hou.paneTabType.SceneViewer)
    geo = hou.node(f"/obj/{node_name}")
    if viewer is None or geo is None:
        return
    geo.setSelected(True, clear_all_selected=True)
    viewer.curViewport().frameSelected()
