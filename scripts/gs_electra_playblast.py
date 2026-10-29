"""hython gs_electra_playblast.py <person_base> <out_dir>: a carved GS person on the Electra camera move (v008;
ELECTRA_PATH=slow: same move, 15 s at constant speed, ~3/4 of it over her head).

<person_base> = .../person/<run>_person (.ply + .json from carve_person.py). She is stood up exactly as in the
crowd shots (P' = s R (P - foot): feet on the floor at the origin, 1.6 m, facing +Z), at full size, in Electra's
place in houdini/electra_arc_v008_240f.hip, and rendered through its arc_cam (9:16, 50 mm, rise + circle)
with Electra and the proxies hidden. Writes <out_dir>/frames and <out_dir>/<run>_electra_240f.mp4.
"""
import json
import math
import os
import subprocess
import sys

import hou

ROOT = os.environ.get("GSPLAT_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))).replace("\\", "/")
sys.path.insert(0, ROOT + "/scripts")
import gs_orbit_reference as ref  # noqa: E402

base, out_dir = sys.argv[1], sys.argv[2]
run = os.path.basename(base).replace("_person", "")
up = json.load(open(base + ".json"))
hou.hipFile.load(ROOT + "/houdini/electra_arc_v008_240f.hip", suppress_save_prompt=True, ignore_load_warnings=True)
for name in ("electra", "proxy_boxes"):
    if hou.node("/obj/" + name):
        hou.node("/obj/" + name).setDisplayFlag(False)

geo = hou.node("/obj").createNode("geo", "GS_PERSON")
f = geo.createNode("file", "ply")
f.parm("file").set(base + ".ply")
drop = geo.createNode("attribdelete", "drop_normals")
drop.setInput(0, f)
drop.parm("ptdel").set("N nx ny nz")
bake = geo.createNode("bakegsplat", "bake")
bake.setInput(0, drop)
bake.parm("sphcoeff").set(True)
fix = geo.createNode("attribwrangle", "undo_colmap_rotation")
fix.setInput(0, bake)
fix.parm("snippet").set("""matrix3 r = ident(); rotate(r, radians(90), {1, 0, 0});
@P *= r;
p@orient = qmultiply(quaternion(r), p@orient);
""")
(r0, r1, r2), f0 = up["R"], up["foot"]
stand = geo.createNode("attribwrangle", "stand_upright")
stand.setInput(0, fix)
stand.parm("snippet").set(f"""matrix3 m = transpose(set(set({r0[0]}, {r0[1]}, {r0[2]}), set({r1[0]}, {r1[1]}, {r1[2]}), set({r2[0]}, {r2[1]}, {r2[2]})));
float s = {up["scale"]};
@P = (@P - set({f0[0]}, {f0[1]}, {f0[2]})) * m * s;
p@orient = qmultiply(quaternion(m), p@orient);
v@scale *= s;""")
stand.setDisplayFlag(True)
stand.setRenderFlag(True)
geo.layoutChildren()

FRAMES = 240
if os.environ.get("ELECTRA_PATH") == "slow":
    # CEO 2026-10-29: v008's turn over her head took ~6 s with smoothstep eases, so the top views (the shot's angle,
    # the one that matters) smeared past. Same rise + circle, 15 s, the camera at one constant speed along the path:
    # the circle at 60 deg is ~4x longer than the rise, so the top gets ~3/4 of the time
    import numpy as np
    FRAMES, EL_TOP, SMOOTH = 360, 60.0, 15
    P = np.array(hou.node("/obj/electra/OUT").geometry().pointFloatAttribValues("P")).reshape(-1, 3)
    lo, hi = P.min(0), P.max(0)
    height, centre = float(hi[1] - lo[1]), (lo + hi) / 2
    cam = hou.node("/obj/arc_cam")
    vfov = 2 * math.atan(cam.parm("aperture").eval() * 16 / 9 / 2 / cam.parm("focal").eval())
    hfov = 2 * math.atan(cam.parm("aperture").eval() / 2 / cam.parm("focal").eval())
    dist = max(height * 1.03 / 2 / math.tan(vfov / 2), float(max(hi[0] - lo[0], hi[2] - lo[2])) * 1.03 / 2 / math.tan(hfov / 2))
    a0 = math.degrees(math.asin((0.92 - 0.5) * height / dist))   # eye level, as v008

    def d(el, az):
        el, az = math.radians(el), math.radians(az)
        return np.array([math.cos(el) * math.sin(az), math.sin(el), math.cos(el) * math.cos(az)])
    way = [(a0, 0.0), (EL_TOP, 0.0), (EL_TOP, -360.0)]   # v008: negative = towards her right first
    dense = np.array([d(e0 + (e1 - e0) * t, z0 + (z1 - z0) * t) for (e0, z0), (e1, z1) in zip(way, way[1:])
                      for t in np.linspace(0, 1, 400, endpoint=False)] + [d(*way[-1])])
    arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(dense, axis=0), axis=1))]
    u = np.linspace(0, arc[-1], FRAMES)
    dirs = np.stack([np.interp(u, arc, dense[:, k]) for k in range(3)], 1)
    pad = np.pad(dirs, ((SMOOTH // 2, SMOOTH // 2), (0, 0)), mode="edge")
    dirs = np.stack([np.convolve(pad[:, k], np.ones(SMOOTH) / SMOOTH, "valid") for k in range(3)], 1)
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    for parm in ("tx", "ty", "tz", "rx", "ry", "rz"):
        cam.parm(parm).deleteAllKeyframes()
    prev = None
    for f, z in enumerate(dirs, 1):
        x = np.cross([0.0, 1.0, 0.0], z)
        x /= np.linalg.norm(x)
        rot = hou.Matrix3((tuple(x), tuple(np.cross(z, x)), tuple(z))).extractRotates()
        rot = [v if prev is None else v + 360 * round((p - v) / 360) for v, p in zip(rot, prev or rot)]
        prev = rot
        for parm, v in zip(("tx", "ty", "tz", "rx", "ry", "rz"), (*(centre + dist * z), *rot)):
            k = hou.Keyframe()
            k.setFrame(f)
            k.setValue(float(v))
            k.setExpression("linear()")
            cam.parm(parm).setKeyframe(k)
    hou.playbar.setFrameRange(1, FRAMES)

os.makedirs(out_dir + "/frames", exist_ok=True)
rop = hou.node("/out/playblast")
rop.parmTuple("f").deleteAllKeyframes()
rop.parmTuple("f").set((1, FRAMES, 1))
rop.parm("picture").set(out_dir + "/frames/$F4.png")
# the Electra hip's ROP is in OpenColorIO mode with no colorspace: Houdini 22 then applies the ACES 2.0 tone-mapped
# view (harsh, dark) and ignores the gamma parm. Plain sRGB instead.
rop.parm("colorcorrect").set("ocio")
rop.parm("ociocolorspace").set("sRGB Encoded Rec.709 (sRGB)")
rop.parm("gamma").set(1.0)
rop.render()
mp4 = f"{out_dir}/{run}_electra_{FRAMES}f.mp4"
subprocess.check_call([ref.FFMPEG, "-hide_banner", "-loglevel", "error", "-nostdin", "-y", "-framerate", "24",
                       "-i", out_dir + "/frames/%04d.png", "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p", mp4])
print("video", mp4)
