"""Build a 360-degree orbit reference: Electra in T-pose, 35 mm camera, OpenGL playblast.

  hython scripts/gs_orbit_reference.py [--frames 120] [--fps 24] [--out khmer_girl/ref_orbit]
                                       [--path circle|boom|grid]

circle (default): the camera turns at an exactly constant speed (360 / frames
degrees per frame) on one level circle. boom: starts 2 m from the face at eye
height, turns slowly in front and faster behind, and booms up to look down on
the crown (see boom_path). grid: whole body in frame, zigzag left/right
up the front, over the head, then zigzag down the back (see grid_path). Either way the clip can drive the camera of an AI
video (Seedance "follow the camera movement of @Video 1") and every frame's
camera is known in advance.

Writes houdini/electra_<path>_ref.hip, <out>/frames/####.png, <out>/electra_<path>_<frames>f.mp4
and <out>/orbit_cameras.json (distance, focal; per-frame views for boom).
"""
import argparse
import json
import math
import os
import subprocess

import sys

import hou

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gs_paths import ROOT, find_ffmpeg  # noqa: E402

FFMPEG = find_ffmpeg()
FOCAL_MM = 35.0
APERTURE_MM = 36.0         # full-frame width, so 35 mm means what a photographer expects
RES = (1920, 1080)
FILL = 0.85                # character height as a fraction of the frame height
CAM_HEIGHT = 2.0           # metres, above the head (Electra is 1.61 m): the shot sees the top of the head
GRID_SIZE = 10.0           # floor grid: gives the AI a real perspective to hold the camera and the body to
GRID_STEP = 0.5
GRID_COLOR = (0.12, 0.12, 0.12)
AMBIENT = 1.3
BACKDROP = "9a9a9a"        # plain mid-grey, like the studio backdrop the AI video should have
# --path boom: the shot is about the face and head, so it starts close and level with them
BOOM_DISTANCE = 2.0        # metres from the face, kept for the whole turn
BOOM_FACE = 0.93           # eye height as a fraction of body height
BOOM_SPEED_RATIO = 3.0     # turn speed at the back / at the front
BOOM_TOP_DEG = 75.0        # final elevation over the face; 90 would flip the camera's up
# --path grid: (from deg, to deg, elevation over the body centre; None = level with the face).
# 0 deg = front, negative = the camera's own left. Front half rising, over the head, back half down.
# Elevation may be (start, end) to change height along the row.
# Zigzag up the front, then over the head to end high behind her, looking down on the back.
GRID_ROWS = (
    # sweeps kept to +-45 deg: wider ones were too fast for the AI video to follow
    (0, -45, None),      # from the face, out to her right
    (-45, 45, 30),       # across the front, higher
    (45, -45, 55),       # back across, high
    (-45, -180, (72, 50)),  # over the head, ending high behind her
)
GRID_TURN = 0.3            # part of the shorter neighbouring row spent changing height at a turn
GRID_MIN_SPEED = 0.35      # time floor for rows near the top, as cos(elevation)
GRID_FRONT_TIME = 1.25     # time per degree of travel: front rows (within +-90 deg) ...
GRID_BACK_TIME = 2.5       # ... and the move over the head, which felt abrupt at the same speed
GRID_RAMP = 0.12           # part of a row spent speeding up / slowing down; constant speed between
GRID_SMOOTH = 25           # frames averaged over, so no change of speed or direction is a jerk

# Upper-arm subtree turned so the shoulder-to-hand line points straight out sideways.
ARMS = {"L_upperarm": ("L_hand", (1.0, 0.0, 0.0)), "R_upperarm": ("R_hand", (-1.0, 0.0, 0.0))}
# Thigh subtree turned outwards so the feet stand shoulder-width apart (ankles at x = +-0.18 m;
# hip joints sit at +-0.092, 0.896 and the ankles 0.06 above the floor).
LEGS = {"L_thigh": ("L_foot", (0.088, -0.833, -0.046)), "R_thigh": ("R_foot", (-0.088, -0.833, -0.046))}

T_POSE_PY = '''
import numpy as np
node = hou.pwd()
geo = node.geometry()
arms = %r
names = {p.attribValue("name"): p for p in geo.points()}
children = {}
for prim in geo.prims():
    pts = [v.point() for v in prim.vertices()]
    for a, b in zip(pts, pts[1:]):
        children.setdefault(a.number(), []).append(b)

def subtree(p):
    out, todo = [], [p]
    while todo:
        q = todo.pop()
        out.append(q)
        todo.extend(children.get(q.number(), []))
    return out

def rotation(a, b):
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    v, c = np.cross(a, b), float(np.dot(a, b))
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx / (1 + c)

for root, (tip, target) in arms.items():
    pivot = np.array(names[root].position())
    rot = rotation(np.array(names[tip].position()) - pivot, np.array(target))
    for p in subtree(names[root]):
        p.setPosition(hou.Vector3(pivot + rot @ (np.array(p.position()) - pivot)))
        m = np.array(p.attribValue("transform")).reshape(3, 3)   # row vectors
        p.setAttribValue("transform", tuple((m @ rot.T).flatten()))
'''


def build_character() -> hou.Node:
    geo = hou.node("/obj").createNode("geo", "electra")
    src = geo.createNode("testgeometry_electra", "electra")
    pose = geo.createNode("python", "t_pose")
    pose.setInput(0, src, 2)
    pose.parm("python").set(T_POSE_PY % ({**ARMS, **LEGS},))
    deform = geo.createNode("bonedeform", "deform")
    deform.setInput(0, src, 0)
    deform.setInput(1, src, 1)
    deform.setInput(2, pose, 0)
    out = geo.createNode("null", "OUT")
    out.setInput(0, deform)
    out.setDisplayFlag(True)
    out.setRenderFlag(True)
    geo.layoutChildren()
    return out


def framing(height_m: float) -> tuple[float, float]:
    """Distance and downward tilt (degrees) that put head and feet FILL of the frame
    apart, centred, from a camera CAM_HEIGHT above the floor."""
    vfov = 2 * math.atan(APERTURE_MM * RES[1] / RES[0] / 2 / FOCAL_MM)
    span = lambda d: math.atan2(height_m - CAM_HEIGHT, d) - math.atan2(-CAM_HEIGHT, d)  # noqa: E731
    lo, hi = 0.1, 50.0
    for _ in range(60):  # the angular span shrinks with distance: bisect for the target
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if span(mid) > FILL * vfov else (lo, mid)
    d = (lo + hi) / 2
    tilt = (math.atan2(height_m - CAM_HEIGHT, d) + math.atan2(-CAM_HEIGHT, d)) / 2
    return d, math.degrees(tilt)


def build_camera(frames: int, height_m: float) -> tuple[hou.Node, float, float]:
    distance, tilt = framing(height_m)
    pivot = hou.node("/obj").createNode("null", "orbit_pivot")
    pivot.parm("ry").setExpression(f"($F - 1) * 360.0 / {frames}")
    pivot.setDisplayFlag(False)  # its axis gizmo otherwise shows under the feet in the playblast
    cam = hou.node("/obj").createNode("cam", "orbit_cam")
    cam.setFirstInput(pivot)
    for parm, value in (("tz", distance), ("ty", CAM_HEIGHT), ("rx", tilt), ("resx", RES[0]), ("resy", RES[1]),
                        ("focal", FOCAL_MM), ("aperture", APERTURE_MM)):
        cam.parm(parm).set(value)
    return cam, distance, tilt


def boom_path(frames: int, height_m: float) -> list[dict]:
    """One turn that starts level with the face and booms up to look down on the crown.

    Turn speed is slow at the front and BOOM_SPEED_RATIO times faster at the back:
    angle = 360 * (u - s * sin(2 pi u) / (2 pi)), whose speed 1 - s cos(2 pi u) is
    lowest at u = 0 (front) and highest at u = 0.5 (back). The camera stays
    BOOM_DISTANCE from the face; its elevation eases from 0 to BOOM_TOP_DEG.
    """
    s = (BOOM_SPEED_RATIO - 1) / (BOOM_SPEED_RATIO + 1)
    face = BOOM_FACE * height_m
    views = []
    for f in range(1, frames + 1):
        u = (f - 1) / frames
        deg = 360.0 * (u - s * math.sin(2 * math.pi * u) / (2 * math.pi))
        t = (f - 1) / (frames - 1)
        elev = BOOM_TOP_DEG * t * t * (3 - 2 * t)  # smoothstep: starts and ends gently
        e = math.radians(elev)
        views.append({"frame": f, "deg": round(deg, 3), "height_m": round(face + BOOM_DISTANCE * math.sin(e), 4),
                      "tilt_deg": round(-elev, 3), "ground_dist_m": round(BOOM_DISTANCE * math.cos(e), 4)})
    return views


def grid_path(frames: int, height_m: float) -> list[dict]:
    """Zigzag over the front half, over the head, then zigzag down the back half,
    so the frames tile every side and the top like a grid (rows in GRID_ROWS).

    Starts level with the face, whole body in frame. Each row runs at constant
    speed with a short ramp at both ends (GRID_RAMP), so the camera only slows
    into the U-turns, and changes height during the turn. Rows get time in
    proportion to how far the camera travels, front rows more (GRID_FRONT_TIME).
    Left = the camera's own left (negative turn).
    """
    centre = height_m / 2
    dist = framing_distance(height_m)
    face = math.degrees(math.asin(min(1.0, (BOOM_FACE * height_m - centre) / dist)))
    rows = []
    for a0, a1, e in GRID_ROWS:
        e0, e1 = e if isinstance(e, tuple) else (e, e)
        rows.append((a0, a1, face if e0 is None else e0, face if e1 is None else e1))

    def row_elev(k: int, s: float) -> float:
        return rows[k][2] + (rows[k][3] - rows[k][2]) * s

    # travel on the sphere: turning shrinks with cos(elevation) (floored so the top does not
    # whip round), plus any change of height along the row
    weights = [math.hypot(abs(a1 - a0) * max(math.cos(math.radians((e0 + e1) / 2)), GRID_MIN_SPEED), e1 - e0)
               * (GRID_FRONT_TIME if max(abs(a0), abs(a1)) <= 90 else GRID_BACK_TIME) for a0, a1, e0, e1 in rows]
    bounds = [sum(weights[:k]) / sum(weights) for k in range(len(rows) + 1)]

    def reverses(j: int) -> bool:
        """The camera stops only where it turns back, or at the start and end."""
        if j <= 0 or j >= len(rows):
            return True
        return (rows[j - 1][1] - rows[j - 1][0]) * (rows[j][1] - rows[j][0]) < 0

    track = []
    for f in range(1, frames + 1):
        t = (f - 1) / (frames - 1)
        k = next(i for i in range(len(rows)) if t <= bounds[i + 1])
        s = (t - bounds[k]) / (bounds[k + 1] - bounds[k])
        r_in = GRID_RAMP if reverses(k) else 0.0   # trapezoid speed: ramp, constant, ramp
        r_out = GRID_RAMP if reverses(k + 1) else 0.0
        vmax = 1 / (1 - (r_in + r_out) / 2)
        if s < r_in:
            p = vmax * s * s / (2 * r_in)
        elif s > 1 - r_out:
            p = 1 - vmax * (1 - s) ** 2 / (2 * r_out)
        else:
            p = vmax * (s - r_in / 2)
        a0, a1 = rows[k][:2]
        deg = a0 + (a1 - a0) * p
        elev = row_elev(k, p)
        for j in range(1, len(rows)):              # change height across the turn at bounds[j]
            half = GRID_TURN * min(bounds[j] - bounds[j - 1], bounds[j + 1] - bounds[j])
            w = (t - bounds[j] + half) / (2 * half)
            if 0 < w < 1:
                # blend between the two rows' own heights at this moment: no jump at the window edges
                lo = row_elev(j - 1, p) if k == j - 1 else rows[j - 1][3]
                hi = row_elev(j, p) if k == j else rows[j][2]
                elev = lo + (hi - lo) * w * w * (3 - 2 * w)
        track.append((deg, elev))
    # moving average: rows meet with different speeds; this turns each speed change into a short ease
    n = GRID_SMOOTH // 2
    padded = [track[0]] * n + track + [track[-1]] * n
    views = []
    for f in range(1, frames + 1):
        win = padded[f - 1:f - 1 + 2 * n + 1]
        deg = sum(d for d, _ in win) / len(win)
        elev = sum(e for _, e in win) / len(win)
        e = math.radians(elev)
        views.append({"frame": f, "deg": round(deg, 3), "height_m": round(centre + dist * math.sin(e), 4),
                      "tilt_deg": round(-elev, 3), "ground_dist_m": round(dist * math.cos(e), 4)})
    return views


def framing_distance(height_m: float) -> float:
    """Distance from the body centre that fits the whole body (FILL of the frame height)."""
    vfov = 2 * math.atan(APERTURE_MM * RES[1] / RES[0] / 2 / FOCAL_MM)
    return height_m / FILL / (2 * math.tan(vfov / 2))


def build_path_camera(views: list[dict]) -> hou.Node:
    """Keyframe every frame: position on the sphere round the aim point, aimed at it.
    Rotate order xyz tilts first, then turns, as the pivot rig does for the circle."""
    cam = hou.node("/obj").createNode("cam", "path_cam")
    for parm, value in (("resx", RES[0]), ("resy", RES[1]), ("focal", FOCAL_MM), ("aperture", APERTURE_MM)):
        cam.parm(parm).set(value)
    for v in views:
        a = math.radians(v["deg"])
        for parm, value in (("tx", v["ground_dist_m"] * math.sin(a)), ("ty", v["height_m"]),
                            ("tz", v["ground_dist_m"] * math.cos(a)), ("rx", v["tilt_deg"]), ("ry", v["deg"])):
            key = hou.Keyframe()
            key.setFrame(v["frame"])
            key.setValue(value)
            key.setExpression("linear()")
            cam.parm(parm).setKeyframe(key)
    return cam


def build_lights() -> None:
    """Flat ambient light: the same brightness from every side of the turn.
    An environment light with no map is a uniform dome; the old Ambient light
    object only feeds the material's ambient term and leaves a playblast black."""
    hou.node("/obj").createNode("envlight", "ambient").parm("light_intensity").set(AMBIENT)


def build_grid() -> None:
    """Floor grid lines (rows and columns as open polylines) under the character."""
    geo = hou.node("/obj").createNode("geo", "floor_grid")
    grid = geo.createNode("grid", "grid")
    lines = int(GRID_SIZE / GRID_STEP) + 1
    for parm, value in (("type", "poly"), ("surftype", "rowcol"), ("rows", lines), ("cols", lines)):
        grid.parm(parm).set(value)
    grid.parmTuple("size").set((GRID_SIZE, GRID_SIZE))
    color = geo.createNode("color", "color")
    color.setInput(0, grid)
    color.parmTuple("color").set(GRID_COLOR)
    color.setDisplayFlag(True)
    color.setRenderFlag(True)
    geo.layoutChildren()


def playblast(cam: hou.Node, frames: int, out_dir: str) -> None:
    os.makedirs(os.path.join(out_dir, "frames"), exist_ok=True)
    rop = hou.node("/out").createNode("opengl", "playblast")
    rop.parm("camera").set(cam.path())
    rop.parm("trange").set(1)
    rop.parmTuple("f").deleteAllKeyframes()
    rop.parmTuple("f").set((1, frames, 1))
    rop.parm("tres").set(True)
    rop.parmTuple("res").set(RES)
    rop.parm("picture").set(os.path.join(out_dir, "frames", "$F4.png").replace("\\", "/"))
    backdrop = os.path.join(out_dir, "backdrop.png")
    subprocess.check_call([FFMPEG, "-hide_banner", "-loglevel", "error", "-nostdin", "-y", "-f", "lavfi",
                           "-i", f"color=c=0x{BACKDROP}:s={RES[0]}x{RES[1]}", "-frames:v", "1", backdrop])
    rop.parm("bgimage").set(backdrop.replace("\\", "/"))
    rop.parm("aamode").set("aa8")
    # Viewport look: no scene lights means a headlight from the camera, so surfaces
    # facing away (back wall) go dark and the body reads in 3D; OCIO matches the viewport.
    # (hqlighting=0 would be closer still, but crashes a headless hython render.)
    rop.parm("alights").set("")
    rop.parm("shadows").set(False)
    rop.parm("colorcorrect").set("ocio")
    rop.render()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=120)
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--out", default=os.path.join(ROOT, "khmer_girl", "ref_orbit"))
    ap.add_argument("--path", default="circle", choices=("circle", "boom", "grid"),
                    help="boom: level with the face at 2 m, slow in front, fast behind, up to the crown; "
                         "grid: zigzag up the front, over the head, zigzag down the back")
    args = ap.parse_args()
    out_dir = os.path.abspath(args.out)

    hou.setFps(args.fps)
    hou.playbar.setFrameRange(1, args.frames)
    character = build_character()
    bb = character.geometry().boundingBox()
    print("T-pose size (m):", [round(v, 3) for v in bb.sizevec()])
    if args.path == "boom":
        views = boom_path(args.frames, bb.sizevec()[1])
        cam = build_path_camera(views)
        info = {"distance_m": BOOM_DISTANCE, "views": views}
        print(f"boom: {BOOM_DISTANCE} m from the face, up to {BOOM_TOP_DEG} deg, back {BOOM_SPEED_RATIO}x faster")
    elif args.path == "grid":
        views = grid_path(args.frames, bb.sizevec()[1])
        cam = build_path_camera(views)
        dist = framing_distance(bb.sizevec()[1])
        info = {"distance_m": round(dist, 4), "rows": [list(r) for r in GRID_ROWS], "views": views}
        print(f"grid: {len(GRID_ROWS)} rows, {dist:.2f} m from the body centre")
    else:
        cam, distance, tilt = build_camera(args.frames, bb.sizevec()[1])
        info = {"deg_per_frame": 360.0 / args.frames, "distance_m": round(distance, 4), "height_m": CAM_HEIGHT,
                "tilt_deg": round(tilt, 3)}
        print(f"camera {CAM_HEIGHT} m high, {distance:.2f} m away, tilted {tilt:.1f} deg")
    build_lights()
    build_grid()
    hip = os.path.join(ROOT, "houdini", f"electra_{args.path}_ref.hip")
    hou.hipFile.save(hip)
    playblast(cam, args.frames, out_dir)

    mp4 = os.path.join(out_dir, f"electra_{args.path}_{args.frames}f.mp4")
    subprocess.check_call([FFMPEG, "-hide_banner", "-loglevel", "error", "-nostdin", "-y", "-framerate", str(args.fps),
                           "-i", os.path.join(out_dir, "frames", "%04d.png"), "-c:v", "libx264", "-crf", "12",
                           "-pix_fmt", "yuv420p", mp4])
    with open(os.path.join(out_dir, "orbit_cameras.json"), "w") as f:
        json.dump({"frames": args.frames, "fps": args.fps, "path": args.path, **info,
                   "focal_mm": FOCAL_MM, "aperture_mm": APERTURE_MM, "resolution": list(RES)}, f, indent=1)
    print("saved", hip)
    print("video", mp4)


if __name__ == "__main__":
    main()
