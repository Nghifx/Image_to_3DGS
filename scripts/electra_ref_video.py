"""v030: v027 (20 s, v025's path, v018's marks, arms 15 deg) with the camera ~10 % closer.

  hython scripts/electra_ref_video.py     (the Genjutsu reference: Electra on the blue screen, v030, 20 s)

CEO (2026-10-29): her arms are in now, so bring the camera in a little - MARGIN 1.1 -> 1.0 (frame height = her
height x MARGIN, or her span from above, whichever needs more room).
"""
import math
import os
import subprocess
import sys

import numpy as np

import hou

ROOT = os.environ.get("GSPLAT_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))).replace("\\", "/")
sys.path.insert(0, ROOT + "/scripts")
import gs_orbit_reference as ref  # noqa: E402

MODE = "sweep"
SRC_HIP = ROOT + "/houdini/electra_grid_apose_240f.hip"
OUT_HIP = ROOT + "/houdini/electra_arc_v030_480f.hip"
OUT_DIR = ROOT + "/khmer_girl/ref_electra_arc_v030_20s"
FRAMES, FPS = 480, 24   # CEO: 20 s
RES = (1080, 1920)
FOCAL, APERTURE = 50.0, 24.0
MARGIN = 1.0              # CEO 2026-10-29: closer, arms are in (v022-v027: 1.1)
EYE = 0.92
AZ_MAX = 80.0             # swings out to her sides (CEO); the shot itself needs -37..+14 deg
LOW_EL, TOP_EL = 44.0, 84.0   # the crowd's band 44..89 deg (the pole itself would spin the camera)
FACE_PART = 0.15          # share of the move spent rising from eye level to LOW_EL (her face)
ARM_OUT = 15.0            # deg from hanging straight down (CEO 2026-10-29: closer in; v018-v026: 29)
THUMB_IN = 20.0           # deg the thumb is turned in against the index finger
SWINGS = 1.75             # left-right swings (CEO: fewer)
CURL = 60.0               # deg per finger joint: four fingers curled (CEO: more than the first 40)
THUMB = 0.0               # thumb kept straight (CEO)
HOLD = 0
TRACK_MARKS = """
import random
geo = hou.pwd().geometry()
geo.addAttrib(hou.attribType.Point, "Cd", (1.0, 1.0, 1.0))
rng = random.Random(3)
ARM, W = 0.07, 0.016   # half length and half width of each bar, metres

def quad(c, u, v):
    pts = []
    for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
        p = geo.createPoint()
        p.setPosition(tuple(c[i] + a * u[i] + b * v[i] for i in range(3)))
        p.setAttribValue("Cd", (0.79, 0.21, 0.2))   # pale red
        pts.append(p)
    poly = geo.createPolygon()
    for p in pts:
        poly.addVertex(p)

def cross(c, ax1, ax2):
    quad(c, [ARM * x for x in ax1], [W * x for x in ax2])
    quad(c, [W * x for x in ax1], [ARM * x for x in ax2])

# the camera works in front of her (+Z): marks there stay clear of her silhouette and paint out easily;
# behind her (-Z) only a few, far out
for i in range(-4, 5):
    for j in range(-4, 6):
        z = j * 0.9 + rng.uniform(-0.25, 0.25)
        x = i * 1.0 + rng.uniform(-0.25, 0.25)
        if abs(x) < 0.7 and abs(z) < 0.7:
            continue   # none under or right by her feet
        if z < 0 and (abs(x) < 2.5 or (i + j) % 3):
            continue   # behind her: only every third mark, and none in line with her
        cross((x, 0.004, z), (1, 0, 0), (0, 0, 1))
for i in range(-4, 5):                                       # denser on the floor just in front of her: in frame
    for j in range(1, 5):                                    # whenever the camera looks down
        x, z = i * 0.55 + rng.uniform(-0.12, 0.12), 0.45 + j * 0.5 + rng.uniform(-0.12, 0.12)
        if abs(x) > 0.5 or z > 0.9:
            cross((x, 0.004, z), (1, 0, 0), (0, 0, 1))
for x, z in ((-0.46, 0.31), (0.5, 0.31), (-0.33, 0.58), (0.2, 0.62)):   # the CEO's 4, in front of her feet
    cross((x, 0.004, z), (1, 0, 0), (0, 0, 1))
for k in range(-4, 5):
    for h in (0.8, 1.7, 2.6, 3.5):
        o = k * 1.1 + rng.uniform(-0.2, 0.2)
        hh = h + rng.uniform(-0.15, 0.15)
        cross((o, hh, 4.99), (1, 0, 0), (0, 1, 0))            # the wall in front of her (behind the camera)
        for side in (-4.99, 4.99):                              # side walls: front half only
            if o > 0:
                cross((side, hh, o), (0, 0, 1), (0, 1, 0))
        if abs(k) >= 3 and h > 2.0:                             # back wall: a few, high and out to the sides
            cross((o, hh, -4.99), (1, 0, 0), (0, 1, 0))
"""
POSE = f"""
import math
def axis_rot(axis, deg):
    axis = np.array(axis, float); axis /= np.linalg.norm(axis); a = math.radians(deg)
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + math.sin(a) * k + (1 - math.cos(a)) * k @ k

def turn(root, rot, pivot=None):
    pivot = np.array(names[root].position()) if pivot is None else pivot
    for p in subtree(names[root]):
        p.setPosition(hou.Vector3(pivot + rot @ (np.array(p.position()) - pivot)))
        m = np.array(p.attribValue("transform")).reshape(3, 3)
        p.setAttribValue("transform", tuple((m @ rot.T).flatten()))

def pos(n):
    return np.array(names[n].position())

def curl(s, finger, deg):
    axis = pos(s + "_index_01") - pos(s + "_pinky_01")      # across the knuckles
    tip0 = abs(pos(s + finger + "_03")[0])
    for j in ("_01", "_02", "_03"):
        turn(s + finger + j, axis_rot(axis, deg))
    if abs(pos(s + finger + "_03")[0]) > tip0:                # opened away from the thigh: undo, close the other way
        for j in ("_03", "_02", "_01"):
            turn(s + finger + j, axis_rot(axis, -deg))
        for j in ("_01", "_02", "_03"):
            turn(s + finger + j, axis_rot(axis, -deg))

for s in ("L", "R"):
    for finger in ("_index", "_middle", "_ring", "_pinky"):
        curl(s, finger, {CURL})
    curl(s, "_thumb", {THUMB})

def thumb_in(s, deg):   # turn the thumb about the palm normal, towards the index finger
    palm = np.cross(pos(s + "_index_01") - pos(s + "_pinky_01"), pos(s + "_middle_01") - pos(s + "_hand"))
    gap = lambda: np.linalg.norm(pos(s + "_thumb_03") - pos(s + "_index_02"))
    g0 = gap()
    turn(s + "_thumb_01", axis_rot(palm, deg))
    if gap() > g0:
        turn(s + "_thumb_01", axis_rot(palm, -2 * deg))

for s in ("L", "R"):
    thumb_in(s, {THUMB_IN})
"""

hou.hipFile.load(SRC_HIP, suppress_save_prompt=True, ignore_load_warnings=True)
hou.setFps(FPS)
hou.playbar.setFrameRange(1, FRAMES)
hou.node("/obj/path_cam").destroy()
tp = hou.node("/obj/electra/t_pose")
down = (math.sin(math.radians(ARM_OUT)), -math.cos(math.radians(ARM_OUT)))
code = tp.parm("python").eval().replace("0.5591929034707469", repr(down[0])).replace("-0.8290375725550417", repr(down[1]))
assert repr(down[0]) in code, "A-pose arm target not found"
tp.parm("python").set(code + POSE)

# blue screen: the room (floor grid + inside-out box) turned chroma blue with a faint mottle, plus red "+" marks
room = hou.node("/obj/floor_grid")
unlit = room.node("unlit")
blue = room.createNode("attribwrangle", "blue_screen")
blue.setInput(0, unlit.input(0))
blue.parm("snippet").set("float n = noise(@P * 0.6) - 0.5; v@Cd = (@P.y < 0.01 ? set(0.09, 0.33, 0.74) : set(0.05, 0.2, 0.57)) * (1 + 0.04 * n);")   # studio blue, floor lighter, barely mottled
marks = room.createNode("python", "track_marks")
marks.parm("python").set(TRACK_MARKS)
merge = room.createNode("merge", "screen")
merge.setInput(0, blue)
merge.setInput(1, marks)
unlit.setInput(0, merge)

P = np.array(hou.node("/obj/electra/OUT").geometry().pointFloatAttribValues("P")).reshape(-1, 3)
lo, hi = P.min(0), P.max(0)
height = float(hi[1] - lo[1])
centre = (lo + hi) / 2
vfov = 2 * math.atan(APERTURE * RES[1] / RES[0] / 2 / FOCAL)
hfov = 2 * math.atan(APERTURE / 2 / FOCAL)
dist = max(height * MARGIN / 2 / math.tan(vfov / 2), float(max(hi[0] - lo[0], hi[2] - lo[2])) * MARGIN / 2 / math.tan(hfov / 2))
a0 = math.degrees(math.asin(min(max((EYE - 0.5) * height / dist, -1), 1)))



def ease(t: float) -> float:
    t = min(max(t, 0.0), 1.0)
    return t * t * (3 - 2 * t)


def direction(el: float, az: float) -> np.ndarray:
    el, az = math.radians(el), math.radians(az)
    return np.array([math.cos(el) * math.sin(az), math.sin(el), math.cos(el) * math.cos(az)])


def shape(t: float) -> tuple:
    """(elevation, azimuth) in degrees along the path, t 0..1: eye level -> LOW_EL over the first FACE_PART, then an
    even climb through the shot's band to TOP_EL, swinging left and right like a pendulum (no corners)."""
    if t < FACE_PART:
        el = a0 + (LOW_EL - a0) * ease(t / FACE_PART)
    else:
        el = LOW_EL + (TOP_EL - LOW_EL) * (t - FACE_PART) / (1 - FACE_PART)
    return el, AZ_MAX * math.sin(2 * math.pi * SWINGS * t)


TAIL_FRAMES, TAIL_EL, TAIL_AZ = 80, 60.0, -180.0   # CEO: 3 s from the end of the swings round to her back


def frame_of(el: float, az: float) -> np.ndarray:
    """Camera rotation as the keyframe loop builds it: aimed at her, world up."""
    z = direction(el, az)
    x = np.cross([0.0, 1.0, 0.0], z)
    x /= np.linalg.norm(x)
    return np.stack([x, np.cross(z, x), z])


def paced(fn, n: int) -> list:
    """n (el, az) samples of fn(0..1) spaced by the turn of the picture (v022), not by travel (v018)."""
    ts = np.linspace(0.0, 1.0, 20001)
    rots = [frame_of(*fn(t)) for t in ts]
    turn = [math.acos(min(1.0, max(-1.0, (np.trace(p @ q.T) - 1) / 2))) for p, q in zip(rots, rots[1:])]
    arc = np.r_[0.0, np.cumsum(turn)]
    return [fn(float(np.interp(u, arc, ts))) for u in np.linspace(0, arc[-1], n)]


end_el, end_az = shape(1.0)
tail = lambda t: (end_el + (TAIL_EL - end_el) * t, end_az + (TAIL_AZ - end_az) * t)   # noqa: E731
POSES = paced(shape, FRAMES - TAIL_FRAMES) + paced(tail, TAIL_FRAMES + 1)[1:]


def pose(f: int) -> tuple:
    """(elevation, azimuth) in degrees at frame f."""
    return POSES[f - 1]


print(f"Electra {height:.2f} m, radius {dist:.2f} m, start {a0:.1f} deg, {MODE}")

cam = hou.node("/obj").createNode("cam", "arc_cam")
for parm, v in (("resx", RES[0]), ("resy", RES[1]), ("focal", FOCAL), ("aperture", APERTURE), ("far", 500)):
    cam.parm(parm).set(v)
prev = None
for f in range(1, FRAMES + 1):
    el, az = pose(f)
    z = direction(el, az)
    x = np.cross([0.0, 1.0, 0.0], z)
    x /= np.linalg.norm(x)
    up = np.cross(z, x)
    rot = hou.Matrix3((tuple(x), tuple(up), tuple(z))).extractRotates()
    rot = [v if prev is None else v + 360 * round((p - v) / 360) for v, p in zip(rot, prev or rot)]
    prev = rot
    for parm, v in zip(("tx", "ty", "tz", "rx", "ry", "rz"), (*(centre + dist * z), *rot)):
        k = hou.Keyframe()
        k.setFrame(f)
        k.setValue(float(v))
        k.setExpression("linear()")
        cam.parm(parm).setKeyframe(k)

# the path, to check in the viewport (object hidden: not in the playblast)
path = [tuple(float(v) for v in centre + dist * direction(*pose(f))) for f in range(1, FRAMES + 1, 2)]
guide = hou.node("/obj").createNode("geo", "cam_path_guide")
py = guide.createNode("python")
py.parm("python").set(f"""geo = hou.pwd().geometry()
poly = geo.createPolygon(False)
for p in {path}:
    q = geo.createPoint()
    q.setPosition(p)
    poly.addVertex(q)
""")
py.setDisplayFlag(True)
guide.setDisplayFlag(False)

os.makedirs(OUT_DIR + "/frames", exist_ok=True)
rop = hou.node("/out/playblast")
rop.parm("camera").set(cam.path())
rop.parmTuple("f").deleteAllKeyframes()
rop.parmTuple("f").set((1, FRAMES, 1))
rop.parm("tres").set(True)
rop.parmTuple("res").set(RES)
rop.parm("picture").set(OUT_DIR + "/frames/$F4.png")
rop.parm("colorcorrect").set("ocio")
rop.parm("ociocolorspace").set("sRGB Encoded Rec.709 (sRGB)")   # plain sRGB, no ACES tone map
rop.parm("gamma").set(1.0)
rop.parm("bgimage").set("")
hou.hipFile.save(OUT_HIP)
rop.render()

mp4 = OUT_DIR + f"/electra_arc_v030_{FRAMES}f.mp4"
subprocess.check_call([ref.FFMPEG, "-hide_banner", "-loglevel", "error", "-nostdin", "-y", "-framerate", str(FPS),
                       "-i", OUT_DIR + "/frames/%04d.png", "-c:v", "libx264", "-crf", "12", "-pix_fmt", "yuv420p", mp4])
print("saved", OUT_HIP)
print("video", mp4)
