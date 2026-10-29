"""hython: grade + matte one chiLinh54 character for training, with the Pa Then COP net as the template.

  hython scripts/batch_grade.py <id> [key]

The net in houdini/pathen_v008_grade_cop.hip (/obj/GRADE_PATHEN) holds the approved look: grade matched to the
boom-up plate A002C010, saturation 0.85 and the matte choked 5 px (no white rim). Only the inputs change:
  rgb   = chars/<id>/frames/$F4.png        (the Seedance video, 720x1280)
  alpha = chars/<id>/alpha/$F4.png         (LTX-2.5 Alpha-Gen matte, same size)
  out   = chars/<id>/graded/$F4.png        (RGBA, straight alpha)
With "key" (a video shot on the blue screen of the Electra v018 path) the matte is pulled here instead: a chroma
key on the blue hue band and a despill of the blue on the edges, unchoked and blurred EDGE_SOFT_PX (soft, generous: screen leftovers are trimmed later).
The red tracking crosses survive the key as small islands; batch_dataset.py final drops them (largest island).
Saved as chars/<id>/<id>_grade_cop.hip so the CEO can open and retouch the grade per character.
"""
import json
import os
import sys

import hou

ROOT = os.environ.get("GSPLAT_ROOT", "E:/DEV/GSplatStudio/Gaussian_Splatting").replace("\\", "/")
# blue screen as Genjutsu renders it (006b, 2026-10-28): hue 197-212 deg, saturation 0.55-1, median RGB below
SCREEN_HUE = (185, 225)
SCREEN_RGB = (0.184, 0.49, 0.631)   # sRGB, 006b; replaced by chars/<id>/screen_rgb.json (batch_dataset.py screen)
EDGE_SOFT_PX = 6.0   # gaussian blur of the matte at 1080 wide (CEO 2026-10-28: soft edge, a blue rim is cleaned later)
CHOKE_PX = 0         # 006b choked 4 px and ate her skirt; leftover screen splats are cheaper to trim than lost cloth
GUIDE_IN_PX = 12     # the AI guide's edge wanders a few px; 12 px in it is surely her
GUIDE_OUT_PX = 10    # hair and fingers the guide clipped still get the key; crosses further out are dropped
GUIDE_BODY_PX = 3    # the guide itself, barely eroded, counts as her whatever the key says (keeps the skirt whole)
cid = sys.argv[1]
key_mode = len(sys.argv) > 2 and sys.argv[2] == "key"
char = f"{ROOT}/chars/{cid}"
if os.path.exists(f"{char}/screen_rgb.json"):
    SCREEN_RGB = tuple(json.load(open(f"{char}/screen_rgb.json")))
frames = len(os.listdir(f"{char}/frames"))
hou.hipFile.load(f"{ROOT}/houdini/pathen_v008_grade_cop.hip", suppress_save_prompt=True, ignore_load_warnings=True)
net = hou.node("/obj/GRADE_PATHEN")
net.setName(f"GRADE_{cid}", unique_name=True)
rgb = net.node("rgb")
rgb.parm("filename").set(f"{char}/frames/$F4.png")
net.node("alpha").parm("filename").set(f"{char}/alpha/$F4.png")
net.node("CHOKE_ALPHA").parm("radius").set(-5)
if key_mode:
    to_rgba = net.createNode("rgbtorgba", "to_rgba")   # the key cannot read a plain RGB layer
    to_rgba.setInput(0, rgb)
    key = net.createNode("chromakey", "BLUE_KEY")
    key.setInput(0, to_rgba)
    key.parm("huecircle1").set(SCREEN_HUE[0])
    key.parm("huecircle2").set(SCREEN_HUE[1])
    # floors from the 006b skirt (grey-blue checks on near-black cloth, 2026-10-28): at saturation 0.3 and no
    # luminance floor the checks and the black shoes were keyed out; the screen itself sits above 0.5 / 0.1
    key.parm("huecircle3").set(0.45)   # min saturation
    key.parm("lumrangex").set(0.07)    # min luminance (linear): darker pixels are her, not the screen
    key.parm("lumrolloff").set(0.03)
    key.parm("rolloff").set(1)
    key.parm("huerolloff").set(8)
    key.parm("satrolloff").set(0.1)
    key.parmTuple("screencolor").set(SCREEN_RGB)
    # the AI guide (batch_dataset.py guide) rides in the old LTX slot: deep inside it is her whatever the colour
    # (blue embroidery), outside its rim the key is dropped (red crosses); the key's own edge lives in between
    guide = net.node("alpha")
    guide.parm("filename").set(f"{char}/guide/$F4.png")
    inside = net.createNode("dilateerode", "GUIDE_INSIDE")
    inside.setInput(0, guide)
    inside.parm("radius").set(-GUIDE_IN_PX)
    near = net.createNode("dilateerode", "GUIDE_NEAR")
    near.setInput(0, guide)
    near.parm("radius").set(GUIDE_OUT_PX)
    key_near = net.createNode("blend", "KEY_NEAR")
    key_near.setInput(0, key, 1)   # output 1 = matte (1 = person)
    key_near.setInput(1, near)
    key_near.parm("mode").set("multiply")
    body = net.createNode("dilateerode", "GUIDE_BODY")
    body.setInput(0, guide)
    body.parm("radius").set(-GUIDE_BODY_PX)
    merged = net.createNode("blend", "KEY_PLUS_BODY")
    merged.setInput(0, key_near)
    merged.setInput(1, body)
    merged.parm("mode").set("max")
    edge_only = net.createNode("invert", "EDGE_ONLY")
    edge_only.setInput(0, inside)
    despill = net.createNode("despill", "DESPILL_BLUE")   # on the edge band only: her blue cloth keeps its colour
    despill.setInput(0, rgb)
    despill.setInput(2, edge_only)
    despill.parmTuple("col").set(SCREEN_RGB)
    choke = net.node("CHOKE_ALPHA")
    choke.setInput(0, merged)
    choke.parm("radius").set(-CHOKE_PX)
    soften = net.createNode("blur", "SOFTEN_EDGE")   # CEO 2026-10-28: the raw key edge reads too hard
    soften.setInput(0, choke)
    soften.parm("filter").set("gaussian")
    soften.parm("units").set("pixels")
    soften.parm("size_pixel").set(EDGE_SOFT_PX)
    net.node("join_rgba").setInput(1, soften)
    net.node("join_rgba").setInput(0, despill)
    net.layoutChildren()
rop = net.node("write_graded")
rop.parmTuple("f").set((1, frames, 1))
rop.parm("copoutput").set(f"{char}/graded/$F4.png")
hou.playbar.setFrameRange(1, frames)
hou.hipFile.save(f"{char}/{cid}_grade_cop.hip")
rop.render()
print("graded", frames, "frames ->", f"{char}/graded")
