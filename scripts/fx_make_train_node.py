"""Recreate the GSplat training node inside a running Houdini (FX) session and start training.

In Houdini: Windows > Python Shell, then
    exec(open(r"E:\\DEV\\GSplatStudio\\Gaussian_Splatting\\scripts\\fx_make_train_node.py").read())

Same settings the GSplat Studio panel uses for step 4. Change NAME for another character.
Training runs in the background, so Houdini stays usable; progress shows in the node's
work item and in GSplats/<NAME>/<NAME>.default/logs/log.txt.
"""
import os
import sys

ROOT = r"E:\DEV\GSplatStudio\Gaussian_Splatting"
NAME = "khmer_01"
STEPS = 30000

os.environ["GSPLAT_ROOT"] = ROOT
for sub in ("scripts", r"houdini\gsplat_studio\python3.13libs"):
    path = os.path.join(ROOT, sub)
    if path not in sys.path:
        sys.path.insert(0, path)

import hou  # noqa: E402
import h22_train_gsplats as trainer  # noqa: E402

try:  # lets outside tools (Claude) drive this session
    import houdini_bridge
    houdini_bridge.start()
except Exception as err:
    print("bridge not started:", err)

old = hou.node(f"/obj/train_{NAME}")
if old is not None:
    old.destroy()

train = trainer.build(os.path.join(ROOT, "colmap", NAME), NAME, "default", STEPS, 1000000,
                      keep_world=True, ckpt_interval=2000, out_dir=os.path.join(ROOT, "GSplats"))
train.parent().layoutChildren()
train.cookWorkItems(block=False)
print("training started:", train.path(), "->", train.parm("outputfolder").eval())
