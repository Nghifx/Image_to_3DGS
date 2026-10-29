"""venv: step 3 of the shot fill - a training set = her video frames + the filled shot views (gs_shot_fill.py).

  tools/venv/Scripts/python scripts/shot_augment.py <run> [copies=1]

colmap/<run>_shot = colmap/<run> (sparse model, image links) plus every GSplats/<run>/shot/fill/*.png as one more
image with its own PINHOLE camera (each fill has its own zoom and lens shift). Their poses come in ply space and
go back to the COLMAP world through the same world turn about X (R_col = R_ply @ turn). <copies> > 1 repeats each
fill under new names, to weigh the shot's angles more against the 200-odd video frames.
Train it as usual: hython scripts/h22_train_gsplats.py colmap/<run>_shot <run>_shot --keep-world ...
"""
import json
import math
import os
import shutil
import sys

import numpy as np
import pycolmap

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def rot_x(deg: float) -> np.ndarray:
    a = math.radians(deg)
    return np.array([[1, 0, 0], [0, math.cos(a), -math.sin(a)], [0, math.sin(a), math.cos(a)]])


def main() -> None:
    run = sys.argv[1]
    copies = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    src, dst = f"{ROOT}/colmap/{run}", f"{ROOT}/colmap/{run}_shot"
    fill_dir = f"{ROOT}/GSplats/{run}/shot/fill"
    fill = json.load(open(f"{fill_dir}/fill.json"))
    turn = rot_x(fill["world_turn_x"])
    # COLMAP 4 keeps rigs and frames next to cameras and images: each fill gets its own camera, rig and frame,
    # appended to the text form of the model, which is then read back and written as binary
    rec = pycolmap.Reconstruction(f"{src}/sparse/0")
    shutil.rmtree(dst, ignore_errors=True)
    os.makedirs(f"{dst}/images")
    for n in os.listdir(f"{src}/images"):
        os.link(f"{src}/images/{n}", f"{dst}/images/{n}")
    txt = f"{dst}/_txt"
    os.makedirs(txt)
    rec.write_text(txt)
    cam_id, img_id = max(rec.cameras) + 1, max(rec.images) + 1
    rig_id = max(r.rig_id for r in rec.rigs.values()) + 1
    frame_id = max(f.frame_id for f in rec.frames.values()) + 1
    out = {k: open(f"{txt}/{k}.txt", "a") for k in ("cameras", "rigs", "frames", "images")}
    for c in range(copies):
        for v in fill["views"]:
            name = v["name"] if c == 0 else v["name"].replace(".png", f"_{c}.png")
            shutil.copy2(f"{fill_dir}/{v['name']}", f"{dst}/images/{name}")
            qx, qy, qz, qw = pycolmap.Rotation3d(np.array(v["R"]) @ turn).quat
            tx, ty, tz = v["t"]
            pose = f"{qw} {qx} {qy} {qz} {tx} {ty} {tz}"
            print(f"{cam_id} PINHOLE {fill['width']} {fill['height']} {v['fx']} {v['fy']} {v['cx']} {v['cy']}", file=out["cameras"])
            print(f"{rig_id} 1 CAMERA {cam_id}", file=out["rigs"])
            print(f"{frame_id} {rig_id} {pose} 1 CAMERA {cam_id} {img_id}", file=out["frames"])
            print(f"{img_id} {pose} {cam_id} {name}", file=out["images"])
            print("", file=out["images"])   # no 2D points
            cam_id, img_id, rig_id, frame_id = cam_id + 1, img_id + 1, rig_id + 1, frame_id + 1
    for f in out.values():
        f.close()
    rec = pycolmap.Reconstruction(txt)
    os.makedirs(f"{dst}/sparse/0")
    rec.write(f"{dst}/sparse/0")
    shutil.rmtree(txt)
    print(f"{dst}: {rec.num_reg_images()} images ({len(fill['views']) * copies} filled shot views), {rec.num_points3D()} points")


if __name__ == "__main__":
    main()
