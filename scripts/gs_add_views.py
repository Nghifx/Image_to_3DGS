"""venv: colmap/<run> + Fixer-cleaned novel views -> colmap/<out_run> (Difix3D+ style: the fixed renders join training).
python gs_add_views.py <run> <novel_dir> <out_run>
novel_dir/views.json: views with R, t (world->camera). Orbit views (novel_views.py): raw/ RGBA render + fixed/ Fixer RGB,
the run's camera 1. Shot views (shot_views.py): fx fy cx cy per view, fixed/ already RGBA."""
import json, os, shutil, sys
import cv2, numpy as np, pycolmap
from scipy.spatial.transform import Rotation
root = os.environ.get("GSPLAT_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))).replace("\\", "/")
run, novel, out_run = sys.argv[1:4]
out = f"{root}/colmap/{out_run}"
shutil.rmtree(out, ignore_errors=True)
os.makedirs(f"{out}/images"); os.makedirs(f"{out}/text")
rec = pycolmap.Reconstruction(f"{root}/colmap/{run}/sparse/0")
nv = json.load(open(f"{novel}/views.json"))
fc = open(f"{out}/text/cameras.txt", "w")
for cid, c in rec.cameras.items():
    fc.write(f"{cid} {c.model.name} {c.width} {c.height} {' '.join(map(str, c.params))}\n")
next_cam = max(rec.cameras) + 1
with open(f"{out}/text/images.txt", "w") as f:
    k = 1
    for im in sorted(rec.images.values(), key=lambda i: i.name):
        q, t = im.cam_from_world().rotation.quat, im.cam_from_world().translation
        f.write(f"{k} {q[3]} {q[0]} {q[1]} {q[2]} {t[0]} {t[1]} {t[2]} {im.camera_id} {im.name}\n\n"); k += 1
        os.link(f"{root}/colmap/{run}/images/{im.name}", f"{out}/images/{im.name}")
    for v in nv["views"]:
        q = Rotation.from_matrix(np.array(v["R"])).as_quat()   # x y z w
        t = v["t"]
        if "fx" in v:
            fc.write(f"{next_cam} PINHOLE {nv['width']} {nv['height']} {v['fx']} {v['fy']} {v['cx']} {v['cy']}\n")
            cid, next_cam = next_cam, next_cam + 1
            shutil.copy2(f"{novel}/fixed/{v['name']}", f"{out}/images/{v['name']}")
        else:
            cid = 1
            rgb = cv2.imread(f"{novel}/fixed/{v['name']}")
            a = cv2.imread(f"{novel}/raw/{v['name']}", cv2.IMREAD_UNCHANGED)[..., 3]
            cv2.imwrite(f"{out}/images/{v['name']}", np.dstack([cv2.resize(rgb, a.shape[::-1]), a]))
        f.write(f"{k} {q[3]} {q[0]} {q[1]} {q[2]} {t[0]} {t[1]} {t[2]} {cid} {v['name']}\n\n"); k += 1
fc.close()
with open(f"{out}/text/points3D.txt", "w") as f:
    for i, p in rec.points3D.items():
        f.write(f"{i} {p.xyz[0]} {p.xyz[1]} {p.xyz[2]} {p.color[0]} {p.color[1]} {p.color[2]} {p.error}\n")
r2 = pycolmap.Reconstruction(f"{out}/text")
os.makedirs(f"{out}/sparse/0"); r2.write(f"{out}/sparse/0"); shutil.rmtree(f"{out}/text")
os.makedirs(f"{root}/colmap/_work/{out_run}", exist_ok=True)
shutil.copy2(f"{root}/colmap/_work/{run}/cameras.json", f"{root}/colmap/_work/{out_run}/")
print(r2.num_images(), "images,", r2.num_cameras(), "cameras ->", out)
