"""Convert a Nerfstudio dataset (transforms.json + images + init.ply) to a COLMAP one.

Used to feed the Houdini-rendered Wukong datasets to Houdini 22's ML Train GSplats,
which reads COLMAP (images/ + sparse/0/*.bin).

  tools/venv/Scripts/python scripts/nerfstudio_to_colmap.py data/body data/wukong_body_colmap
"""
import json
import os
import shutil
import sys
import tempfile

import numpy as np
import pycolmap

GL_TO_CV = np.diag([1.0, -1.0, -1.0])  # camera axes: OpenGL (-Z forward, +Y up) -> OpenCV (+Z forward, +Y down)


def quat_wxyz(r: np.ndarray) -> np.ndarray:
    """Rotation matrix -> unit quaternion (w, x, y, z)."""
    w = np.sqrt(max(0.0, 1.0 + r[0, 0] + r[1, 1] + r[2, 2])) / 2.0
    x = np.sqrt(max(0.0, 1.0 + r[0, 0] - r[1, 1] - r[2, 2])) / 2.0
    y = np.sqrt(max(0.0, 1.0 - r[0, 0] + r[1, 1] - r[2, 2])) / 2.0
    z = np.sqrt(max(0.0, 1.0 - r[0, 0] - r[1, 1] + r[2, 2])) / 2.0
    x = np.copysign(x, r[2, 1] - r[1, 2])
    y = np.copysign(y, r[0, 2] - r[2, 0])
    z = np.copysign(z, r[1, 0] - r[0, 1])
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


def read_ply_points(path: str) -> np.ndarray:
    with open(path, "rb") as f:
        n = 0
        while True:
            line = f.readline().decode().strip()
            if line.startswith("element vertex"):
                n = int(line.split()[-1])
            elif line == "end_header":
                break
        rec = np.frombuffer(f.read(), dtype=[("p", "<f4", 3), ("c", "u1", 3)], count=n)
    return rec["p"].astype(np.float64)


def write_text_model(src: str, text_dir: str, n_points: int) -> None:
    meta = json.load(open(os.path.join(src, "transforms.json")))
    w, h = int(meta["w"]), int(meta["h"])
    with open(os.path.join(text_dir, "cameras.txt"), "w") as f:
        f.write(f"1 PINHOLE {w} {h} {meta['fl_x']} {meta['fl_y']} {meta['cx']} {meta['cy']}\n")
    with open(os.path.join(text_dir, "images.txt"), "w") as f:
        for i, frame in enumerate(meta["frames"], start=1):
            c2w = np.array(frame["transform_matrix"], dtype=np.float64)
            r_c2w = c2w[:3, :3] @ GL_TO_CV
            r_w2c = r_c2w.T
            t_w2c = -r_w2c @ c2w[:3, 3]
            q = quat_wxyz(r_w2c)
            name = os.path.basename(frame["file_path"])
            f.write(f"{i} {q[0]} {q[1]} {q[2]} {q[3]} {t_w2c[0]} {t_w2c[1]} {t_w2c[2]} 1 {name}\n\n")
    pts = read_ply_points(os.path.join(src, "init.ply"))
    rng = np.random.default_rng(0)
    pts = pts[rng.choice(len(pts), size=min(n_points, len(pts)), replace=False)]
    with open(os.path.join(text_dir, "points3D.txt"), "w") as f:
        for i, p in enumerate(pts, start=1):
            f.write(f"{i} {p[0]} {p[1]} {p[2]} 128 128 128 0\n")


def main(src: str, dst: str, n_points: int = 100_000) -> None:
    sparse = os.path.join(dst, "sparse", "0")
    os.makedirs(sparse, exist_ok=True)
    with tempfile.TemporaryDirectory() as text_dir:
        write_text_model(src, text_dir, n_points)
        rec = pycolmap.Reconstruction(text_dir)
        rec.write(sparse)  # binary
    print(f"cameras {rec.num_cameras()}, images {rec.num_images()}, points {rec.num_points3D()}")
    shutil.copytree(os.path.join(src, "images"), os.path.join(dst, "images"), dirs_exist_ok=True)
    print("dataset:", dst)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
