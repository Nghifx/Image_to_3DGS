"""Step 3: train Gaussian splats for one part with Brush (GPU).

  python gs_train.py body
  python gs_train.py cloth --steps 30000

Output: output/<part>/<part>_<iter>.ply  (standard 3DGS ply, SH degree 3)
"""
import argparse
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gs_config as cfg  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("part", choices=cfg.PARTS)
    ap.add_argument("--steps", type=int, default=30000)
    args = ap.parse_args()

    data_dir = os.path.join(cfg.DATA_DIR, args.part)
    with open(os.path.join(data_dir, "transforms.json")) as f:
        views = len(json.load(f)["frames"])
    missing = [v for v in range(views)
               if not os.path.exists(os.path.join(data_dir, "images", f"{cfg.CAM_START_FRAME + v:04d}.png"))]
    if missing:
        sys.exit(f"{len(missing)} of {views} views not rendered yet for '{args.part}'")

    out_dir = os.path.join(cfg.OUTPUT_DIR, args.part)
    os.makedirs(out_dir, exist_ok=True)
    cmd = [
        cfg.BRUSH_EXE, data_dir,
        "--total-steps", str(args.steps),
        "--refine-every", str(views),        # ~ one pass over the views
        "--max-resolution", str(cfg.RESOLUTION),
        "--sh-degree", "3",
        "--eval-split-every", "45",          # 6 held-out views to measure quality
        "--eval-every", "5000",
        "--eval-save-to-disk",
        "--export-every", "10000",
        "--export-path", out_dir,
        "--export-name", f"{args.part}_{{iter}}.ply",
    ]
    print(" ".join(cmd), flush=True)
    sys.exit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
