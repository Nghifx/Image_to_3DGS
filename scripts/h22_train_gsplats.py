"""Train Gaussian splats with Houdini 22's native ML Train GSplats TOP.

  hython h22_train_gsplats.py <colmap_dataset_dir> <run_name> [--strategy default|mcmc]
                              [--steps 30000] [--cap 1000000]

The dataset must be COLMAP style: <dir>/images/*.png (RGBA ok) + <dir>/sparse/0/*.bin.
Writes a .hip with the TOP network to houdini/<run_name>_h22.hip and the trained
splats under ml/<run_name>/... (see the printed GSplats folder).
"""
import argparse
import os
import subprocess
import sys
import time

import hou
import pdg

ROOT = os.environ.get("GSPLAT_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def auto_batch() -> int:
    """Images per training step, sized to this machine's GPU.

    Bigger batches give smoother gradients and keep the GPU busier, but every
    image in flight costs VRAM on top of the cached data set, so a card that is
    too small would run out of memory mid-training.
    """
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            text=True)
        gb = int(out.splitlines()[0].strip()) / 1024.0
    except Exception:
        return 1                       # unknown card: the safe default
    if gb >= 22:
        return 6
    if gb >= 15:
        return 4
    if gb >= 11:
        return 3
    return 1


PNG_MAGIC = bytes([137]) + b"PNG" + bytes([13, 10, 26, 10])
# training resolution: 1920 keeps full-frame 1080p video (the uncropped route) at full
# size; at 1600 it was halved to 960 and lost detail. ~120 such frames cached in VRAM
# are under 1 GB.
TRAIN_MAX_SIDE = 1920


def _png_size(path: str) -> tuple[int, int] | None:
    """Width and height straight out of a PNG header, no image library needed."""
    try:
        with open(path, "rb") as f:
            head = f.read(24)
        if head[:8] != PNG_MAGIC or head[12:16] != b"IHDR":
            return None
        return (int.from_bytes(head[16:20], "big"), int.from_bytes(head[20:24], "big"))
    except OSError:
        return None


def auto_downscale(dataset: str) -> int:
    """How much to shrink the training images.

    Caching images in VRAM is what makes training fast, so the images have to
    be small enough to fit there. 4K stills from a phone do not; half-size ones
    do, and the detail lost is far less than the time saved.
    """
    images = os.path.join(dataset, "images")
    if not os.path.isdir(images):
        return 1
    names = sorted(n for n in os.listdir(images) if n.lower().endswith(".png"))
    size = _png_size(os.path.join(images, names[0])) if names else None
    if size is None:
        return 1
    factor = 1
    while max(size) / factor > TRAIN_MAX_SIDE and factor < 8:
        factor *= 2
    return factor


def build(dataset: str, run: str, strategy: str, steps: int, cap: int, keep_world: bool,
          ckpt_interval: int = 2000, batch: int = 0, out_dir: str = "") -> hou.Node:
    topnet = hou.node("/obj").createNode("topnet", f"train_{run}")
    train = topnet.createNode("ml_traingsplats", "train")
    parms = {
        "basedirectory": (out_dir or os.path.join(ROOT, "ml")).replace("\\", "/"),
        "basename": run,
        "runname": f"{run}.{strategy}",
        "datasetfolder": os.path.dirname(dataset).replace("\\", "/"),
        "datasetbasename": os.path.basename(dataset),
        "source": "colmap",
        "datafactor": auto_downscale(dataset),
        "strategy": strategy,
        "capmax": cap,
        "maxsteps": steps,
        "alphasupervision": "2",   # random background composite: images carry alpha
        "autoopenviewer": False,
        "enableviewer": False,
        "savevideo": False,
        # Testing renders and writes a frame every few steps. It costs real
        # training time and disk, and nobody looks at those frames.
        "testgsplats": False,
        # The biggest speed win by far: hold the training images in VRAM instead
        # of reading them off disk again on every step.
        "cacheimages": True,
        # Loader threads feeding the GPU; the default only uses a quarter of the
        # cores, which starves training on a many-core machine.
        "numworkersmenu": "-1",        # equal to CPU count less one
        "maxbatchsize": batch or auto_batch(),
        # Progress lines on stdout drive the GSplat Studio progress bar, and the
        # periodic .ply lets the panel show the splats in the viewport while
        # training is still running.
        "logtostandardoutput": True,
        "ckptply": True,
        "plyckptinterval": ckpt_interval,
        # For COLMAP data, normalization also re-orients the scene (auto up axis);
        # off keeps the splats in the dataset's own (already aligned) world space.
        "normalize": not keep_world,
    }
    for name, value in parms.items():
        parm = train.parm(name)
        if parm is None:
            sys.exit(f"parameter {name} not found on ml_traingsplats")
        parm.set(value)
    train.setDisplayFlag(True)
    return train


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("run")
    ap.add_argument("--strategy", default="default", choices=("default", "mcmc"))
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--cap", type=int, default=1000000)
    ap.add_argument("--keep-world", action="store_true", help="no normalization / auto alignment")
    ap.add_argument("--ckpt-interval", type=int, default=2000,
                    help="steps between preview .ply checkpoints")
    ap.add_argument("--batch", type=int, default=0,
                    help="images per training step (0 = pick from GPU memory)")
    ap.add_argument("--out-dir", default="", help="parent folder of the run (default <root>/ml)")
    args = ap.parse_args()

    train = build(os.path.abspath(args.dataset), args.run, args.strategy, args.steps, args.cap,
                  args.keep_world, args.ckpt_interval, args.batch, os.path.abspath(args.out_dir) if args.out_dir else "")
    hip = os.path.join(ROOT, "houdini", f"{args.run}_h22.hip")
    hou.hipFile.save(hip)
    print("saved", hip)
    print("gsplats folder:", train.parm("outputfolder").eval())
    print("batch size:", train.parm("maxbatchsize").eval(),
          "| image downscale:", train.parm("datafactor").eval(),
          "| images cached to VRAM:", bool(train.parm("cacheimages").eval()), flush=True)

    t0 = time.time()
    train.cookWorkItems(block=True)
    items = train.getPDGNode().workItems
    for item in items:
        print("work item", item.name, item.state)
    print(f"training finished in {(time.time() - t0) / 60:.1f} min")
    hou.hipFile.save(hip)
    if not items or any(item.state != pdg.workItemState.CookedSuccess for item in items):
        sys.exit("training did not finish - see the log above")


if __name__ == "__main__":
    main()
