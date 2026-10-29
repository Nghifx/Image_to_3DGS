"""Zip the project for another workstation: code, plugin, source media, finished datasets.

  tools/venv/Scripts/python scripts/make_package.py [--out D:\\GSplatStudio_package] [--name khmer_01 --name ghost_01]

Left out on purpose (setup.bat or Houdini rebuild them, or they are old experiments):
tools/venv, ml/, output/, logs/, data/_work, colmap/_work, colmap/<name>/images
(setup.bat links them back from data/<name>/images), Brush/LichtFeld leftovers.
"""
import argparse
import datetime
import os
import sys
import zipfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gs_paths import find_ffmpeg  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRISM_FFMPEG = find_ffmpeg()   # whichever ffmpeg this machine has (Prism's copy is gone, 2026-10-29)
REMBG_MODEL = os.path.join(os.path.expanduser("~"), ".rembg", "models", "isnet-general-use", "isnet-general-use.onnx")
STORED = (".png", ".jpg", ".jpeg", ".mp4", ".webp", ".onnx", ".zip", ".ply", ".bin")  # already compressed or binary
MAX_HIP_MB = 5
SKIP_DIRS = {"__pycache__", ".claude"}


def walk(rel: str):
    base = os.path.join(ROOT, rel)
    for folder, dirs, files in os.walk(base):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for f in files:
            full = os.path.join(folder, f)
            yield full, os.path.relpath(full, ROOT)


def collect(names: list[str]) -> list[tuple[str, str]]:
    items = []
    for f in ("setup.bat", "setup.ps1", "requirements.txt", "SETUP.md"):
        items.append((os.path.join(ROOT, f), f))
    for rel in ("scripts", "houdini/gsplat_studio", "houdini/video", "khmer_girl", "ghost_wraith"):
        items += list(walk(rel))
    for f in os.listdir(os.path.join(ROOT, "houdini")):
        full = os.path.join(ROOT, "houdini", f)
        if f.endswith((".hip", ".hipnc")) and os.path.getsize(full) < MAX_HIP_MB * 2**20:
            items.append((full, os.path.join("houdini", f)))
    for name in names:
        items += list(walk(os.path.join("data", name)))
        items += list(walk(os.path.join("colmap", name, "sparse")))
        items += list(walk(os.path.join("GSplats", name, f"{name}.default", "gsplats")))
    items.append((PRISM_FFMPEG, os.path.join("tools", "ffmpeg", "ffmpeg.exe")))
    items.append((REMBG_MODEL, os.path.join("tools", "models", "isnet-general-use.onnx")))
    missing = [src for src, _ in items if not os.path.isfile(src)]
    if missing:
        raise SystemExit("missing: " + ", ".join(missing))
    return items


def collect_workflow() -> list[tuple[str, str]]:
    items = []
    for f in ("setup.bat", "setup.ps1", "requirements.txt", "SETUP.md", "setup_fixer.bat", "QUY_TRINH_NHAN_VAT.txt"):
        items.append((os.path.join(ROOT, f), f))
    items += [i for i in walk("scripts") if ".bak_" not in i[1]]
    items += list(walk("houdini/gsplat_studio"))
    for hip in ("electra_grid_apose_240f.hip", "electra_arc_v008_240f.hip", "electra_arc_v030_480f.hip"):
        items.append((os.path.join(ROOT, "houdini", hip), os.path.join("houdini", hip)))
    ref = os.path.join("khmer_girl", "ref_electra_arc_v030_20s", "electra_arc_v030_480f.mp4")
    items.append((os.path.join(ROOT, ref), ref))
    items += [i for i in walk(os.path.join("chars", "011", "genjutsu_inputs")) if "_cu" not in i[1]]
    items += [i for i in walk(os.path.join("tools", "Fixer")) if os.sep + ".git" + os.sep not in i[0]]
    items.append((PRISM_FFMPEG, os.path.join("tools", "ffmpeg", "ffmpeg.exe")))
    items.append((REMBG_MODEL, os.path.join("tools", "models", "isnet-general-use.onnx")))
    missing = [src for src, _ in items if not os.path.isfile(src)]
    if missing:
        raise SystemExit("missing: " + ", ".join(missing))
    return items


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(os.path.splitdrive(ROOT)[0] + os.sep, "GSplatStudio_package"))
    ap.add_argument("--name", action="append", default=None, help="finished dataset to include (repeatable)")
    ap.add_argument("--workflow", action="store_true", help="new-character workflow only (see QUY_TRINH_NHAN_VAT.txt)")
    args = ap.parse_args()
    items = collect_workflow() if args.workflow else collect(args.name or ["khmer_01", "ghost_01"])
    os.makedirs(args.out, exist_ok=True)
    zip_path = os.path.join(args.out, f"GSplatStudio_{'workflow_' if args.workflow else ''}{datetime.date.today():%Y-%m-%d}.zip")
    total = sum(os.path.getsize(src) for src, _ in items)
    print(f"{len(items)} files, {total / 2**20:.0f} MB -> {zip_path}")
    with zipfile.ZipFile(zip_path, "w", allowZip64=True) as z:
        for i, (src, arc) in enumerate(items, 1):
            method = zipfile.ZIP_STORED if arc.lower().endswith(STORED) else zipfile.ZIP_DEFLATED
            z.write(src, os.path.join("Gaussian_Splatting", arc), compress_type=method)
            if i % 200 == 0:
                print(f"  {i}/{len(items)}", flush=True)
    with open(os.path.join(args.out, "manifest.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(sorted(arc.replace("\\", "/") for _, arc in items)))
    print(f"done: {os.path.getsize(zip_path) / 2**20:.0f} MB")


if __name__ == "__main__":
    main()
