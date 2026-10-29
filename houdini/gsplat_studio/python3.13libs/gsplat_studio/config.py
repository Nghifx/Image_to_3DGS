"""Paths, defaults and license limits for GSplat Studio."""
import os

import hou

# project folder = four levels above this file (<root>/houdini/gsplat_studio/python3.13libs/gsplat_studio)
ROOT = os.environ.get("GSPLAT_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), *[".."] * 4))
SCRIPTS = os.path.join(ROOT, "scripts")
DATA_DIR = os.path.join(ROOT, "data")
ML_DIR = os.path.join(ROOT, "ml")
COLMAP_DIR = os.path.join(ROOT, "colmap")
GSPLATS_DIR = os.path.join(ROOT, "GSplats")
LOGS_DIR = os.path.join(ROOT, "logs")

HYTHON = os.path.join(hou.getenv("HFS") or "", "bin", "hython.exe")
VENV_PYTHON = os.path.join(ROOT, "tools", "venv", "Scripts", "python.exe")

# Dataset defaults. Kept deliberately few: everything else is derived.
DEFAULT_FRAMES = 160          # stills pulled out of a 360 video
DEFAULT_VIEWS = 160           # camera positions around an object
DEFAULT_HEIGHT_M = 1.65       # real-world height, so the splat lands at Houdini scale
# Chosen on khmer_girl (2026-10-23): every frame, background kept, mcmc, 50k steps
# was clearly the sharpest; ~1 h per character on an RTX 5070 Ti. The kept set
# (checker room) has to be cut away from the splat afterwards.
DEFAULT_STEPS = 50000
DEFAULT_STRATEGY = "mcmc"
KEEP_BG = True                # steps 2-3 keep the set: no background cut
CKPT_INTERVAL = 2000          # how often a preview .ply is written while training
AI_ROUNDS = 3                 # most times Claude may correct the camera angles

# Render quality presets: (samples per pixel, extra close-up views as a fraction)
QUALITY = {
    "Nhanh": (24, 0.0),
    "Vua": (64, 0.5),
    "Cao": (128, 0.75),
}
DEFAULT_QUALITY = "Vua"

# Apprentice renders top out at 1280x720 and get a watermark burned into the
# bottom-right corner. We render at the cap and blank the watermark's alpha, so
# the trainer treats that patch as background instead of learning it as geometry.
APPRENTICE_RES = (1280, 720)
WATERMARK_BOX = (980, 620, 1280, 720)  # x0, y0, x1, y1 in the 1280x720 frame
FULL_RES = (2048, 2048)


def is_apprentice() -> bool:
    return hou.licenseCategory() == hou.licenseCategoryType.Apprentice


def render_resolution() -> tuple[int, int]:
    return APPRENTICE_RES if is_apprentice() else FULL_RES


def run_dir(name: str, strategy: str = DEFAULT_STRATEGY, base: str = ML_DIR) -> str:
    """Folder the trainer writes a run into."""
    return os.path.join(base, name, f"{name}.{strategy}")
