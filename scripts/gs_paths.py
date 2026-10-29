"""Where the project and its outside tools live, so the folder can move to another machine.

ROOT is the folder that holds scripts/, so nothing depends on D:\\Gaussian_Splatting.
"""
import os
import shutil

ROOT = os.environ.get("GSPLAT_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_FFMPEG_CANDIDATES = (
    os.environ.get("GSS_FFMPEG", ""),
    os.path.join(ROOT, "tools", "ffmpeg", "ffmpeg.exe"),     # shipped with the project package
    shutil.which("ffmpeg") or "",
    r"C:\Program Files\Prism2\Tools\FFmpeg\bin\ffmpeg.exe",  # the original workstation
)


def find_ffmpeg() -> str:
    """First ffmpeg that exists; plain "ffmpeg" (a clear error at first use) if none does."""
    return next((c for c in _FFMPEG_CANDIDATES if c and os.path.isfile(c)), "ffmpeg")
