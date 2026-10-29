"""Clean Gaussian-splat renders with NVIDIA Fixer (Cosmos-Predict 0.6B, single-step diffusion).

Usage (Fixer venv python, NOT tools/venv or hython):
    tools/fixer_venv/Scripts/python scripts/fixer_run.py <in_dir> <out_dir> [--timestep 250] [--bg 0.5]

Input : <in_dir>/*.png|*.jpg  (RGBA ok: composited on a flat grey of value --bg)
Output: <out_dir>/<same name>.png, RGB, at the input's original size.

Needs tools/Fixer (git clone of nv-tlabs/Fixer + `models/` from hf nvidia/Fixer, which is
NVIDIA Open Model License, commercial OK) and tools/fixer_venv (torch cu128 + cosmos-predict2
installed --no-deps). transformer-engine and megatron-core don't build on Windows, so
tools/Fixer/win_shims supplies pure-PyTorch stand-ins (RMSNorm, SDPA attention, RoPE).

The model works at 576x1024. Upstream feeds landscape 1024x576; our renders are portrait, so we
feed portrait 576 wide x 1024 high (DiT uses RoPE, any multiple of 16 works) and stretch back.
torch.compile is skipped (needs triton on Windows); bf16 eager is fast enough here.
"""
import argparse
import glob
import os
import sys
import time
import types

ROOT = os.environ.get("GSPLAT_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIXER = os.path.join(ROOT, "tools", "Fixer")
sys.path[:0] = [os.path.join(FIXER, "win_shims"), os.path.join(FIXER, "src")]

# The Qwen VLM text encoder asserts flash-attn at import; Fixer never uses a text encoder,
# so hand cosmos a placeholder module instead of the real one.
_qwen = types.ModuleType("imaginaire.networks.qwen2_5_vl")
_qwen.Qwen2_5_VisionTransformerPretrainedModel = _qwen.Qwen2_5_VLModel = _qwen.get_rope_index = None
sys.modules["imaginaire.networks.qwen2_5_vl"] = _qwen

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

W, H = 576, 1024  # model input, portrait


def load_model(timestep):
    import pix2pix_turbo_nocond_cosmos_base_faster_tokenizer as fx

    fx.config.dit_path = os.path.join(FIXER, "models", "base", "model_fast_tokenizer.pt")
    fx.config.tokenizer["vae_pth"] = os.path.join(FIXER, "models", "base", "tokenizer_fast.pth")
    model = fx.Pix2Pix_Turbo(
        pretrained_path=os.path.join(FIXER, "models", "pretrained", "pretrained_fixer.pkl"),
        timestep=timestep,
    ).to(device="cuda", dtype=torch.bfloat16)
    model.set_eval()
    return model


def to_input(path, bg):
    im = Image.open(path)
    size = im.size
    rgba = np.asarray(im.convert("RGBA"), dtype=np.float32) / 255.0
    rgb = rgba[..., :3] * rgba[..., 3:] + bg * (1.0 - rgba[..., 3:])
    rgb = Image.fromarray((rgb * 255 + 0.5).astype(np.uint8)).resize((W, H), Image.BICUBIC)
    t = torch.from_numpy(np.asarray(rgb, dtype=np.float32) / 127.5 - 1.0).permute(2, 0, 1)[None]
    return t.to("cuda", torch.bfloat16), size


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("in_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--timestep", type=int, default=250, help="250 = upstream training/inference value")
    ap.add_argument("--bg", type=float, default=0.5, help="grey level behind transparent pixels")
    a = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(a.in_dir, "*.png")) + glob.glob(os.path.join(a.in_dir, "*.jpg")))
    if not paths:
        sys.exit(f"no images in {a.in_dir}")
    os.makedirs(a.out_dir, exist_ok=True)

    t0 = time.time()
    model = load_model(a.timestep)
    print(f"model loaded in {time.time() - t0:.1f}s", flush=True)

    times = []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for i, p in enumerate(paths):
            x, size = to_input(p, a.bg)
            torch.cuda.synchronize()
            t = time.time()
            y = model(x)
            torch.cuda.synchronize()
            times.append(time.time() - t)
            y = ((y[0].float().clamp(-1, 1) + 1) * 127.5 + 0.5).byte().permute(1, 2, 0).cpu().numpy()
            name = os.path.splitext(os.path.basename(p))[0] + ".png"
            Image.fromarray(y).resize(size, Image.BICUBIC).save(os.path.join(a.out_dir, name))
            print(f"GSS_PROGRESS {100 * (i + 1) // len(paths)} {name} {times[-1]:.2f}s", flush=True)

    steady = times[1:] or times
    print(f"done {len(paths)} images, first {times[0]:.2f}s, then {sum(steady) / len(steady):.3f}s/img, "
          f"peak VRAM alloc {torch.cuda.max_memory_allocated() / 2**30:.2f} GB, "
          f"reserved {torch.cuda.max_memory_reserved() / 2**30:.2f} GB")


if __name__ == "__main__":
    main()
