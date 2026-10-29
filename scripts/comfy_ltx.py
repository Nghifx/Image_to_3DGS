"""venv: LTX-2.5 IC-LoRA on Comfy Cloud (Alpha-Gen matte by default, or the 2x Pixel-Spatial-Upscaler) for a video.

  COMFY_CLOUD_API_KEY=... python scripts/comfy_ltx.py <video.mp4> <out_dir> <lora_name> [chunk=121] [max_frames]
  (env COMFY_OUT_SCALE=2 + COMFY_PROMPT with the Pixel-Spatial-Upscaler LoRA turns it into a 2x video upscaler)
  (env COMFY_START=<0-based frame> runs just that one chunk, COMFY_SEED=<n> changes the noise seed,
   COMFY_RESUME=1 carries on from the frames already in out_dir after a failed run)

The graph is the official LTX-2.5_V2V_ICLoRA_Single_Stage_Distilled example flattened to API
format (no prompt enhancer, no start image): the source frames go in as the IC-LoRA guide,
an empty audio track fills the audio half of the AV latent, 8 distilled steps, tiled decode,
int8 transformer and text encoder (the bf16 pair hit Comfy Cloud's job time limit on 121 frames),
PNG frames out. The video is cut into chunks of `chunk` frames (8n+1; mattes drift past ~145)
that overlap by one frame; frames come back as out_dir/0001.png ...
"""
import json
import os
import sys
import time
import urllib.parse

import requests

BASE = "https://cloud.comfy.org"
KEY = os.environ["COMFY_CLOUD_API_KEY"]
HDR = {"X-API-Key": KEY}
MAX_WAIT = int(os.environ.get("COMFY_MAX_WAIT", "1200"))   # seconds per chunk before we cancel it
SCALE = int(os.environ.get("COMFY_OUT_SCALE", "1"))   # 2 for the Pixel-Spatial-Upscaler IC-LoRA (output = 2x the guide)
PROMPT = os.environ.get("COMFY_PROMPT", "")            # the matte needs none; the upscaler gets a short description
PREFIX = os.environ.get("COMFY_PREFIX", "pathen_alpha")
SIGMAS = "1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0"


def graph(video: str, lora: str, start: int, length: int, width: int, height: int, fps: float) -> dict:
    n = lambda cls, **inputs: {"class_type": cls, "inputs": inputs}   # noqa: E731
    return {
        "1": n("UNETLoader", unet_name="ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors", weight_dtype="default"),
        "2": n("LTXICLoRALoaderModelOnly", model=["1", 0], lora_name=lora, strength_model=1.0),
        "3": n("VAELoader", vae_name="ltx-2.5-video-vae-bf16.safetensors"),
        "4": n("VAELoader", vae_name="ltx-2.5-audio-vae-bf16.safetensors"),
        "5": n("CLIPLoader", clip_name="gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors", type="ltxv"),
        "6": n("CLIPTextEncode", clip=["5", 0], text=PROMPT),
        "7": n("CLIPTextEncode", clip=["5", 0], text=""),
        "8": n("LTXVConditioning", positive=["6", 0], negative=["7", 0], frame_rate=fps),
        "10": n("LoadVideo", file=video),
        "11": n("GetVideoComponents", video=["10", 0]),
        "12": n("ImageFromBatch", image=["11", 0], batch_index=start, length=length),
        "13": n("EmptyLTXVLatentVideo", width=width * SCALE, height=height * SCALE, length=length, batch_size=1),
        "14": n("LTXAddVideoICLoRAGuide", positive=["8", 0], negative=["8", 1], vae=["3", 0], latent=["13", 0],
                image=["12", 0], frame_idx=0, strength=1.0, latent_downscale_factor=["2", 1], crop="disabled",
                use_tiled_encode=False, tile_size=256, tile_overlap=64),
        "15": n("EmptyAudio", duration=length / fps, sample_rate=44100, channels=2),
        "16": n("VAEEncodeAudio", audio=["15", 0], vae=["4", 0]),
        "17": n("LTXVSetAudioRefTokens", positive=["14", 0], negative=["14", 1], audio_latent=["16", 0]),
        "18": n("LTXVConcatAVLatent", video_latent=["14", 2], audio_latent=["17", 2]),
        "20": n("RandomNoise", noise_seed=int(os.environ.get("COMFY_SEED", "42"))),   # another seed redoes a chunk that came back as noise
        "21": n("CFGGuider", model=["2", 0], positive=["17", 0], negative=["17", 1], cfg=1.0),
        "22": n("KSamplerSelect", sampler_name="euler_ancestral"),
        "23": n("ManualSigmas", sigmas=SIGMAS),
        "24": n("SamplerCustomAdvanced", noise=["20", 0], guider=["21", 0], sampler=["22", 0], sigmas=["23", 0],
                latent_image=["18", 0]),
        "25": n("LTXVSeparateAVLatent", av_latent=["24", 0]),
        "26": n("LTXVCropGuides", positive=["17", 0], negative=["17", 1], latent=["25", 0]),
        "27": n("VAEDecodeTiled", samples=["26", 2], vae=["3", 0], tile_size=512, overlap=64, temporal_size=64,
                temporal_overlap=8),
        "28": n("SaveImage", images=["27", 0], filename_prefix=f"{PREFIX}_{start:04d}"),
    }


def run(prompt: dict) -> list:
    r = requests.post(f"{BASE}/api/prompt", headers=HDR, json={"prompt": prompt}, timeout=60)
    if r.status_code != 200:
        sys.exit(f"submit failed {r.status_code}: {r.text[:2000]}")
    pid = r.json()["prompt_id"]
    print("job", pid)
    t0 = time.time()
    while True:
        if time.time() - t0 > MAX_WAIT:   # don't burn GPU credits on a stuck job
            requests.post(f"{BASE}/api/interrupt", headers=HDR, timeout=30)
            requests.post(f"{BASE}/api/queue", headers=HDR, json={"delete": [pid]}, timeout=30)
            sys.exit(f"job {pid} passed {MAX_WAIT}s - interrupted")
        s = requests.get(f"{BASE}/api/job/{pid}/status", headers=HDR, timeout=30).json()
        st = s.get("status")
        if st in ("completed", "success"):
            break
        if st in ("failed", "error", "cancelled") or "error" in str(st):   # incl. non_retryable_error (bad input)
            sys.exit(f"job {st}: {json.dumps(s)[:2000]}")
        time.sleep(10)
        print(f"  {st} {time.time() - t0:.0f}s", flush=True) if int(time.time() - t0) % 120 < 10 else None
    h = requests.get(f"{BASE}/api/history_v2/{pid}", headers=HDR, timeout=60).json()
    entry = h.get(pid, h)
    # only the SaveImage node, in the order it saved them: cloud file names are content hashes, so sorting by
    # name scrambles the frames (and node 10 lists the input video too)
    return entry.get("outputs", {}).get("28", {}).get("images", [])


video, out_dir, lora = sys.argv[1:4]
chunk = int(sys.argv[4]) if len(sys.argv) > 4 else 121
max_frames = int(sys.argv[5]) if len(sys.argv) > 5 else None   # quick tests on the first frames only
import cv2  # noqa: E402

cap = cv2.VideoCapture(video)
total, fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), cap.get(cv2.CAP_PROP_FPS)
w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
cap.release()
total = min(total, max_frames) if max_frames else total
os.makedirs(out_dir, exist_ok=True)
with open(video, "rb") as f:
    up = requests.post(f"{BASE}/api/upload/image", headers=HDR, files={"image": (os.path.basename(video), f, "video/mp4")},
                       data={"overwrite": "true"}, timeout=300)
if up.status_code != 200:
    sys.exit(f"upload failed {up.status_code}: {up.text[:1000]}")
name = up.json()["name"]
print(f"uploaded {name}: {total} frames {w}x{h} @ {fps}")
start = int(os.environ.get("COMFY_START", "0"))   # 0-based first frame: redo one bad chunk with COMFY_START + chunk
if os.environ.get("COMFY_RESUME"):   # carry on after a failed chunk: from the last frame already saved (chunks overlap by one)
    start = max(0, len([n for n in os.listdir(out_dir) if n.endswith(".png")]) - 1)
while start < total - 1:
    length = min(chunk, total - start)
    if length < 9:   # a 1-frame tail breaks the empty-audio branch: redo the last 9 frames instead
        start, length = total - 9, 9
    length = (length - 1) // 8 * 8 + 1                                     # LTX wants 8n+1 frames
    imgs = run(graph(name, lora, start, length, w, h, fps))
    imgs = imgs[:length]   # the decode can hand back one frame more than asked
    for k, im in enumerate(imgs):
        q = urllib.parse.urlencode({"filename": im["filename"], "subfolder": im.get("subfolder", ""), "type": im.get("type", "output")})
        data = requests.get(f"{BASE}/api/view?{q}", headers=HDR, timeout=120).content
        open(os.path.join(out_dir, f"{start + k + 1:04d}.png"), "wb").write(data)   # overlap frame: later chunk wins
    print(f"frames {start + 1}-{start + len(imgs)} saved")
    start += length - 1
    if os.environ.get("COMFY_START"):   # redo mode: that one chunk only
        break
print("done", out_dir)
