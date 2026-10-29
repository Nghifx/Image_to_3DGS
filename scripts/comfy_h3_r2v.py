"""venv: MiniMax H3 (open weights) Reference-to-Video on Comfy Cloud, the cheap stand-in for the Seedance step.

  COMFY_CLOUD_API_KEY=... python scripts/comfy_h3_r2v.py <char_id> <seconds> <out.mp4> [width height steps]

Inputs as for Seedance (scripts/batch_gen.sh): chars/<id>/refs/ref_1..4.jpg, chars/<id>/portrait_seedream5pro.png,
the Electra camera move (khmer_girl/ref_electra_arc_v008_10s, retimed to <seconds>) and chars/<id>/seedance_prompt.txt
with Seedance's @Image n / @Video 1 tags turned into H3's <Picture n> / <Video 1>.
Graph = the Comfy template video_minimax_h3_r2v (int8 ref2va model, Qwen3-VL encoder, res_multistep, simple
schedule, no CFG), its audio output kept because the template's decode expects it. Default 544x960, 20 steps.
"""
import json
import os
import subprocess
import sys
import time

import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gs_paths  # noqa: E402

BASE = "https://cloud.comfy.org"
HDR = {"X-API-Key": os.environ["COMFY_CLOUD_API_KEY"]}
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAX_WAIT = int(os.environ.get("COMFY_MAX_WAIT", "1500"))
# the camera move; env H3_REF_VIDEO picks another one (e.g. the 15 s v013 sweep), retimed to <seconds> either way
REF_VIDEO = os.environ.get("H3_REF_VIDEO") or os.path.join(ROOT, "khmer_girl", "ref_electra_arc_v008_10s", "electra_arc_v008_240f.mp4")


def upload(path: str, mime: str) -> str:
    with open(path, "rb") as f:
        r = requests.post(f"{BASE}/api/upload/image", headers=HDR, files={"image": (os.path.basename(path), f, mime)},
                          data={"overwrite": "true"}, timeout=300)
    r.raise_for_status()
    return r.json()["name"]


def frames_for(seconds: float) -> int:
    """The template's length rule: 24 fps, snapped up to the model's 17k + 5 grid."""
    n = max(5, round(seconds * 24))
    return n + (5 - n % 17) % 17


def graph(refs: list, video: str, prompt: str, w: int, h: int, length: int, steps: int) -> dict:
    n = lambda cls, **inputs: {"class_type": cls, "inputs": inputs}   # noqa: E731
    g = {
        "119": n("VAELoader", vae_name="minimax_h3_video_vae_int8_convrot.safetensors"),
        "120": n("VAELoader", vae_name="minimax_h3_audio_vae_fp32.safetensors"),
        "127": n("UNETLoader", unet_name="minimax_h3_ref2va_pruned_int8_convrot.safetensors", weight_dtype="default"),
        "128": n("CLIPLoader", clip_name="qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors", type="minimax", device="default"),
        "150": n("LoadVideo", file=video),
        "151": n("GetVideoComponents", video=["150", 0]),
        "123": n("KSamplerSelect", sampler_name="res_multistep"),
        "124": n("BasicScheduler", model=["127", 0], scheduler="simple", steps=steps, denoise=1.0),
        "129": n("RandomNoise", noise_seed=int(os.environ.get("COMFY_SEED", "42"))),
        "126": n("BasicGuider", model=["127", 0], conditioning=["136", 0]),
        "125": n("SamplerCustomAdvanced", noise=["129", 0], guider=["126", 0], sampler=["123", 0], sigmas=["124", 0],
                 latent_image=["136", 1]),
        "122": n("VAEDecode", samples=["125", 0], vae=["119", 0]),
        "121": n("VAEDecodeAudio", samples=["125", 0], vae=["120", 0]),
        "130": n("CreateVideo", images=["122", 0], audio=["121", 0], fps=24),
        "92": n("SaveVideo", video=["130", 0], filename_prefix="h3_r2v", format="auto", codec="auto"),
    }
    r2v = {"clip": ["128", 0], "vae": ["119", 0], "audio_vae": ["120", 0], "prompt": prompt, "width": w, "height": h,
           "length": length, "ref_image_size": "match", "ref_videos.ref_video_0": ["151", 0]}
    for i, name in enumerate(refs):
        g[f"16{i}"] = n("LoadImage", image=name)
        r2v[f"ref_images.ref_image_{i}"] = [f"16{i}", 0]
    g["136"] = n("MiniMaxH3ReferenceToVideo", **r2v)
    return g


def run(prompt: dict) -> list:
    r = requests.post(f"{BASE}/api/prompt", headers=HDR, json={"prompt": prompt}, timeout=60)
    if r.status_code != 200:
        sys.exit(f"submit failed {r.status_code}: {r.text[:2000]}")
    pid, t0 = r.json()["prompt_id"], time.time()
    print("job", pid, flush=True)
    while True:
        if time.time() - t0 > MAX_WAIT:   # don't burn GPU credits on a stuck job
            requests.post(f"{BASE}/api/interrupt", headers=HDR, timeout=30)
            sys.exit(f"job {pid} passed {MAX_WAIT}s - interrupted")
        st = requests.get(f"{BASE}/api/job/{pid}/status", headers=HDR, timeout=30).json()
        if st.get("status") in ("completed", "success"):
            break
        if "error" in str(st.get("status")) or st.get("status") in ("failed", "cancelled"):
            sys.exit(f"job failed: {json.dumps(st)[:2000]}")
        time.sleep(10)
    h = requests.get(f"{BASE}/api/history_v2/{pid}", headers=HDR, timeout=60).json()
    out = h.get(pid, h).get("outputs", {}).get("92", {})
    print(f"done in {time.time() - t0:.0f}s")
    return [f for v in out.values() if isinstance(v, list) for f in v if isinstance(f, dict) and "filename" in f]


def main() -> None:
    cid, seconds, out = sys.argv[1], float(sys.argv[2]), sys.argv[3]
    w, h, steps = (int(v) for v in sys.argv[4:7]) if len(sys.argv) > 6 else (544, 960, 20)
    char = os.path.join(ROOT, "chars", cid)
    # the camera move retimed to the clip length (its own length -> <seconds>)
    import cv2
    cap = cv2.VideoCapture(REF_VIDEO)
    src_seconds = cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    ref_vid = os.path.join(char, f"ref_cam_{seconds:g}s.mp4")
    subprocess.check_call([gs_paths.find_ffmpeg(), "-hide_banner", "-loglevel", "error", "-y", "-i", REF_VIDEO,
                           "-vf", f"setpts=PTS*{seconds / src_seconds},fps=24,scale=544:-2", "-an", "-c:v", "libx264",
                           "-crf", "14", ref_vid])
    refs = [upload(os.path.join(char, "refs", f"ref_{i}.jpg"), "image/jpeg") for i in (1, 2, 3, 4)]
    refs.append(upload(os.path.join(char, "portrait_seedream5pro.png"), "image/png"))
    video = upload(ref_vid, "video/mp4")
    prompt = open(os.path.join(char, "seedance_prompt.txt"), encoding="utf-8").read()
    for i in range(1, 6):
        prompt = prompt.replace(f"@Image {i}", f"<Picture {i}>")
    prompt = prompt.replace("@Video 1", "<Video 1>")
    files = run(graph(refs, video, prompt, w, h, frames_for(seconds), steps))
    if not files:
        sys.exit("no video in the job output")
    f = files[0]
    q = {"filename": f["filename"], "subfolder": f.get("subfolder", ""), "type": f.get("type", "output")}
    with open(out, "wb") as fh:
        fh.write(requests.get(f"{BASE}/api/view", headers=HDR, params=q, timeout=300).content)
    print("saved", out)


if __name__ == "__main__":
    main()
