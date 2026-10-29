# Steps 3-5 of the chiLinh54 batch for one character, after the CEO approved its Seedance video.
#
#   COMFY_CLOUD_API_KEY=... bash scripts/batch_char.sh <id>        e.g. 001
#
# Input:  chars/<id>/video.mp4 (or the older name seedance_720p_10s.mp4); any size, frames are trained at full size
# 3  in parallel: COLMAP on the frames with the checker room (local)  |  LTX-2.5 Alpha-Gen matte (Comfy Cloud)
# 4  Houdini COP: matte + choke 5 px + the plate grade -> chars/<id>/graded   (batch_grade.py)
# 5  train 50k steps on graded frames + COLMAP cameras, carve the person out -> GSplats/c<id>/person/
# Blue-screen video (Electra v018 path): MATTE=key bash scripts/batch_char.sh <id>  - no LTX, no cloud: a local AI
# guide (batch_dataset.py guide) + the chroma key in the Houdini net (batch_grade.py key), then COLMAP on her alone
# (batch_dataset.py masked + gs_colmap_ai.py, full size).
# Log: chars/<id>/batch.log. Each step skips itself if its output is already there, so a rerun resumes.
set -e
cid=$1
R=$(cd "$(dirname "$0")/.." && pwd -W)   # E:/... form: the Windows Pythons cannot read /e/...
H=${HYTHON:-$(ls -d "/c/Program Files/Side Effects Software/Houdini 22."*/bin/hython.exe | tail -1)}   # newest Houdini 22
PY="$R/tools/venv/Scripts/python"
C=$R/chars/$cid
N=c$cid
FF=$("$PY" -c "import sys;sys.path.insert(0,r'$R/scripts');import gs_paths;print(gs_paths.find_ffmpeg())")
ALPHA_LORA=ltx-2.5-22b-ic-lora-alpha-gen-0.9.safetensors
cd "$R"
exec > >(tee -a "$C/batch.log") 2>&1
echo "=== $cid $(date '+%F %T')"

# frames (0001.png ...)
VIDEO=$(ls "$C"/video.mp4 "$C"/seedance_720p_10s.mp4 2>/dev/null | head -1)
if [ ! -f "$C/frames/0001.png" ] && [ "${PICK:-1}" = 1 ]; then
    # CEO 2026-10-29: good frames from every angle, not all of them - sharp ones, each sharing ~50-62 % with the
    # last pick (pick_frames.py; 161 of a 20 s Genjutsu video). PICK=0 falls back to every FRAME_STEP-th frame
    "$PY" scripts/pick_frames.py "$VIDEO" "$cid" --min "${PICK_MIN:-0.5}" --max "${PICK_MAX:-0.62}" | tail -1
fi
if [ ! -f "$C/frames/0001.png" ]; then
    mkdir -p "$C/frames"
    # every FRAME_STEP-th frame (CEO 2026-10-28: the videos are sharp enough now; 360 -> 180 frames halves COLMAP's
    # matching, the matte and training)
    "$FF" -hide_banner -loglevel error -i "$VIDEO" -vf "select='not(mod(n\,${FRAME_STEP:-2}))'" \
        -fps_mode vfr -start_number 1 "$C/frames/%04d.png"
fi
echo "frames: $(ls "$C/frames" | wc -l)"

# 3a COLMAP with the background (background job) - LTX route only. The blue-screen route solves on her alone after
# the key (step 4): the AI-drawn room drifts against her (0751, 2026-10-29: room solve 21 dB ghosted, her 27.7 dB)
(
    if [ "${MATTE:-ltx}" != key ] && [ ! -f "colmap/${N}_raw/sparse/0/images.bin" ]; then
        "$PY" scripts/batch_dataset.py raw "$cid"
        "$PY" scripts/gs_colmap_ai.py "${N}_raw" --keep-bg 2>&1 | grep -v GSS_PROGRESS | tail -3
    fi
) &
colmap_job=$!

# 3b matte on Comfy Cloud: the LoRA wants sizes divisible by 32, so a 720-wide Seedance video is padded to 736
# and cropped back (an H3 video is already 736 wide: no pad, no crop)
if [ "${MATTE:-ltx}" = key ]; then
    # blue-screen video (Electra v018 path): the matte is pulled in Houdini (step 4), steered by a local AI guide
    [ "$(ls "$C/guide" 2>/dev/null | wc -l)" -ge "$(ls "$C/frames" | wc -l)" ] || "$PY" scripts/batch_dataset.py guide "$cid"
    [ -f "$C/screen_rgb.json" ] || "$PY" scripts/batch_dataset.py screen "$cid"   # this video's own blue
else
read FW FH < <("$PY" -c "import cv2;h,w=cv2.imread(r'$C/frames/0001.png').shape[:2];print(w,h)")
FH=${FH//[^0-9]/}   # Windows Python ends the line with a carriage return
W=$((FW * 1280 / FH))   # the matte runs at 1280 high (a Full HD video, e.g. Genjutsu, is scaled down and back)
OFF=$(( ((W + 31) / 32 * 32 - W) / 2 ))
if [ "$(ls "$C/alpha" 2>/dev/null | wc -l)" -lt "$(ls "$C/frames" | wc -l)" ]; then
    mkdir -p "$C/alpha_736" "$C/alpha"
    # built from the kept frames, so the matte has exactly one frame per training frame
    "$FF" -hide_banner -loglevel error -y -framerate 24 -i "$C/frames/%04d.png" -vf "scale=$W:1280,pad=$((W + 2 * OFF)):1280:$OFF:0" \
        -c:v libx264 -crf 12 -pix_fmt yuv420p -an "$C/seedance_736x1280.mp4"
    # 41-frame chunks: 121-frame jobs hung past 25 min twice on 2026-10-26 (41-frame upscale chunks took ~1 min)
    # up to 3 tries, each resuming after the frames already saved (the cloud sometimes fails a model download)
    for try in 1 2 3; do
        [ "$(ls "$C/alpha_736" | wc -l)" -ge "$(ls "$C/frames" | wc -l)" ] && break
        COMFY_RESUME=1 COMFY_MAX_WAIT=900 COMFY_PREFIX="${N}_alpha" "$PY" scripts/comfy_ltx.py "$C/seedance_736x1280.mp4" \
            "$C/alpha_736" "$ALPHA_LORA" 41 2>&1 | grep -E "saved|done|failed|interrupted|error" || true
    done
    for f in "$C"/alpha_736/*.png; do
        "$FF" -hide_banner -loglevel error -y -i "$f" -vf "crop=$W:1280:$OFF:0,scale=$FW:$FH:flags=bicubic,format=gray" "$C/alpha/$(basename "$f")"
    done
fi
echo "alpha: $(ls "$C/alpha" | wc -l)"
[ "$(ls "$C/alpha" | wc -l)" -ge "$(ls "$C/frames" | wc -l)" ] || { echo "matte incomplete - stopping"; exit 1; }
fi
wait $colmap_job

# 4 grade + matte in Houdini, then the cameras
KEY_ARG=$([ "${MATTE:-ltx}" = key ] && echo key || true)
[ -f "$C/graded/0001.png" ] || "$H" scripts/batch_grade.py "$cid" $KEY_ARG 2>&1 | grep -E "graded|Error"
if [ "${MATTE:-ltx}" = key ]; then
    "$PY" scripts/batch_dataset.py masked "$cid"
    [ -f "colmap/$N/sparse/0/images.bin" ] || "$PY" scripts/gs_colmap_ai.py "$N" 2>&1 | grep -v GSS_PROGRESS | tail -3
else
    echo "colmap: $(ls colmap/${N}_raw/sparse/0)"
    "$PY" scripts/batch_dataset.py final "$cid"
fi

# 5 train + carve
if ! ls GSplats/$N/*/gsplats/*.49999.ply >/dev/null 2>&1; then
    "$H" scripts/h22_train_gsplats.py colmap/$N $N --keep-world --strategy default --steps 50000 \
        --ckpt-interval 5000 --out-dir GSplats 2>&1 | tail -2
fi
P=$(ls GSplats/$N/*/gsplats/*.49999.ply)
mkdir -p GSplats/$N/person
"$PY" scripts/carve_person.py "$P" colmap/_work/$N/cameras.json data/$N \
    GSplats/$N/person/${N}_person.ply GSplats/$N/person/${N}_person.json noroll 1.6 2>&1 | tail -2
echo "=== $cid done $(date '+%F %T')"
