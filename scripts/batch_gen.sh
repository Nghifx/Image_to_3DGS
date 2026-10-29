# Steps 1-2 of the chiLinh54 batch for one character: Seedream portrait, then the Seedance orbit video.
#
#   bash scripts/batch_gen.sh <id>
#
# Needs chars/<id>/refs (scripts/batch_refs.py), chars/<id>/portrait_prompt.txt and seedance_prompt.txt.
# Writes chars/<id>/portrait_seedream5pro.png (Seedream 5.0 Pro 1.5K, 3:4, ~2.5 credits) and
# chars/<id>/seedance_720p_10s.mp4 (GEN=h3: MiniMax H3 on Comfy Cloud, else Seedance 2.5 omni-reference, 720p 9:16 10 s, 70 credits): @Image 1-4 = the
# face-blurred photos, @Image 5 = the portrait, @Video 1 = the Electra camera move (v008, arms by the prompt).
# Skips what already exists; exits non-zero if a job gives no file.
set -e
cid=$1
R=$(cd "$(dirname "$0")/.." && pwd -W)
C=$R/chars/$cid
REF_VIDEO=$R/khmer_girl/ref_electra_arc_v008_10s/electra_arc_v008_240f.mp4
cd "$C"
exec > >(tee -a "$C/batch.log") 2>&1
echo "=== gen $cid $(date '+%F %T')"

if [ ! -f portrait_seedream5pro.png ]; then
    U=$(higgsfield generate create seedream_v5_pro --prompt "$(cat portrait_prompt.txt)" --image refs/front_upper_blur.jpg \
        --aspect_ratio 3:4 --resolution 1.5k --wait --json 2>&1 | grep -oE 'https://[^" ]+[0-9a-f]\.png' | head -1)
    [ -n "$U" ] || { echo "portrait failed"; exit 1; }
    curl -s -o portrait_seedream5pro.png "$U"
fi
echo "portrait ok"

if [ ! -f seedance_720p_10s.mp4 ] && [ "$GEN" = h3 ]; then
    # MiniMax H3 on Comfy Cloud (~$1.6, ~21 min) instead of Seedance: same refs, prompt and camera move, 736x1280
    # 243 frames; kept under the Seedance file name so batch_char.sh picks it up unchanged
    "$R/tools/venv/Scripts/python" "$R/scripts/comfy_h3_r2v.py" "$cid" 10 seedance_720p_10s.mp4 736 1280 20 || exit 1
elif [ ! -f seedance_720p_10s.mp4 ]; then
    U=$(higgsfield generate create seedance_2_5 --mode omni_reference --image refs/ref_1.jpg --image refs/ref_2.jpg \
        --image refs/ref_3.jpg --image refs/ref_4.jpg --image portrait_seedream5pro.png --video "$REF_VIDEO" \
        --duration 10 --resolution 720p --generate_audio false --aspect_ratio 9:16 \
        --prompt "$(cat seedance_prompt.txt)" --wait --wait-timeout 30m 2>&1 | grep -oE 'https://[^" ]+\.mp4|[Ee]rror.*|ip_detected|failed' | head -1)
    case "$U" in https*) curl -s -o seedance_720p_10s.mp4 "$U" ;; *) echo "video failed: $U"; exit 1 ;; esac
fi
echo "video ok"
