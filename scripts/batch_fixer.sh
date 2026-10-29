# Fixer pass for one character, after batch_char.sh: the angles its video barely filmed are rendered from its GS,
# cleaned with NVIDIA Fixer and trained in with the video frames (Difix3D+ style) -> GSplats/c<id>x/person/
#
#   bash scripts/batch_fixer.sh <id>          e.g. 0751c
#
# Two rings of views at the training cameras' distance (gs_novel_views.py):
#   over her head, all round (55 and 75 deg, every 15 deg)   - the boom-up shot's angles (c011x, 2026-10-29)
#   her back half at eye level (0 and 25 deg, 75..285 deg)    - the front half is filmed; Fixer's softer copies of it
#                                                                would only blur her face (c011y)
# Needs ml/venv.gsplat.localscheduler (Houdini's trainer env, built on the first train) and tools/fixer_venv
# (setup_fixer.bat). Log: chars/<id>/batch.log.
set -e
cid=$1
R=$(cd "$(dirname "$0")/.." && pwd -W)
H=${HYTHON:-$(ls -d "/c/Program Files/Side Effects Software/Houdini 22."*/bin/hython.exe | tail -1)}
PY="$R/tools/venv/Scripts/python"
TPY="$R/ml/venv.gsplat.localscheduler/Scripts/python.exe"
FPY="$R/tools/fixer_venv/Scripts/python"
C=chars/$cid
N=c$cid
cd "$R"
exec > >(tee -a "$C/batch.log") 2>&1
echo "=== fixer $cid $(date '+%F %T')"
P=$(ls GSplats/$N/*/gsplats/*.ply | sort -V | tail -1)   # the trained splat (whole run, before the carve)
PJ=GSplats/$N/person/${N}_person.json

NV_AZ=$(seq -s, 0 15 345) NV_EL=55,75 NV_START=2001 "$TPY" scripts/gs_novel_views.py $N "$P" $PJ $C/fixer_top | tail -1
NV_AZ=$(seq -s, 75 15 285) NV_EL=0,25 NV_START=4001 "$TPY" scripts/gs_novel_views.py $N "$P" $PJ $C/fixer_side | tail -1
for ring in fixer_top fixer_side; do
    "$FPY" scripts/fixer_run.py $C/$ring/raw $C/$ring/fixed --bg 0.5 | tail -1
done
"$PY" scripts/gs_add_views.py $N $C/fixer_top ${N}_t | tail -1
"$PY" scripts/gs_add_views.py ${N}_t $C/fixer_side ${N}x | tail -1

"$H" scripts/h22_train_gsplats.py colmap/${N}x ${N}x --keep-world --strategy default --steps 50000 \
    --ckpt-interval 5000 --out-dir GSplats 2>&1 | tail -2
PX=$(ls GSplats/${N}x/*/gsplats/*.49999.ply)
mkdir -p GSplats/${N}x/person
"$PY" scripts/carve_person.py "$PX" colmap/_work/${N}x/cameras.json data/$N \
    GSplats/${N}x/person/${N}x_person.ply GSplats/${N}x/person/${N}x_person.json noroll 1.6 2>&1 | tail -1
ELECTRA_PATH=slow "$H" scripts/gs_electra_playblast.py GSplats/${N}x/person/${N}x_person $C/pb_${N}x 2>&1 | tail -1
echo "=== fixer $cid done $(date '+%F %T')"
