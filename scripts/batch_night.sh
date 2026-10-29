# Unattended chiLinh54 run for several characters: all cloud generation first, then the local steps one by one.
#
#   COMFY_CLOUD_API_KEY=... bash scripts/batch_night.sh <wait_for_id> <id> [<id> ...]
#
# 1. batch_gen.sh for every <id> (portrait + Seedance video; the cloud jobs run while the GPU is busy elsewhere)
# 2. waits until <wait_for_id>'s batch_char.sh run has finished (one training at a time on the GPU)
# 3. batch_char.sh for every <id> that got its video (matte, COLMAP, grade, train, carve)
# A character that fails is logged and skipped; the next one still runs. Summary in chars/night_<date>.log.
R=$(cd "$(dirname "$0")/.." && pwd -W)
LOG=$R/chars/night_$(date +%Y%m%d).log
wait_for=$1
shift
say() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }

ok=()
for cid in "$@"; do
    if bash "$R/scripts/batch_gen.sh" "$cid" >/dev/null 2>&1; then ok+=("$cid"); say "$cid portrait + video ok"
    else say "$cid generation FAILED (see chars/$cid/batch.log)"; fi
done

# done line in its log, or its process gone (a step that failed under set -e writes no line)
until grep -qE "=== $wait_for done|stopping" "$R/chars/$wait_for/batch.log" 2>/dev/null \
      || ! ps -ef | grep -q "[b]atch_char.sh $wait_for"; do sleep 60; done
say "$wait_for finished, starting local steps"

for cid in "${ok[@]}"; do
    if bash "$R/scripts/batch_char.sh" "$cid" >/dev/null 2>&1; then say "$cid GS done: GSplats/c$cid/person"
    else say "$cid local steps FAILED (see chars/$cid/batch.log)"; fi
done
say "night run finished"
