#!/bin/bash
# f01-07 윈도우 추가 시드 1개: export → 상위20 무한평면 쐐기 판정 → json 삭제.
# 인자 = k (seed = k*101)
K="$1"
S=$((K*101))
SP="C:/Users/user/AppData/Local/Temp/claude/c--Users-user-OneDrive-2026-1-3D-DFN-modeling/c501f425-c178-491d-b844-fdbebd260c97/scratchpad"
cd "c:/Users/user/OneDrive/2026-1/3D DFN modeling/handoffv2" || exit 1
JSON="$SP/wedgemc/dfn_f0107_seed$S.json"
if [ ! -f "$JSON" ]; then
  TS=$(cat demo_output/win_f01-07/target_sets.txt)
  PYTHONPATH=. python -X utf8 -m dfn_analysis.export_domain_dfn_json \
    --pipeline-dir demo_output/win_f01-07 --sets $TS --rmax-local 25 \
    --lmin-det 0.5 --seed $S --halo 5 \
    --out "$JSON" > /dev/null 2>&1
fi
PYTHONPATH=. python -X utf8 "$SP/wedge_top20.py" "$JSON" 20 2>/dev/null | grep -E "^SUM"
rm -f "$JSON"
