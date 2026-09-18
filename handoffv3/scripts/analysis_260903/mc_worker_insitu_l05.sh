#!/bin/bash
# 전방 미굴착 도메인(x 11.368~21.368) 제자리 블록 MC 실현 1개.
# 인자 = k (seed = k*101)
K="$1"
S=$((K*101))
SP="C:/Users/user/AppData/Local/Temp/claude/c--Users-user-OneDrive-2026-1-3D-DFN-modeling/c501f425-c178-491d-b844-fdbebd260c97/scratchpad"
cd "c:/Users/user/OneDrive/2026-1/3D DFN modeling/handoffv2" || exit 1
if [ -f "$SP/insitumc_l05/blk_seed${S}_labels.npz" ]; then
  echo "seed $S skip"
  exit 0
fi
if [ -f "$SP/__nocache__/dfn_seed$S.json" ]; then
  JSON="$SP/__nocache__/dfn_seed$S.json"
else
  JSON="$SP/insitumc_l05/dfn_seed$S.json"
  PYTHONPATH=. python -X utf8 -m dfn_analysis.export_domain_dfn_json \
    --pipeline-dir demo_output/win_f01-06_l05 --sets 1 2 3 --rmax-local 25 \
    --lmin-det 0.5 --seed $S --halo 5 --halo-z 7.48 \
    --out "$JSON" > /dev/null 2>&1
fi
PYTHONPATH=. python -X utf8 "$SP/detect_interior_labels.py" \
  "$JSON" 0.1 "$SP/insitumc_l05/blk_seed${S}_labels.npz" 2>&1 | grep "result" | sed "s/^/seed $S /"
rm -f "$SP/insitumc_l05/dfn_seed$S.json"
