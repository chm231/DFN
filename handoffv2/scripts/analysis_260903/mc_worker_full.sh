#!/bin/bash
# 전체 도메인(x -0.145 ~ 21.368, 1~6면 시나리오) MC 실현 1개.
# 인자 = k (seed = k*101)
K="$1"
S=$((K*101))
SP="C:/Users/user/AppData/Local/Temp/claude/c--Users-user-OneDrive-2026-1-3D-DFN-modeling/c501f425-c178-491d-b844-fdbebd260c97/scratchpad"
cd "c:/Users/user/OneDrive/2026-1/3D DFN modeling/handoffv2" || exit 1
if [ -f "$SP/fullmc/blk_seed${S}_labels.npz" ]; then
  echo "seed $S skip"
  exit 0
fi
PYTHONPATH=. python -X utf8 -m dfn_analysis.export_domain_dfn_json \
  --pipeline-dir demo_output/win_f01-06 --sets 1 2 3 --rmax-local 25 \
  --lmin-det 0.5 --seed $S --halo 5 --halo-z 7.48 \
  --x-range -0.145 21.368 \
  --out "$SP/fullmc/dfn_seed$S.json" > /dev/null 2>&1
PYTHONPATH=. python -X utf8 -m dfn_analysis.detect_blocks_from_domain_json \
  --json "$SP/fullmc/dfn_seed$S.json" --method edgecut --voxel 0.1 \
  --out-prefix "$SP/fullmc/blk_seed$S" 2>&1 | grep "result" | sed "s/^/seed $S /"
rm -f "$SP/fullmc/dfn_seed$S.json"
