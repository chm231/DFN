#!/bin/bash
# f06 전방 도메인 MC 실현 1개: export → 블록 판정(edgecut, 굴착 가정) → json 삭제.
# 인자 = k (seed = k*101). 관측 원판(막장면 복원 포함)은 매 실현 동일하게 포함됨.
K="$1"
S=$((K*101))
SP="C:/Users/user/AppData/Local/Temp/claude/c--Users-user-OneDrive-2026-1-3D-DFN-modeling/c501f425-c178-491d-b844-fdbebd260c97/scratchpad"
cd "c:/Users/user/OneDrive/2026-1/3D DFN modeling/handoffv2" || exit 1
if [ -f "$SP/f06mc/blk_seed${S}_labels.npz" ]; then
  echo "seed $S skip (이미 완료)"
  exit 0
fi
PYTHONPATH=. python -X utf8 -m dfn_analysis.export_domain_dfn_json \
  --pipeline-dir demo_output/win_f01-06 --sets 1 2 3 --rmax-local 25 \
  --lmin-det 0.5 --seed $S --halo 5 --halo-z 7.48 \
  --out "$SP/f06mc/dfn_seed$S.json" > /dev/null 2>&1
PYTHONPATH=. python -X utf8 -m dfn_analysis.detect_blocks_from_domain_json \
  --json "$SP/f06mc/dfn_seed$S.json" --method edgecut --voxel 0.1 \
  --out-prefix "$SP/f06mc/blk_seed$S" 2>&1 | grep "result" | sed "s/^/seed $S /"
rm -f "$SP/f06mc/dfn_seed$S.json"
