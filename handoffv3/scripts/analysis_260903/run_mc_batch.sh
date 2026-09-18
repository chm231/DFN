#!/bin/bash
# 몬테카를로 배치: 시드별 도메인 JSON export + 블록 판정 (full 모드)
# 사용: bash run_mc_batch.sh <seed1> [seed2] [seed3]
set -e
H="c:/Users/user/OneDrive/2026-1/3D DFN modeling/handoffv1"
SC="C:/Users/user/AppData/Local/Temp/claude/c--Users-user-OneDrive-2026-1-3D-DFN-modeling/c501f425-c178-491d-b844-fdbebd260c97/scratchpad"
JSON="$H/demo_output/dfm_demo/export/dfn_domain_x22.204-32.204_halo5.json"
mkdir -p "$SC/mc"
cd "$H"
for S in "$@"; do
  echo "=== MC seed $S ==="
  PYTHONPATH=. python -X utf8 -m dfn_analysis.export_domain_dfn_json \
    --pipeline-dir demo_output/dfm_demo --sets 1 2 3 4 \
    --rmax-local 25 --lmin-det 0.5 --seed "$S" 2>&1 | tail -1
  cp "$JSON" "$SC/mc/dfn_seed${S}.json"
  PYTHONPATH=. python -X utf8 -m dfn_analysis.detect_blocks_from_domain_json \
    --json "$SC/mc/dfn_seed${S}.json" --voxel 0.1 --connectivity 6 \
    --out-prefix "$SC/blocks/mc_seed${S}_full" 2>&1 | grep -E "^\[result\]"
done
# 데모 원본(seed 2026) 복원
cp "$SC/dfn_domain_seed2026.json" "$JSON"
echo "BATCH_DONE ($*)"
