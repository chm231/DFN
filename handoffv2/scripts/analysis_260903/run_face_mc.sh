#!/bin/bash
# 막장면 쐐기 시드 민감도: 각 실현 JSON에 대해 --labels all 판정
# 사용: bash run_face_mc.sh <json1> <json2> ...
set -e
H="c:/Users/user/OneDrive/2026-1/3D DFN modeling/handoffv2"
SC="C:/Users/user/AppData/Local/Temp/claude/c--Users-user-OneDrive-2026-1-3D-DFN-modeling/c501f425-c178-491d-b844-fdbebd260c97/scratchpad"
mkdir -p "$SC/face_mc"
cd "$H"
for J in "$@"; do
  TAG=$(basename "$J" .json)
  echo "=== $TAG ==="
  PYTHONPATH=. python -X utf8 -m dfn_analysis.detect_face_blocks \
    --json "$J" --voxel 0.05 --labels all \
    --out-prefix "$SC/face_mc/$TAG" 2>&1 | grep -E "^\[face\]"
done
echo "BATCH_DONE"
