#!/bin/bash
# 4면 윈도우 B단계: 시드별 도메인 DFN JSON export → 전달 폴더 저장
# 사용: bash run_window_export.sh <tag> <seed1> [seed2] [seed3]
set -e
TAG="$1"; shift
H2="c:/Users/user/OneDrive/2026-1/3D DFN modeling/handoffv2"
PIPE="demo_output/win_${TAG}"
OUT="c:/Users/user/OneDrive/2026-1/3D DFN modeling/deliverables/260902_한양대_DFN_누적윈도우/${TAG}"
mkdir -p "$OUT"
cd "$H2"
TS=$(cat "demo_output/win_${TAG}/target_sets.txt")
for S in "$@"; do
  echo "=== $TAG seed $S (sets: $TS) ==="
  PYTHONPATH=. python -X utf8 -m dfn_analysis.export_domain_dfn_json \
    --pipeline-dir "$PIPE" --sets $TS \
    --rmax-local 25 --lmin-det 0.5 --seed "$S" 2>&1 | tail -2
  J=$(ls -t "demo_output/win_${TAG}/export/"dfn_domain_*.json | head -1)
  B=$(basename "$J" .json)
  cp "$J" "$OUT/${B/dfn_domain/dfn_${TAG}}_seed${S}.json"
done
echo "EXPORT_DONE ${TAG} ($*)"
