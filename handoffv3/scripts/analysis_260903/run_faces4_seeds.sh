#!/bin/bash
# 4개 막장면(09~12) 역산 결과로 시드별 도메인 DFN JSON 생성 → 전달 폴더에 시드명으로 저장
set -e
H="c:/Users/user/OneDrive/2026-1/3D DFN modeling/handoffv2"
PIPE="../handoffv1/demo_output/dfm_demo_faces4"
OUT="c:/Users/user/OneDrive/2026-1/3D DFN modeling/deliverables/260902_한양대_DFN_faces4"
mkdir -p "$OUT"
cd "$H"
for S in "$@"; do
  echo "=== seed $S ==="
  PYTHONPATH=. python -X utf8 -m dfn_analysis.export_domain_dfn_json \
    --pipeline-dir "$PIPE" --sets 1 2 3 \
    --rmax-local 25 --lmin-det 0.5 --seed "$S" 2>&1 | tail -2
  cp "$PIPE/export/dfn_domain_x22.343-32.343_halo5.json" \
     "$OUT/dfn_faces4_x22.343-32.343_halo5_seed${S}.json"
done
echo "BATCH_DONE ($*)"
