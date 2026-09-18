#!/bin/bash
# 슬라이딩 윈도우 MC 실현 1개: export → 제자리 블록 라벨 + 대상면 절리선 예측.
# 인자: <tag> <pipeline_dir_name> <target_face_x> <k>  (seed = k*101)
TAG="$1"
WIN="$2"
TX="$3"
K="$4"
S=$((K*101))
SP="C:/Users/user/AppData/Local/Temp/claude/c--Users-user-OneDrive-2026-1-3D-DFN-modeling/c501f425-c178-491d-b844-fdbebd260c97/scratchpad"
cd "c:/Users/user/OneDrive/2026-1/3D DFN modeling/handoffv2" || exit 1
OUTD="$SP/slide/$TAG"
mkdir -p "$OUTD"
if [ -f "$OUTD/blk_seed${S}_labels.npz" ] && [ -f "$OUTD/tr_seed${S}.npz" ]; then
  echo "$TAG seed $S skip"
  exit 0
fi
TS=$(cat demo_output/$WIN/target_sets.txt)
JSON="$OUTD/dfn_seed$S.json"
PYTHONPATH=. python -X utf8 -m dfn_analysis.export_domain_dfn_json \
  --pipeline-dir demo_output/$WIN --sets $TS --rmax-local 25 \
  --lmin-det 0.5 --seed $S --halo 5 --halo-z 7.48 \
  --out "$JSON" > /dev/null 2>&1
PYTHONPATH=. python -X utf8 "$SP/detect_interior_labels.py" \
  "$JSON" 0.1 "$OUTD/blk_seed${S}_labels.npz" 2>&1 | grep result | sed "s/^/$TAG seed $S /"
PYTHONPATH=. python -X utf8 "$SP/compute_face_plane_traces.py" \
  "$JSON" "$TX" "$OUTD/tr_seed${S}.npz" > /dev/null 2>&1
rm -f "$JSON"
