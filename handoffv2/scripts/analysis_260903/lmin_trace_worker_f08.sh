#!/bin/bash
# lmin 스윕용: w14 윈도우 실현 1개 export → 면05 평면 절리선 예측만 (블록 생략).
# 인자: <pipeline_dir_name> <out_subdir> <k>
WIN="$1"
SUB="$2"
K="$3"
S=$((K*101))
SP="C:/Users/user/AppData/Local/Temp/claude/c--Users-user-OneDrive-2026-1-3D-DFN-modeling/c501f425-c178-491d-b844-fdbebd260c97/scratchpad"
cd "c:/Users/user/OneDrive/2026-1/3D DFN modeling/handoffv2" || exit 1
OUTD="$SP/slide/$SUB"
mkdir -p "$OUTD"
if [ -f "$OUTD/tr_seed${S}.npz" ]; then exit 0; fi
TS=$(cat demo_output/$WIN/target_sets.txt)
JSON="$OUTD/dfn_seed$S.json"
PYTHONPATH=. python -X utf8 -m dfn_analysis.export_domain_dfn_json \
  --pipeline-dir demo_output/$WIN --sets $TS --rmax-local 25 \
  --lmin-det 0.5 --seed $S --halo 5 --halo-z 7.48 \
  --out "$JSON" > /dev/null 2>&1
PYTHONPATH=. python -X utf8 "$SP/compute_face_plane_traces.py" \
  "$JSON" 15.4338 "$OUTD/tr_seed${S}.npz" > /dev/null 2>&1
rm -f "$JSON"
