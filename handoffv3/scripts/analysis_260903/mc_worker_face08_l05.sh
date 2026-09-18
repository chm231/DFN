#!/bin/bash
# f06 시나리오 MC 실현 1개의 면08(x=15.4338) 예측 절리선 통계.
# 인자 = k (seed = k*101)
K="$1"
S=$((K*101))
SP="C:/Users/user/AppData/Local/Temp/claude/c--Users-user-OneDrive-2026-1-3D-DFN-modeling/c501f425-c178-491d-b844-fdbebd260c97/scratchpad"
cd "c:/Users/user/OneDrive/2026-1/3D DFN modeling/handoffv2" || exit 1
if [ -f "$SP/face08mc_l05/tr_seed${S}.npz" ]; then
  echo "seed $S skip"
  exit 0
fi
JSON="$SP/face08mc_l05/dfn_seed$S.json"
PYTHONPATH=. python -X utf8 -m dfn_analysis.export_domain_dfn_json \
  --pipeline-dir demo_output/win_f01-06_l05 --sets 1 2 3 --rmax-local 25 \
  --lmin-det 0.5 --seed $S --halo 5 --halo-z 7.48 \
  --out "$JSON" > /dev/null 2>&1
PYTHONPATH=. python -X utf8 "$SP/compute_face_plane_traces.py" \
  "$JSON" 15.4338 "$SP/face08mc_l05/tr_seed${S}.npz" 2>&1 | grep "result" | sed "s/^/seed $S /"
rm -f "$JSON"
