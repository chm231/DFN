#!/bin/bash
# 쐐기 케이스 1개 전체 빌드: 판정+메쉬 → 경계평면 CSV → 색 → pvsm → PNG 3장 → zip
# 인자: <json> <case_dir_name>
set -e
JSON="$1"
NAME="$2"
NTOP="${3:-20}"
SP="C:/Users/user/AppData/Local/Temp/claude/c--Users-user-OneDrive-2026-1-3D-DFN-modeling/c501f425-c178-491d-b844-fdbebd260c97/scratchpad"
DEL="c:/Users/user/OneDrive/2026-1/3D DFN modeling/deliverables/260903_한양대_쐐기케이스_top20"
CASE="$DEL/$NAME"
PV="/c/Program Files/ParaView 6.1.1/bin/pvpython.exe"
cd "c:/Users/user/OneDrive/2026-1/3D DFN modeling/handoffv2"
PYTHONPATH=. python -X utf8 "$SP/build_wedge_case.py" "$JSON" "$CASE" "$NTOP" 2>&1 | grep -E "wedge|recon"
PYTHONPATH=. python -X utf8 "$SP/make_bounding_planes.py" "$JSON" "$CASE/wedge_bounding_planes.csv" "$NTOP" 2>&1 | grep saved
python -X utf8 "$SP/bake_block_rgb.py" "$CASE/wedges_polyhedra.vtp" "$CASE/wedges_polyhedra.vtp" 2>&1 | grep baked
"$PV" "$SP/make_pvsm_wedgecase.py" "$CASE" "$CASE/wedge_case.pvsm" 2>&1 | grep saved
"$PV" "$SP/render_wedge_views.py" "$CASE" 2>&1 | grep saved | tail -1
cd "$DEL"
rm -f "$NAME.zip"
python -X utf8 -c "import shutil,sys; shutil.make_archive('$NAME','zip','.','$NAME'); print('zipped $NAME')"
