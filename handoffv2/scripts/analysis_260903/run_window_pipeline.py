# -*- coding: utf-8 -*-
"""다면 윈도우 파이프라인 A단계: 변환→kr→config→P32→복원 (export 제외).

사용: python run_window_pipeline.py <tag> <face1> <face2> <face3> <face4>
예:   python run_window_pipeline.py f01-04 01 02 03 04
"""
import shutil
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np

TAG = sys.argv[1]
FACES = sys.argv[2:]
SC = Path(__file__).parent
H2 = Path("c:/Users/user/OneDrive/2026-1/3D DFN modeling/handoffv2")
DFM = Path("c:/Users/user/OneDrive/2026-1/3D DFN modeling/handoffv2/demodata/DFM_Export/DFM_Export")
OUT = Path("c:/Users/user/OneDrive/2026-1/3D DFN modeling/handoffv2/demo_output") / f"win_{TAG}"


def run(cmd):
    print("+", " ".join(cmd), flush=True)
    r = subprocess.run(cmd, cwd=str(H2), capture_output=True, text=True,
                       encoding="utf-8", errors="replace",
                       env={**__import__("os").environ, "PYTHONPATH": ".",
                            "PYTHONIOENCODING": "utf-8"})
    print(r.stdout[-1500:], flush=True)
    if r.returncode != 0:
        print(r.stderr[-2000:], flush=True)
        sys.exit(f"FAILED: {cmd}")


# 0) 입력 사본 (해당 4개 면 폴더만)
indir = SC / "win_input" / TAG
if indir.exists():
    shutil.rmtree(indir)
indir.mkdir(parents=True)
for f in FACES:
    shutil.copytree(DFM / f, indir / f)

# 1) 변환
run([sys.executable, "-X", "utf8", "scripts/convert_dfm_export_to_trace_dataset.py",
     "--dfm-dir", str(indir), "--outdir", str(OUT)])

# 2) 대상 절리군 자동 선택: 절리선 >= 100 그리고 3개 면 이상 관측
h5p = OUT / "trace_dataset/trace_dataset_3d.h5"
with h5py.File(h5p) as f:
    sid = f["traces/set_id"][:].ravel()
    fx = f["traces/face_x_m"][:].ravel() if "traces/face_x_m" in f else None
targets = []
for s in sorted(set(sid.tolist())):
    m = sid == s
    n_faces = len(np.unique(np.round(fx[m], 3))) if fx is not None else 4
    if m.sum() >= 100 and n_faces >= 3:
        targets.append(int(s))
if not targets:
    sys.exit("no target sets")
ts = [str(t) for t in targets]
print(f"[{TAG}] target sets = {ts}", flush=True)
(OUT / "target_sets.txt").write_text(" ".join(ts), encoding="utf-8")

# 3~6) kr → config → P32 → 복원
run([sys.executable, "-X", "utf8", "-m", "dfn_analysis.estimate_kr",
     "--trace-h5", str(h5p), "--dfn-model", f"win_{TAG}",
     "--generation-rmin", "0.5", "--target-set", *ts, "--outdir", str(OUT / "kr")])
run([sys.executable, "-X", "utf8", "-m", "dfn_analysis.build_dataset_config_from_traces",
     "--trace-h5", str(h5p), "--kr-summary-csv", str(OUT / "kr/kr_summary_by_set.csv"),
     "--dataset-name", f"win_{TAG}", "--target-set", *ts,
     "--out", str(OUT / "dataset_config.json")])
run([sys.executable, "-X", "utf8", "-m", "dfn_analysis.estimate_p32_mc_calibrated",
     "--trace-h5", str(h5p), "--config", str(OUT / "dataset_config.json"),
     "--site", f"win_{TAG}", "--target-set", *ts,
     "--kr-summary-csv", str(OUT / "kr/kr_summary_by_set.csv"),
     "--bootstrap-csv", str(OUT / "kr/kr_summary_by_set.csv"),
     "--dfn-h5", str(OUT / "dfn_export_for_python.h5"),
     "--outcsv", str(OUT / "p32/p32_summary.csv")])
run([sys.executable, "-X", "utf8", "-m", "dfn_analysis.reconstruct_discs_from_traces",
     "--trace-h5", str(h5p), "--kr-summary-csv", str(OUT / "kr/kr_summary_by_set.csv"),
     "--target-set", *ts, "--max-centroid-sep", "4.5",
     "--radius-mode", "sample", "--radius-seed", "2026",
     "--out-csv", str(OUT / "reconstruct/reconstructed_discs.csv")])
print(f"PHASE_A_DONE {TAG}", flush=True)
