# -*- coding: utf-8 -*-
"""단일면 법선 수정 후 창 17개 복원 재실행.

인자 정확성 검증: 패치는 군집을 바꾸지 않으므로 (disc 수, set_id/n_faces/n_traces 수열)이
백업과 일치해야 한다. 일치하면 인자가 맞는 것으로 보고 채택, 불일치면 백업을 되돌린다.
"""
import csv, os, shutil, subprocess, sys, time
from pathlib import Path
import numpy as np
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

H2 = Path(r"c:/Users/user/OneDrive/2026-1/3D DFN modeling/handoffv2")
OUT = H2 / "demo_output"
# 백업 위치는 스크립트 위치와 무관하게 고정한다. __file__ 기준으로 두면 스크립트를
# 옮긴 뒤 재실행할 때 새 빈 폴더가 생겨 "수정 전" 기준이 현재 파일로 덮어써진다.
BK = Path(os.environ.get("RERUN_BACKUP_DIR", OUT.parent / "_rerun_backup"))
BK.mkdir(exist_ok=True)

def key(p):
    rows = list(csv.DictReader(open(p, newline="", encoding="utf-8")))
    return [(r["set_id"], r["n_faces"], r["n_traces"]) for r in rows], rows

def normals(rows):
    N = np.array([[float(r["nx"]), float(r["ny"]), float(r["nz"])] for r in rows])
    return N / np.linalg.norm(N, axis=1, keepdims=True)

wins = sorted(d.name for d in OUT.iterdir() if d.name.startswith("win_"))
env = {**os.environ, "PYTHONPATH": ".", "PYTHONIOENCODING": "utf-8"}
print(f"{'창':<18} {'인자':<8} {'disc':>6} {'법선변경':>8} {'각 중앙':>8} {'각 최대':>8} 판정")
tot_ch = tot_d = 0
for w in wins:
    csvp = OUT / w / "reconstruct" / "reconstructed_discs.csv"
    if not csvp.exists():
        print(f"{w:<18} 복원 CSV 없음 — 건너뜀"); continue
    bkp = BK / f"{w}.csv"
    if not bkp.exists():
        shutil.copy2(csvp, bkp)
    k0, rows0 = key(bkp)
    ts = (OUT / w / "target_sets.txt").read_text().split()
    ok = None
    for opt, name in ((["--adaptive-sep"], "적응"), ([], "고정4.5"),
                      (["--max-centroid-sep", "3.5"], "고정3.5")):
        cmd = [sys.executable, "-X", "utf8", "-m", "dfn_analysis.reconstruct_discs_from_traces",
               "--trace-h5", str(OUT / w / "trace_dataset/trace_dataset_3d.h5"),
               "--kr-summary-csv", str(OUT / w / "kr/kr_summary_by_set.csv"),
               "--target-set", *ts, *opt,
               "--radius-mode", "sample", "--radius-seed", "2026",
               "--out-csv", str(csvp)]
        r = subprocess.run(cmd, cwd=str(H2), capture_output=True, text=True,
                           encoding="utf-8", errors="replace", env=env)
        if r.returncode != 0:
            print(f"{w:<18} 실행 실패 ({name}): {r.stderr[-200:]}"); break
        k1, rows1 = key(csvp)
        if k1 == k0:
            ok = (name, rows0, rows1); break
    if ok is None:
        shutil.copy2(bkp, csvp)
        print(f"{w:<18} {'-':<8} {len(k0):>6} {'-':>8} {'-':>8} {'-':>8} 인자 불일치 → 백업 복원")
        continue
    name, rows0, rows1 = ok
    N0, N1 = normals(rows0), normals(rows1)
    ang = np.degrees(np.arccos(np.clip(np.abs(np.einsum("ij,ij->i", N0, N1)), 0, 1)))
    ch = ang > 0.1
    tot_ch += int(ch.sum()); tot_d += len(rows1)
    print(f"{w:<18} {name:<8} {len(rows1):>6} {int(ch.sum()):>8} "
          f"{(np.median(ang[ch]) if ch.any() else 0):>7.1f}° {(ang[ch].max() if ch.any() else 0):>7.1f}° 갱신")
print(f"\n합계 disc {tot_d}개 중 법선 변경 {tot_ch}개 ({tot_ch/max(tot_d,1)*100:.1f}%)")
