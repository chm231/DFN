# -*- coding: utf-8 -*-
"""직선성 조건을 풀고 끝점 근접만으로 연쇄(polyline) 연결 — 상대 가설의 가장 강한 형태."""
import csv, os, sys
import numpy as np
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = r"c:/Users/user/OneDrive/2026-1/3D DFN modeling/handoffv2/demodata/DFM_Export/DFM_Export"
faces = sorted(d for d in os.listdir(ROOT) if d.isdigit())
D = {}
for face in faces:
    ds, a, b = [], [], []
    with open(os.path.join(ROOT, face, "Export 3D DS Data", "3D_disk_data.csv"), newline="") as f:
        for r in csv.DictReader(f):
            ds.append(int(r["DS"]))
            a.append([float(r[f"TraceP1{c}"]) for c in "XYZ"])
            b.append([float(r[f"TraceP2{c}"]) for c in "XYZ"])
    D[face] = (np.array(ds), np.array(a), np.array(b))

def run(gap_max, ang_deg):
    cosA = np.cos(np.radians(ang_deg)) if ang_deg is not None else None
    n_in = n_out = 0; SL = []; SZ = []; Lall = []
    for face in faces:
        ds, A, B = D[face]
        V = B - A; L = np.linalg.norm(V, axis=1); U = V / np.maximum(L[:, None], 1e-12)
        parent = list(range(len(L)))
        def find(i):
            while parent[i] != i: parent[i] = parent[parent[i]]; i = parent[i]
            return i
        for d in np.unique(ds):
            idx = np.nonzero(ds == d)[0]
            P = np.vstack([A[idx], B[idx]])
            for ii in range(len(idx)):
                i = idx[ii]
                for j in idx[ii+1:]:
                    if cosA is not None and abs(float(U[i] @ U[j])) < cosA: continue
                    gap = min(np.linalg.norm(A[i]-A[j]), np.linalg.norm(A[i]-B[j]),
                              np.linalg.norm(B[i]-A[j]), np.linalg.norm(B[i]-B[j]))
                    if gap <= gap_max:
                        ri, rj = find(i), find(j)
                        if ri != rj: parent[ri] = rj
        grp = {}
        for i in range(len(L)): grp.setdefault(find(i), []).append(i)
        for g in grp.values():
            SL.append(float(L[g].sum()))   # 연쇄 총 길이 (polyline 길이)
            SZ.append(len(g))
        n_in += len(L); n_out += len(grp); Lall.extend(L.tolist())
    SL = np.array(SL); SZ = np.array(SZ); Lall = np.array(Lall)
    return n_in, n_out, np.median(Lall), np.median(SL), SL.mean(), SL.max(), SZ.max(), (SZ > 1).sum()

print(f"{'간격':>6} {'각도게이트':>9} {'원수':>6} {'연결후':>6} {'감소':>7} {'L중앙':>7} {'연쇄L중앙':>9} {'연쇄L평균':>9} {'연쇄L최대':>9} {'최대조각':>7} {'2개이상':>7}")
for gap, ang in ((0.05,45),(0.10,45),(0.20,45),(0.10,None),(0.20,None),(0.50,None)):
    n_in,n_out,mL,mS,aS,xS,mx,n2 = run(gap, ang)
    print(f"{gap:>6.2f} {str(ang)+'°' if ang else '없음':>9} {n_in:>6} {n_out:>6} {(1-n_out/n_in)*100:>6.1f}% "
          f"{mL:>7.3f} {mS:>9.3f} {aS:>9.3f} {xS:>9.3f} {mx:>7} {n2:>7}")
