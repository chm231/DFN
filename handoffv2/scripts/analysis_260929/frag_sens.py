# -*- coding: utf-8 -*-
"""조각 연결 게이트(간격)를 넓혀가며 절리선 수·길이 변화를 본다. 동일 직선성(수직 이격)으로 판정."""
import csv, os, sys
import numpy as np
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = r"c:/Users/user/OneDrive/2026-1/3D DFN modeling/handoffv2/demodata/DFM_Export/DFM_Export"

def load(face):
    p3 = os.path.join(ROOT, face, "Export 3D DS Data", "3D_disk_data.csv")
    ds, a, b = [], [], []
    with open(p3, newline="") as f:
        for row in csv.DictReader(f):
            ds.append(int(row["DS"]))
            a.append([float(row[f"TraceP1{c}"]) for c in "XYZ"])
            b.append([float(row[f"TraceP2{c}"]) for c in "XYZ"])
    return np.array(ds), np.array(a), np.array(b)

faces = sorted(d for d in os.listdir(ROOT) if d.isdigit())
DATA = {f: load(f) for f in faces}
OFF, ANG = 0.05, np.cos(np.radians(15.0))   # 수직 이격 5 cm, 축간각 15도

def run(gap_max, same_ds=True):
    n_in = n_out = 0
    all_L, all_S = [], []
    for face in faces:
        ds, A, B = DATA[face]
        V = B - A; L = np.linalg.norm(V, axis=1)
        U = V / np.maximum(L[:, None], 1e-12)
        parent = list(range(len(L)))
        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]; i = parent[i]
            return i
        keys = np.unique(ds) if same_ds else np.array([0])
        for d in keys:
            idx = np.nonzero(ds == d)[0] if same_ds else np.arange(len(L))
            for ii in range(len(idx)):
                i = idx[ii]
                for j in idx[ii + 1:]:
                    if abs(float(U[i] @ U[j])) < ANG:
                        continue
                    # j의 두 끝점이 i의 직선에서 벗어난 수직 거리
                    for P in (A[j], B[j]):
                        w = P - A[i]; t = w @ U[i]
                        if np.linalg.norm(w - t * U[i]) > OFF:
                            break
                    else:
                        gap = min(np.linalg.norm(A[i]-A[j]), np.linalg.norm(A[i]-B[j]),
                                  np.linalg.norm(B[i]-A[j]), np.linalg.norm(B[i]-B[j]))
                        if gap <= gap_max:
                            ri, rj = find(i), find(j)
                            if ri != rj: parent[ri] = rj
        groups = {}
        for i in range(len(L)): groups.setdefault(find(i), []).append(i)
        for g in groups.values():
            P = np.vstack([A[g], B[g]]); c = P.mean(0); M = P - c
            u = np.linalg.svd(M, full_matrices=False)[2][0]; t = M @ u
            all_S.append(float(t.max() - t.min()))
        n_in += len(L); n_out += len(groups); all_L.extend(L.tolist())
    L = np.array(all_L); S = np.array(all_S)
    return n_in, n_out, np.median(L), np.median(S), S.mean(), S.max()

print(f"{'간격게이트':>9} {'DS제한':>7} {'원수':>6} {'연결후':>6} {'감소율':>7} {'L중앙':>7} {'연결L중앙':>9} {'연결L평균':>9} {'연결L최대':>9}")
for gap in (0.05, 0.10, 0.20, 0.50, 1.00):
    n_in, n_out, mL, mS, aS, xS = run(gap)
    print(f"{gap:>9.2f} {'예':>7} {n_in:>6} {n_out:>6} {(1-n_out/n_in)*100:>6.1f}% {mL:>7.3f} {mS:>9.3f} {aS:>9.3f} {xS:>9.3f}")
n_in, n_out, mL, mS, aS, xS = run(0.50, same_ds=False)
print(f"{0.50:>9.2f} {'아니오':>7} {n_in:>6} {n_out:>6} {(1-n_out/n_in)*100:>6.1f}% {mL:>7.3f} {mS:>9.3f} {aS:>9.3f} {xS:>9.3f}")
