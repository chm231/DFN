# -*- coding: utf-8 -*-
"""DFM export: 절리선이 '조각'인지, 이어붙이면 길이가 얼마나 변하는지 정량 확인."""
import csv, os, sys
import numpy as np
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = r"c:/Users/user/OneDrive/2026-1/3D DFN modeling/handoffv2/demodata/DFM_Export/DFM_Export"

def load(face):
    p3 = os.path.join(ROOT, face, "Export 3D DS Data", "3D_disk_data.csv")
    ds, r, a, b, n = [], [], [], [], []
    with open(p3, newline="") as f:
        for row in csv.DictReader(f):
            ds.append(int(row["DS"])); r.append(float(row["Radius"]))
            a.append([float(row[f"TraceP1{c}"]) for c in "XYZ"])
            b.append([float(row[f"TraceP2{c}"]) for c in "XYZ"])
            n.append([float(row[f"Normal{c}"]) for c in "XYZ"])
    return (np.array(ds), np.array(r), np.array(a), np.array(b), np.array(n))

faces = sorted(d for d in os.listdir(ROOT) if d.isdigit())
print(f"{'면':>3} {'개수':>5} {'L중앙':>7} {'L평균':>7} {'L최대':>7} {'R/반길이중앙':>12} "
      f"{'연결후개수':>9} {'연결후L중앙':>11} {'연결후L최대':>11} {'최대조각수':>9}")
tot = {}
for face in faces:
    ds, r, A, B, N = load(face)
    V = B - A
    L = np.linalg.norm(V, axis=1)
    ratio = r / (L / 2.0)

    # 같은 면·같은 DS 안에서 끝점이 맞닿고 방향이 같은 조각을 연결 (union-find)
    parent = list(range(len(L)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    U = V / np.maximum(L[:, None], 1e-12)
    GAP, ANG = 0.05, np.cos(np.radians(15.0))   # 5 cm, 축간각 15도
    for d in np.unique(ds):
        idx = np.nonzero(ds == d)[0]
        for ii in range(len(idx)):
            i = idx[ii]
            for j in idx[ii + 1:]:
                if abs(float(U[i] @ U[j])) < ANG:
                    continue
                gap = min(np.linalg.norm(A[i] - A[j]), np.linalg.norm(A[i] - B[j]),
                          np.linalg.norm(B[i] - A[j]), np.linalg.norm(B[i] - B[j]))
                if gap <= GAP:
                    ri, rj = find(i), find(j)
                    if ri != rj:
                        parent[ri] = rj
    groups = {}
    for i in range(len(L)):
        groups.setdefault(find(i), []).append(i)
    spans, sizes = [], []
    for g in groups.values():
        P = np.vstack([A[g], B[g]])
        # 주축 방향 사영 길이 = 이은 절리선의 끝-끝 길이
        c = P.mean(0); M = P - c
        u = np.linalg.svd(M, full_matrices=False)[2][0]
        t = M @ u
        spans.append(float(t.max() - t.min())); sizes.append(len(g))
    spans = np.array(spans); sizes = np.array(sizes)
    print(f"{face:>3} {len(L):>5} {np.median(L):>7.3f} {L.mean():>7.3f} {L.max():>7.3f} "
          f"{np.median(ratio):>12.3f} {len(spans):>9} {np.median(spans):>11.3f} "
          f"{spans.max():>11.3f} {sizes.max():>9}")
    tot[face] = (L, ratio, spans, sizes)

L = np.concatenate([v[0] for v in tot.values()])
ratio = np.concatenate([v[1] for v in tot.values()])
spans = np.concatenate([v[2] for v in tot.values()])
sizes = np.concatenate([v[3] for v in tot.values()])
print("\n[전체]")
print(f"  원자료 절리선 {len(L)}개, 길이 중앙 {np.median(L):.3f} m, 평균 {L.mean():.3f} m, 최대 {L.max():.3f} m")
print(f"  Radius / 반길이 : 중앙 {np.median(ratio):.3f}, 5~95% {np.percentile(ratio,5):.3f}~{np.percentile(ratio,95):.3f}")
print(f"  연결 후 {len(spans)}개 ({len(spans)/len(L)*100:.1f}%), 길이 중앙 {np.median(spans):.3f} m, "
      f"평균 {spans.mean():.3f} m, 최대 {spans.max():.3f} m")
print(f"  조각 1개로 남은 것 {int((sizes==1).sum())}개 ({(sizes==1).mean()*100:.1f}%), "
      f"2개 이상 {int((sizes>1).sum())}개, 최대 조각수 {sizes.max()}")
print(f"  길이 중앙값 배율 {np.median(spans)/np.median(L):.2f}배, 평균 배율 {spans.mean()/L.mean():.2f}배")
