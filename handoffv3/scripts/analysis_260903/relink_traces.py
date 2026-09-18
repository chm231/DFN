# -*- coding: utf-8 -*-
"""절리선 조각 이어붙이기(relink): 같은 면·군, 동일선상(오프셋<5cm, 방향<2.5도),
끝점 간격<=tol 인 조각들을 연쇄 병합해 kr 적합용 CSV 생성.

사용: python relink_traces.py <trace_h5> <tol_m> <out_csv>   (tol=0 이면 병합 없음)
"""
import sys
import csv as csvmod
import h5py
import numpy as np

h5p, tol, out = sys.argv[1], float(sys.argv[2]), sys.argv[3]
with h5py.File(h5p) as f:
    g = f["traces"]
    p0 = g["p0_xyz"][:].astype(float)
    p1 = g["p1_xyz"][:].astype(float)
    L = g["observed_length_m"][:].astype(float)
    cen = g["censoring_class"][:].astype(int)
    sid = g["set_id"][:].astype(int)
    fid = g["face_id"][:].astype(int)
    e0 = g["p0_endpoint_type"][:]
    e1 = g["p1_endpoint_type"][:]

def boundary(e):  # tunnel_boundary(관측창 잘림) 여부
    s = e.decode() if isinstance(e, bytes) else str(e)
    return "tunnel" in s

parent = np.arange(len(L))
def find(a):
    while parent[a] != a:
        parent[a] = parent[parent[a]]
        a = parent[a]
    return a

n_pairs = 0
for key in set(zip(fid.tolist(), sid.tolist())):
    idx = np.nonzero((fid == key[0]) & (sid == key[1]))[0]
    if len(idx) < 2 or tol <= 0:
        continue
    q0, q1 = p0[idx][:, 1:3], p1[idx][:, 1:3]
    d = q1 - q0
    ln = np.linalg.norm(d, axis=1)
    u = d / np.maximum(ln[:, None], 1e-12)
    mid = 0.5 * (q0 + q1)
    for a in range(len(idx)):
        for b in range(a + 1, len(idx)):
            if abs(u[a] @ u[b]) < 0.999:
                continue
            dm = mid[b] - mid[a]
            if abs(u[a][0] * dm[1] - u[a][1] * dm[0]) > 0.05:
                continue
            gap = min(np.linalg.norm(x - y) for x in (q0[a], q1[a]) for y in (q0[b], q1[b]))
            if gap <= tol:
                ra, rb = find(idx[a]), find(idx[b])
                if ra != rb:
                    parent[rb] = ra
                    n_pairs += 1

groups = {}
for i in range(len(L)):
    groups.setdefault(find(i), []).append(i)

rows = []
for root, members in groups.items():
    if len(members) == 1:
        i = members[0]
        rows.append((sid[i], fid[i], L[i], cen[i],
                     p0[i, 1], p0[i, 2], p1[i, 1], p1[i, 2]))
        continue
    pts, ends = [], []
    for i in members:
        pts += [p0[i, 1:3], p1[i, 1:3]]
        ends += [boundary(e0[i]), boundary(e1[i])]
    pts = np.array(pts)
    dirm = pts[np.argmax(np.linalg.norm(pts - pts.mean(0), axis=1))] - pts.mean(0)
    dirm /= np.linalg.norm(dirm)
    t = (pts - pts.mean(0)) @ dirm
    i_lo, i_hi = int(np.argmin(t)), int(np.argmax(t))
    length = t[i_hi] - t[i_lo]
    cen_new = int(ends[i_lo]) + int(ends[i_hi])  # 0/1/2 관측창 잘림 끝점 수
    i0 = members[0]
    rows.append((sid[i0], fid[i0], length, cen_new,
                 pts[i_lo][0], pts[i_lo][1], pts[i_hi][0], pts[i_hi][1]))

with open(out, "w", newline="", encoding="utf-8") as fh:
    w = csvmod.writer(fh)
    w.writerow(["set_id", "face_id", "observed_length_m", "censoring_class",
                "p0_y", "p0_z", "p1_y", "p1_z"])
    w.writerows(rows)
print(f"tol={tol}: 병합 {n_pairs}쌍 → 절리선 {len(L)} → {len(rows)}")
