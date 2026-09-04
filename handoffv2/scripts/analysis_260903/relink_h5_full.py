# -*- coding: utf-8 -*-
"""relink 완전판: 원본 trace h5 스키마 전체를 복제하며 조각을 병합.

사용: python relink_h5_full.py <src_h5> <tol_m> <out_h5>
"""
import sys
import h5py
import numpy as np

src, tol, out = sys.argv[1], float(sys.argv[2]), sys.argv[3]
with h5py.File(src) as f:
    g = f["traces"]
    names = list(g.keys())
    data = {k: g[k][:] for k in names}
    meta = {k: f["meta"][k][:] for k in f["meta"].keys()}
p0 = data["p0_xyz"].astype(float)
p1 = data["p1_xyz"].astype(float)
sid = data["set_id"].astype(int)
fid = data["face_id"].astype(int)
n = len(sid)

parent = np.arange(n)
def find(a):
    while parent[a] != a:
        parent[a] = parent[parent[a]]
        a = parent[a]
    return a

for key in set(zip(fid.tolist(), sid.tolist())):
    idx = np.nonzero((fid == key[0]) & (sid == key[1]))[0]
    if len(idx) < 2:
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

groups = {}
for i in range(n):
    groups.setdefault(find(i), []).append(i)

def boundary(e):
    s = e.decode() if isinstance(e, bytes) else str(e)
    return "tunnel" in s

new = {k: [] for k in names}
for root, mem in groups.items():
    if len(mem) == 1:
        i = mem[0]
        for k in names:
            new[k].append(data[k][i])
        continue
    # 극단 끝점 탐색 (yz 투영 기준)
    epts, owner = [], []  # (member, which_end)
    for i in mem:
        epts += [p0[i], p1[i]]
        owner += [(i, 0), (i, 1)]
    epts = np.array(epts)
    c = epts.mean(0)
    dirv = epts[np.argmax(np.linalg.norm(epts - c, axis=1))] - c
    dirv /= np.linalg.norm(dirv)
    t = (epts - c) @ dirv
    i_lo, i_hi = int(np.argmin(t)), int(np.argmax(t))
    m_lo, e_lo = owner[i_lo]
    m_hi, e_hi = owner[i_hi]
    P0, P1 = epts[i_lo], epts[i_hi]
    L = float(np.linalg.norm(P1 - P0))
    lw = data["observed_length_m"][mem].astype(float)
    nvec = (data["trace_normal_xyz"][mem].astype(float) * lw[:, None]).sum(0)
    nvec /= max(np.linalg.norm(nvec), 1e-12)
    et_lo = data["p0_endpoint_type"][m_lo] if e_lo == 0 else data["p1_endpoint_type"][m_lo]
    et_hi = data["p0_endpoint_type"][m_hi] if e_hi == 0 else data["p1_endpoint_type"][m_hi]
    base = mem[0]
    for k in names:
        if k == "p0_xyz": v = P0.astype(np.float32)
        elif k == "p1_xyz": v = P1.astype(np.float32)
        elif k == "observed_length_m": v = np.float32(L)
        elif k == "censoring_class": v = np.uint8(int(boundary(et_lo)) + int(boundary(et_hi)))
        elif k == "trace_normal_xyz": v = nvec.astype(np.float32)
        elif k == "trace_normal_valid": v = np.uint8(1)
        elif k == "trace_normal_quality": v = np.float32(1.0)
        elif k == "trace_normal_reason": v = np.bytes_(b"relinked")
        elif k == "p0_endpoint_type": v = et_lo
        elif k == "p1_endpoint_type": v = et_hi
        elif k == "is_closed_loop": v = np.uint8(0)
        elif k in ("polyline_vertex_start", "polyline_vertex_count", "polyline_vertices_xyz"):
            v = None  # 아래에서 일괄 재구성
        else:
            v = data[k][base]
        if v is not None or k not in ("polyline_vertex_start", "polyline_vertex_count", "polyline_vertices_xyz"):
            new[k].append(v)

m = len(new["set_id"])
# polyline 일괄 재구성 (모든 trace = 2점)
new["polyline_vertex_start"] = (2 * np.arange(m)).astype(np.int32)
new["polyline_vertex_count"] = np.full(m, 2, np.int32)
pl = np.empty((2 * m, 3), np.float32)
pl[0::2] = np.array(new["p0_xyz"], dtype=np.float32)
pl[1::2] = np.array(new["p1_xyz"], dtype=np.float32)
new["polyline_vertices_xyz"] = pl
new["trace_id"] = np.arange(m, dtype=data["trace_id"].dtype)
new["fracture_id"] = np.arange(m, dtype=data["fracture_id"].dtype)

with h5py.File(out, "w") as f:
    g = f.create_group("traces")
    for k in names:
        arr = new[k]
        if not isinstance(arr, np.ndarray):
            arr = np.array(arr, dtype=data[k].dtype)
        g.create_dataset(k, data=arr)
    mg = f.create_group("meta")
    for k, v in meta.items():
        mg.create_dataset(k, data=v)
print(f"tol={tol}: {n} -> {m} traces, out={out}")
