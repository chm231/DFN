# -*- coding: utf-8 -*-
"""진단: 현재 kr 이 관측된 '관통 면 수 분포'를 설명하는가?

[착안] 한 균열이 면 3·4·5 에 걸쳐 나타났다면 그 균열은 최소한 그 면 간격을 덮을
  만큼 크다 — 반지름에 대한 강한 제약이다. 현행 우도는 그 세 절리선을 무관한
  세 표본으로 취급해 이 정보를 전혀 쓰지 않는다.

[방법] kr 격자마다 원판을 3D 에 생성하고(반지름 powerlaw, 법선은 관측 트레이스
  법선 재표집, 중심 균등) 각 막장면에서 현을 계산·관측창 클리핑·검출하한 적용 →
  관통 면 수 k 의 모형 분포 P(k|kr) 를 얻는다. 관측 k 분포(복원 association)와
  다항우도로 대조한다.

[읽는 법] 현행 kr 이 관측 k 분포를 잘 맞히면 새 정보가 없다는 뜻이고,
  다면 관통을 과소예측하면 현행 kr 이 과대(큰 균열 과소)라는 뜻이다.

사용: (handoffv3) python scripts/multiface/kspan_diagnostic.py
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from dfn_analysis import reconstruct_discs_from_traces as R
from dfn_analysis.radius_powerlaw_likelihood import clip_segments_to_convex_polygon_vectorized

PIPE = ROOT / "demo_output/_seq2/cum_7"
H5 = PIPE / "trace_dataset/trace_dataset_3d.h5"
LMIN = 0.5           # 검출 하한 [m] — 관측자료와 동일 기준
RMIN, RMAX = 0.5, 25.0
NSIM = 400_000       # kr 당 생성 원판 수
KR_GRID = np.round(np.arange(2.6, 5.01, 0.1), 2)


def ccw(poly):
    a = 0.5 * np.sum(poly[:, 0] * np.roll(poly[:, 1], -1) - np.roll(poly[:, 0], -1) * poly[:, 1])
    return poly if a > 0 else poly[::-1]


def observed_k(traces):
    """복원 association 으로 원판별 '검출된 면 수' 분포를 센다."""
    lab = R.associate_agglomerative(traces, 15.0, 0.15, 4.5, adaptive_sep=True,
                                    sep_safety=1.4, sep_cap=8.0)
    per = defaultdict(lambda: defaultdict(float))
    sid = {}
    for i, l in enumerate(lab):
        t = traces[i]
        # 같은 면 조각은 하나의 현으로 합산(모형은 (원판,면)당 현 1개를 예측한다)
        per[l][round(float(t["face_x"]), 3)] += float(t["chord"])
        sid[l] = t["set_id"]
    out = defaultdict(lambda: defaultdict(int))
    for l, faces in per.items():
        k = sum(1 for v in faces.values() if v >= LMIN)
        if k >= 1:
            out[sid[l]][k] += 1
    return out


def model_k(kr, normals_pool, face_xs, poly, rng):
    """kr 에 대한 관통 면 수 분포 P(k|kr) (검출=관측창 내 가시길이>=LMIN)."""
    a = kr + 1.0
    u = rng.uniform(0.0, 1.0, NSIM)
    r = (RMIN ** (1 - a) + u * (RMAX ** (1 - a) - RMIN ** (1 - a))) ** (1 / (1 - a))
    n = normals_pool[rng.integers(0, len(normals_pool), NSIM)]
    n = n / np.linalg.norm(n, axis=1, keepdims=True)
    ymin, ymax = poly[:, 0].min(), poly[:, 0].max()
    zmin, zmax = poly[:, 1].min(), poly[:, 1].max()
    cx = rng.uniform(min(face_xs) - RMAX, max(face_xs) + RMAX, NSIM)
    cy = rng.uniform(ymin - RMAX, ymax + RMAX, NSIM)
    cz = rng.uniform(zmin - RMAX, zmax + RMAX, NSIM)
    nx = n[:, 0]
    sin_phi = np.sqrt(np.maximum(1.0 - nx * nx, 1e-12))
    # 면 내 현 방향 d = normalize(n x ex)
    d = np.stack([np.zeros(NSIM), n[:, 2], -n[:, 1]], axis=1) / sin_phi[:, None]
    k_count = np.zeros(NSIM, dtype=np.int32)
    for fx in face_xs:
        t = np.abs(cx - fx) / sin_phi
        hit = t < r
        if not hit.any():
            continue
        half = np.sqrt(r[hit] ** 2 - t[hit] ** 2)
        # 현 중점: 원판 중심을 면 위로 평면내 투영 (u_hat = (ex - nx*n)/sin_phi 방향)
        s = (fx - cx[hit]) / sin_phi[hit]
        ux = (1.0 - nx[hit] * n[hit, 0]) / sin_phi[hit]
        uy = (0.0 - nx[hit] * n[hit, 1]) / sin_phi[hit]
        uz = (0.0 - nx[hit] * n[hit, 2]) / sin_phi[hit]
        my = cy[hit] + s * uy
        mz = cz[hit] + s * uz
        mid = np.stack([my, mz], axis=1)
        dirs = d[hit][:, 1:3]
        dirs = dirs / np.linalg.norm(dirs, axis=1, keepdims=True)
        vis, cls = clip_segments_to_convex_polygon_vectorized(mid, dirs, 2.0 * half, poly)
        det = (cls >= 0) & (vis >= LMIN)
        idx = np.nonzero(hit)[0][det]
        k_count[idx] += 1
    k = k_count[k_count >= 1]
    kmax = 6
    h = np.array([(k == j).sum() for j in range(1, kmax + 1)], dtype=np.float64)
    return h / h.sum()


def main():
    with h5py.File(H5) as f:
        poly = ccw(np.array(f["meta/tunnel_poly_yz"], dtype=float))
        face_xs = sorted({round(float(v), 3) for v in f["traces/face_x_m"][:].ravel()})
    traces = R.load_traces(str(H5))
    obs = observed_k(traces)
    pools = defaultdict(list)
    for t in traces:
        pools[t["set_id"]].append(t["normal"])

    print(f"막장면 {len(face_xs)}개  x = {face_xs}")
    print(f"검출 하한 {LMIN} m,  kr 격자 {KR_GRID[0]}~{KR_GRID[-1]} (간격 0.1), 원판 {NSIM:,}/kr\n")

    kr_hat = {}
    with open(PIPE / "kr/kr_summary_by_set.csv", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            kr_hat[int(row["set_id"])] = float(row["kr_hat"])

    for s in sorted(obs):
        pool = np.array(pools[s], dtype=float)
        rng = np.random.default_rng(1000 + s)
        o = np.array([obs[s].get(j, 0) for j in range(1, 7)], dtype=float)
        print(f"{'='*72}\nset {s}   복원 원판 {int(o.sum()):,}개   현행 kr_hat = {kr_hat.get(s, float('nan')):.2f}")
        print(f"  관측 k 분포: " + "  ".join(f"k={j}:{int(o[j-1]):,}" for j in range(1, 5)))
        best, bll = None, -np.inf
        rows = []
        for kr in KR_GRID:
            p = model_k(float(kr), pool, face_xs, poly, rng)
            ll = float(np.sum(o * np.log(np.clip(p, 1e-12, 1))))
            rows.append((kr, p, ll))
            if ll > bll:
                bll, best = ll, (kr, p)
        print(f"  {'kr':>5s} {'P(k=1)':>8s} {'P(k=2)':>8s} {'P(k=3)':>8s} {'P(k>=4)':>8s} {'loglik':>12s}")
        print(f"  {'관측':>5s} " + "".join(f"{v:8.4f}" for v in
              (o[0]/o.sum(), o[1]/o.sum(), o[2]/o.sum(), o[3:].sum()/o.sum())) + f"{'':>12s}")
        for kr, p, ll in rows:
            mark = "  <-- 현행" if abs(kr - kr_hat.get(s, -9)) < 0.051 else ("  <== k-최적" if kr == best[0] else "")
            if abs(kr - kr_hat.get(s, -9)) < 0.051 or kr == best[0] or int(round(kr * 10)) % 5 == 0:
                print(f"  {kr:5.1f} {p[0]:8.4f} {p[1]:8.4f} {p[2]:8.4f} {p[3:].sum():8.4f} {ll:12.1f}{mark}")
        print(f"  → k 분포만으로 본 최적 kr = {best[0]:.1f}   (현행 {kr_hat.get(s, float('nan')):.2f})")


if __name__ == "__main__":
    main()
