"""3DEC식 완전 절단(polyhedral split) 블록 판별 프로토타입.

3DEC `block cut dfn` 절차 차용:
  1) 도메인 박스(볼록 다면체 1개)에서 시작
  2) 균열(원판)을 반지름 내림차순 정렬 (--top-n 개 사용)
  3) 각 원판에 대해: 유한 원판이 실제로 닿는 블록만 골라 (안 닿으면 스킵)
     그 블록을 원판의 무한 평면으로 완전 분할 (부분 균열도 블록 경계까지 연장)
  4) sliver 방지: 분할 결과 한쪽 부피 < --min-volume 이면 그 절단은 기각 (3DEC block tolerance)
  5) 분류: 도메인 외곽 접촉 / 터널 내부 / 터널 벽 걸침(straddle) / 암반 내부
     제거 후보(candidate) = 터널 벽 걸침 & 도메인 외곽 비접촉  (복셀 필터와 동일 논리)

주의(모델 특성): 완전 절단은 부분 관통 균열을 과절단하므로 블록 수가 보수적(과대)이다.
실제 3DEC 에서는 원판 밖 접촉면에 rock bridge 물성을 줘 이를 보정한다. 복셀 CCA
(detect_blocks_from_domain_json)가 기하학적 하한, 본 방식이 3DEC 정의의 상한에 해당.

터널 처리 근사: 터널을 파내지 않고, 절단 후 블록 꼭짓점·중심의 터널 프리즘(단면 폴리곤
x 도메인 전 구간) 내외 판정으로 분류한다. 꼭짓점이 모두 밖이면서 면이 벽을 스치는
블록은 놓칠 수 있다(프로토타입 한계).

실행 예:
  python -m dfn_analysis.cut_blocks_polyhedral --json example_io/dfn_domain_*.json \
      --top-n 30 --min-volume 0.01 --out-prefix example_io/blocks3dec/top30
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull, QhullError


# ----------------------------------------------------------------------
# 볼록 다면체 유틸 (vertex 표현)
# ----------------------------------------------------------------------

def hull_of(verts_xyz: np.ndarray):
    """중복 제거 후 ConvexHull. 퇴화(부피 0)면 None."""
    verts_xyz = np.unique(np.round(verts_xyz, 9), axis=0)
    if len(verts_xyz) < 4:
        return None
    try:
        return ConvexHull(verts_xyz)
    except QhullError:
        return None


def hull_edges(hull: ConvexHull):
    """hull의 꼭짓점 인덱스 edge 집합."""
    edges = set()
    for simplex in hull.simplices:
        n = len(simplex)
        for i in range(n):
            a, b = int(simplex[i]), int(simplex[(i + 1) % n])
            edges.add((min(a, b), max(a, b)))
    return edges


def split_by_plane(hull: ConvexHull, p0_xyz: np.ndarray, n_xyz: np.ndarray,
                   eps: float = 1e-9):
    """볼록 다면체를 평면으로 분할. (음측 hull, 양측 hull, 단면 교점) 반환.

    평면이 다면체를 가로지르지 않으면 (None, None, None).
    """
    verts = hull.points[hull.vertices]
    d = (verts - p0_xyz) @ n_xyz
    if d.min() > -eps or d.max() < eps:
        return None, None, None  # 전부 한쪽

    # hull 전체 점 기준으로 edge-평면 교점 계산
    pts = hull.points
    dp = (pts - p0_xyz) @ n_xyz
    ipts = []
    for a, b in hull_edges(hull):
        da, db = dp[a], dp[b]
        if (da > eps and db < -eps) or (da < -eps and db > eps):
            t = da / (da - db)
            ipts.append(pts[a] + t * (pts[b] - pts[a]))
    if len(ipts) < 3:
        return None, None, None
    ipts = np.asarray(ipts)

    neg = np.vstack([pts[dp <= eps], ipts])
    pos = np.vstack([pts[dp >= -eps], ipts])
    return hull_of(neg), hull_of(pos), ipts


def disc_touches_section(ipts_xyz: np.ndarray, c_xyz: np.ndarray,
                         n_xyz: np.ndarray, r: float) -> bool:
    """평면 단면(교점들의 볼록다각형)과 유한 원판(중심 c, 반지름 r)의 교차 여부.

    둘 다 같은 평면 위에 있으므로 2D 판정: 원 중심이 다각형 내부이거나,
    다각형 경계까지 거리 <= r 이면 닿는다.
    """
    ref = np.array([0.0, 0.0, 1.0]) if abs(n_xyz[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = np.cross(n_xyz, ref); u /= np.linalg.norm(u)
    v = np.cross(n_xyz, u)
    q2 = (ipts_xyz - c_xyz) @ np.column_stack([u, v])  # (m,2), 원 중심이 원점
    try:
        h2 = ConvexHull(q2)
    except QhullError:
        return False
    poly = q2[h2.vertices]
    # 원점이 다각형 내부인가 (볼록: 모든 edge 에 대해 같은 쪽)
    inside = True
    m = len(poly)
    mind = np.inf
    for i in range(m):
        a, b = poly[i], poly[(i + 1) % m]
        e = b - a
        # 원점까지 선분 거리
        t = np.clip(-(a @ e) / (e @ e), 0.0, 1.0)
        mind = min(mind, float(np.linalg.norm(a + t * e)))
        cross = e[0] * (-a[1]) - e[1] * (-a[0])
        if cross < 0:
            inside = False
    return inside or mind <= r


# ----------------------------------------------------------------------
# 터널/도메인 분류
# ----------------------------------------------------------------------

def points_in_polygon_yz(pts_yz: np.ndarray, poly_yz: np.ndarray) -> np.ndarray:
    """ray casting (벡터화)."""
    y, z = pts_yz[:, 0], pts_yz[:, 1]
    inside = np.zeros(len(pts_yz), dtype=bool)
    n = len(poly_yz)
    j = n - 1
    for i in range(n):
        yi, zi = poly_yz[i]
        yj, zj = poly_yz[j]
        cond = ((zi > z) != (zj > z)) & (y < (yj - yi) * (z - zi) / (zj - zi + 1e-15) + yi)
        inside ^= cond
        j = i
    return inside


def classify_block(hull: ConvexHull, box, poly_yz, tol=1e-6):
    """(touches_boundary, tunnel_rel) 반환. tunnel_rel ∈ {inside, straddle, outside}."""
    verts = hull.points[hull.vertices]
    x0, x1, y0, y1, z0, z1 = box
    touches_boundary = bool(
        (np.abs(verts[:, 0] - x0) < tol).any() or (np.abs(verts[:, 0] - x1) < tol).any()
        or (np.abs(verts[:, 1] - y0) < tol).any() or (np.abs(verts[:, 1] - y1) < tol).any()
        or (np.abs(verts[:, 2] - z0) < tol).any() or (np.abs(verts[:, 2] - z1) < tol).any()
    )
    # 꼭짓점 + 중심으로 터널 내외 판정 (프로토타입 근사)
    test = np.vstack([verts, verts.mean(axis=0, keepdims=True)])
    ins = points_in_polygon_yz(test[:, 1:3], poly_yz)
    if ins.all():
        rel = "inside"
    elif ins.any():
        rel = "straddle"
    else:
        rel = "outside"
    return touches_boundary, rel


# ----------------------------------------------------------------------
# 메인
# ----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--json", required=True, help="dfn_domain_*.json")
    ap.add_argument("--top-n", type=int, default=30, help="반지름 상위 N개 균열 (기본 30)")
    ap.add_argument("--min-volume", type=float, default=0.01,
                    help="sliver 기각 최소 블록 부피 [m³] (3DEC block tolerance 역할)")
    ap.add_argument("--out-prefix", required=True, help="<prefix>_blocks.csv/.vtp 출력")
    args = ap.parse_args()

    with open(args.json, encoding="utf-8") as f:
        data = json.load(f)
    dom = data["meta"]["domain"]
    x0, x1 = [float(v) for v in dom["x_range_m"]]
    yz = dom["yz_bounds_m"]
    box = (x0, x1, yz["y_min"], yz["y_max"], yz["z_min"], yz["z_max"])
    poly_yz = np.asarray(dom["tunnel_polygon_yz_m"], dtype=float)

    fr = sorted(data["fractures"], key=lambda f_: -f_["radius_m"])[:args.top_n]
    print(f"[input] 균열 {len(fr)}개 (r {fr[-1]['radius_m']:.2f}~{fr[0]['radius_m']:.2f} m), "
          f"도메인 x[{x0:.1f},{x1:.1f}]")

    # 시작 블록 = 도메인 박스
    corners = np.array([[x, y, z] for x in box[:2] for y in box[2:4] for z in box[4:6]])
    blocks = [dict(hull=hull_of(corners), cut_by=[])]

    n_cut_total, n_skip_no_touch, n_rej_sliver = 0, 0, 0
    for k, f_ in enumerate(fr):
        c = np.asarray(f_["center_xyz_m"]); n = np.asarray(f_["normal_xyz"], dtype=float)
        n /= np.linalg.norm(n); r = float(f_["radius_m"])
        new_blocks = []
        for blk in blocks:
            neg, pos, ipts = split_by_plane(blk["hull"], c, n)
            if neg is None:
                new_blocks.append(blk); continue
            # 유한 원판이 이 블록 단면에 실제로 닿는가 (3DEC: 안 닿는 블록은 hide)
            if not disc_touches_section(ipts, c, n, r):
                new_blocks.append(blk); n_skip_no_touch += 1; continue
            # sliver 기각 (block tolerance)
            if neg.volume < args.min_volume or pos.volume < args.min_volume:
                new_blocks.append(blk); n_rej_sliver += 1; continue
            new_blocks.append(dict(hull=neg, cut_by=blk["cut_by"] + [f_["id"]]))
            new_blocks.append(dict(hull=pos, cut_by=blk["cut_by"] + [f_["id"]]))
            n_cut_total += 1
        blocks = new_blocks
    print(f"[cut] 절단 {n_cut_total}회, 원판 미접촉 스킵 {n_skip_no_touch}회, "
          f"sliver 기각 {n_rej_sliver}회 → 블록 {len(blocks)}개")

    # 분류
    rows = []
    for i, blk in enumerate(blocks):
        tb, rel = classify_block(blk["hull"], box, poly_yz)
        cand = (rel == "straddle") and not tb
        cen = blk["hull"].points[blk["hull"].vertices].mean(axis=0)
        rows.append(dict(block_id=i, volume_m3=blk["hull"].volume,
                         n_cuts=len(blk["cut_by"]), tunnel_rel=rel,
                         touches_boundary=int(tb), candidate=int(cand),
                         cx=cen[0], cy=cen[1], cz=cen[2], hull=blk["hull"]))
    n_in = sum(1 for r_ in rows if r_["tunnel_rel"] == "inside")
    n_st = sum(1 for r_ in rows if r_["tunnel_rel"] == "straddle")
    n_cand = sum(r_["candidate"] for r_ in rows)
    cand_v = [r_["volume_m3"] for r_ in rows if r_["candidate"]]
    print(f"[classify] 터널 내부 {n_in} / 벽 걸침 {n_st} / 제거 후보(벽 걸침·외곽 비접촉) {n_cand}")
    if cand_v:
        print(f"[candidate] 부피 합 {sum(cand_v):.2f} m³, 최대 {max(cand_v):.2f}, "
              f"중앙값 {float(np.median(cand_v)):.2f}")

    prefix = Path(args.out_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    with open(f"{prefix}_blocks.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["block_id", "volume_m3", "n_cuts", "tunnel_rel",
                    "touches_boundary", "candidate", "cx", "cy", "cz"])
        for r_ in sorted(rows, key=lambda q: -q["volume_m3"]):
            w.writerow([r_["block_id"], f"{r_['volume_m3']:.4f}", r_["n_cuts"],
                        r_["tunnel_rel"], r_["touches_boundary"], r_["candidate"],
                        f"{r_['cx']:.3f}", f"{r_['cy']:.3f}", f"{r_['cz']:.3f}"])

    # VTK 내보내기 (블록 표면, cell data: block_id / candidate / volume)
    try:
        import pyvista as pv
        parts = []
        for r_ in rows:
            h = r_["hull"]
            faces = np.column_stack([np.full(len(h.simplices), 3), h.simplices]).ravel()
            m = pv.PolyData(h.points, faces=faces)
            m.cell_data["block_id"] = np.full(m.n_cells, r_["block_id"], dtype=np.int32)
            m.cell_data["candidate"] = np.full(m.n_cells, r_["candidate"], dtype=np.int8)
            m.cell_data["volume_m3"] = np.full(m.n_cells, r_["volume_m3"], dtype=np.float32)
            m.cell_data["tunnel_rel"] = np.full(
                m.n_cells, {"inside": 0, "straddle": 1, "outside": 2}[r_["tunnel_rel"]],
                dtype=np.int8)
            parts.append(m)
        merged = parts[0].merge(parts[1:]) if len(parts) > 1 else parts[0]
        merged.save(f"{prefix}_blocks.vtp")
        print(f"[out] {prefix}_blocks.csv / _blocks.vtp")
    except ImportError:
        print(f"[out] {prefix}_blocks.csv (pyvista 없음 — vtp 생략)")


if __name__ == "__main__":
    main()
