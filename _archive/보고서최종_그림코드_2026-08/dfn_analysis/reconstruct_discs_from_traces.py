"""
관측 절리선(trace) → 복원 원판(disc) 생성.

조건부 DFN 생성(generate_conditional_hidden_dfn.py)의 입력인
`reconstructed_discs.csv`(가시 disc 목록)를 관측 trace 로부터 만든다.

방법(핸드오프 답변 ①과 동일):
  1) trace 별 국소 평면 법선 = trace_normal_xyz(무효 시 폴리라인 SVD).
  2) 연결(association, 기본 agglomerative): 후보쌍(같은 set·법선축각·공면·근접)을
     근접순으로 병합하되, 병합할 때마다 (a) 결합 평면 SVD 잔차 ≤ resid_tol,
     (b) "한 균열=면당 chord 1개" 물리제약을 검증해 통과할 때만 확정 → 연쇄/과대병합
     억제(oracle 대비 trace 순도 ~95%). (geometric=단순 Union-Find, oracle=fracture_id)
  3) cluster 의 모든 3D trace 점에 SVD 평면적합 → 법선·중심(=disc center).
  4) 반지름: disc_boundary 경계호가 충분(arc≥arc_min)하면 원적합(determined);
     아니면 --kr-summary-csv 제공 시 모집단 멱법칙 g_R∝r^{-kr} 사후평균으로
     축소추정(shrinkage, empirical-Bayes) → 참 반지름 대비 오차·편향 대폭 감소하고
     튜닝 자유파라미터가 없어 과적합에 강함. kr 미제공 시 보수적 하한 0.5·chord.
     주의: 관측은 censored 되어 있어, 추정력↑(참값 회수) 은 censored 관측 대비 P21 을
           높인다(둘은 상반된 목표).
  5) adoption:  잔차 작고 지지 충분 → deterministic_disc / 그 외 → orientation_only
     / 퇴화(법선 무효·점 부족) → rejected.  조건화는 keep-adoptions 로 필터한다.

좌표계: x=East(터널축), y=North, z=Up. 막장면 = x=상수 평면.
출력 컬럼: set_id, cx, cy, cz, nx, ny, nz, radius, adoption, n_traces, n_faces, residual_m
"""
import argparse
import csv
import math
import os

import h5py
import numpy as np


def _fit_plane_svd(points: np.ndarray):
    """SVD 최소제곱 평면적합 → (unit normal, centroid, RMSE 잔차)."""
    centroid = points.mean(axis=0)
    if len(points) < 3:
        return np.array([1.0, 0.0, 0.0]), centroid, 0.0
    _, _, vt = np.linalg.svd(points - centroid)
    normal = vt[2, :]
    nrm = np.linalg.norm(normal)
    normal = normal / nrm if nrm > 1e-12 else np.array([1.0, 0.0, 0.0])
    resid = float(np.sqrt(np.mean((np.dot(points - centroid, normal)) ** 2)))
    return normal, centroid, resid


def _axial_angle_deg(n0: np.ndarray, n1: np.ndarray) -> float:
    """법선 축각(부호 무시) [deg]."""
    c = abs(float(np.clip(np.dot(n0, n1), -1.0, 1.0)))
    return math.degrees(math.acos(c))


class _UF:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, a):
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def _mean_axial_normal(normals, fallback: np.ndarray) -> np.ndarray:
    """축성(axial) 단위법선들의 평균. n 과 -n 을 같은 방향으로 보고 정렬한 뒤 평균."""
    A = np.asarray(normals, dtype=np.float64)
    nrm = np.linalg.norm(A, axis=1)
    A = A[nrm > 1e-12] / nrm[nrm > 1e-12, None]
    if len(A) == 0:
        return fallback
    s = np.sign(A @ A[0])
    s[s == 0] = 1.0
    m = (A * s[:, None]).mean(axis=0)
    nm = float(np.linalg.norm(m))
    return m / nm if nm > 1e-12 else fallback


def _plane_basis(normal: np.ndarray):
    """평면 법선에 직교하는 정규직교 기저 (u, v)."""
    a = np.array([1.0, 0.0, 0.0]) if abs(normal[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = a - (a @ normal) * normal
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    return u, v


def _circle_fit_kasa(P2: np.ndarray):
    """평면 내 2D 점군에 대수적(Kåsa) 원 적합 → (center2d, R, RMSE 잔차)."""
    x, y = P2[:, 0], P2[:, 1]
    A = np.column_stack([x, y, np.ones(len(x))])
    sol, *_ = np.linalg.lstsq(A, x * x + y * y, rcond=None)
    cx, cy = sol[0] / 2.0, sol[1] / 2.0
    R = math.sqrt(max(sol[2] + cx * cx + cy * cy, 0.0))
    resid = float(np.sqrt(np.mean((np.hypot(x - cx, y - cy) - R) ** 2)))
    return np.array([cx, cy]), R, resid


def radius_posterior_mean(a: float, kr: float, rmax: float, n: int = 200) -> float:
    """반지름 축소추정(empirical-Bayes): chord 하한 a 만 알 때, 크기편향 멱법칙
    g_R(r) ∝ r^{-kr} 를 사전으로 한 사후평균 E[R | R>=a, kr].
    (chord offset 균일 가정. R=a·cosh(u) 치환으로 R→a 특이점 제거.)
    kr(모집단 정보)만 쓰고 튜닝 자유파라미터가 없어 과적합에 강하다."""
    a = max(float(a), 1e-6)
    if not math.isfinite(kr) or rmax <= a:
        return a
    umax = math.acosh(rmax / a)
    u = np.linspace(0.0, umax, n)
    Rr = a * np.cosh(u)
    den = np.trapezoid(Rr ** (-(kr + 1.0)), u)   # ∝ ∫ R^{-(kr+1)}
    num = np.trapezoid(Rr ** (-kr), u)           # ∝ ∫ R^{-kr}
    return float(num / den) if den > 0 else a


def _arc_coverage_deg(P2: np.ndarray, c2: np.ndarray) -> float:
    """원 중심 기준 경계점들이 덮는 호(arc) 각도 [deg] (반지름 안정성 판정용)."""
    ang = np.sort(np.arctan2(P2[:, 1] - c2[1], P2[:, 0] - c2[0]))
    if len(ang) < 2:
        return 0.0
    gaps = np.diff(np.concatenate([ang, [ang[0] + 2 * math.pi]]))
    return math.degrees(2 * math.pi - gaps.max())


def load_traces(trace_h5: str):
    """per-trace: vertices, normal, centroid, set_id, face_x, fracture_id,
    boundary_pts(=disc_boundary 끝점 3D), chord(=|p1-p0|)."""
    with h5py.File(trace_h5, "r") as f:
        g = f["traces"]
        starts = g["polyline_vertex_start"][...]
        counts = g["polyline_vertex_count"][...]
        verts_all = g["polyline_vertices_xyz"][...].astype(np.float64)
        normals = g["trace_normal_xyz"][...].astype(np.float64)
        nvalid = g["trace_normal_valid"][...]
        set_id = g["set_id"][...].astype(int)
        face_x = g["face_x_m"][...].astype(float)
        frac_id = g["fracture_id"][...].astype(int)
        p0 = g["p0_xyz"][...].astype(np.float64)
        p1 = g["p1_xyz"][...].astype(np.float64)
        t0 = [s.decode() for s in g["p0_endpoint_type"][...]]
        t1 = [s.decode() for s in g["p1_endpoint_type"][...]]
    traces = []
    for i in range(len(set_id)):
        v = verts_all[starts[i]:starts[i] + counts[i]]
        if len(v) < 2:
            continue
        n = normals[i]
        if not nvalid[i] or np.linalg.norm(n) < 1e-9:
            n, _, _ = _fit_plane_svd(v)  # 폴리라인 SVD 대체
        n = n / np.linalg.norm(n)
        # 균열 경계(disc_boundary)로 판정된 끝점만 원 적합에 사용
        bpts = []
        if t0[i] == "disc_boundary":
            bpts.append(p0[i])
        if t1[i] == "disc_boundary":
            bpts.append(p1[i])
        traces.append(dict(idx=i, verts=v, normal=n, centroid=v.mean(axis=0),
                           mid=0.5 * (p0[i] + p1[i]),
                           set_id=int(set_id[i]), face_x=float(face_x[i]),
                           frac_id=int(frac_id[i]),
                           boundary=bpts, chord=float(np.linalg.norm(p1[i] - p0[i]))))
    return traces


def associate_geometric(traces, angle_deg, coplanar_m, max_sep_m):
    """같은 set·근접 공면 trace 들을 Union-Find 로 묶어 cluster index 배열 반환."""
    n = len(traces)
    uf = _UF(n)
    for i in range(n):
        ti = traces[i]
        for j in range(i + 1, n):
            tj = traces[j]
            if ti["set_id"] != tj["set_id"]:
                continue
            if np.linalg.norm(ti["centroid"] - tj["centroid"]) > max_sep_m:
                continue
            if _axial_angle_deg(ti["normal"], tj["normal"]) > angle_deg:
                continue
            d_ij = abs(float(np.dot(tj["centroid"] - ti["centroid"], ti["normal"])))
            d_ji = abs(float(np.dot(ti["centroid"] - tj["centroid"], tj["normal"])))
            if max(d_ij, d_ji) > coplanar_m:
                continue
            uf.union(i, j)
    return [uf.find(i) for i in range(n)]


def associate_oracle(traces):
    """검증용: 참 fracture_id 로 묶음."""
    return [t["frac_id"] for t in traces]


def _predicted_chord_on_face(m_a, n_a, x_b, eps=1e-6):
    """면 x_a 의 chord(중점 m_a, 법선 n_a)를 갖는 원판이 면 x_b 에서 만들
    chord 의 (예측 점 q, 방향 d). 원판–평면 교선은 x 에 따라 선형 이동한다."""
    ex = np.array([1.0, 0.0, 0.0])
    d = np.cross(n_a, ex)
    nd = np.linalg.norm(d)
    if nd < eps:  # 원판 평면이 면과 평행 → chord 정의 불가
        return None, None
    d = d / nd
    g = ex - (ex @ n_a) * n_a            # 평면 내 x축 성분(교선의 x-이동 방향)
    if abs(g[0]) < eps:                   # 원판 평면이 x축을 포함 → 같은 y-z 위치
        q = np.array([x_b, m_a[1], m_a[2]])
    else:
        s = (x_b - m_a[0]) / g[0]
        q = m_a + s * g
    return q, d


def associate_predictive(traces, angle_deg, pos_tol_m, max_gap):
    """막장면 간 기하 연속성 예측 + 인접(±gap) 면 1:1 최적매칭(Hungarian).
    한 면의 trace 로부터 다음 면 chord 위치를 예측해 가장 잘 맞는 것과만 매칭한다."""
    from scipy.optimize import linear_sum_assignment

    n = len(traces)
    uf = _UF(n)
    # (set_id, face_x) 별 trace 인덱스 그룹
    by_sf = {}
    for i, t in enumerate(traces):
        by_sf.setdefault((t["set_id"], round(t["face_x"], 3)), []).append(i)
    sets = sorted({t["set_id"] for t in traces})
    cos_thr = math.cos(math.radians(angle_deg))
    BIG = 1e6

    for sid in sets:
        faces = sorted(fx for (s, fx) in by_sf if s == sid)
        for fi, xa in enumerate(faces):
            for xb in faces[fi + 1:fi + 1 + max_gap]:  # 인접 + gap 허용
                A = by_sf[(sid, xa)]
                B = by_sf[(sid, xb)]
                if not A or not B:
                    continue
                cost = np.full((len(A), len(B)), BIG)
                for ia, gi in enumerate(A):
                    ta = traces[gi]
                    m_a = ta["mid"]
                    q, d = _predicted_chord_on_face(m_a, ta["normal"], xb)
                    for ib, gj in enumerate(B):
                        tb = traces[gj]
                        if abs(float(np.dot(ta["normal"], tb["normal"]))) < cos_thr:
                            continue
                        if q is None:
                            continue
                        # 예측 chord 선까지 tb 중점의 수직거리
                        w = tb["mid"] - q
                        perp = float(np.linalg.norm(w - (w @ d) * d))
                        if perp <= pos_tol_m:
                            cost[ia, ib] = perp
                if not np.isfinite(cost).any() or (cost < BIG).sum() == 0:
                    continue
                ri, ci = linear_sum_assignment(cost)
                for ia, ib in zip(ri, ci):
                    if cost[ia, ib] < BIG:
                        uf.union(A[ia], B[ib])
    return [uf.find(i) for i in range(n)]


def _same_face_one_chord(members, normal, chord_tol):
    """물리제약: 한 균열은 면당 chord 1개. 같은 면 멤버들의 중점이 chord 수직방향으로
    chord_tol 이내로 모여있어야(=하나의 chord) True. 벌어져 있으면 서로 다른 균열."""
    ex = np.array([1.0, 0.0, 0.0])
    d = np.cross(normal, ex)
    nd = np.linalg.norm(d)
    if nd < 1e-6:
        return True
    d = d / nd
    perp_axis = np.cross(normal, d)  # 면 내 chord 수직축
    by_face = {}
    for m in members:
        by_face.setdefault(round(m["face_x"], 3), []).append(m["mid"])
    for mids in by_face.values():
        if len(mids) < 2:
            continue
        proj = [float(mm @ perp_axis) for mm in mids]
        if max(proj) - min(proj) > chord_tol:
            return False
    return True


def associate_agglomerative(traces, angle_deg, coplanar_m, max_sep_m,
                            resid_tol=0.08, chord_tol=0.25):
    """검증형 응집: 후보 간선을 근접순으로 병합하되, 병합 후 (a) 결합 평면 잔차 ≤ resid_tol,
    (b) 면당 chord 1개 제약을 만족할 때만 확정 → 연쇄 병합/과대병합 억제."""
    n = len(traces)
    uf = _UF(n)
    members = {i: [traces[i]] for i in range(n)}  # 대표 idx -> 멤버 리스트

    # 후보 간선: 같은 set, 법선 축각·공면·근접 게이트 통과쌍 (근접순 정렬)
    edges = []
    for i in range(n):
        ti = traces[i]
        for j in range(i + 1, n):
            tj = traces[j]
            if ti["set_id"] != tj["set_id"]:
                continue
            sep = float(np.linalg.norm(ti["centroid"] - tj["centroid"]))
            if sep > max_sep_m:
                continue
            if _axial_angle_deg(ti["normal"], tj["normal"]) > angle_deg:
                continue
            d_ij = abs(float(np.dot(tj["centroid"] - ti["centroid"], ti["normal"])))
            d_ji = abs(float(np.dot(ti["centroid"] - tj["centroid"], tj["normal"])))
            if max(d_ij, d_ji) > coplanar_m:
                continue
            edges.append((max(d_ij, d_ji), i, j))
    edges.sort()

    for _, i, j in edges:
        ri, rj = uf.find(i), uf.find(j)
        if ri == rj:
            continue
        merged = members[ri] + members[rj]
        pts = np.vstack([m["verts"] for m in merged])
        normal, _, resid = _fit_plane_svd(pts)
        if resid > resid_tol:
            continue
        if not _same_face_one_chord(merged, normal, chord_tol):
            continue
        uf.union(i, j)
        root = uf.find(i)
        other = rj if root == ri else ri
        members[root] = merged
        members.pop(other, None)
    return [uf.find(i) for i in range(n)]


def reconstruct(trace_h5, association, angle_deg, coplanar_m, max_sep_m,
                pos_tol_m=0.4, max_gap=2, target_sets=None, arc_min=90.0,
                kr_map=None, rmax=250.0, exclude_faces=None):
    traces = load_traces(trace_h5)
    if target_sets:
        traces = [t for t in traces if t["set_id"] in set(target_sets)]
    if exclude_faces:  # LOFO 검증: 특정 막장면 trace 를 학습에서 제외(hold-out)
        ex = {round(float(x), 3) for x in exclude_faces}
        traces = [t for t in traces if round(t["face_x"], 3) not in ex]
    if not traces:
        return []
    if association == "oracle":
        labels = associate_oracle(traces)
    elif association == "predictive":
        labels = associate_predictive(traces, angle_deg, pos_tol_m, max_gap)
    elif association == "agglomerative":
        labels = associate_agglomerative(traces, angle_deg, coplanar_m, max_sep_m)
    else:
        labels = associate_geometric(traces, angle_deg, coplanar_m, max_sep_m)

    clusters = {}
    for t, lab in zip(traces, labels):
        clusters.setdefault(lab, []).append(t)

    discs = []
    for members in clusters.values():
        pts = np.vstack([m["verts"] for m in members])
        set_id = members[0]["set_id"]
        faces = sorted({round(m["face_x"], 3) for m in members})
        normal, centroid, resid = _fit_plane_svd(pts)
        if len(faces) < 2:
            # 단일 막장면 성분은 끝점이 모두 x=상수 평면에 놓이므로 SVD 가 막장면
            # 자체를 되돌려 준다(법선 = x축). 이때는 절리선에 딸린 3차원 법선을 쓴다.
            normal = _mean_axial_normal([m["normal"] for m in members], normal)
            resid = float(np.sqrt(np.mean(((pts - centroid) @ normal) ** 2)))
        if normal[0] < 0:  # x축 부호로 일관성
            normal = -normal

        # (1) 균열 경계(disc_boundary) 끝점에 평면 내 원 적합 → 참 center/radius
        bpts = [b for m in members for b in m["boundary"]]
        center, radius, radius_status = centroid, 0.5, "lower_bound"
        arc = 0.0
        if len(bpts) >= 3:
            u, vv = _plane_basis(normal)
            B = np.asarray(bpts) - centroid
            B2 = np.column_stack([B @ u, B @ vv])
            c2, r_fit, r_resid = _circle_fit_kasa(B2)
            arc = _arc_coverage_deg(B2, c2)
            if arc >= arc_min and r_resid <= 0.2 and r_fit >= 0.3:
                center = centroid + c2[0] * u + c2[1] * vv
                radius = max(r_fit, 0.5)
                radius_status = "determined"

        # (2) 경계 부족 → kr 있으면 축소추정(모집단 정규화), 없으면 보수적 하한
        if radius_status != "determined":
            a_lower = 0.5 * max(m["chord"] for m in members)
            kr = (kr_map or {}).get(set_id)
            if kr is not None and math.isfinite(kr):
                radius = max(radius_posterior_mean(a_lower, kr, rmax), a_lower, 0.5)
                radius_status = "shrinkage"
            else:
                radius = max(a_lower, 0.5)

        # adoption: 반지름 정보가 있는 disc(determined/shrinkage) → deterministic, 그 외 하한 → orientation_only
        adoption = "deterministic_disc" if radius_status in ("determined", "shrinkage") else "orientation_only"

        discs.append(dict(
            set_id=set_id, cx=center[0], cy=center[1], cz=center[2],
            nx=normal[0], ny=normal[1], nz=normal[2], radius=radius,
            adoption=adoption, n_traces=len(members), n_faces=len(faces),
            residual_m=resid, radius_status=radius_status, arc_deg=round(arc, 1),
        ))
    return discs


def write_csv(discs, out_csv):
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    cols = ["set_id", "cx", "cy", "cz", "nx", "ny", "nz", "radius",
            "adoption", "radius_status", "arc_deg", "n_traces", "n_faces", "residual_m"]
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for d in discs:
            w.writerow(d)


def main():
    ap = argparse.ArgumentParser(description="관측 trace → 복원 disc(reconstructed_discs.csv)")
    ap.add_argument("--trace-h5", required=True)
    ap.add_argument("--out-csv", required=True,
                    help="예: <pdir>/reconstruct/reconstructed_discs.csv")
    ap.add_argument("--association",
                    choices=["agglomerative", "geometric", "predictive", "oracle"],
                    default="agglomerative")
    ap.add_argument("--normal-angle-deg", type=float, default=15.0)
    ap.add_argument("--coplanar-dist", type=float, default=0.15,
                    help="geometric 모드: 공면 허용거리 [m]")
    ap.add_argument("--max-centroid-sep", type=float, default=2.0,
                    help="geometric 모드: 중심 간 최대거리 [m]")
    ap.add_argument("--pos-tol", type=float, default=0.4,
                    help="predictive 모드: 예측 chord 선까지 허용 수직거리 [m]")
    ap.add_argument("--max-gap", type=int, default=2,
                    help="predictive 모드: 매칭 허용 면 간격(스킵 포함)")
    ap.add_argument("--arc-min", type=float, default=120.0,
                    help="원적합 반지름 채택 최소 호(arc) 커버리지 [deg]. 낮으면 부분호로 반지름 과대")
    ap.add_argument("--kr-summary-csv", default=None,
                    help="set별 kr_hat CSV. 주면 경계 부족 disc 반지름을 모집단 멱법칙으로 "
                         "축소추정(empirical-Bayes)해 추정력↑·과적합↓. 없으면 보수적 하한 사용.")
    ap.add_argument("--rmax", type=float, default=250.0, help="반지름 상한 [m] (축소추정 적분범위)")
    ap.add_argument("--target-set", nargs="+", type=int, default=None,
                    help="복원 대상 set. Laxemar 공정 비교시 Set 4(지수분포) 제외: --target-set 1 2 3 5")
    args = ap.parse_args()

    kr_map = None
    if args.kr_summary_csv:
        kr_map = {int(r["set_id"]): float(r["kr_hat"])
                  for r in csv.DictReader(open(args.kr_summary_csv))
                  if r.get("kr_hat") not in (None, "", "nan")}
    discs = reconstruct(args.trace_h5, args.association, args.normal_angle_deg,
                        args.coplanar_dist, args.max_centroid_sep,
                        args.pos_tol, args.max_gap, args.target_set, args.arc_min,
                        kr_map, args.rmax)
    write_csv(discs, args.out_csv)

    # 진단
    from collections import Counter
    by_set = Counter(d["set_id"] for d in discs)
    by_rstat = Counter(d["radius_status"] for d in discs)
    multi = sum(1 for d in discs if d["n_faces"] >= 2)
    print(f"[reconstruct] association={args.association}  복원 disc {len(discs)}개 "
          f"-> {args.out_csv}")
    print(f"  set별: {dict(sorted(by_set.items()))}")
    print(f"  반지름: {dict(by_rstat)} "
          f"(determined=원적합, shrinkage=kr축소추정, lower_bound=관측하한)  "
          f"| 다면 관통 disc {multi}개")
    if discs:
        rr = np.array([d["residual_m"] for d in discs])
        print(f"  잔차(m) median={np.median(rr):.3f} p90={np.percentile(rr,90):.3f}")


if __name__ == "__main__":
    main()
