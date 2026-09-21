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
# =============================================================================
# 파일 역할:
#   관측 trace(HDF5) → 복원 원판(disc) CSV. 조건부 DFN 생성기의 입력인
#   reconstructed_discs.csv(가시 disc 목록)를 관측 trace 만으로 만든다(자립 모듈).
#
# 주요 입력:
#   - --trace-h5 : trace HDF5. /traces 그룹에서 polyline_vertices_xyz,
#                  trace_normal_xyz(+valid), set_id, face_x_m, fracture_id,
#                  p0_xyz/p1_xyz, p0/p1_endpoint_type 를 읽는다.
#   - --association {agglomerative(기본)|geometric|predictive|oracle} : 연결(association) 방식
#   - --kr-summary-csv (선택): set별 kr_hat. 경계 부족 disc 반지름 축소추정에 사용
#   - --target-set / --rmax / --arc-min / --normal-angle-deg / --coplanar-dist /
#     --max-centroid-sep / --pos-tol / --max-gap : 연결·반지름 제어 파라미터
#
# 주요 출력:
#   - --out-csv (reconstructed_discs.csv). 컬럼:
#     set_id, cx,cy,cz, nx,ny,nz, radius, adoption, radius_status, arc_deg,
#     n_traces, n_faces, residual_m
#   - 콘솔 진단: set별 disc 수, 반지름 상태 분포(determined/shrinkage/lower_bound),
#     다면 관통 disc 수, 평면적합 잔차 median/p90
#
# 핵심 처리 흐름:
#   1) trace 로드: 폴리라인 정점/법선(무효 시 SVD 대체)/경계끝점/현 길이 계산
#   2) 연결(association): 같은 set·법선축각·공면·근접 trace 를 cluster 로 묶는다
#      (agglomerative=병합마다 결합평면 잔차·"면당 chord 1개" 검증, geometric=Union-Find)
#   3) cluster 의 모든 3D 점에 SVD 평면적합 → 법선·중심(=disc center)
#   4) 반지름: 경계호 충분→원적합(determined) / 아니면 kr 축소추정(shrinkage) /
#      kr 없으면 보수적 하한 0.5·chord(lower_bound)
#   5) adoption 판정(deterministic_disc/orientation_only/rejected) 후 CSV 기록 + 진단 출력
# =============================================================================
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


def radius_posterior_sample(a: float, kr: float, rmax: float, rng: np.random.Generator,
                            rmin_pop: float = 0.5, n: int = 400) -> float:
    """radius_posterior_mean과 같은 사후분포에서 반지름 '표본'을 추출한다.
    점추정(하한 절단된 사후평균)이 만드는 0.5 m 스파이크를 없애고, 모집단 지지구간
    [max(a, rmin_pop), rmax]과 일치하는 조건부 크기분포를 그대로 재현한다.
    사후밀도(u-치환, R=a·cosh u): p(u) ∝ R(u)^{-(kr+1)}."""
    a = max(float(a), 1e-6)
    lo = max(a, rmin_pop)
    if not math.isfinite(kr) or rmax <= lo:
        return lo
    u_lo = math.acosh(lo / a) if lo > a else 0.0
    u_hi = math.acosh(rmax / a)
    u = np.linspace(u_lo, u_hi, n)
    pdf = (a * np.cosh(u)) ** (-(kr + 1.0))
    cdf = np.cumsum(pdf)
    cdf /= cdf[-1]
    return float(a * math.cosh(np.interp(rng.random(), cdf, u)))


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
                            resid_tol=0.08, chord_tol=0.25,
                            adaptive_sep=False, sep_safety=1.4, sep_cap=8.0,
                            same_face_sep=2.0, adjacent_dx_max=3.2):
    """검증형 응집: 후보 간선을 근접순으로 병합하되, 병합 후 (a) 결합 평면 잔차 ≤ resid_tol,
    (b) 면당 chord 1개 제약을 만족할 때만 확정 → 연쇄 병합/과대병합 억제.

    adaptive_sep=True: 근접 게이트를 방향·면간격 적응형으로. 원판이 막장면과
    평행할수록(|nx|→1) 두 면의 현 중심이 멀어지므로 게이트를 넓힌다:
        gate = clamp( sep_safety * |Δface_x| / sinθ , same_face_sep, sep_cap )
    (θ=원판 법선과 x축의 각, sinθ=√(1−nx²); 같은 면 조각은 same_face_sep,
     상한 sep_cap 으로 비인접 과병합 억제). max_sep_m 은 무시된다."""
    n = len(traces)
    uf = _UF(n)
    members = {i: [traces[i]] for i in range(n)}  # 대표 idx -> 멤버 리스트

    # 후보 간선: 같은 set, 법선 축각·공면·근접 게이트 통과쌍 (근접순 정렬)
    # 판정식은 쌍별 루프와 동일하고 '누구를 언제 검사하느냐'만 바꾼다:
    #   (a) 블록 분할 — set 과 면(face_x) 라벨만으로 탈락이 정해지는 쌍은 만들지 않는다.
    #       절리선은 이산 막장면 위에만 있으므로 |Δface_x| > adjacent_dx_max 인
    #       (면 p, 면 q) 블록은 통째로 건너뛴다(적응 게이트일 때만 해당 조건이 있다).
    #   (b) 벡터화 — 남은 블록 안의 쌍은 numpy 배열 연산 1회씩으로 일괄 판정한다.
    #       sqrt/acos 는 단조성을 이용해 제곱거리·코사인 비교로 대체한다.
    C_all = np.array([t["centroid"] for t in traces], dtype=np.float64)
    N_all = np.array([t["normal"] for t in traces], dtype=np.float64)
    FX_all = np.array([float(t["face_x"]) for t in traces], dtype=np.float64)
    SID_all = np.array([t["set_id"] for t in traces])
    cos_tol = math.cos(math.radians(angle_deg))  # 축각 <= angle_deg  <=>  |cos| >= cos_tol

    def _block_edges(a, b):
        """블록 내 쌍 (a, b) 를 벡터 판정해 (공면거리, i, j) 리스트를 만든다 (i < j)."""
        if not len(a):
            return []
        d = C_all[b] - C_all[a]
        sep2 = np.einsum("ij,ij->i", d, d)  # 제곱거리 (sqrt 생략)
        if adaptive_sep:
            dfx = np.abs(FX_all[a] - FX_all[b])
            nx = np.minimum(0.97, np.maximum(np.abs(N_all[a][:, 0]),
                                             np.abs(N_all[b][:, 0])))
            sin_t = np.sqrt(np.maximum(1.0 - nx * nx, 1e-6))
            gate = np.clip(sep_safety * dfx / sin_t, same_face_sep, sep_cap)
        else:
            gate = np.full(len(a), float(max_sep_m))
        m = sep2 <= gate * gate
        m &= np.abs(np.einsum("ij,ij->i", N_all[a], N_all[b])) >= cos_tol
        a, b, d = a[m], b[m], d[m]
        if not len(a):
            return []
        # 공면거리는 a,b 교환에 불변(d 부호만 바뀜) → 간선 라벨만 i<j 로 정규화한다
        dmax = np.maximum(np.abs(np.einsum("ij,ij->i", d, N_all[a])),
                          np.abs(np.einsum("ij,ij->i", d, N_all[b])))
        k = dmax <= coplanar_m
        return list(zip(dmax[k].tolist(),
                        np.minimum(a[k], b[k]).tolist(),
                        np.maximum(a[k], b[k]).tolist()))

    edges = []
    for sid_val in np.unique(SID_all):
        idx_s = np.nonzero(SID_all == sid_val)[0]
        if len(idx_s) < 2:
            continue
        if adaptive_sep:
            fkey = np.round(FX_all[idx_s], 3)
            faces_u = np.unique(fkey)
            by_face = [idx_s[fkey == fv] for fv in faces_u]
            for p in range(len(faces_u)):
                for q in range(p, len(faces_u)):
                    if abs(float(faces_u[q] - faces_u[p])) > adjacent_dx_max:
                        continue  # 비인접 면 블록: 쌍을 만들지 않는다
                    I, J = by_face[p], by_face[q]
                    if p == q:
                        ii, jj = np.triu_indices(len(I), k=1)
                        edges += _block_edges(I[ii], I[jj])
                    elif len(I) and len(J):
                        edges += _block_edges(np.repeat(I, len(J)),
                                              np.tile(J, len(I)))
        else:
            ii, jj = np.triu_indices(len(idx_s), k=1)
            edges += _block_edges(idx_s[ii], idx_s[jj])
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


def _chord_on_other_face(center, normal, radius, face_x):
    """disc가 평면 x=face_x 에 남기는 현 길이 [m] (관측창 클리핑 전, 0=교차 없음)."""
    sin_phi = math.sqrt(max(1.0 - float(normal[0]) ** 2, 0.0))
    if sin_phi < 1e-9:  # 면과 평행한 disc → 다른 면과 교차하지 않음
        return 0.0
    t = abs(face_x - float(center[0])) / sin_phi
    if t >= radius:
        return 0.0
    return 2.0 * math.sqrt(radius ** 2 - t ** 2)


def _chord_fit_multi(members, normal, centroid):
    """다면 클러스터 현 보존 적합: (p, q, r) 최소제곱.
    disc 평면 좌표계에서 면 교차선 ℓ_i = {v·û = h_i} (전부 평행, 방향 d).
    관측: 각 면의 chord 반길이 a_i, 중점의 d-좌표 u_i.
    모델: 현 반길이 √(r²−(h_i−q)²), 현 중점 d-좌표 p.
    p* = mean(u_i)로 분리되고, q는 격자 탐색, r²(q) = mean_i[(h_i−q)²+a_i²]."""
    sin_phi = math.sqrt(max(1.0 - float(normal[0]) ** 2, 0.0))
    if sin_phi < 1e-6:
        return None
    ex = np.array([1.0, 0.0, 0.0])
    d = np.cross(normal, ex)
    d /= np.linalg.norm(d)
    u_hat = (ex - float(normal[0]) * normal) / sin_phi   # 평면 내 x 증가 방향
    # 면별 증거 집계: 같은 면 멤버(공선 조각)는 끝점 투영의 '합집합 구간'이 관측 현이다.
    by_face = {}
    for m in members:
        lo = float((m["verts"][0] - centroid) @ d)
        hi = float((m["verts"][-1] - centroid) @ d)
        lo, hi = min(lo, hi), max(lo, hi)
        k = round(float(m["face_x"]), 3)
        if k in by_face:
            by_face[k] = (min(by_face[k][0], lo), max(by_face[k][1], hi), by_face[k][2])
        else:
            by_face[k] = (lo, hi, float(m["face_x"]))
    h = np.array([(v[2] - centroid[0]) / sin_phi for v in by_face.values()])
    a = np.array([0.5 * (v[1] - v[0]) for v in by_face.values()])
    u = np.array([0.5 * (v[1] + v[0]) for v in by_face.values()])
    p_star = float(u.mean())

    def loss(q):
        r2 = float(np.mean((h - q) ** 2 + a ** 2))
        half = np.sqrt(np.maximum(r2 - (h - q) ** 2, 0.0))
        return float(np.sum((half - a) ** 2)), math.sqrt(r2)

    lo, hi = h.min() - 3.0, h.max() + 3.0
    best_q, best_l, best_r = 0.0, float("inf"), None
    for _ in range(3):  # 3단계 격자 세분
        qs = np.linspace(lo, hi, 61)
        for q in qs:
            l, r = loss(q)
            if l < best_l:
                best_l, best_q, best_r = l, float(q), r
        span = (hi - lo) / 60
        lo, hi = best_q - span, best_q + span
    if best_r is None or not math.isfinite(best_r):
        return None
    center = centroid + p_star * d + best_q * u_hat
    return center, best_r


def reconstruct(trace_h5, association, angle_deg, coplanar_m, max_sep_m,
                pos_tol_m=0.4, max_gap=2, target_sets=None, arc_min=90.0,
                kr_map=None, rmax=250.0, exclude_faces=None,
                radius_mode="mean", radius_seed=None, absence_lmin=0.5,
                multi_chord_fit=False, lower_bound="fragment",
                adaptive_sep=False, sep_safety=1.4, sep_cap=8.0):
    rng = np.random.default_rng(radius_seed)  # radius_mode="sample"에서만 사용
    all_face_xs = []
    if radius_mode == "sample":
        with h5py.File(trace_h5, "r") as f:
            if "meta" in f and "face_x_positions_m" in f["meta"]:
                all_face_xs = [float(v) for v in f["meta/face_x_positions_m"][:]]
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
        labels = associate_agglomerative(traces, angle_deg, coplanar_m, max_sep_m,
                                         adaptive_sep=adaptive_sep,
                                         sep_safety=sep_safety, sep_cap=sep_cap)
    else:
        labels = associate_geometric(traces, angle_deg, coplanar_m, max_sep_m)

    clusters = {}
    for t, lab in zip(traces, labels):
        clusters.setdefault(lab, []).append(t)

    discs = []
    for members in clusters.values():
        pts = np.vstack([m["verts"] for m in members])
        set_id = members[0]["set_id"]
        normal, centroid, resid = _fit_plane_svd(pts)
        if len(pts) < 3:
            # 단일 절리선(2점 폴리라인)은 점 SVD가 퇴화(placeholder [1,0,0] 반환)
            # → 실측 trace 법선을 disc 법선으로 사용한다.
            normal = np.asarray(members[0]["normal"], dtype=np.float64)
            normal = normal / np.linalg.norm(normal)
            resid = 0.0
        if normal[0] < 0:  # x축 부호로 일관성
            normal = -normal
        faces = sorted({round(m["face_x"], 3) for m in members})

        # (0) [실험: --multi-chord-fit] 다면 클러스터의 현 보존 적합.
        # 면 교차선들은 disc 평면 내에서 모두 평행(방향 d = n×x̂)이므로,
        # (현방향 위치 p, 수직 위치 q, 반지름 r)의 최소제곱으로 각 멤버 면의
        # 현 길이(2a_i)·위치(u_i)를 동시에 정합시킨다.
        if multi_chord_fit and len(members) >= 2:
            fitted = _chord_fit_multi(members, normal, centroid)
            if fitted is not None:
                center, radius = fitted
                discs.append(dict(
                    set_id=set_id, cx=center[0], cy=center[1], cz=center[2],
                    nx=normal[0], ny=normal[1], nz=normal[2], radius=radius,
                    adoption="deterministic_disc", n_traces=len(members),
                    n_faces=len(faces), residual_m=resid,
                    radius_status="chord_fit", arc_deg=0.0,
                ))
                continue

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
            if lower_bound == "span":
                # 같은면 조각들의 끝-끝 스팬(틈 포함)을 하나의 현으로 본다:
                # association 이 맞다면 균열은 틈을 가로질러야 하므로 유효한 하한.
                spans = []
                by_face_mem = {}
                for m in members:
                    by_face_mem.setdefault(round(float(m["face_x"]), 3), []).append(m)
                for fmem in by_face_mem.values():
                    if len(fmem) == 1:
                        spans.append(float(fmem[0]["chord"]))
                    else:
                        P = np.vstack([m["verts"] for m in fmem])
                        spans.append(float(np.max(np.linalg.norm(
                            P[:, None, :] - P[None, :, :], axis=2))))
                a_lower = 0.5 * max(spans)
            else:
                a_lower = 0.5 * max(m["chord"] for m in members)
            kr = (kr_map or {}).get(set_id)
            if kr is not None and math.isfinite(kr):
                if radius_mode == "sample":
                    # 사후분포 표본 + 현 보존 배치: 관측 chord(2a)를 정확히 재현하도록
                    # 중심을 disc 평면 내 chord 수직방향으로 d=sqrt(r^2-a^2) 이동.
                    # 부재(absence) 조건화: 다른 관측 면에 absence_lmin 이상 현을 남기는
                    # (r, 위치) 표본은 기각·재추출 (hidden 제거 기준과 동일 논리).
                    radius_status = "shrinkage_sampled"
                    own_faces = {round(float(m["face_x"]), 3) for m in members}
                    other_faces = [x for x in all_face_xs
                                   if round(x, 3) not in own_faces]
                    radius = max(a_lower, 0.5)
                    if len(members) == 1:
                        # 현 보존 배치: 하류 모듈 규약(이상화 평면 x=face_x)에서 자기 면
                        # 현이 관측 chord(2a)와 같도록 중심 오프셋 s를 해석적으로 푼다.
                        #   |face_x - c_x(s)| = d*·sinφ,  c(s) = mid + s·ŵ,  d* = √(r²-a²)
                        # 가드: 관측 절리선이 disc 안에 있도록 |s| ≤ d*.
                        fx_own = float(members[0]["face_x"])
                        v = members[0]["verts"]
                        chord_dir = v[-1] - v[0]
                        w = np.cross(normal, chord_dir)
                        w_n = np.linalg.norm(w)
                        w_hat = w / w_n if w_n > 1e-9 else None
                        sin_phi = math.sqrt(max(1.0 - float(normal[0]) ** 2, 0.0))
                        delta = fx_own - float(centroid[0])
                        for _try in range(30):
                            r_try = radius_posterior_sample(a_lower, kr, rmax, rng)
                            d_star = math.sqrt(max(r_try ** 2 - a_lower ** 2, 0.0))
                            if w_hat is None or abs(w_hat[0]) < 1e-6:
                                break  # 오프셋으로 x를 못 움직임(면 평행 disc 등) → 폴백
                            sigmas = [1.0, -1.0] if rng.random() < 0.5 else [-1.0, 1.0]
                            placed = False
                            for sg in sigmas:
                                s = (delta - sg * d_star * sin_phi) / w_hat[0]
                                if abs(s) > d_star + 1e-9:
                                    continue  # 절리선이 disc 밖으로 나감
                                c_try = centroid + s * w_hat
                                if all(_chord_on_other_face(c_try, normal, r_try, xf)
                                       < absence_lmin for xf in other_faces):
                                    radius, center, placed = r_try, c_try, True
                                    break
                            if placed:
                                break
                        # 실패 시 폴백: 최소 disc(r=max(a,0.5)), 중심=절리선 중점
                    else:
                        radius = radius_posterior_sample(a_lower, kr, rmax, rng)
                else:
                    radius = max(radius_posterior_mean(a_lower, kr, rmax), a_lower, 0.5)
                    radius_status = "shrinkage"
            else:
                radius = max(a_lower, 0.5)

        # adoption: 반지름 정보가 있는 disc(determined/shrinkage) → deterministic, 그 외 하한 → orientation_only
        adoption = ("deterministic_disc"
                    if radius_status in ("determined", "shrinkage", "shrinkage_sampled")
                    else "orientation_only")

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
    # ------------------------------------------------------------------------
    # [CLI 인자 요약] — 자세한 도움말: --help
    #   --trace-h5         : 필수
    #   --out-csv          : 필수 — 예: <pdir>/reconstruct/reconstructed_discs.csv
    #   --association      : {agglomerative,geometric,predictive,oracle}, 기본
    #                          agglomerative
    #   --normal-angle-deg : 기본 15.0
    #   --coplanar-dist    : 기본 0.15 — geometric 모드: 공면 허용거리 [m]
    #   --max-centroid-sep : 기본 4.5 — 후보쌍 중심 간 최대거리 [m] (면 간격보다 커야 다면 매칭)
    #   --pos-tol          : 기본 0.4 — predictive 모드: 예측 chord 선까지 허용 수직거리 [m]
    #   --max-gap          : 기본 2 — predictive 모드: 매칭 허용 면 간격(스킵 포함)
    #   --arc-min          : 기본 120.0 — 원적합 반지름 채택 최소 호(arc) 커버리지 [deg]. 낮으면 부분호로 반지름 과대
    #   --kr-summary-csv   : set별 kr_hat CSV. 주면 경계 부족 disc 반지름을 모집단 멱법칙으로
    #                          축소추정(empirical-Bayes)해 추정력↑·과적합↓. 없으면 보수적 하한 사용.
    #   --rmax             : 기본 250.0 — 반지름 상한 [m] (축소추정 적분범위)
    #   --target-set       : 복수값 — 복원 대상 set. Laxemar 공정 비교시 Set 4(지수분포) 제외: --target-
    #                          set 1 2 3 5
    #   --radius-mode      : {mean,sample}, 기본 mean — 경계 부족 disc 반지름: mean=사후평균(하한 0.5
    #                          절단, 기존), sample=사후분포 표본 + 현 보존 배치(관측 chord 정확 재현; 안정성 해석
    #                          입력 권장)
    #   --radius-seed      : sample 모드 난수 시드 (재현성)
    # ------------------------------------------------------------------------
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
    ap.add_argument("--max-centroid-sep", type=float, default=4.5,
                    help="후보쌍 중심 간 최대거리 [m]. 다면 매칭에 임계적: 이 값이 "
                         "면 간격보다 작으면 서로 다른 면 trace 쌍이 후보에서 전부 "
                         "탈락해 다면 disc 가 0개가 된다. 기본 4.5는 면 간격 최대치"
                         "(~2.8m)를 여유있게 커버하며 비인접(면 건너뜀) 매칭은 0. "
                         "6.0 이상은 비인접 과병합 위험.")
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
    ap.add_argument("--radius-mode", choices=["mean", "sample"], default="mean",
                    help="경계 부족 disc 반지름: mean=사후평균(하한 0.5 절단, 기존), "
                         "sample=사후분포 표본 + 현 보존 배치(관측 chord 정확 재현; "
                         "안정성 해석 입력 권장)")
    ap.add_argument("--radius-seed", type=int, default=None,
                    help="sample 모드 난수 시드 (재현성)")
    ap.add_argument("--lower-bound", choices=["fragment", "span"], default="fragment",
                    help="shrinkage 반지름 하한: fragment=조각 최대 현(기존), "
                         "span=같은면 병합 스팬(틈 포함 끝-끝 길이)")
    ap.add_argument("--adaptive-sep", action="store_true",
                    help="근접 게이트를 방향·면간격 적응형으로: gate=clamp("
                         "sep_safety*|Δface_x|/sinθ, same_face, sep_cap). 막장면과 "
                         "평행한(|nx|↑) 절리군일수록 게이트를 넓혀 방향 편향을 줄인다. "
                         "적용 시 --max-centroid-sep 무시.")
    ap.add_argument("--sep-safety", type=float, default=1.4,
                    help="adaptive-sep 안전계수 k (기본 1.4)")
    ap.add_argument("--sep-cap", type=float, default=8.0,
                    help="adaptive-sep 게이트 상한 [m] (비인접 과병합 억제, 기본 8.0)")
    ap.add_argument("--multi-chord-fit", action="store_true",
                    help="[실험] 다면 클러스터의 (중심,반지름)을 멤버 면 현 길이·위치와 "
                         "정합하는 최소제곱으로 결정 (원적합/표본추출 대체)")
    args = ap.parse_args()

    kr_map = None
    if args.kr_summary_csv:
        kr_map = {int(r["set_id"]): float(r["kr_hat"])
                  for r in csv.DictReader(open(args.kr_summary_csv))
                  if r.get("kr_hat") not in (None, "", "nan")}
    discs = reconstruct(args.trace_h5, args.association, args.normal_angle_deg,
                        args.coplanar_dist, args.max_centroid_sep,
                        args.pos_tol, args.max_gap, args.target_set, args.arc_min,
                        kr_map, args.rmax,
                        radius_mode=args.radius_mode, radius_seed=args.radius_seed,
                        multi_chord_fit=args.multi_chord_fit,
                        lower_bound=args.lower_bound,
                        adaptive_sep=args.adaptive_sep, sep_safety=args.sep_safety,
                        sep_cap=args.sep_cap)
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
