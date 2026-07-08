"""Conditional hidden DFN generation (MVP: remove-and-resample).

SKB DFN-R style geometric conditioning. Combines
  * VISIBLE discs   — directly reconstructed from observed traces (kept as-is)
  * HIDDEN discs     — stochastically generated from INVERTED set parameters,
                       with any disc that intersects an observed face removed
                       (that region is already explained by the visible discs).

Output = visible ∪ hidden, plus a SKB Fig. 4-1 style comparison figure
(observed traces = blue, conditioned traces = red) and P21/P32 diagnostics.

Conditioning is LOCAL: hidden discs are generated only in a box around the
observed faces (conditioning is inherently a local operation; the full 250 m
domain holds ~1.8e7 fractures and is neither needed nor tractable here).

Ground truth (dfn_export_for_python.h5) is used for VALIDATION ONLY — never
inside the generation recipe (project policy D013).

Coordinate convention (from the generator): x = East = tunnel advance,
y = North, z = Up. Observation faces are planes of constant x.

Reused, unmodified: sampling functions from ``dfn generator v1/python/generate_dfn.py``.

Usage
-----
    python dfn_analysis/generate_conditional_hidden_dfn.py \
        --pipeline-dir storage/output/pipeline_test_laxemar
"""
# ======================================================================
# [파일 역할]
#   관측된 트레이스(터널 면에 나타난 균열 흔적)로부터 조건부(conditional)
#   3D DFN(이산 균열망)을 생성한다. SKB DFN-R 방식의 기하학적 조건화.
#     - VISIBLE(가시) 디스크 : 관측 트레이스에서 직접 복원된 디스크(그대로 유지)
#     - HIDDEN(은닉) 디스크   : 역산된 세트 파라미터로 확률적으로 생성하되,
#                               관측 면과 교차하는 디스크는 제거(그 영역은 이미
#                               가시 디스크가 설명하므로 remove-and-resample)
#   결과 = 가시 ∪ 은닉.
#
# [주요 입력] (--pipeline-dir 아래)
#   - reconstruct/reconstructed_discs.csv : 복원된 가시 디스크
#   - trace_dataset/trace_dataset_3d.csv  : 관측 트레이스(3D 좌표)
#   - kr/kr_summary_by_set.csv, p32/p32_summary.csv : 역산 파라미터(kr, P32 등)
#   - dfn_export_for_python.h5            : 터널 단면 폴리곤(검증/기하 참조용)
#
# [주요 출력] (--pipeline-dir/conditional_hidden 아래)
#   - conditional_dfn.csv                    : 조건부 DFN 디스크 목록
#   - observed_vs_conditioned_traces.png     : 관측(파랑) vs 조건화(빨강) 비교 그림
#   - 콘솔 진단 로그(P21/P32, 디스크 개수 등)
#
# [핵심 처리 흐름]
#   1) 입력 로드(가시 디스크 / 관측 트레이스 / 역산 파라미터 / 터널 폴리곤)
#   2) 관측 면 주변 국소 박스 정의(조건화는 본질적으로 국소 연산)
#   3) 세트별 은닉 디스크 확률 생성 → 관측 면 교차 디스크 제거
#   4) 가시 디스크를 면에 재투영해 조건화 트레이스 계산
#   5) 진단 출력 + CSV/그림 저장
#
# [좌표 규약] x = East = 터널 굴진 방향, y = North, z = Up.
#             관측 면은 x = 상수 평면.
# ======================================================================
from __future__ import annotations

import argparse
import csv
import importlib.util
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

REPO = Path(__file__).resolve().parent.parent
OBSERVED_COLOR = "tab:blue"
CONDITIONED_COLOR = "tab:red"
POWERLAW_SETS = (1, 2, 3, 5)  # Set 4 is exponential (D002) — excluded from inversion


# ----------------------------------------------------------------------
# Reuse the generator's sampling functions (single source of truth)
# ----------------------------------------------------------------------
# 제너레이터(generate_dfn.py)를 동적 로드하여 샘플링 함수들을 재사용한다.
#   반환값: 로드된 generate_dfn 모듈 객체(단일 진실 공급원)
def _load_generator():
    # generate_dfn.py 파일 경로를 지정하고 importlib로 모듈 스펙 생성/실행
    path = REPO / "dfn generator v1" / "python" / "generate_dfn.py"
    spec = importlib.util.spec_from_file_location("generate_dfn", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


GEN = _load_generator()


# ----------------------------------------------------------------------
# Geometry: disc ∩ face plane, segment ∩ convex polygon window
# ----------------------------------------------------------------------
# 디스크(center, normal, radius)를 평면 x=xf로 절단했을 때 생기는 현(chord)의
# 3D 끝점 두 개를 반환한다. 교차하지 않으면 None.
#   인자: center 중심[xyz], normal 법선, radius 반경, xf 면 위치, tol 허용오차
#   반환: (현 시작점, 현 끝점) 또는 None
def disc_face_chord(
    center: np.ndarray, normal: np.ndarray, radius: float, xf: float, tol: float = 1e-9
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Full chord of a disc cut by the plane x = xf, or None if it misses."""
    # 법선 정규화 및 x축 단위벡터 준비
    n = normal / np.linalg.norm(normal)
    ex = np.array([1.0, 0.0, 0.0])
    # 면까지의 부호있는 거리(중심의 x - 면의 x). 반경보다 멀면 교차 안 함
    dist = float(center[0] - xf)  # signed distance along +x
    if abs(dist) >= radius - tol:
        return None
    # 현 절반 길이(피타고라스) 및 현 방향(디스크 평면과 면 평면의 교선 방향)
    half = math.sqrt(max(radius**2 - dist**2, 0.0))
    chord_dir = np.cross(n, ex)
    m = np.linalg.norm(chord_dir)
    if m < tol:  # disc normal ∥ x → disc plane parallel to face, no line
        return None
    chord_dir /= m
    # 면 법선(x축)의 디스크 평면 내 성분 방향(현 중점을 찾기 위한 방향)
    n_face_proj = ex - (ex @ n) * n  # in-disc component of face normal
    mm = np.linalg.norm(n_face_proj)
    if mm < tol:
        return None
    n_face_proj /= mm
    # 중심에서 면으로 내린 발(foot)을 이용해 현의 중점(mid) 계산
    foot = center - dist * ex
    d_in = float((foot - center) @ n_face_proj)
    mid = center + d_in * n_face_proj
    # 중점에서 현 방향으로 ±half 이동한 두 끝점 반환
    return mid - half * chord_dir, mid + half * chord_dir


# 폴리곤 정점을 반시계(CCW) 방향으로 정렬해 반환한다(Cyrus-Beck 클리핑 전제).
#   인자: poly (N,2) 정점 배열 / 반환: CCW 정렬된 정점 배열
def _ccw_polygon(poly: np.ndarray) -> np.ndarray:
    """Return polygon vertices in counter-clockwise order."""
    # 부호있는 면적(신발끈 공식)의 부호로 방향 판정, 시계면 뒤집기
    x, y = poly[:, 0], poly[:, 1]
    area2 = np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))
    return poly if area2 >= 0 else poly[::-1].copy()


# 2D 선분(y-z 평면)을 볼록 CCW 폴리곤(관측 창)으로 Cyrus-Beck 클리핑한다.
#   인자: p0_yz,p1_yz 선분 끝점(yz), poly_ccw 볼록 폴리곤, tol 허용오차
#   반환: 폴리곤 내부로 잘린 선분 (시작,끝) 또는 완전히 밖이면 None
def clip_segment_to_convex_polygon(
    p0_yz: np.ndarray, p1_yz: np.ndarray, poly_ccw: np.ndarray, tol: float = 1e-12
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Cyrus-Beck clip of a 2D segment against a convex CCW polygon (y-z)."""
    # 선분 방향 d와 진입/이탈 파라미터 t_enter/t_leave 초기화
    d = p1_yz - p0_yz
    t_enter, t_leave = 0.0, 1.0
    n = len(poly_ccw)
    # 각 폴리곤 변에 대해 내향 법선으로 선분을 잘라 t 구간을 좁힌다
    for i in range(n):
        a = poly_ccw[i]
        b = poly_ccw[(i + 1) % n]
        edge = b - a
        # 변의 내향(왼쪽) 법선과 선분 시작점의 내/외 판정값 c0, 방향성분 cd
        inward = np.array([-edge[1], edge[0]])  # left normal (interior side, CCW)
        c0 = float(inward @ (p0_yz - a))
        cd = float(inward @ d)
        # 선분이 변과 평행한 경우: 밖이면 버리고, 안이면 이 변은 건너뜀
        if abs(cd) < tol:
            if c0 < 0:
                return None  # parallel and outside this edge
            continue
        # 변과의 교차 파라미터 t. cd 부호에 따라 진입/이탈 경계 갱신
        t = -c0 / cd
        if cd > 0:
            t_enter = max(t_enter, t)
        else:
            t_leave = min(t_leave, t)
        # 진입이 이탈을 넘어서면 폴리곤과 겹치는 구간 없음
        if t_enter > t_leave:
            return None
    # 최종 진입/이탈 파라미터로 잘린 선분 끝점 반환
    return p0_yz + t_enter * d, p0_yz + t_leave * d


# 디스크가 면(x=xf)에 남기는, 관측 창으로 클리핑된 트레이스의 3D 끝점을 반환.
#   흐름: 디스크∩면 현 계산 → yz로 창 클리핑 → 유효 길이면 3D 끝점 복원
#   반환: (끝점 q0, 끝점 q1) 또는 트레이스 없음 시 None
def visible_trace_on_face(
    center: np.ndarray,
    normal: np.ndarray,
    radius: float,
    xf: float,
    poly_ccw: np.ndarray,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Return the window-clipped trace (3D endpoints) of a disc on face x=xf."""
    # 1) 디스크와 면의 교차 현 계산(없으면 종료)
    chord = disc_face_chord(center, normal, radius, xf)
    if chord is None:
        return None
    # 2) 현의 yz 성분을 관측 창 폴리곤으로 클리핑
    p0, p1 = chord
    clipped = clip_segment_to_convex_polygon(p0[1:3], p1[1:3], poly_ccw)
    if clipped is None:
        return None
    # 3) 클리핑 결과가 사실상 점이면 트레이스로 취급하지 않음
    q0_yz, q1_yz = clipped
    if np.linalg.norm(q1_yz - q0_yz) < 1e-6:
        return None
    # 4) yz 끝점에 면 위치 xf를 붙여 3D 끝점으로 복원
    q0 = np.array([xf, q0_yz[0], q0_yz[1]])
    q1 = np.array([xf, q1_yz[0], q1_yz[1]])
    return q0, q1


# ----------------------------------------------------------------------
# Input loaders
# ----------------------------------------------------------------------
# 복원된 디스크 CSV를 읽어, 지정된 adoption 종류만 가시(visible) 디스크로 로드.
#   인자: path CSV 경로, keep_adoptions 유지할 adoption 문자열 튜플
#   반환: 디스크 dict 리스트(set_id, center, normal, radius, source, adoption)
def load_visible_discs(path: Path, keep_adoptions: Tuple[str, ...]) -> List[dict]:
    """Reconstructed discs kept as the visible (observed) part of the DFN."""
    # CSV 각 행에서 keep_adoptions에 해당하는 디스크만 선별해 담는다
    discs = []
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            if row["adoption"] not in keep_adoptions:
                continue
            discs.append(dict(
                set_id=int(row["set_id"]),
                center=np.array([float(row["cx"]), float(row["cy"]), float(row["cz"])]),
                normal=np.array([float(row["nx"]), float(row["ny"]), float(row["nz"])]),
                radius=float(row["radius"]),
                source="visible",
                adoption=row["adoption"],
            ))
    return discs


# 관측 트레이스 CSV를 읽어 (면 x별 트레이스 목록, 세트별 총 트레이스 길이)로 반환.
#   세트별 길이는 P21 비교를 복원 대상 세트로 한정하기 위해 사용(Set 4 제외).
#   인자: path 트레이스 CSV 경로
#   반환: (by_face: {면x: [(p0,p1),...]}, len_by_set: {set_id: 총길이})
def load_observed_traces(
    path: Path,
) -> Tuple[Dict[float, List[Tuple[np.ndarray, np.ndarray]]], Dict[int, float]]:
    """Observed traces grouped by face x, plus total observed length per set.

    The per-set length lets the P21 comparison be restricted to the sets that
    are actually reconstructed/conditioned (Set 4 is exponential and excluded,
    so including it on the observed side only would bias the comparison).
    """
    by_face: Dict[float, List[Tuple[np.ndarray, np.ndarray]]] = defaultdict(list)
    len_by_set: Dict[int, float] = defaultdict(float)
    with open(path, newline="") as fh:
        # 각 행: 면 x(반올림)로 그룹화, 3D 끝점 저장, 세트별 길이 누적
        for row in csv.DictReader(fh):
            xf = round(float(row["face_x_m"]), 3)
            p0 = np.array([float(row["p0_x"]), float(row["p0_y"]), float(row["p0_z"])])
            p1 = np.array([float(row["p1_x"]), float(row["p1_y"]), float(row["p1_z"])])
            by_face[xf].append((p0, p1))
            len_by_set[int(row["set_id"])] += float(np.linalg.norm(p1 - p0))
    return by_face, len_by_set


# 평균 결과 길이 Rbar로부터 Fisher 분포 집중 파라미터 kappa를 근사한다.
#   인자: Rbar 평균 벡터의 크기(0~1) / 반환: kappa 추정값
def _fisher_kappa(Rbar: float) -> float:
    Rbar = min(Rbar, 0.999999)
    return Rbar * (3.0 - Rbar**2) / (1.0 - Rbar**2)


# 한 세트의 복원된 법선들로부터 평균 극(pole)과 Fisher kappa를 추정한다.
#   인자: discs 디스크 리스트, set_id 대상 세트
#   반환: (평균 단위법선, kappa) 또는 증거 부족 시 None
def set_orientation_from_discs(discs: List[dict], set_id: int) -> Optional[Tuple[np.ndarray, float]]:
    """Mean pole + Fisher kappa for a set, estimated from reconstructed normals."""
    # 해당 세트 법선 수집(2개 미만이면 추정 불가)
    normals = np.array([d["normal"] for d in discs if d["set_id"] == set_id])
    if len(normals) < 2:
        return None
    # 정규화 후 첫 법선 기준으로 부호 정렬(법선 방향 모호성 제거)
    normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)
    ref = normals[0]
    signs = np.sign(normals @ ref)
    signs[signs == 0] = 1.0
    aligned = normals * signs[:, None]
    # 평균 벡터와 그 크기(Rbar)로 평균 극·kappa 산출
    mean_vec = aligned.mean(axis=0)
    Rbar = float(np.linalg.norm(mean_vec))
    if Rbar < 1e-6:
        return None
    return mean_vec / Rbar, _fisher_kappa(Rbar)


# 세트별 역산 파라미터(kr, P32, 유효 rmin, 생성 rmax)를 두 CSV에서 읽어 합친다.
#   인자: kr_csv (kr/rmin/rmax 요약), p32_csv (P32 요약)
#   반환: {set_id: {"kr","rmin","rmax_gen","P32"}}
def load_inverted_params(kr_csv: Path, p32_csv: Path) -> Dict[int, dict]:
    """Per-set inverted kr, P32, effective rmin, generation rmax."""
    # kr 요약에서 kr/rmin/rmax_gen을 세트별로 채운다
    params: Dict[int, dict] = {}
    with open(kr_csv, newline="") as fh:
        for row in csv.DictReader(fh):
            sid = int(row["set_id"])
            params.setdefault(sid, {})
            params[sid]["kr"] = float(row["kr_hat"])
            params[sid]["rmin"] = float(row["set_effective_generation_rmin"])
            params[sid]["rmax_gen"] = float(row["generation_rmax"])
    # P32 요약에서 세트별 P32를 채운다
    with open(p32_csv, newline="") as fh:
        for row in csv.DictReader(fh):
            sid = int(row["set_id"])
            params.setdefault(sid, {})
            params[sid]["P32"] = float(row["P32_hat"])
    return params


# ----------------------------------------------------------------------
# Hidden generation + conditioning
# ----------------------------------------------------------------------
# 국소 박스 안에서 세트별로 조건화 이전의 확률적(은닉) 디스크를 생성한다.
#   인자: params 역산 파라미터, visible_discs 방향 추정용 가시 디스크,
#         box 국소 박스, rmax_local 국소 생성 반경 상한, seed 난수 시드
#   반환: 은닉 디스크 dict 리스트(아직 면 교차 제거 전)
def generate_hidden_discs(
    params: Dict[int, dict],
    visible_discs: List[dict],
    box: dict,
    rmax_local: float,
    seed: int,
) -> List[dict]:
    """Generate unconditioned stochastic discs per set inside the local box."""
    # 박스 부피 V(개수 계산에 사용)
    V = box["dx"] * box["dy"] * box["dz"]
    hidden: List[dict] = []
    # 멱함수 세트(1,2,3,5)에 대해서만 생성(Set 4 지수분포 제외)
    for sid in POWERLAW_SETS:
        # 필수 파라미터(P32, kr) 확인
        p = params.get(sid)
        if p is None or "P32" not in p or "kr" not in p:
            continue
        # 가시 디스크로부터 이 세트의 평균 방향/kappa 추정(부족하면 건너뜀)
        ori = set_orientation_from_discs(visible_discs, sid)
        if ori is None:
            print(f"  [set {sid}] skipped hidden gen: insufficient orientation evidence")
            continue
        mean_n, kappa = ori
        # 크기 분포(멱함수) 정의 후 목표 P32에서 생성 개수 N 산출
        size_dist = {"type": "powerlaw", "kr": p["kr"], "rmin": p["rmin"], "rmax": rmax_local}
        N = GEN.compute_num_fractures_from_P32(p["P32"], size_dist, V)
        if N <= 0:
            continue
        # 세트별로 재현 가능한 시드 파생 후 반경/법선/중심 샘플링
        base = seed + sid * 1000
        radii = GEN.sample_radius(size_dist, N, seed=base)
        normals = GEN.sample_fisher_normals(mean_n, kappa, N, seed=base + 1)
        strike_u, dip_u = GEN.normal_to_strike_dip_basis_vectorized(normals)
        centers = GEN.sample_centers_from_surface_points(
            box, radii, strike_u, dip_u, "area_uniform", seed=base + 2
        )
        # 생성된 N개 디스크를 은닉(hidden)으로 리스트에 추가
        for j in range(N):
            hidden.append(dict(
                set_id=sid, center=centers[j], normal=normals[j],
                radius=float(radii[j]), source="hidden", adoption="stochastic",
            ))
        print(f"  [set {sid}] hidden generated: N={N:,}  (P32={p['P32']:.3f}, kr={p['kr']:.2f}, "
              f"rmin={p['rmin']:.2f}, kappa={kappa:.1f})")
    return hidden


# remove-and-resample의 핵심: 어떤 관측 면에라도 트레이스를 남기는 디스크는 제거.
#   (그 영역은 이미 가시 디스크가 설명하므로 은닉에서 뺀다)
#   인자: discs 은닉 디스크, face_xs 관측 면 x목록, poly_ccw 관측 창
#   반환: (제거되고 남은 디스크 리스트, 제거된 개수)
def remove_face_intersecting(
    discs: List[dict], face_xs: List[float], poly_ccw: np.ndarray
) -> Tuple[List[dict], int]:
    """Drop discs that produce a visible trace on any observed face."""
    # 각 디스크가 관측 면 중 하나라도 교차하면 제거, 아니면 유지
    kept, removed = [], 0
    for d in discs:
        intersects = any(
            visible_trace_on_face(d["center"], d["normal"], d["radius"], xf, poly_ccw) is not None
            for xf in face_xs
        )
        if intersects:
            removed += 1
        else:
            kept.append(d)
    return kept, removed


# ----------------------------------------------------------------------
# Diagnostics + figure
# ----------------------------------------------------------------------
# 폴리곤 면적(신발끈 공식)을 반환한다. P21 계산의 관측 창 면적에 사용.
#   인자: poly (N,2) 정점 / 반환: 면적(양수)
def _polygon_area(poly: np.ndarray) -> float:
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


# 가시 디스크가 각 면에 남기는 트레이스(=조건화 트레이스, 빨강)를 계산한다.
#   인자: visible 가시 디스크, face_xs 면 x목록, poly_ccw 관측 창
#   반환: {면x: [(p0,p1),...]}
def conditioned_traces(
    visible: List[dict], face_xs: List[float], poly_ccw: np.ndarray
) -> Dict[float, List[Tuple[np.ndarray, np.ndarray]]]:
    """Traces the VISIBLE discs leave on each face (the conditioned traces)."""
    # 각 가시 디스크를 모든 면에 재투영하여 면별 트레이스로 모은다
    by_face: Dict[float, List[Tuple[np.ndarray, np.ndarray]]] = defaultdict(list)
    for d in visible:
        for xf in face_xs:
            seg = visible_trace_on_face(d["center"], d["normal"], d["radius"], xf, poly_ccw)
            if seg is not None:
                by_face[xf].append(seg)
    return by_face


# 조건부 DFN의 진단 지표(디스크 개수, 트레이스 개수, P21/P32 등)를 콘솔에 출력.
#   인자: 면 목록/창/관측·조건화 트레이스/세트별 관측길이/각종 개수/파라미터
#   반환: 없음(표준출력으로 진단 로그만 출력)
def print_diagnostics(
    face_xs, poly_ccw, observed, cond, obs_len_by_set, n_visible, n_hidden_gen,
    n_hidden_removed, n_hidden_kept, params,
):
    # 관측 창 총 면적(창 면적 × 면 개수)과 트레이스 총 길이/개수 집계
    window_area = _polygon_area(poly_ccw) * len(face_xs)
    cond_len = sum(np.linalg.norm(p1 - p0) for segs in cond.values() for p0, p1 in segs)
    n_obs = sum(len(s) for s in observed.values())
    n_cond = sum(len(s) for s in cond.values())

    # 관측 P21은 전체 세트 기준과, 복원 대상 세트로 한정한 기준을 함께 계산
    obs_len_all = sum(obs_len_by_set.values())
    # Conditioned traces come only from reconstructed sets, so the fair P21
    # comparison restricts the observed side to the same sets (Set 4 excluded).
    obs_len_matched = sum(L for s, L in obs_len_by_set.items() if s in POWERLAW_SETS)
    p21_obs_all = obs_len_all / window_area if window_area else 0.0
    p21_obs_matched = obs_len_matched / window_area if window_area else 0.0
    p21_cond = cond_len / window_area if window_area else 0.0
    p32_target = sum(params[s]["P32"] for s in POWERLAW_SETS if s in params and "P32" in params[s])

    # 진단 결과 표 형태로 출력(디스크 개수 → 트레이스/P21 → 세트한정 비교 → 목표 P32)
    print("-" * 64)
    print("Conditional hidden DFN - diagnostics")
    print("-" * 64)
    print(f"  observation faces        : {face_xs}")
    print(f"  visible discs (kept)     : {n_visible}")
    print(f"  hidden discs generated   : {n_hidden_gen:,}")
    print(f"  hidden removed (face-hit) : {n_hidden_removed:,}")
    print(f"  hidden discs kept        : {n_hidden_kept:,}")
    print(f"  total conditional discs  : {n_visible + n_hidden_kept:,}")
    print("  " + "-" * 40)
    print(f"  observed traces (blue)   : {n_obs}  | P21(all sets)      = {p21_obs_all:.3f} 1/m")
    print(f"  conditioned traces (red) : {n_cond}  | P21               = {p21_cond:.3f} 1/m")
    print("  " + "-" * 40)
    print(f"  set-matched comparison (reconstructed sets {POWERLAW_SETS}, Set 4 excluded both sides):")
    print(f"    observed  P21 (matched): {p21_obs_matched:.3f} 1/m")
    print(f"    conditioned P21        : {p21_cond:.3f} 1/m")
    if p21_obs_matched > 0:
        err = (p21_cond - p21_obs_matched) / p21_obs_matched * 100
        print(f"    P21 error (matched)    : {err:+.1f} %")
    print(f"  target P32 (sets {POWERLAW_SETS}) : {p32_target:.3f} 1/m")
    print("-" * 64)


# SKB Fig.4-1 스타일 3D 비교 그림 저장: 관측(파랑) vs 조건화(빨강) 트레이스.
#   인자: 관측/조건화 트레이스, 면 목록, 관측 창, 은닉 디스크, 저장 경로, 시드,
#         show_hidden(은닉 중심 오버레이 여부)
#   반환: 없음(PNG 파일 저장)
def plot_fig(observed, cond, face_xs, poly_ccw, hidden_kept, out_path, seed, show_hidden=False):
    # 3D 축 생성 및 관측 창 y/z 범위 계산
    fig = plt.figure(figsize=(13, 6))
    ax = fig.add_subplot(111, projection="3d")

    ymin, ymax = poly_ccw[:, 0].min(), poly_ccw[:, 0].max()
    zmin, zmax = poly_ccw[:, 1].min(), poly_ccw[:, 1].max()

    # Observation window on each face
    for xf in face_xs:
        loop = np.vstack([poly_ccw, poly_ccw[0]])
        ax.plot(np.full(len(loop), xf), loop[:, 0], loop[:, 1], color="0.6", lw=0.8, alpha=0.7)

    # Optional: hidden disc centers near the window slab (to show volume filling)
    if show_hidden and hidden_kept:
        hc = np.array([d["center"] for d in hidden_kept])
        m = ((hc[:, 1] >= ymin - 1) & (hc[:, 1] <= ymax + 1)
             & (hc[:, 2] >= zmin - 1) & (hc[:, 2] <= zmax + 1))
        ax.scatter(hc[m, 0], hc[m, 1], hc[m, 2], s=2, color="0.8", alpha=0.2)

    # 관측 트레이스(파랑)와 조건화 트레이스(빨강)를 각각 선으로 그림
    for segs in observed.values():
        for p0, p1 in segs:
            ax.plot([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]],
                    color=OBSERVED_COLOR, lw=1.3, alpha=0.9)
    for segs in cond.values():
        for p0, p1 in segs:
            ax.plot([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]],
                    color=CONDITIONED_COLOR, lw=1.3, alpha=0.9)

    # 축 라벨/제목(좌표 규약 명시)
    ax.set_xlabel("x — tunnel advance [m]")
    ax.set_ylabel("y — North [m]")
    ax.set_zlabel("z — Up [m]")
    ax.set_title(f"Observed (blue) vs conditioned (red) traces on tunnel faces  (seed={seed})")

    # Zoom to the tunnel window slab so traces fill the frame
    ax.set_xlim(min(face_xs) - 0.5, max(face_xs) + 0.5)
    ax.set_ylim(ymin - 0.5, ymax + 0.5)
    ax.set_zlim(zmin - 0.5, zmax + 0.5)
    ax.set_box_aspect((max(face_xs) - min(face_xs) + 1, ymax - ymin + 1, zmax - zmin + 1))
    ax.view_init(elev=18, azim=-72)
    ax.legend(handles=[
        Line2D([0], [0], color=OBSERVED_COLOR, lw=1.5, label="Observed traces"),
        Line2D([0], [0], color=CONDITIONED_COLOR, lw=1.5, label="Conditioned traces (visible discs)"),
        Line2D([0], [0], color="0.6", lw=1.0, label="Observation window"),
    ], loc="upper right")

    # 출력 폴더 생성 후 그림 저장
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Figure written to {out_path}")


# 최종 조건부 DFN(가시 + 유지된 은닉)을 conditional_dfn.csv로 저장한다.
#   인자: visible 가시 디스크, hidden_kept 유지된 은닉 디스크, out_path 저장 경로
#   반환: 없음(CSV 저장)
def write_dfn_csv(visible, hidden_kept, out_path: Path):
    # 헤더 기록 후 가시+은닉 디스크를 한 행씩 기록
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["source", "set_id", "cx", "cy", "cz", "nx", "ny", "nz", "radius", "adoption"])
        for d in visible + hidden_kept:
            c, n = d["center"], d["normal"]
            w.writerow([d["source"], d["set_id"], c[0], c[1], c[2],
                        n[0], n[1], n[2], d["radius"], d["adoption"]])
    print(f"Conditional DFN written to {out_path}")


# ----------------------------------------------------------------------
# 엔트리 포인트: 인자 파싱 → 입력 로드 → 국소 박스 정의 → 은닉 생성/조건화 →
#                진단 출력 → CSV/그림 저장까지 전체 파이프라인을 실행한다.
def main() -> None:
    # CLI 인자 정의(파이프라인 폴더, 국소 반경/여유, 시드, 유지할 adoption 등)
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pipeline-dir", type=Path,
                    default=REPO / "storage/output/pipeline_test_laxemar")
    ap.add_argument("--rmax-local", type=float, default=10.0,
                    help="Upper radius cutoff for local hidden generation [m].")
    ap.add_argument("--margin", type=float, default=None,
                    help="Local box margin around faces/window [m]. Default = rmax-local.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--keep-adoptions", default="deterministic_disc,orientation_only",
                    help="Comma list of reconstructed-disc adoptions to keep as visible.")
    ap.add_argument("--show-hidden", action="store_true",
                    help="Overlay hidden disc centers near the window (default off).")
    args = ap.parse_args()

    # 파이프라인 폴더와 국소 박스 여유(margin, 기본값=rmax_local) 결정
    pdir = args.pipeline_dir
    margin = args.margin if args.margin is not None else args.rmax_local

    # --- Load inputs ---
    # 가시 디스크 / 관측 트레이스 / 역산 파라미터 로드
    keep = tuple(a.strip() for a in args.keep_adoptions.split(","))
    visible = load_visible_discs(pdir / "reconstruct/reconstructed_discs.csv", keep)
    observed, obs_len_by_set = load_observed_traces(pdir / "trace_dataset/trace_dataset_3d.csv")
    params = load_inverted_params(pdir / "kr/kr_summary_by_set.csv", pdir / "p32/p32_summary.csv")

    # 관측 면 x목록과 터널 단면 폴리곤(YZ) 로드 후 CCW 정렬
    face_xs = sorted(observed.keys())
    with h5py.File(pdir / "dfn_export_for_python.h5", "r") as f:
        poly = np.array(f["tunnel/poly_YZ"])
    poly_ccw = _ccw_polygon(poly)

    # --- Local conditioning box around the observed faces ---
    # 관측 창 y/z 범위에 margin을 더해 국소 조건화 박스 정의(x는 면 범위 기준)
    ymin, ymax = poly_ccw[:, 0].min(), poly_ccw[:, 0].max()
    zmin, zmax = poly_ccw[:, 1].min(), poly_ccw[:, 1].max()
    box = dict(
        x0=min(face_xs) - margin, dx=(max(face_xs) - min(face_xs)) + 2 * margin,
        y0=ymin - margin, dy=(ymax - ymin) + 2 * margin,
        z0=zmin - margin, dz=(zmax - zmin) + 2 * margin,
    )
    print(f"Local conditioning box: x[{box['x0']:.1f},{box['x0']+box['dx']:.1f}] "
          f"y[{box['y0']:.1f},{box['y0']+box['dy']:.1f}] z[{box['z0']:.1f},{box['z0']+box['dz']:.1f}]")
    print(f"Visible discs kept ({keep}): {len(visible)}")

    # --- Generate + condition ---
    # 은닉 디스크 생성 후, 관측 면과 교차하는 것을 제거(remove-and-resample)
    hidden_all = generate_hidden_discs(params, visible, box, args.rmax_local, args.seed)
    hidden_kept, n_removed = remove_face_intersecting(hidden_all, face_xs, poly_ccw)

    # --- Conditioned traces (red) = visible discs re-projected onto faces ---
    # 가시 디스크를 면에 재투영해 조건화 트레이스 계산
    cond = conditioned_traces(visible, face_xs, poly_ccw)

    # --- Diagnostics + outputs ---
    # 진단 출력 후 조건부 DFN CSV와 비교 그림 저장
    print_diagnostics(face_xs, poly_ccw, observed, cond, obs_len_by_set, len(visible),
                      len(hidden_all), n_removed, len(hidden_kept), params)
    out_dir = pdir / "conditional_hidden"
    write_dfn_csv(visible, hidden_kept, out_dir / "conditional_dfn.csv")
    plot_fig(observed, cond, face_xs, poly_ccw, hidden_kept,
             out_dir / "observed_vs_conditioned_traces.png", args.seed, args.show_hidden)


if __name__ == "__main__":
    main()
