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
# [파일 역할] 조건부(conditional) 은닉 DFN 생성 (MVP: 제거-후-재샘플링 방식)
#   - VISIBLE(가시) disc: 관측 트레이스에서 직접 복원된 disc를 그대로 유지
#   - HIDDEN(은닉) disc: 역산된 set 파라미터로 확률적으로 생성하되,
#     관측면(face)과 교차하는 disc는 제거(그 영역은 이미 가시 disc가 설명함)
#   - SKB DFN-R 스타일의 기하학적 조건화(geometric conditioning)를 구현
#
# [주요 입력]
#   - reconstruct/reconstructed_discs.csv : 복원된 가시 disc
#   - trace_dataset/trace_dataset_3d.csv  : 관측 트레이스(3D)
#   - kr/kr_summary_by_set.csv            : set별 역산 kr / rmin / rmax
#   - p32/p32_summary.csv                 : set별 역산 P32
#   - dfn_export_for_python.h5            : 터널 관측창 폴리곤(검증/기하용)
#
# [주요 출력]
#   - conditional_hidden/conditional_dfn.csv               : visible ∪ hidden disc
#   - conditional_hidden/observed_vs_conditioned_traces.png: 관측(파랑) vs 조건화(빨강) 비교 그림
#   - 콘솔 진단: 트레이스 개수, P21/P32 등
#
# [핵심 처리 흐름]
#   입력 로드 → 로컬 조건화 박스 설정 → 은닉 disc 생성(set별 분포타입 분기)
#   → 관측면 교차 disc 제거 → 조건화 트레이스 계산 → 진단/CSV/그림 출력
#
# [주의] 대상 set은 하드코딩이 아니라 데이터(역산 파라미터)에서 자동 유도된다.
#        set별 크기분포는 params["dist_type"](powerlaw/exponential)로 분기한다.
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
from scipy.integrate import quad
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

REPO = Path(__file__).resolve().parent.parent
OBSERVED_COLOR = "tab:blue"
CONDITIONED_COLOR = "tab:red"
# 대상 set은 하드코딩하지 않고 역산 파라미터가 존재하는 set에서 자동 유도한다
# (예: Laxemar는 Set4=지수분포라 kr 미보유 → 자동 제외; Forsmark는 5개 모두 powerlaw → 전부 포함).
# --sets / --exclude-sets 로 수동 지정 가능. per-set 분포타입은 params["dist_type"](기본 powerlaw).


# ----------------------------------------------------------------------
# Reuse the generator's sampling functions (single source of truth)
# ----------------------------------------------------------------------
# [함수] 생성기(generate_dfn.py) 모듈을 동적으로 로드한다.
#   - 목적: 샘플링 로직(반지름/방향/중심)을 원본 생성기와 동일하게 재사용(단일 소스)
#   - 인자: 없음
#   - 반환: 로드된 generate_dfn 모듈 객체
def _load_generator():
    path = REPO / "dfn generator v1" / "python" / "generate_dfn.py"
    spec = importlib.util.spec_from_file_location("generate_dfn", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


GEN = _load_generator()


# ----------------------------------------------------------------------
# Geometry: disc ∩ face plane, segment ∩ convex polygon window
# ----------------------------------------------------------------------
# [함수] disc가 관측면(x=xf 평면)에 의해 잘려 생기는 현(chord, 전체 선분)을 계산한다.
#   - 주요 인자: center/normal/radius (disc 중심/법선/반지름), xf (관측면의 x좌표)
#   - 반환: 현의 양 끝점(3D) 튜플, disc가 면을 만나지 않으면 None
def disc_face_chord(
    center: np.ndarray, normal: np.ndarray, radius: float, xf: float, tol: float = 1e-9
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Full chord of a disc cut by the plane x = xf, or None if it misses."""
    n = normal / np.linalg.norm(normal)
    ex = np.array([1.0, 0.0, 0.0])
    dist = float(center[0] - xf)  # signed distance along +x
    # 중심에서 면까지의 x거리가 반지름 이상이면 disc가 면에 닿지 않음
    if abs(dist) >= radius - tol:
        return None
    # 현의 반길이(half): 반지름과 면까지 거리로 피타고라스 계산
    half = math.sqrt(max(radius**2 - dist**2, 0.0))
    # 현의 방향: disc 법선과 x축의 외적 (면과 disc 평면의 교선 방향)
    chord_dir = np.cross(n, ex)
    m = np.linalg.norm(chord_dir)
    if m < tol:  # disc normal ∥ x → disc plane parallel to face, no line
        return None
    chord_dir /= m
    # disc 평면 안에서 면 법선(x축)의 성분 방향 (현 중점을 찾기 위한 축)
    n_face_proj = ex - (ex @ n) * n  # in-disc component of face normal
    mm = np.linalg.norm(n_face_proj)
    if mm < tol:
        return None
    n_face_proj /= mm
    # 중심을 면에 수직 투영한 발(foot) → 현의 중점(mid) 계산
    foot = center - dist * ex
    d_in = float((foot - center) @ n_face_proj)
    mid = center + d_in * n_face_proj
    # 현 중점에서 현 방향으로 ±half 이동한 두 끝점 반환
    return mid - half * chord_dir, mid + half * chord_dir


# [함수] 폴리곤 정점을 반시계(CCW) 방향으로 정렬해 반환한다.
#   - 인자: poly (Nx2 정점 배열), 반환: CCW 정렬된 정점 배열
#   - 부호 있는 면적(area2)이 음수면 순서를 뒤집는다.
def _ccw_polygon(poly: np.ndarray) -> np.ndarray:
    """Return polygon vertices in counter-clockwise order."""
    x, y = poly[:, 0], poly[:, 1]
    area2 = np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))
    return poly if area2 >= 0 else poly[::-1].copy()


# [함수] 2D 선분을 볼록(convex) 폴리곤 관측창에 대해 잘라낸다(Cyrus-Beck 클리핑).
#   - 주요 인자: p0_yz/p1_yz (선분 양끝, y-z 면 좌표), poly_ccw (CCW 볼록 폴리곤)
#   - 반환: 관측창 내부로 잘린 선분 양끝점, 완전히 밖이면 None
def clip_segment_to_convex_polygon(
    p0_yz: np.ndarray, p1_yz: np.ndarray, poly_ccw: np.ndarray, tol: float = 1e-12
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Cyrus-Beck clip of a 2D segment against a convex CCW polygon (y-z)."""
    d = p1_yz - p0_yz
    # 매개변수 t 구간 [t_enter, t_leave]을 각 에지로 조여 나간다
    t_enter, t_leave = 0.0, 1.0
    n = len(poly_ccw)
    for i in range(n):
        # 폴리곤의 각 에지(a→b)와 그 안쪽 법선(inward) 계산
        a = poly_ccw[i]
        b = poly_ccw[(i + 1) % n]
        edge = b - a
        inward = np.array([-edge[1], edge[0]])  # left normal (interior side, CCW)
        c0 = float(inward @ (p0_yz - a))
        cd = float(inward @ d)
        # 선분이 에지와 평행한 경우: 바깥쪽이면 탈락, 안쪽이면 무시
        if abs(cd) < tol:
            if c0 < 0:
                return None  # parallel and outside this edge
            continue
        # 에지와의 교차 매개변수 t; 진입/이탈 여부로 구간 갱신
        t = -c0 / cd
        if cd > 0:
            t_enter = max(t_enter, t)
        else:
            t_leave = min(t_leave, t)
        # 진입이 이탈을 넘어서면 내부 구간 없음 → 탈락
        if t_enter > t_leave:
            return None
    return p0_yz + t_enter * d, p0_yz + t_leave * d


# [함수] disc가 관측면(x=xf) 위에 남기는, 관측창으로 잘린 트레이스를 계산한다.
#   - 처리: (1) disc∩면 현 계산 → (2) 현을 관측창 폴리곤으로 클리핑
#   - 반환: 트레이스 양끝점(3D, x=xf 고정), 교차/가시 트레이스가 없으면 None
def visible_trace_on_face(
    center: np.ndarray,
    normal: np.ndarray,
    radius: float,
    xf: float,
    poly_ccw: np.ndarray,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Return the window-clipped trace (3D endpoints) of a disc on face x=xf."""
    # (1) disc와 면의 교차 현(전체 선분) 계산
    chord = disc_face_chord(center, normal, radius, xf)
    if chord is None:
        return None
    # (2) 현의 y-z 성분을 관측창 폴리곤으로 클리핑
    p0, p1 = chord
    clipped = clip_segment_to_convex_polygon(p0[1:3], p1[1:3], poly_ccw)
    if clipped is None:
        return None
    # 길이가 거의 0인 트레이스는 무시
    q0_yz, q1_yz = clipped
    if np.linalg.norm(q1_yz - q0_yz) < 1e-6:
        return None
    # y-z 끝점을 x=xf를 붙여 3D 좌표로 복원
    q0 = np.array([xf, q0_yz[0], q0_yz[1]])
    q1 = np.array([xf, q1_yz[0], q1_yz[1]])
    return q0, q1


# ----------------------------------------------------------------------
# Input loaders
# ----------------------------------------------------------------------
# [함수] 복원된 disc CSV를 읽어, 지정한 adoption 유형만 가시(visible) disc로 로드한다.
#   - 주요 인자: path (reconstructed_discs.csv), keep_adoptions (유지할 adoption 튜플)
#   - 반환: disc dict 리스트 (set_id/center/normal/radius/source/adoption)
def load_visible_discs(path: Path, keep_adoptions: Tuple[str, ...]) -> List[dict]:
    """Reconstructed discs kept as the visible (observed) part of the DFN."""
    discs = []
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            # 유지 대상 adoption이 아니면 건너뜀
            if row["adoption"] not in keep_adoptions:
                continue
            d = dict(
                set_id=int(row["set_id"]),
                center=np.array([float(row["cx"]), float(row["cy"]), float(row["cz"])]),
                normal=np.array([float(row["nx"]), float(row["ny"]), float(row["nz"])]),
                radius=float(row["radius"]),
                source="visible",
                adoption=row["adoption"],
            )
            # 반지름 사후표본용 관측 정보(있을 때만; 구버전 CSV 호환)
            if row.get("a_half"):
                d.update(radius_status=row["radius_status"], a_half=float(row["a_half"]),
                         censored=bool(int(row["censored"])),
                         chord_mid=np.array([float(row[k]) for k in ("chord_mx", "chord_my", "chord_mz")]),
                         chord_dir=np.array([float(row[k]) for k in ("chord_ux", "chord_uy", "chord_uz")]),
                         faces=[float(x) for x in row["faces"].split()])
            discs.append(d)
    return discs


# [함수] 관측 반현 a 에 대한 반지름 사후표본 1개.
#   사전 g(R) ∝ R^-kr (크기편향 멱법칙), 현 offset 균일 가정.
#   우도: 비절단 trace  p(c = a | R) = a / (R sqrt(R^2 - a^2))
#         절단 trace    P(c >= a | R) = sqrt(1 - a^2 / R^2)   (관측 길이는 현의 하한)
#   R = a cosh(u) 치환으로 R→a 특이점 제거, u 격자 역CDF 표본.
def sample_radius_posterior(a, kr, censored, rmax, rng, rmin=0.0, n_grid=400):
    if rmax <= a:
        return a
    # 모집단 하한 rmin (a 보다 크면 R >= rmin 으로 절단)
    u = np.linspace(math.acosh(max(rmin, a) / a), math.acosh(rmax / a), n_grid)
    R = a * np.cosh(u)
    w = R ** (-kr - 1.0) if not censored else R ** (-kr) * np.tanh(u) * np.sinh(u)
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (w[1:] + w[:-1]) * np.diff(u))])
    return float(a * np.cosh(np.interp(rng.uniform() * cdf[-1], cdf, u)))


# [함수] 앙상블 실현 1개용: shrinkage 반지름 가시 disc 의 반지름·중심을 사후분포에서 다시 뽑는다.
#   - 표본 disc 는 관측 최장 현을 포함하도록 중심을 둔다 (비절단: 현 = 관측 현, 절단: 현 ⊇ 관측 현,
#     절단 쪽 연장은 좌우 균일 가정).
#   - 관측 정합성: disc 가 관측되지 않은 막장면에 lmin_det 이상 trace 를 만들거나, 관측된 막장면에
#     trace 를 만들지 못하면 기각. max_tries 초과 시 원래(점추정) disc 유지.
#   - determined / lower_bound(kr 없음) disc 와 관측 정보가 없는 disc 는 그대로 둔다.
#   - 반환: (새 disc 리스트, 통계 dict)
def sample_visible_discs(visible, params, face_xs, poly_ccw, lmin_det, rmax, seed, max_tries=200):
    rng = np.random.default_rng(seed + 777)
    out, n_s, n_fb = [], 0, 0
    for d in visible:
        kr = params.get(d["set_id"], {}).get("kr")
        rmin = params.get(d["set_id"], {}).get("rmin", 0.0)
        if d.get("radius_status") != "shrinkage" or kr is None:
            out.append(d)
            continue
        n = d["normal"] / np.linalg.norm(d["normal"])
        t = d["chord_dir"] - (d["chord_dir"] @ n) * n
        t /= np.linalg.norm(t)
        q = np.cross(n, t)
        a = d["a_half"]
        own = {round(x, 3) for x in d["faces"]}
        for _ in range(max_tries):
            R = sample_radius_posterior(a, kr, d["censored"], rmax, rng, rmin=rmin)
            if d["censored"]:
                off = rng.uniform(0.0, math.sqrt(max(R * R - a * a, 0.0)))
                c = math.sqrt(max(R * R - off * off, 0.0))
                shift = rng.uniform(-(c - a), c - a)
            else:
                off, shift = math.sqrt(max(R * R - a * a, 0.0)), 0.0
            center = d["chord_mid"] + shift * t + (1 if rng.uniform() < 0.5 else -1) * off * q
            ok = True
            for xf in face_xs:
                seg = visible_trace_on_face(center, n, R, xf, poly_ccw)
                length = 0.0 if seg is None else float(np.linalg.norm(seg[1] - seg[0]))
                if (round(xf, 3) in own and length <= 0.0) or (round(xf, 3) not in own and length >= lmin_det):
                    ok = False
                    break
            if ok:
                out.append(dict(d, center=center, radius=R))
                n_s += 1
                break
        else:
            out.append(d)
            n_fb += 1
    return out, dict(sampled=n_s, fallback=n_fb)


# [함수] 관측 트레이스 CSV를 읽어 (a) 면(x)별 트레이스, (b) set별 총 트레이스 길이를 반환한다.
#   - 주요 인자: path (trace_dataset_3d.csv)
#   - 반환: (by_face: 면 x -> [(p0,p1)...], len_by_set: set_id -> 총길이)
#   - set별 길이는 조건화된 set으로만 P21 비교를 한정하는 데 사용(공정 비교)
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
        for row in csv.DictReader(fh):
            # 면 x좌표(반올림)를 키로, 트레이스 양끝점을 3D로 파싱
            xf = round(float(row["face_x_m"]), 3)
            p0 = np.array([float(row["p0_x"]), float(row["p0_y"]), float(row["p0_z"])])
            p1 = np.array([float(row["p1_x"]), float(row["p1_y"]), float(row["p1_z"])])
            by_face[xf].append((p0, p1))
            # set별 트레이스 길이 누적
            len_by_set[int(row["set_id"])] += float(np.linalg.norm(p1 - p0))
    return by_face, len_by_set


# [함수] 평균 결과길이 Rbar로부터 Fisher 분포 집중도(kappa)를 추정한다(근사식).
#   - 인자: Rbar (단위벡터 평균의 크기, 0~1), 반환: kappa 추정값
def _fisher_kappa(Rbar: float) -> float:
    Rbar = min(Rbar, 0.999999)
    return Rbar * (3.0 - Rbar**2) / (1.0 - Rbar**2)


# [함수] 특정 set의 복원 disc 법선들로부터 평균 극(pole)과 Fisher kappa를 추정한다.
#   - 주요 인자: discs (disc 리스트), set_id (대상 set)
#   - 반환: (평균 단위법선, kappa) 튜플, 법선이 2개 미만이면 None
def set_orientation_from_discs(discs: List[dict], set_id: int) -> Optional[Tuple[np.ndarray, float]]:
    """Mean pole + Fisher kappa for a set, estimated from reconstructed normals."""
    normals = np.array([d["normal"] for d in discs if d["set_id"] == set_id])
    if len(normals) < 2:
        return None
    normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)
    # 법선 부호를 기준벡터(ref)에 맞춰 정렬 (양극/음극 혼재로 평균이 상쇄되는 것 방지)
    ref = normals[0]
    signs = np.sign(normals @ ref)
    signs[signs == 0] = 1.0
    aligned = normals * signs[:, None]
    # 평균벡터의 크기(Rbar) → kappa, 방향 → 평균 극
    mean_vec = aligned.mean(axis=0)
    Rbar = float(np.linalg.norm(mean_vec))
    if Rbar < 1e-6:
        return None
    return mean_vec / Rbar, _fisher_kappa(Rbar)


# [함수] set별 역산 파라미터(kr/P32/rmin/rmax)를 두 CSV에서 읽어 하나의 dict로 합친다.
#   - 주요 인자: kr_csv (kr_summary_by_set.csv), p32_csv (p32_summary.csv)
#   - 반환: {set_id: {kr, rmin, rmax_gen, P32, dist_type}} 딕셔너리
def load_inverted_params(kr_csv: Path, p32_csv: Path) -> Dict[int, dict]:
    """Per-set inverted kr, P32, effective rmin, generation rmax."""
    params: Dict[int, dict] = {}
    # kr CSV: set별 멱함수 지수 kr, 유효 rmin, 생성 rmax 로드
    with open(kr_csv, newline="") as fh:
        for row in csv.DictReader(fh):
            sid = int(row["set_id"])
            params.setdefault(sid, {})
            params[sid]["kr"] = float(row["kr_hat"])
            params[sid]["rmin"] = float(row["set_effective_generation_rmin"])
            params[sid]["rmax_gen"] = float(row["generation_rmax"])
    # P32 CSV: set별 역산 P32(면적 밀도) 로드
    with open(p32_csv, newline="") as fh:
        for row in csv.DictReader(fh):
            sid = int(row["set_id"])
            params.setdefault(sid, {})
            params[sid]["P32"] = float(row["P32_hat"])
    # 분포 타입 기본값(powerlaw); 외부 config로 set별 override 가능(예: 지수분포)
    for sid in params:
        params[sid].setdefault("dist_type", "powerlaw")
    return params


# ----------------------------------------------------------------------
# Hidden generation + conditioning
# ----------------------------------------------------------------------
# [함수] 로컬 박스 안에서 set별로 조건화 전(unconditioned) 확률적 은닉 disc를 생성한다.
#   - 주요 인자: params (set별 역산 파라미터), visible_discs (방향 추정용 복원 disc),
#     box (로컬 생성 박스), rmax_local (로컬 최대 반지름), seed, target_sets (대상 set)
#   - 반환: 생성된 은닉 disc dict 리스트
#   - 핵심: target_sets는 데이터 유도값(상위에서 전달), 크기분포는 dist_type로 분기
def generate_hidden_discs(
    params: Dict[int, dict],
    visible_discs: List[dict],
    box: dict,
    rmax_local: float,
    seed: int,
    target_sets: List[int],
    rmax_far: Optional[float] = None,
) -> List[dict]:
    """Generate unconditioned stochastic discs per set inside the local box.

    rmax_far (선택): 주면 2단 생성. 반지름 (rmax_local, rmax_far] 균열을 box 를 사방으로
    (rmax_far - rmax_local) 만큼 넓힌 박스에서 추가 생성한다(box 여유 = rmax_local 가정).
    set P32 는 [rmin, rmax_far] 절단 분포 기준으로 두 단에 면적 모멘트 비율대로 나눈다.
    None 이면 종전과 동일(rmax_local 절단, P32 전량을 국소 박스에).

    target_sets는 하드코딩이 아니라 상위(main)에서 데이터로부터 유도해 전달한다.
    per-set 분포타입(params["dist_type"])에 따라 powerlaw/exponential 크기분포로 생성한다.
    """
    # 로컬 박스 부피 V: P32로부터 생성 개수 N 계산에 사용
    V = box["dx"] * box["dy"] * box["dz"]
    hidden: List[dict] = []
    for sid in target_sets:
        # 해당 set의 역산 파라미터가 없거나 P32가 없으면 건너뜀
        p = params.get(sid)
        if p is None or "P32" not in p:
            continue
        # set별 크기분포 타입 결정 (기본 powerlaw, 지수분포 등 override 가능)
        dist_type = p.get("dist_type", "powerlaw")
        # 크기분포 구성 (powerlaw는 kr, exponential은 r0 필요)
        if dist_type == "exponential":
            r0 = p.get("r0")
            if r0 is None:
                print(f"  [set {sid}] exponential set needs r0 (via --config); skipped")
                continue
            size_dist = {"type": "exponential", "r0": float(r0),
                         "rmin": p.get("rmin", 0.5), "rmax": rmax_local}
            size_desc = f"exp r0={float(r0):.2f}"
        else:  # powerlaw
            if "kr" not in p:
                print(f"  [set {sid}] powerlaw set has no kr; skipped")
                continue
            size_dist = {"type": "powerlaw", "kr": p["kr"], "rmin": p["rmin"], "rmax": rmax_local}
            size_desc = f"kr={p['kr']:.2f}"
        # 방향(orientation): config 에 trace 기반 추정값(trend/plunge/kappa)이 있으면 그것을 쓰고,
        # 없으면 복원 disc 법선에서 추정한다. 복원 disc 는 법선을 공유하는 경우가 많아
        # (예: 한 set 의 disc 전부 동일 법선) 후자는 kappa 를 과대 추정할 수 있다.
        # config 규약(build_dataset_config_from_traces): pole = [cosP cosT, cosP sinT, -sinP]
        if "orientation" in p:
            t, pl, k = np.radians(p["orientation"][0]), np.radians(p["orientation"][1]), p["orientation"][2]
            ori = (np.array([np.cos(pl) * np.cos(t), np.cos(pl) * np.sin(t), -np.sin(pl)]), k)
        else:
            ori = set_orientation_from_discs(visible_discs, sid)
        if ori is None:
            print(f"  [set {sid}] skipped hidden gen: insufficient orientation evidence")
            continue
        mean_n, kappa = ori
        # 2단 생성: [rmin, rmax_far] 절단 분포의 면적 모멘트로 P32 를 국소/원거리 단에 배분
        tiers = [(size_dist, box, p["P32"], 0)]
        if rmax_far is not None and rmax_far > rmax_local:
            full = dict(size_dist, rmax=rmax_far)
            m2 = lambda a, b: quad(lambda r: r * r * GEN.size_pdf_truncated(np.array([r]), full)[0], a, b)[0]
            frac_local = m2(size_dist["rmin"], rmax_local) / m2(size_dist["rmin"], rmax_far)
            g = rmax_far - rmax_local
            far_box = dict(x0=box["x0"] - g, dx=box["dx"] + 2 * g, y0=box["y0"] - g, dy=box["dy"] + 2 * g,
                           z0=box["z0"] - g, dz=box["dz"] + 2 * g)
            tiers = [(size_dist, box, p["P32"] * frac_local, 0),
                     (dict(size_dist, rmin=rmax_local, rmax=rmax_far), far_box,
                      p["P32"] * (1.0 - frac_local), 500)]
        for t_dist, t_box, t_p32, t_off in tiers:
            # 목표 P32와 크기분포로부터 박스 내 생성 개수 N 산출
            N = GEN.compute_num_fractures_from_P32(t_p32, t_dist, t_box["dx"] * t_box["dy"] * t_box["dz"])
            if N <= 0:
                continue
            # set별 재현성 있는 seed 파생 (반지름/법선/중심 샘플링에 각각 사용)
            base = seed + sid * 1000 + t_off
            # 반지름/법선(Fisher)/중심을 원본 생성기 함수로 샘플링
            radii = GEN.sample_radius(t_dist, N, seed=base)
            normals = GEN.sample_fisher_normals(mean_n, kappa, N, seed=base + 1)
            strike_u, dip_u = GEN.normal_to_strike_dip_basis_vectorized(normals)
            centers = GEN.sample_centers_from_surface_points(
                t_box, radii, strike_u, dip_u, "area_uniform", seed=base + 2
            )
            # 샘플들을 은닉 disc dict로 축적
            for j in range(N):
                hidden.append(dict(
                    set_id=sid, center=centers[j], normal=normals[j],
                    radius=float(radii[j]), source="hidden", adoption="stochastic",
                ))
            print(f"  [set {sid}] hidden generated: N={N:,}  (P32={t_p32:.3f}, {size_desc}, "
                  f"r=[{t_dist['rmin']:.2f},{t_dist['rmax']:g}], kappa={kappa:.1f})")
    return hidden


# [함수] 관측면에서 '검출되었어야 하는' 은닉 disc를 제거(조건화 핵심 단계).
#   - 주요 인자: discs (은닉 disc), face_xs (관측면 x목록), poly_ccw (관측창),
#     lmin_det (검출 하한 [m]; 관측자료에 적용한 것과 동일 기준)
#   - 반환: (유지된 disc 리스트, 제거된 개수)
#   - 이유: 관측면에서 검출되었을 영역은 이미 가시 disc가 설명하므로 중복 제거.
#     가시길이가 lmin_det 미만인 균열은 실제 매핑에서도 검출되지 않았을 것이므로
#     관측과 모순되지 않아 남긴다(제거하면 관측면 부근 hidden 밀도가 과소해진다).
def remove_face_intersecting(
    discs: List[dict], face_xs: List[float], poly_ccw: np.ndarray, lmin_det: float = 0.0
) -> Tuple[List[dict], int]:
    """Drop discs that would have been detected as a trace on an observed face."""
    kept, removed = [], 0
    for d in discs:
        # 어느 한 관측면에서라도 검출 하한 이상으로 보이면 제거 대상
        detected = False
        for xf in face_xs:
            seg = visible_trace_on_face(d["center"], d["normal"], d["radius"], xf, poly_ccw)
            if seg is None:
                continue
            if float(np.linalg.norm(seg[1] - seg[0])) >= lmin_det:
                detected = True
                break
        if detected:
            removed += 1
        else:
            kept.append(d)
    return kept, removed


# ----------------------------------------------------------------------
# Diagnostics + figure
# ----------------------------------------------------------------------
# [함수] 폴리곤 면적을 신발끈(shoelace) 공식으로 계산한다.
#   - 인자: poly (Nx2 정점), 반환: 면적(양수). P21 계산의 관측창 면적에 사용.
def _polygon_area(poly: np.ndarray) -> float:
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


# [함수] 가시(visible) disc가 각 관측면에 남기는 트레이스(=조건화 트레이스)를 계산한다.
#   - 주요 인자: visible (가시 disc), face_xs (관측면 x), poly_ccw (관측창)
#   - 반환: 면 x -> [(p0,p1)...] 딕셔너리. 관측 트레이스(파랑)와의 비교 대상(빨강).
def conditioned_traces(
    visible: List[dict], face_xs: List[float], poly_ccw: np.ndarray
) -> Dict[float, List[Tuple[np.ndarray, np.ndarray]]]:
    """Traces the VISIBLE discs leave on each face (the conditioned traces)."""
    by_face: Dict[float, List[Tuple[np.ndarray, np.ndarray]]] = defaultdict(list)
    for d in visible:
        for xf in face_xs:
            seg = visible_trace_on_face(d["center"], d["normal"], d["radius"], xf, poly_ccw)
            if seg is not None:
                by_face[xf].append(seg)
    return by_face


# [함수] 조건화 결과의 진단 지표(트레이스 개수, P21/P32 등)를 콘솔에 출력한다.
#   - 주요 인자: 관측/조건화 트레이스, set별 관측길이, disc 개수들, params, target_sets
#   - 반환: 없음(표준출력). P21은 (트레이스 총길이 / 관측창 총면적).
def print_diagnostics(
    face_xs, poly_ccw, observed, cond, obs_len_by_set, n_visible, n_hidden_gen,
    n_hidden_removed, n_hidden_kept, params, target_sets, visible=None,
):
    # 관측창 총면적 = 한 면 면적 × 관측면 개수
    window_area = _polygon_area(poly_ccw) * len(face_xs)
    # 조건화 트레이스 총길이 및 트레이스 개수 집계
    cond_len = sum(np.linalg.norm(p1 - p0) for segs in cond.values() for p0, p1 in segs)
    n_obs = sum(len(s) for s in observed.values())
    n_cond = sum(len(s) for s in cond.values())

    # set별 복원(visible) 재생성 길이 (P21 을 set 별로 공정 비교하기 위함)
    cond_len_by_set: Dict[int, float] = defaultdict(float)
    for d in (visible or []):
        for xf in face_xs:
            seg = visible_trace_on_face(d["center"], d["normal"], d["radius"], xf, poly_ccw)
            if seg is not None:
                cond_len_by_set[int(d["set_id"])] += float(np.linalg.norm(seg[1] - seg[0]))

    obs_len_all = sum(obs_len_by_set.values())
    # Conditioned traces come only from the conditioned sets, so the fair P21
    # comparison restricts the observed side to the same sets.
    ts = set(target_sets)
    # 조건화 set에 한정한 관측 길이(공정 비교용) 및 각종 P21/P32 계산
    obs_len_matched = sum(L for s, L in obs_len_by_set.items() if s in ts)
    p21_obs_all = obs_len_all / window_area if window_area else 0.0
    p21_obs_matched = obs_len_matched / window_area if window_area else 0.0
    p21_cond = cond_len / window_area if window_area else 0.0
    p32_target = sum(params[s]["P32"] for s in target_sets if s in params and "P32" in params[s])

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
    print(f"  set-matched comparison (conditioned sets {sorted(target_sets)}):")
    print(f"    observed  P21 (matched): {p21_obs_matched:.3f} 1/m")
    print(f"    conditioned P21        : {p21_cond:.3f} 1/m")
    if p21_obs_matched > 0:
        err = (p21_cond - p21_obs_matched) / p21_obs_matched * 100
        print(f"    P21 error (matched)    : {err:+.1f} %")
    print(f"  target P32 (sets {sorted(target_sets)}) : {p32_target:.3f} 1/m")

    # set별 P21 (관측 vs 복원 visible) — set 을 섞지 않는 공정 비교.
    # 혼합 'matched' 비교는 conditioned(전 set visible)와 observed(target set만)의
    # 비대칭 때문에 왜곡될 수 있어, set 별 표를 권장 지표로 함께 출력한다.
    if visible is not None:
        ts = set(target_sets)
        all_sets = sorted(set(obs_len_by_set) | set(cond_len_by_set))
        print("  " + "-" * 40)
        print("  per-set P21 [1/m]  (관측 vs 복원 visible):")
        print(f"    {'set':>4} {'observed':>9} {'reconstructed':>13} {'err':>8}  role")
        for s in all_sets:
            po = obs_len_by_set.get(s, 0.0) / window_area if window_area else 0.0
            pc = cond_len_by_set.get(s, 0.0) / window_area if window_area else 0.0
            role = "conditioned(visible+hidden)" if s in ts else "visible-only"
            err_s = f"{(pc - po) / po * 100:+.1f}%" if po > 0 else "  n/a"
            print(f"    {s:>4} {po:>9.3f} {pc:>13.3f} {err_s:>8}  {role}")
        if p21_obs_all > 0:
            err_all = (p21_cond - p21_obs_all) / p21_obs_all * 100
            print(f"    {'ALL':>4} {p21_obs_all:>9.3f} {p21_cond:>13.3f} {err_all:>+7.1f}%  "
                  f"(대칭: 전 set vs 전 set)")
    print("-" * 64)


# [함수] SKB Fig.4-1 스타일의 3D 비교 그림(관측=파랑, 조건화=빨강)을 생성/저장한다.
#   - 주요 인자: observed/cond 트레이스, face_xs, poly_ccw, hidden_kept, out_path, seed
#   - 반환: 없음(PNG 파일 저장). show_hidden=True면 은닉 disc 중심도 옅게 표시.
def plot_fig(observed, cond, face_xs, poly_ccw, hidden_kept, out_path, seed, show_hidden=False):
    fig = plt.figure(figsize=(13, 6))
    ax = fig.add_subplot(111, projection="3d")

    ymin, ymax = poly_ccw[:, 0].min(), poly_ccw[:, 0].max()
    zmin, zmax = poly_ccw[:, 1].min(), poly_ccw[:, 1].max()

    # 각 관측면에 관측창 폴리곤(회색 루프)을 그림
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

    # 관측 트레이스(파랑)와 조건화 트레이스(빨강)를 3D 선분으로 겹쳐 그림
    for segs in observed.values():
        for p0, p1 in segs:
            ax.plot([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]],
                    color=OBSERVED_COLOR, lw=1.3, alpha=0.9)
    for segs in cond.values():
        for p0, p1 in segs:
            ax.plot([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]],
                    color=CONDITIONED_COLOR, lw=1.3, alpha=0.9)

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

    # 출력 폴더 생성 후 PNG 저장
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Figure written to {out_path}")


# [함수] 최종 조건부 DFN(visible ∪ hidden_kept)을 CSV로 기록한다.
#   - 주요 인자: visible/hidden_kept disc 리스트, out_path (conditional_dfn.csv)
#   - 반환: 없음(CSV 저장). 열: source/set_id/중심/법선/radius/adoption.
def write_dfn_csv(visible, hidden_kept, out_path: Path):
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
# [함수] 엔트리포인트: 인자 파싱 → 입력 로드 → 은닉 생성/조건화 → 진단/출력.
#   - 인자: 없음(CLI 인자로 제어), 반환: 없음
#   - 전체 파이프라인을 순서대로 조율하는 오케스트레이션 함수.
def main() -> None:
    # CLI 인자 정의 (파이프라인 경로, 로컬 rmax, seed, 대상/제외 set 등)
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pipeline-dir", type=Path,
                    default=REPO / "storage/output/pipeline_test_laxemar")
    ap.add_argument("--rmax-local", type=float, default=10.0,
                    help="Upper radius cutoff for local hidden generation [m].")
    ap.add_argument("--margin", type=float, default=None,
                    help="Local box margin around faces/window [m]. Default = rmax-local.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lmin-det", type=float, default=0.5,
                    help="검출 하한 [m]. 관측 막장면에서 이 길이 이상으로 보였어야 하는 "
                         "hidden disc 만 제거한다(관측자료와 동일 기준).")
    ap.add_argument("--keep-adoptions", default="deterministic_disc,orientation_only",
                    help="Comma list of reconstructed-disc adoptions to keep as visible.")
    ap.add_argument("--show-hidden", action="store_true",
                    help="Overlay hidden disc centers near the window (default off).")
    ap.add_argument("--sets", nargs="+", type=int, default=None,
                    help="Sets to condition (default: all sets with inverted params).")
    ap.add_argument("--exclude-sets", nargs="+", type=int, default=[],
                    help="Sets to exclude from conditioning.")
    ap.add_argument("--config", default=None,
                    help="Optional dataset JSON: per-set dist_type/r0 (e.g. exponential sets).")
    args = ap.parse_args()

    pdir = args.pipeline_dir
    # 로컬 박스 여유(margin): 미지정 시 rmax-local과 동일하게 사용
    margin = args.margin if args.margin is not None else args.rmax_local

    # --- Load inputs ---
    # 입력 로드: 가시 disc / 관측 트레이스 / 역산 파라미터
    keep = tuple(a.strip() for a in args.keep_adoptions.split(","))
    visible = load_visible_discs(pdir / "reconstruct/reconstructed_discs.csv", keep)
    observed, obs_len_by_set = load_observed_traces(pdir / "trace_dataset/trace_dataset_3d.csv")
    params = load_inverted_params(pdir / "kr/kr_summary_by_set.csv", pdir / "p32/p32_summary.csv")

    # per-set 분포타입/r0 override (지수분포 set 등)
    if args.config:
        import json
        cfg = json.load(open(args.config, encoding="utf-8"))
        # config의 set별 dist_type/r0를 역산 파라미터에 덮어씀
        for sid_str, s in cfg.get("sets", {}).items():
            sid = int(sid_str)
            params.setdefault(sid, {})
            if "dist_type" in s:
                params[sid]["dist_type"] = str(s["dist_type"])
            if "r0" in s:
                params[sid]["r0"] = float(s["r0"])
            if {"trend", "plunge", "kappa"} <= s.keys():
                params[sid]["orientation"] = (float(s["trend"]), float(s["plunge"]), float(s["kappa"]))

    # 대상 set: 지정 없으면 역산 파라미터가 있는 모든 set (하드코딩 (1,2,3,5) 제거)
    exclude = set(args.exclude_sets)
    target_sets = args.sets if args.sets is not None else sorted(params.keys())
    target_sets = [s for s in target_sets if s not in exclude]
    print(f"Target sets (conditioned): {target_sets}")

    # 관측면 x목록과 터널 관측창 폴리곤(y-z) 로드 → CCW 정렬
    face_xs = sorted(observed.keys())
    with h5py.File(pdir / "dfn_export_for_python.h5", "r") as f:
        poly = np.array(f["tunnel/poly_YZ"])
    poly_ccw = _ccw_polygon(poly)

    # --- Local conditioning box around the observed faces ---
    # 관측면 주변에 margin을 둔 로컬 조건화 박스 정의(전체 도메인은 비현실적)
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
    # 은닉 disc 생성 후, 관측면과 교차하는 disc를 제거(조건화)
    hidden_all = generate_hidden_discs(params, visible, box, args.rmax_local, args.seed, target_sets)
    hidden_kept, n_removed = remove_face_intersecting(hidden_all, face_xs, poly_ccw,
                                                      args.lmin_det)
    print(f"Hidden removed (detected on observed faces, lmin_det={args.lmin_det:g} m): {n_removed}")

    # --- Conditioned traces (red) = visible discs re-projected onto faces ---
    # 조건화 트레이스(빨강) = 가시 disc를 관측면에 다시 투영한 트레이스
    cond = conditioned_traces(visible, face_xs, poly_ccw)

    # --- Diagnostics + outputs ---
    # 진단 출력 후 최종 DFN CSV와 비교 그림 저장

    print_diagnostics(face_xs, poly_ccw, observed, cond, obs_len_by_set, len(visible),
                      len(hidden_all), n_removed, len(hidden_kept), params, target_sets,
                      visible=visible)
    out_dir = pdir / "conditional_hidden"
    write_dfn_csv(visible, hidden_kept, out_dir / "conditional_dfn.csv")
    plot_fig(observed, cond, face_xs, poly_ccw, hidden_kept,
             out_dir / "observed_vs_conditioned_traces.png", args.seed, args.show_hidden)


if __name__ == "__main__":
    main()
