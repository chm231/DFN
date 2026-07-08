# ---------------------------------------------------------------------------
# [파일 역할]
#   터널 관측창(window)을 고려한 몬테카를로(MC) 우도 기반으로 균열 반경(radius)
#   멱법칙(power-law) 지수 kr 을 추정하는 스크립트.
#   관측창 클리핑(폴리곤/바운딩박스)과 검열(censoring) 효과를 순방향 모델로
#   재현하여, 각 set 별로 kr 를 프로파일 우도로 최적화한다.
#
# [주요 입력]
#   - 트레이스 데이터: HDF5(/traces, /meta/tunnel_poly_yz) 또는 CSV
#     각 트레이스의 관측 길이(observed_length_m), 검열등급(censoring_class),
#     face-local YZ 끝점(p0_y,p0_z,p1_y,p1_z), (옵션) 실제 반경 radius_m.
#   - 관측창 폴리곤(tunnel_poly_yz, YZ 평면), 사이트(forsmark/laxemar) 정보,
#     kr 격자 범위/해상도, MC 표본 수, 반경 지지구간(rmin/rmax) 등 CLI 인자.
#
# [주요 출력] (outdir 하위 CSV/JSON)
#   - window_mc_fit_by_set.csv/.json: set별 kr_hat 추정 및 진단 지표
#   - window_mc_profile_likelihood.csv: kr 격자에 대한 프로파일 우도
#   - window_mc_posterior_predictive.csv: 검열등급 사후예측 점검
#   - bbox/polygon 비교, center-weighting 비교, 우도 분해, class-weight 민감도,
#     예측 생존곡선(survival) 등 부가 진단 CSV
#
# [핵심 처리 흐름]
#   1) 트레이스/메타 로드 및 set별 그룹화, rmin 일관성 점검
#   2) 각 (set, lmin_fit)에 대해 kr 격자를 순회하며:
#      - kr로 size-biased 반경 표본 → 참(true) 현(chord) 길이 표본
#      - 방향(경험적/방위조건부) 표본 → 관측창 클리핑으로 가시 길이/검열등급 산출
#      - 길이×등급 확률표를 만들어 관측 히스토그램과 로그우도 비교
#   3) 최댓값을 주는 kr_hat 선택, 프로파일 폭/사후예측/부트스트랩 CI 진단
#   4) 채택(adoption) 상태 판정 후 결과 CSV/JSON 기록
# ---------------------------------------------------------------------------
import argparse
import csv
import json
import os
from typing import Dict, List, Optional, Sequence, Set, Tuple

import h5py
import numpy as np


# 수치 안정용 미소값 및 결과에 부착할 경고 문자열 상수들.
# (폴리곤 창은 거친 실제 면 형상을 아직 직접 모델링하지 않음, bbox 창은 근사 진단용,
#  Set 4는 검열이 심해 최종 상태를 provisional_ok 로 상한 처리한다는 안내)
EPS = 1e-12
WINDOW_WARNING_POLYGON = "v4.1 polygon window MC uses tunnel polygon clipping; rough face mesh geometry is not directly modeled yet"
WINDOW_WARNING_BBOX = "bbox window MC is a fallback/diagnostic approximation, not final window-aware likelihood"
SET4_WARNING = "Set 4 is high-censoring; final status capped at provisional_ok"

# ---------------------------------------------------------------------------
# Fisher orientation parameters derived from DFN normals
# (trend deg, plunge deg, kappa) — tunnel X-axis, lower-hemisphere convention
# ---------------------------------------------------------------------------
# 사이트/세트별 Fisher 방위 분포 파라미터: (trend 방위각, plunge 경사각, kappa 집중도).
# 방위조건부(direction_mode=orientation_conditioned) 방향 표본 생성에 사용.
FORSMARK_FISHER_PARAMS: Dict[int, Tuple[float, float, float]] = {
    1: (182.8, -1.7, 22.1),
    2: (134.8, -2.7, 21.8),
    3: (229.4, -2.2, 24.0),
    4: (252.1,  0.7,  3.1),
    5: (176.9, 18.0,  0.9),
}
LAXEMAR_FISHER_PARAMS: Dict[int, Tuple[float, float, float]] = {
    1: (118.3,  4.3,  4.7),
    2: (169.6, -0.2, 20.1),
    3: (233.9,  0.9,  7.0),
    4: (186.4,-11.8,  0.8),
    5: (207.0, 24.4, 24.0),
}
SITE_FISHER_PARAMS: Dict[str, Dict[int, Tuple[float, float, float]]] = {
    "forsmark": FORSMARK_FISHER_PARAMS,
    "laxemar":  LAXEMAR_FISHER_PARAMS,
}
# 사이트/세트별 반경 분포 유형(powerlaw/exponential)과 표(table)상의 하한 r0.
# set별 우도 반경 지지구간(effective rmin) 계산의 근거로 사용.
SITE_SET_SUPPORT_INFO: Dict[str, Dict[int, Dict[str, float | str]]] = {
    "forsmark": {
        1: {"type": "powerlaw", "table_r0": 0.28},
        2: {"type": "powerlaw", "table_r0": 0.25},
        3: {"type": "powerlaw", "table_r0": 0.14},
        4: {"type": "powerlaw", "table_r0": 0.15},
        5: {"type": "powerlaw", "table_r0": 0.25},
    },
    "laxemar": {
        1: {"type": "powerlaw", "table_r0": 0.328},
        2: {"type": "powerlaw", "table_r0": 0.977},
        3: {"type": "powerlaw", "table_r0": 0.858},
        4: {"type": "exponential", "table_r0": 4.0},
        5: {"type": "powerlaw", "table_r0": 0.400},
    },
}

# ---------------------------------------------------------------------------
# Recovery / adoption status thresholds
# ---------------------------------------------------------------------------
# kr 참값 회복(recovery) 판정 임계값: |kr_hat - kr_true| 오차가
# 0.3 이하이면 good, 0.6 이하이면 moderate, 그 외 failed 로 분류.
_RECOVERY_GOOD_THRESHOLD      = 0.3
_RECOVERY_MODERATE_THRESHOLD  = 0.6


# (trend, plunge) 방위/경사(도 단위)를 단위 극(pole) 벡터 [x,y,z]로 변환한다.
# 인자: trend_deg 방위각, plunge_deg 경사각. 반환: 단위 극 벡터(z 상향 규약).
def mean_pole_from_trend_plunge(trend_deg: float, plunge_deg: float) -> np.ndarray:
    """Convert (trend, plunge) in degrees to a unit pole vector [x, y, z]."""
    t = np.radians(trend_deg)
    p = np.radians(plunge_deg)
    # NED convention → x=East, y=North, z=Up  (here we use xyz with z upward)
    # pole unit vector: (cos p cos t, cos p sin t, -sin p)
    return np.array([np.cos(p) * np.cos(t), np.cos(p) * np.sin(t), -np.sin(p)])


# 구면 S^2 위 Von Mises-Fisher 분포에서 n개의 단위 법선을 표본추출한다.
# 인자: mean_pole 평균 방향, kappa 집중도, n 표본수, rng 난수생성기.
# 반환: (n,3) 단위 법선 배열. kappa~0이면 균등 구면, 아니면 Wood(1994) 기각표본법.
def sample_fisher_normals(mean_pole: np.ndarray, kappa: float, n: int, rng: np.random.Generator) -> np.ndarray:
    """
    Sample n unit normals from a Von Mises-Fisher distribution on S^2
    with mean direction mean_pole and concentration kappa.
    Uses the Wood (1994) rejection-sampler when kappa > 0,
    falls back to uniform sphere when kappa == 0.
    """
    # 평균 방향을 단위벡터로 정규화. kappa가 매우 작으면 방향성이 없으므로 균등 구면으로 처리.
    mu = mean_pole / np.linalg.norm(mean_pole)
    if kappa < 1e-4:
        # Uniform sphere
        v = rng.standard_normal((n, 3))
        norms = np.linalg.norm(v, axis=1, keepdims=True)
        return v / np.maximum(norms, 1e-12)

    # Rejection sampler (Ulrich 1984 / Wood 1994)
    # b, x0, c: 기각표본법에 필요한 사전 계산 상수(mu 방향 성분 w의 제안분포 파라미터).
    samples = np.empty((n, 3), dtype=np.float64)
    generated = 0
    b = (-2.0 * kappa + np.sqrt(4.0 * kappa ** 2 + 4.0)) / 2.0
    x0 = (1.0 - b) / (1.0 + b)
    c = kappa * x0 + 2.0 * np.log(1.0 - x0 ** 2)

    # 목표 개수 n을 채울 때까지 배치 단위로 후보를 생성/기각 반복.
    while generated < n:
        # 제안분포에서 mu 축 방향 성분 W를 뽑고, 기각 기준을 만족하는 것만 채택.
        batch = max(n - generated, 256)
        Z = rng.beta(1.0, 1.0, size=batch)
        W = (1.0 - (1.0 + b) * Z) / (1.0 - (1.0 - b) * Z)
        U = rng.uniform(0.0, 1.0, size=batch)
        accept = kappa * W + 2.0 * np.log(1.0 - x0 * W) - c >= np.log(U)
        w = W[accept]
        if len(w) == 0:
            continue
        # Random unit vectors in the plane perp to mu
        v_raw = rng.standard_normal((len(w), 3))
        v_raw -= (v_raw @ mu)[:, None] * mu
        nrm = np.linalg.norm(v_raw, axis=1, keepdims=True)
        v_hat = v_raw / np.maximum(nrm, 1e-12)
        # 채택된 축방향 성분 w와 수직평면 단위벡터 v_hat를 합쳐 최종 구면 표본 pts 구성.
        s = np.sqrt(np.maximum(1.0 - w ** 2, 0.0))
        pts = s[:, None] * v_hat + w[:, None] * mu
        take = min(len(pts), n - generated)
        samples[generated : generated + take] = pts[:take]
        generated += take

    return samples


# 각 균열 법선에 대해 터널 면(X=상수 평면)과의 교선(트레이스) YZ 방향을 계산한다.
# 인자: normals (N,3) 법선 배열. 반환: (directions_yz 단위 YZ 방향, valid X축 평행 아님 마스크).
def normals_to_trace_directions_yz(normals: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the YZ-plane trace direction for each fracture normal when
    intersected by the tunnel face (X = const plane, normal = [1, 0, 0]).

    trace direction = n × x_axis   (cross product gives strike line on face)

    Returns:
        directions_yz : (M, 2) array of unit YZ directions
        valid         : (N,) bool mask — False if n is parallel to X-axis
    """
    # n × x_axis 로 면 위 방향을 구하고, 크기가 0에 가까운(법선이 X축에 평행) 경우는 무효 처리.
    x_axis = np.array([1.0, 0.0, 0.0])
    crosses = np.cross(normals, x_axis)          # (N, 3)
    norms = np.linalg.norm(crosses, axis=1)      # (N,)
    valid = norms > 1e-6
    directions_yz = np.zeros((len(normals), 2), dtype=np.float64)
    directions_yz[valid] = crosses[valid, 1:] / norms[valid, np.newaxis]
    # 방향 부호 모호성 제거: 첫 비영 성분이 양수가 되도록 정규(canonical) 방향으로 뒤집음.
    # Canonical orientation: flip so first non-zero component is positive
    for i in np.where(valid)[0]:
        d = directions_yz[i]
        if d[0] < 0.0 or (abs(d[0]) < 1e-12 and d[1] < 0.0):
            directions_yz[i] *= -1.0
    return directions_yz, valid


# 방위조건부 모델에서 YZ 면 위 트레이스 방향 n개를 표본추출한다.
# (Fisher 극 표본 → X축과의 외적으로 YZ 방향 변환 → X축 평행 법선 기각)
# 인자: set_id, site, n 표본수, rng. 반환: (M,2) 단위 방향 배열(M<=n).
def orientation_conditioned_trace_directions_yz(
    set_id: int,
    site: str,
    n: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Sample n trace directions on the YZ face from the orientation-conditioned model:
      1. Sample fracture poles from Fisher(mean_pole, kappa)
      2. Convert to YZ trace directions via cross product with X-axis
      3. Reject normals parallel to X-axis (prob ≈ 0 for typical kappa)
    Returns (M, 2) array of unit directions, M <= n.
    """
    params = SITE_FISHER_PARAMS.get(site, {}).get(set_id)
    if params is None:
        raise ValueError(f"No Fisher params for site={site}, set_id={set_id}")
    trend, plunge, kappa = params
    mu = mean_pole_from_trend_plunge(trend, plunge)

    # 기각으로 인한 개수 부족을 대비해 과표본(oversample)한 뒤 유효 방향만 남김.
    # Oversample to account for rejection
    oversample = max(n * 2, 1000)
    normals = sample_fisher_normals(mu, kappa, oversample, rng)
    dirs_yz, valid = normals_to_trace_directions_yz(normals)
    dirs_yz = dirs_yz[valid]
    if len(dirs_yz) == 0:
        # Fallback: uniform directions
        angles = rng.uniform(0, np.pi, n)
        dirs_yz = np.column_stack([np.cos(angles), np.sin(angles)])
    return dirs_yz


# 추정 kr_hat와 참값 kr_true의 절대오차로 회복 상태를 분류한다.
# 반환: kr_true가 없으면 "unknown", 임계값에 따라 good/moderate/failed_recovery.
def determine_recovery_status(kr_hat: float, kr_true: Optional[float]) -> str:
    if kr_true is None or not np.isfinite(kr_true):
        return "unknown"
    err = abs(kr_hat - kr_true)
    if err <= _RECOVERY_GOOD_THRESHOLD:
        return "good_recovery"
    if err <= _RECOVERY_MODERATE_THRESHOLD:
        return "moderate_recovery"
    return "failed_recovery"


# 적합(fit) 상태와 회복(recovery) 상태를 결합해 최종 채택(adoption) 판정을 내린다.
# 인자: fit_status, recovery_status, rmin_support_status(지지구간 일치 여부).
# 반환: accepted / provisional_accepted / rejected / (지지구간 불일치 시) 진단전용.
def determine_adoption_status(
    fit_status: str,
    recovery_status: str,
    rmin_support_status: str = "matched",
) -> str:
    if rmin_support_status != "matched":
        return "diagnostic_only_rmin_support_mismatch"
    good_fit = fit_status in ("ok", "provisional_ok")
    if good_fit and recovery_status == "good_recovery":
        return "accepted"
    if good_fit and recovery_status in ("moderate_recovery", "unknown"):
        return "provisional_accepted"
    return "rejected"


# HDF5(/traces)에서 트레이스 행 목록과 관측창 폴리곤(/meta/tunnel_poly_yz)을 읽는다.
# 인자: h5_path. 반환: (트레이스 dict 리스트, 폴리곤 YZ 배열 또는 None).
def load_trace_data_from_h5(h5_path: str) -> tuple[List[dict], Optional[np.ndarray]]:
    rows: List[dict] = []
    polygon_yz = None
    with h5py.File(h5_path, "r") as f:
        if "traces" not in f:
            raise ValueError(f"Could not find /traces in: {h5_path}")
        if "meta" in f and "tunnel_poly_yz" in f["meta"]:
            polygon_yz = f["meta/tunnel_poly_yz"][:].astype(np.float64)

        # 트레이스 끝점 3D 좌표(p0/p1_xyz)와 (있으면) 실제 반경을 일괄 로드.
        grp = f["traces"]
        p0 = grp["p0_xyz"][:].astype(np.float64)
        p1 = grp["p1_xyz"][:].astype(np.float64)
        radius_m = grp["radius_m"][:].astype(np.float64) if "radius_m" in grp else None
        # 각 트레이스를 dict로 변환: 3D 끝점에서 YZ 성분만 취해 face-local 좌표로 저장.
        for idx in range(len(grp["set_id"])):
            rows.append(
                {
                    "set_id": int(grp["set_id"][idx]),
                    "face_id": int(grp["face_id"][idx]),
                    "observed_length_m": float(grp["observed_length_m"][idx]),
                    "censoring_class": int(grp["censoring_class"][idx]),
                    "radius_m": float(radius_m[idx]) if radius_m is not None else float("nan"),
                    "p0_y": float(p0[idx, 1]),
                    "p0_z": float(p0[idx, 2]),
                    "p1_y": float(p1[idx, 1]),
                    "p1_z": float(p1[idx, 2]),
                }
            )
    return rows, polygon_yz


# HDF5 /meta에서 DFN 생성 시의 반경 지지구간(rmin/rmax) 및 set별 메타를 읽는다.
# 인자: h5_path. 반환: 생성 rmin/rmax와 set_ids/table_r0/effective_rmin 등을 담은 dict.
def load_trace_rmin_metadata_from_h5(h5_path: str) -> dict:
    metadata = {
        "generation_rmin": None,
        "generation_rmax": None,
        "set_ids": None,
        "set_table_r0": None,
        "set_generation_rmin": None,
        "set_effective_rmin": None,
    }
    with h5py.File(h5_path, "r") as f:
        if "meta" not in f:
            return metadata
        meta = f["meta"]
        if "generation_rmin" in meta:
            metadata["generation_rmin"] = float(np.asarray(meta["generation_rmin"][()]).ravel()[0])
        if "generation_rmax" in meta:
            metadata["generation_rmax"] = float(np.asarray(meta["generation_rmax"][()]).ravel()[0])
        for key in ("set_ids", "set_table_r0", "set_generation_rmin", "set_effective_rmin"):
            if key in meta:
                metadata[key] = meta[key][:].ravel()
    return metadata


# set별 반경 하한(rmin) 조회표를 구성한다. HDF5 메타를 우선 사용하고,
# 없으면 SITE_SET_SUPPORT_INFO의 분포유형/table_r0로 유효 rmin을 보완한다.
# 인자: site, 전역 생성 rmin, HDF5 메타. 반환: {set_id: {table_r0, generation_rmin, effective_generation_rmin}}.
def build_set_rmin_lookup(
    site: str,
    global_generation_rmin: float,
    trace_rmin_metadata: dict,
) -> Dict[int, dict]:
    lookup: Dict[int, dict] = {}
    meta_ids = trace_rmin_metadata.get("set_ids")
    meta_table_r0 = trace_rmin_metadata.get("set_table_r0")
    meta_generation = trace_rmin_metadata.get("set_generation_rmin")
    meta_effective = trace_rmin_metadata.get("set_effective_rmin")

    # 1순위: HDF5 메타에 set별 값이 있으면 그대로 조회표에 등록.
    if meta_ids is not None:
        for idx, set_id in enumerate(np.asarray(meta_ids, dtype=np.int32)):
            lookup[int(set_id)] = {
                "table_r0": float(meta_table_r0[idx]) if meta_table_r0 is not None else float("nan"),
                "generation_rmin": float(meta_generation[idx]) if meta_generation is not None else float(global_generation_rmin),
                "effective_generation_rmin": float(meta_effective[idx]) if meta_effective is not None else float(global_generation_rmin),
            }

    # 2순위: 메타에 없는 set은 사이트 표 기반으로 보완(setdefault로 기존값 보존).
    # powerlaw는 table_r0와 전역 rmin 중 큰 값을, 그 외(지수분포)는 전역 rmin을 사용.
    for set_id, info in SITE_SET_SUPPORT_INFO.get(site, {}).items():
        table_r0 = float(info["table_r0"])
        dist_type = str(info["type"])
        if dist_type == "powerlaw":
            effective_rmin = max(float(global_generation_rmin), table_r0)
        else:
            effective_rmin = float(global_generation_rmin)
        lookup.setdefault(
            int(set_id),
            {
                "table_r0": table_r0,
                "generation_rmin": effective_rmin,
                "effective_generation_rmin": effective_rmin,
            },
        )
    return lookup


# set별 우도 계산에 쓸 반경 하한(likelihood_rmin)을 모드에 따라 결정한다.
# 모드: effective_generation / table_r0 / global. 또한 생성 지지구간과 비교해
# 지지 상태(matched / estimator_lower_or_higher_than_generated_support)를 판정.
# 반환: (likelihood_rmin, effective_generation_rmin, table_r0, support_status).
def resolve_set_likelihood_rmin(
    set_id: int,
    set_rmin_mode: str,
    global_rmin: float,
    set_rmin_lookup: Dict[int, dict],
) -> tuple[float, float, float, str]:
    set_meta = set_rmin_lookup.get(int(set_id), {})
    table_r0 = float(set_meta.get("table_r0", np.nan))
    effective_generation_rmin = float(set_meta.get("effective_generation_rmin", global_rmin))

    if set_rmin_mode == "effective_generation":
        likelihood_rmin = effective_generation_rmin
    elif set_rmin_mode == "table_r0":
        likelihood_rmin = table_r0 if np.isfinite(table_r0) else float(global_rmin)
    else:
        likelihood_rmin = float(global_rmin)

    tolerance = 1e-6
    if likelihood_rmin < effective_generation_rmin - tolerance:
        support_status = "estimator_lower_than_generated_support"
    elif abs(likelihood_rmin - effective_generation_rmin) <= tolerance:
        support_status = "matched"
    else:
        support_status = "estimator_higher_than_generated_support"
    return likelihood_rmin, effective_generation_rmin, table_r0, support_status


# CSV에서 트레이스 행 목록을 읽는다(필수 필드 검증 포함, 폴리곤은 제공하지 않음).
# 인자: csv_path. 반환: (트레이스 dict 리스트, None).
def load_trace_data_from_csv(csv_path: str) -> tuple[List[dict], Optional[np.ndarray]]:
    rows: List[dict] = []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"set_id", "face_id", "observed_length_m", "censoring_class", "p0_y", "p0_z", "p1_y", "p1_z"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing required CSV fields: {sorted(missing)}")
        for row in reader:
            rows.append(
                {
                    "set_id": int(row["set_id"]),
                    "face_id": int(row["face_id"]),
                    "observed_length_m": float(row["observed_length_m"]),
                    "censoring_class": int(row["censoring_class"]),
                    "radius_m": float(row["radius_m"]) if "radius_m" in row and row["radius_m"] else float("nan"),
                    "p0_y": float(row["p0_y"]),
                    "p0_z": float(row["p0_z"]),
                    "p1_y": float(row["p1_y"]),
                    "p1_z": float(row["p1_z"]),
                }
            )
    return rows, None


# 트레이스 행들을 set_id 기준으로 묶는다(target_sets 지정 시 해당 set만).
# 인자: rows, target_sets(None이면 전체). 반환: set_id 오름차순 {set_id: 행 리스트}.
def group_rows_by_set(rows: Sequence[dict], target_sets: Optional[Set[int]]) -> Dict[int, List[dict]]:
    grouped: Dict[int, List[dict]] = {}
    for row in rows:
        set_id = int(row["set_id"])
        if target_sets is not None and set_id not in target_sets:
            continue
        grouped.setdefault(set_id, []).append(row)
    return {set_id: grouped[set_id] for set_id in sorted(grouped)}


# 관측 트레이스 끝점으로부터 경험적 YZ 단위 방향들을 계산한다(정규 방향으로 통일).
# 인자: rows. 반환: (K,2) 단위 방향 배열(유효 방향이 없으면 [[1,0]]).
def empirical_trace_directions_yz(rows: Sequence[dict]) -> np.ndarray:
    directions = []
    for row in rows:
        dy = float(row["p1_y"]) - float(row["p0_y"])
        dz = float(row["p1_z"]) - float(row["p0_z"])
        norm = float(np.hypot(dy, dz))
        if norm <= 0.0:
            continue
        direction = np.array([dy / norm, dz / norm], dtype=np.float64)
        if direction[0] < 0.0 or (abs(direction[0]) < 1e-12 and direction[1] < 0.0):
            direction *= -1.0
        directions.append(direction)
    if not directions:
        return np.array([[1.0, 0.0]], dtype=np.float64)
    return np.vstack(directions)


# 점이 폴리곤 내부에 있는지 ray-casting(홀짝 규칙)으로 판정한다.
# 인자: point_yz 검사점, polygon_yz 정점 배열. 반환: 내부이면 True.
def point_in_polygon(point_yz: np.ndarray, polygon_yz: np.ndarray) -> bool:
    y, z = float(point_yz[0]), float(point_yz[1])
    inside = False
    n = len(polygon_yz)
    for i in range(n):
        y0, z0 = polygon_yz[i]
        y1, z1 = polygon_yz[(i + 1) % n]
        crosses = (z0 > z) != (z1 > z)
        if crosses:
            y_intersect = (y1 - y0) * (z - z0) / (z1 - z0 + EPS) + y0
            if y < y_intersect:
                inside = not inside
    return inside


# 선분 p0-p1 과 선분 q0-q1 의 교차 매개변수 t를 구한다(교차 없으면 None).
# 인자: 두 선분의 끝점. 반환: p0-p1 위 교차점의 t∈[0,1] 또는 None.
def _segment_intersection_t(p0: np.ndarray, p1: np.ndarray, q0: np.ndarray, q1: np.ndarray) -> Optional[float]:
    r = p1 - p0
    s = q1 - q0
    denom = r[0] * s[1] - r[1] * s[0]
    if abs(denom) < 1e-12:
        return None
    qp = q0 - p0
    t = (qp[0] * s[1] - qp[1] * s[0]) / denom
    u = (qp[0] * r[1] - qp[1] * r[0]) / denom
    if -1e-10 <= t <= 1.0 + 1e-10 and -1e-10 <= u <= 1.0 + 1e-10:
        return float(np.clip(t, 0.0, 1.0))
    return None


# 하나의 선분을 폴리곤 관측창으로 클리핑해 가시 길이와 검열등급을 산출한다.
# 인자: p0_yz/p1_yz 선분 끝점, polygon_yz 창. 반환: (가시 길이, 검열등급 -1/0/1/2).
# 등급: -1 창밖, 0 양끝 내부(비검열), 1 한끝 검열, 2 양끝 검열.
def clip_segment_to_polygon(p0_yz: np.ndarray, p1_yz: np.ndarray, polygon_yz: np.ndarray) -> tuple[float, int]:
    # 선분과 폴리곤 각 변의 교차 매개변수 t를 모아 정렬(양끝 0,1 포함).
    t_values = [0.0, 1.0]
    for idx in range(len(polygon_yz)):
        t = _segment_intersection_t(p0_yz, p1_yz, polygon_yz[idx], polygon_yz[(idx + 1) % len(polygon_yz)])
        if t is not None:
            t_values.append(t)
    t_values = sorted(set(round(t, 12) for t in t_values))

    # 인접한 t 구간별 중점이 폴리곤 내부이면 그 구간 길이를 가시 길이에 누적.
    visible_length = 0.0
    segment_length = float(np.linalg.norm(p1_yz - p0_yz))
    for a, b in zip(t_values[:-1], t_values[1:]):
        if b <= a:
            continue
        mid = p0_yz + 0.5 * (a + b) * (p1_yz - p0_yz)
        if point_in_polygon(mid, polygon_yz):
            visible_length += (b - a) * segment_length

    # 양끝점의 내부 여부로 검열등급을 결정.
    p0_inside = point_in_polygon(p0_yz, polygon_yz)
    p1_inside = point_in_polygon(p1_yz, polygon_yz)
    if visible_length <= 1e-10:
        return 0.0, -1
    if p0_inside and p1_inside:
        return visible_length, 0
    if p0_inside or p1_inside:
        return visible_length, 1
    return visible_length, 2


# 다수의 선분을 축정렬 바운딩박스(bbox)로 벡터화 클리핑한다(슬랩 알고리즘).
# 인자: 중심, 방향, 참길이, bbox_min/max. 반환: (가시 길이 배열, 검열등급 배열).
def clip_segments_to_bbox_vectorized(
    centers_yz: np.ndarray,
    directions_yz: np.ndarray,
    true_lengths: np.ndarray,
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    # 중심±(참길이/2)·방향 으로 각 선분의 양끝을 구성하고 진입/진출 매개변수 t0,t1 초기화.
    p0 = centers_yz - 0.5 * true_lengths[:, None] * directions_yz
    p1 = centers_yz + 0.5 * true_lengths[:, None] * directions_yz
    d = p1 - p0
    t0 = np.zeros(len(true_lengths), dtype=np.float64)
    t1 = np.ones(len(true_lengths), dtype=np.float64)
    valid = np.ones(len(true_lengths), dtype=bool)

    # 각 축(Y,Z)의 슬랩과 교차해 진입/진출 t 구간을 좁힘(축 평행+범위 밖은 무효).
    for axis in range(2):
        parallel = np.abs(d[:, axis]) < 1e-12
        valid &= ~(parallel & ((p0[:, axis] < bbox_min[axis]) | (p0[:, axis] > bbox_max[axis])))
        non_parallel = ~parallel
        inv_d = np.zeros(len(true_lengths), dtype=np.float64)
        inv_d[non_parallel] = 1.0 / d[non_parallel, axis]
        ta = (bbox_min[axis] - p0[:, axis]) * inv_d
        tb = (bbox_max[axis] - p0[:, axis]) * inv_d
        t_enter = np.minimum(ta, tb)
        t_exit = np.maximum(ta, tb)
        t0 = np.where(non_parallel, np.maximum(t0, t_enter), t0)
        t1 = np.where(non_parallel, np.minimum(t1, t_exit), t1)

    # 유효 구간 길이로 가시 길이 계산, 양끝 내부 여부로 검열등급(0/1/2) 부여.
    valid &= t1 > t0
    visible_lengths = np.where(valid, (t1 - t0) * true_lengths, 0.0)
    p0_inside = np.all((p0 >= bbox_min) & (p0 <= bbox_max), axis=1)
    p1_inside = np.all((p1 >= bbox_min) & (p1 <= bbox_max), axis=1)
    classes = np.full(len(true_lengths), -1, dtype=np.int32)
    classes[valid & p0_inside & p1_inside] = 0
    classes[valid & (p0_inside ^ p1_inside)] = 1
    classes[valid & ~(p0_inside | p1_inside)] = 2
    return visible_lengths, classes


# 다수 선분을 폴리곤 창으로 클리핑(루프 버전; clip_segment_to_polygon을 반복 호출).
# 인자: 중심, 방향, 참길이, 폴리곤. 반환: (가시 길이 배열, 검열등급 배열).
def clip_segments_to_polygon_loop(
    centers_yz: np.ndarray,
    directions_yz: np.ndarray,
    true_lengths: np.ndarray,
    polygon_yz: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    visible_lengths = np.zeros(len(true_lengths), dtype=np.float64)
    classes = np.full(len(true_lengths), -1, dtype=np.int32)
    p0 = centers_yz - 0.5 * true_lengths[:, None] * directions_yz
    p1 = centers_yz + 0.5 * true_lengths[:, None] * directions_yz
    for idx in range(len(true_lengths)):
        visible_length, cls = clip_segment_to_polygon(p0[idx], p1[idx], polygon_yz)
        visible_lengths[idx] = visible_length
        classes[idx] = cls
    return visible_lengths, classes


# 폴리곤의 부호있는 면적(shoelace)을 계산한다(정점 순서/방향 판정에 사용).
# 인자: polygon_yz. 반환: 부호있는 면적(양수=반시계, 음수=시계).
def signed_polygon_area(polygon_yz: np.ndarray) -> float:
    y = polygon_yz[:, 0]
    z = polygon_yz[:, 1]
    return 0.5 * float(np.dot(y, np.roll(z, -1)) - np.dot(z, np.roll(y, -1)))


# 다수 선분을 볼록 폴리곤 창으로 벡터화 클리핑한다(Cyrus-Beck 방식).
# 인자: 중심, 방향, 참길이, 볼록 폴리곤. 반환: (가시 길이 배열, 검열등급 배열).
def clip_segments_to_convex_polygon_vectorized(
    centers_yz: np.ndarray,
    directions_yz: np.ndarray,
    true_lengths: np.ndarray,
    polygon_yz: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    p0 = centers_yz - 0.5 * true_lengths[:, None] * directions_yz
    p1 = centers_yz + 0.5 * true_lengths[:, None] * directions_yz
    d = p1 - p0
    t0 = np.zeros(len(true_lengths), dtype=np.float64)
    t1 = np.ones(len(true_lengths), dtype=np.float64)
    valid = np.ones(len(true_lengths), dtype=bool)
    # 정점 순서(시계/반시계)에 따라 내부 방향 부호를 통일하기 위한 orientation.
    orientation = 1.0 if signed_polygon_area(polygon_yz) >= 0.0 else -1.0

    # 각 변을 반평면으로 보고 진입/진출 t를 갱신해 유효 구간 [t0,t1]을 좁힘.
    for idx in range(len(polygon_yz)):
        v0 = polygon_yz[idx]
        v1 = polygon_yz[(idx + 1) % len(polygon_yz)]
        edge = v1 - v0
        a = orientation * (edge[0] * (p0[:, 1] - v0[1]) - edge[1] * (p0[:, 0] - v0[0]))
        b = orientation * (edge[0] * d[:, 1] - edge[1] * d[:, 0])
        parallel = np.abs(b) < 1e-12
        valid &= ~(parallel & (a < 0.0))

        non_parallel = ~parallel
        t_cross = np.zeros(len(true_lengths), dtype=np.float64)
        t_cross[non_parallel] = -a[non_parallel] / b[non_parallel]
        entering = non_parallel & (b > 0.0)
        exiting = non_parallel & (b < 0.0)
        t0 = np.where(entering, np.maximum(t0, t_cross), t0)
        t1 = np.where(exiting, np.minimum(t1, t_cross), t1)

    # 유효 구간 길이로 가시 길이 계산. 이어서 양끝점의 폴리곤 내부 여부를 모든 변에 대해 판정.
    valid &= t1 > t0
    visible_lengths = np.where(valid, (t1 - t0) * true_lengths, 0.0)
    p0_inside = np.all(
        [
            orientation
            * (
                (polygon_yz[(idx + 1) % len(polygon_yz), 0] - polygon_yz[idx, 0]) * (p0[:, 1] - polygon_yz[idx, 1])
                - (polygon_yz[(idx + 1) % len(polygon_yz), 1] - polygon_yz[idx, 1]) * (p0[:, 0] - polygon_yz[idx, 0])
            )
            >= -1e-10
            for idx in range(len(polygon_yz))
        ],
        axis=0,
    )
    p1_inside = np.all(
        [
            orientation
            * (
                (polygon_yz[(idx + 1) % len(polygon_yz), 0] - polygon_yz[idx, 0]) * (p1[:, 1] - polygon_yz[idx, 1])
                - (polygon_yz[(idx + 1) % len(polygon_yz), 1] - polygon_yz[idx, 1]) * (p1[:, 0] - polygon_yz[idx, 0])
            )
            >= -1e-10
            for idx in range(len(polygon_yz))
        ],
        axis=0,
    )
    classes = np.full(len(true_lengths), -1, dtype=np.int32)
    classes[valid & p0_inside & p1_inside] = 0
    classes[valid & (p0_inside ^ p1_inside)] = 1
    classes[valid & ~(p0_inside | p1_inside)] = 2
    return visible_lengths, classes


# 면과 교차하는 균열의 크기편향(size-biased) 반경을 표본추출한다(역변환 샘플링).
# 원본 f_R(r)∝r^-(kr+1)가 교차 확률로 편향되어 g_R(r)∝r^-kr 가 됨을 반영.
# 인자: kr 멱지수, rmin/rmax 지지구간, size, rng. 반환: 반경 표본 배열.
def sample_size_biased_radius(kr: float, rmin: float, rmax: float, size: int, rng: np.random.Generator) -> np.ndarray:
    # Original f_R(r) is proportional to r^-(kr+1); intersected g_R(r) is proportional to r^-kr.
    exponent = 1.0 - kr
    u = rng.uniform(0.0, 1.0, size=size)
    if abs(exponent) < 1e-12:
        return rmin * np.exp(u * np.log(rmax / rmin))
    return (u * (rmax**exponent - rmin**exponent) + rmin**exponent) ** (1.0 / exponent)


# 원판(disc) 반경에서 면과의 교선(현, chord)의 참 길이를 표본추출한다.
# 중심-현 거리(offset)를 균등추출해 현 길이 2·sqrt(r^2-offset^2)를 계산.
# 인자: radii 반경 배열, rng. 반환: 참 현 길이 배열.
def sample_true_chords(radii: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    offsets = rng.uniform(0.0, radii)
    return 2.0 * np.sqrt(np.maximum(radii * radii - offsets * offsets, 0.0))


# 주어진 반경 표본에 대해 관측창에서 채택된 가시 트레이스를 시뮬레이션한다.
# 처리: 참 현 길이 표본 → 방향 표본 → 창 안 무작위 중심 배치 → 창 클리핑 → 채택 필터.
# 인자: 폴리곤, 방향풀, 반경, rng, window_mode(polygon/bbox), direction_mode, set_id, site.
# 반환: (가시 길이, 검열등급, 제안면적 proposal_area, 채택된 반경) — 모두 채택분만.
def simulate_window_samples(
    polygon_yz: np.ndarray,
    directions_yz: np.ndarray,
    radii: np.ndarray,
    rng: np.random.Generator,
    window_mode: str = "polygon",
    direction_mode: str = "empirical_trace",
    set_id: int = 0,
    site: str = "",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Simulate accepted visible traces for a provided radius sample."""
    # 창의 바운딩박스 폭/높이와 각 표본의 참 현 길이를 준비.
    bbox_min = np.min(polygon_yz, axis=0)
    bbox_max = np.max(polygon_yz, axis=0)
    w_bbox = bbox_max[0] - bbox_min[0]
    h_bbox = bbox_max[1] - bbox_min[1]
    true_lengths = sample_true_chords(radii, rng)
    n_samples = len(radii)

    # 방향 표본 선택: 방위조건부는 Fisher 기반 풀에서, 기본은 경험적 방향 풀에서 무작위 추출.
    if direction_mode == "orientation_conditioned":
        dir_pool = orientation_conditioned_trace_directions_yz(set_id, site, n_samples * 3, rng)
        direction_idx = rng.integers(0, len(dir_pool), size=n_samples)
        directions = dir_pool[direction_idx]
    else:
        direction_idx = rng.integers(0, len(directions_yz), size=n_samples)
        directions = directions_yz[direction_idx]

    # 중심 제안영역: 선분이 창에 걸칠 수 있도록 반길이만큼 bbox를 확장한 영역에서 균등 배치.
    # proposal_areas: 이 확장영역 면적(중요도 가중치용); centers: 확장영역 내 무작위 중심.
    proposal_areas = (w_bbox + true_lengths * np.abs(directions[:, 0])) * (h_bbox + true_lengths * np.abs(directions[:, 1]))
    expand = 0.5 * true_lengths[:, None]
    centers = rng.uniform(bbox_min - expand, bbox_max + expand)
    # 선택된 창 모드로 클리핑해 가시 길이/검열등급을 얻고, 유효(가시>0)인 것만 채택.
    if window_mode == "bbox":
        visible_lengths, classes = clip_segments_to_bbox_vectorized(centers, directions, true_lengths, bbox_min, bbox_max)
    elif window_mode == "polygon":
        visible_lengths, classes = clip_segments_to_convex_polygon_vectorized(centers, directions, true_lengths, polygon_yz)
    else:
        raise ValueError(f"Unsupported window_mode: {window_mode}")
    accepted = (classes >= 0) & (visible_lengths > 0.0)
    return visible_lengths[accepted], classes[accepted], proposal_areas[accepted], radii[accepted]


# 주어진 kr에서 반경을 샘플링한 뒤 창 관측 트레이스를 시뮬레이션한다.
# (sample_size_biased_radius → simulate_window_samples 를 연결하는 래퍼)
# 인자: kr, rmin/rmax, 폴리곤, 방향, n_samples, rng, window/direction_mode, set_id, site.
# 반환: (가시 길이, 검열등급, 제안면적).
def simulate_window_observations(
    kr: float,
    rmin: float,
    rmax: float,
    polygon_yz: np.ndarray,
    directions_yz: np.ndarray,
    n_samples: int,
    rng: np.random.Generator,
    window_mode: str = "polygon",
    direction_mode: str = "empirical_trace",
    set_id: int = 0,
    site: str = "",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulate window-observed trace lengths, censoring classes, and their center proposal areas.

    direction_mode:
        "empirical_trace"       — resample from observed directions_yz (default)
        "orientation_conditioned" — generate directions from Fisher(mean_pole, kappa)
                                    using SITE_FISHER_PARAMS[site][set_id]
    """
    radii = sample_size_biased_radius(kr, rmin, rmax, n_samples, rng)
    visible_lengths, classes, proposal_areas, _ = simulate_window_samples(
        polygon_yz,
        directions_yz,
        radii,
        rng,
        window_mode=window_mode,
        direction_mode=direction_mode,
        set_id=set_id,
        site=site,
    )
    return visible_lengths, classes, proposal_areas


# 트레이스 길이 히스토그램의 구간 경계(edges)를 만든다(log 또는 linear).
# 인자: lengths, lmin_fit 하한, bin_count, mode. 반환: (bin_count+1,) 경계 배열.
def make_length_edges(lengths: np.ndarray, lmin_fit: float, bin_count: int, mode: str) -> np.ndarray:
    upper = max(float(np.max(lengths)) if len(lengths) else lmin_fit * 1.01, lmin_fit * 1.01)
    if mode == "log":
        return np.geomspace(lmin_fit, upper * 1.000001, bin_count + 1)
    return np.linspace(lmin_fit, upper * 1.000001, bin_count + 1)


# 길이×검열등급(3열) 2차원 히스토그램 카운트를 만든다(가중치 선택 가능).
# 인자: lengths, classes(0/1/2), edges 구간경계, weights. 반환: (bins,3) 카운트 배열.
def binned_counts(
    lengths: np.ndarray,
    classes: np.ndarray,
    edges: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    # 하한/유효등급 필터 후, 각 길이가 속한 구간 인덱스를 계산해 범위로 클립.
    table = np.zeros((len(edges) - 1, 3), dtype=np.float64)
    mask = (lengths >= edges[0]) & (classes >= 0) & (classes <= 2)
    bin_idx = np.searchsorted(edges, lengths[mask], side="right") - 1
    bin_idx = np.clip(bin_idx, 0, len(edges) - 2)
    
    if weights is None:
        for b, c in zip(bin_idx, classes[mask]):
            table[int(b), int(c)] += 1.0
    else:
        w_mask = weights[mask]
        for b, c, w in zip(bin_idx, classes[mask], w_mask):
            table[int(b), int(c)] += float(w)
    return table


# 카운트를 정규화해 (길이×등급) 결합 확률표를 만든다.
# 인자: lengths, classes, edges, weights. 반환: (확률표, 총합). 총합 0이면 균등분포 반환.
def probability_table(
    lengths: np.ndarray,
    classes: np.ndarray,
    edges: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, float]:
    counts = binned_counts(lengths, classes, edges, weights=weights)
    total = float(np.sum(counts))
    if total == 0.0:
        return np.full_like(counts, 1.0 / counts.size), 0.0
    return counts / total, total


# 결합 확률표에서 길이 주변확률(등급 방향 합)을 구한다.
def marginal_length_probability(joint_prob: np.ndarray) -> np.ndarray:
    return np.sum(joint_prob, axis=1)


# 결합 확률표에서 검열등급 주변확률(길이 방향 합)을 구한다.
def marginal_class_probability(joint_prob: np.ndarray) -> np.ndarray:
    return np.sum(joint_prob, axis=0)


# 관측 카운트와 모델 확률표로 결합 로그우도를 계산한다(log 0 방지 클립).
def loglik_from_tables(observed_counts: np.ndarray, model_prob: np.ndarray) -> float:
    return float(np.sum(observed_counts * np.log(np.clip(model_prob, 1e-12, 1.0))))


# 주변 카운트와 주변 모델 확률로 로그우도를 계산한다(길이/등급 단독 성분용).
def loglik_from_marginal_counts(observed_counts: np.ndarray, model_prob: np.ndarray) -> float:
    return float(np.sum(observed_counts * np.log(np.clip(model_prob, 1e-12, 1.0))))


# 우도 목적함수 선택에 따라 후보 kr의 점수(로그우도)를 계산한다.
# length_only/class_only는 해당 성분만, 그 외는 결합(또는 length+가중치·class).
# 인자: 세 성분 로그우도, likelihood_component, class_likelihood_weight. 반환: 점수.
def score_candidate(
    loglik_joint: float,
    loglik_length: float,
    loglik_class: float,
    likelihood_component: str,
    class_likelihood_weight: float,
) -> float:
    if likelihood_component == "length_only":
        return loglik_length
    if likelihood_component == "class_only":
        return loglik_class
    if abs(class_likelihood_weight - 1.0) > 1e-12:
        return loglik_length + class_likelihood_weight * loglik_class
    return loglik_joint


# 프로파일 우도 곡선을 요약한다: 최대 로그우도, Δ<=2 신뢰구간 폭, 약식별성 플래그.
# 인자: profile_rows(각 kr의 loglik), kr_min/max 격자범위. 반환: 요약 dict.
def summarize_profile(profile_rows: Sequence[dict], kr_min: float, kr_max: float) -> dict:
    loglik = np.asarray([float(row["loglik"]) for row in profile_rows], dtype=np.float64)
    kr = np.asarray([float(row["kr_window_mc"]) for row in profile_rows], dtype=np.float64)
    max_loglik = float(np.max(loglik))
    inside = kr[(max_loglik - loglik) <= 2.0]
    width = float(np.max(inside) - np.min(inside)) if len(inside) else float("nan")
    return {
        "max_loglik": max_loglik,
        "profile_width_delta2": width,
        "weak_identifiability_flag": bool(np.isfinite(width) and width > 0.5 * (kr_max - kr_min)),
    }


# 검열등급 배열에서 등급 0/1/2 각각의 비율을 계산한다.
# 인자: classes. 반환: (등급0 비율, 등급1 비율, 등급2 비율); 비면 NaN.
def fraction_by_class(classes: np.ndarray) -> tuple[float, float, float]:
    if len(classes) == 0:
        return float("nan"), float("nan"), float("nan")
    return tuple(float(np.mean(classes == cls)) for cls in (0, 1, 2))


# 적합 진단 지표들로 최종 적합 상태(fit_status)와 기각 여부를 판정한다.
# 점검: MC 채택수 부족 / 약식별성 / 사후예측 분위수비(q90,q95) / 검열등급 L1 오차.
# 인자: set_id, MC 채택수, q90/q95 비, class_l1, 약식별성 플래그, window_mode.
# 반환: (status, rejected 여부, 사유, 경고 리스트). Set 4는 provisional_ok로 상한.
def determine_status(
    set_id: int,
    n_model_accepted: int,
    q90_ratio: float,
    q95_ratio: float,
    class_l1: float,
    weak_identifiability: bool,
    window_mode: str,
) -> tuple[str, bool, str, List[str]]:
    warnings = [WINDOW_WARNING_POLYGON if window_mode == "polygon" else WINDOW_WARNING_BBOX]
    if n_model_accepted < 5000:
        return "low_mc_acceptance", True, "MC accepted samples < 5000", warnings
    if weak_identifiability:
        return "weak_identifiability", True, "profile likelihood delta<=2 interval too wide", warnings
    if np.isfinite(q90_ratio) and (q90_ratio < 1.0 / 3.0 or q90_ratio > 3.0):
        return "posterior_predictive_failed", True, "q90_model/q90_observed outside [1/3, 3]", warnings
    if np.isfinite(q95_ratio) and (q95_ratio < 1.0 / 3.0 or q95_ratio > 3.0):
        return "posterior_predictive_failed", True, "q95_model/q95_observed outside [1/3, 3]", warnings
    if class_l1 > 0.30:
        return "class_fraction_mismatch", True, "class_fraction_l1_error > 0.30", warnings
    if set_id == 4:
        warnings.append(SET4_WARNING)
        return "provisional_ok", False, "Set 4 capped at provisional_ok", warnings
    return "ok", False, "", warnings


# 중심 가중 방식의 상태 라벨을 반환한다(proposal_area=권장, 그 외=레거시 진단).
def center_weighting_status(center_weighting: str) -> str:
    if center_weighting == "proposal_area":
        return "preferred_for_window_mc"
    return "legacy_diagnostic"


# 길이/등급/결합 우도로 각각 얻은 kr_hat를 참값과 비교해 편향(bias)의 원인을 분류한다.
# 예: 등급우도가 kr을 오염, 검열과정 편향, 길이과정 편향, 순방향모델 편향 등.
# 인자: kr_true, 세 성분의 kr_hat. 반환: 편향 원인 문자열.
def determine_bias_source(
    kr_true: Optional[float],
    kr_hat_length_only: float,
    kr_hat_class_only: float,
    kr_hat_joint: float,
) -> str:
    if kr_true is None or not np.isfinite(kr_true):
        return "unknown"
    err_length = abs(kr_hat_length_only - kr_true)
    err_class = abs(kr_hat_class_only - kr_true)
    err_joint = abs(kr_hat_joint - kr_true)
    if err_length <= _RECOVERY_MODERATE_THRESHOLD and err_joint > _RECOVERY_MODERATE_THRESHOLD:
        return "class_likelihood_contaminates_kr"
    if kr_hat_class_only < kr_true - _RECOVERY_MODERATE_THRESHOLD and kr_hat_joint < kr_true - _RECOVERY_MODERATE_THRESHOLD:
        return "censoring_class_process_bias"
    if kr_hat_length_only < kr_true - _RECOVERY_MODERATE_THRESHOLD:
        return "length_process_bias"
    if err_length > _RECOVERY_MODERATE_THRESHOLD and err_class > _RECOVERY_MODERATE_THRESHOLD:
        return "visibility_forward_model_bias"
    return "mixed_or_reduced_bias"


# 하나의 후보(candidate)에서 회복/채택 상태를 계산해 요약 행(dict)을 만든다.
# 인자: candidate(kr 및 진단지표), kr_true. 반환: kr_hat/오차/상태 등을 담은 dict.
def build_component_row(candidate: dict, kr_true: Optional[float]) -> dict:
    recovery_status = determine_recovery_status(float(candidate["kr"]), kr_true)
    adoption_status = determine_adoption_status(str(candidate["fit_status"]), recovery_status)
    return {
        "kr_hat": float(candidate["kr"]),
        "kr_abs_error": abs(float(candidate["kr"]) - kr_true) if kr_true is not None and np.isfinite(kr_true) else float("nan"),
        "q90_ratio": float(candidate["q90_ratio"]),
        "q95_ratio": float(candidate["q95_ratio"]),
        "class_l1": float(candidate["class_l1"]),
        "fit_status": str(candidate["fit_status"]),
        "recovery_status": recovery_status,
        "adoption_status": adoption_status,
    }


# [핵심 함수] 하나의 (set, lmin_fit) 조합에 대해 kr 격자를 순회하며 창-MC 우도를
#   최대화하는 kr_hat를 추정하고 진단 지표를 계산한다.
# 인자(주요): set_id, set_rows 관측 트레이스, polygon_yz 창, kr_grid 후보 격자,
#   rmin/rmax 반경 지지구간, lmin_fit 길이 하한, mc_samples_per_grid, bin 설정,
#   window/direction_mode, kr_true(회복평가), center_weighting, likelihood_component,
#   oracle_radius_mode(진단), run_bootstrap/n_bootstrap.
# 반환: (fit_row 요약, profile_rows 프로파일, pp_rows 사후예측, candidate_rows 후보들).
def fit_set_lmin(
    set_id: int,
    set_rows: Sequence[dict],
    polygon_yz: np.ndarray,
    kr_grid: np.ndarray,
    rmin: float,
    rmax: float,
    lmin_fit: float,
    mc_samples_per_grid: int,
    bin_count: int,
    bin_mode: str,
    window_mode: str,
    direction_mode: str = "empirical_trace",
    site: str = "",
    kr_true: Optional[float] = None,
    center_weighting: str = "unweighted",
    likelihood_component: str = "joint",
    class_likelihood_weight: float = 1.0,
    oracle_radius_mode: str = "none",
    run_bootstrap: bool = False,
    n_bootstrap: int = 100,
) -> tuple[dict, List[dict], List[dict], List[dict]]:
    # 관측 길이/등급을 배열로 모으고, lmin_fit 이상만 적합에 사용(관측 히스토그램/주변 카운트 준비).
    lengths_all = np.asarray([float(row["observed_length_m"]) for row in set_rows], dtype=np.float64)
    classes_all = np.asarray([int(row["censoring_class"]) for row in set_rows], dtype=np.int32)
    used = lengths_all >= lmin_fit
    obs_lengths = lengths_all[used]
    obs_classes = classes_all[used]
    edges = make_length_edges(obs_lengths, lmin_fit, bin_count, bin_mode)
    obs_counts = binned_counts(obs_lengths, obs_classes, edges)
    obs_length_counts = np.sum(obs_counts, axis=1)
    obs_class_counts = np.sum(obs_counts, axis=0)
    # 관측 방향 풀과 (진단용) 관측 트레이스의 실제 반경 목록을 준비.
    directions = empirical_trace_directions_yz(set_rows)
    oracle_radii_all = np.asarray(
        [float(row.get("radius_m", float("nan"))) for row in set_rows if np.isfinite(float(row.get("radius_m", float("nan"))))],
        dtype=np.float64,
    )

    profile_rows: List[dict] = []
    pp_rows: List[dict] = []
    rng_base = 91000 + set_id * 1000 + int(round(lmin_fit * 1000.0))
    best = None
    candidate_rows: List[dict] = []
    
    # Store precomputed likelihood summaries for bootstrap and decomposition
    grid_prob_summaries = []
    
    # kr 격자를 순회하며 각 후보에 대해 창-MC 시뮬레이션과 로그우도를 계산.
    for idx, kr in enumerate(kr_grid):
        rng = np.random.default_rng(rng_base + idx)
        # 반경 소스: oracle 모드는 관측 반경 재표본(진단), 아니면 kr로 size-biased 표본.
        if oracle_radius_mode == "observed_trace_radii":
            if len(oracle_radii_all) == 0:
                raise ValueError("oracle_radius_mode=observed_trace_radii requires radius_m in observed trace rows.")
            oracle_radii = rng.choice(oracle_radii_all, size=mc_samples_per_grid, replace=True)
            sim_lengths, sim_classes, sim_areas, sim_radii = simulate_window_samples(
                polygon_yz,
                directions,
                oracle_radii,
                rng,
                window_mode=window_mode,
                direction_mode=direction_mode,
                set_id=set_id,
                site=site,
            )
        else:
            sim_radii = None
            sim_lengths, sim_classes, sim_areas = simulate_window_observations(
                float(kr), rmin, rmax, polygon_yz, directions, mc_samples_per_grid, rng,
                window_mode=window_mode, direction_mode=direction_mode,
                set_id=set_id, site=site,
            )
        # 시뮬레이션 트레이스도 lmin_fit 이상만 사용해 관측과 동일 조건으로 비교.
        sim_used = sim_lengths >= lmin_fit
        sim_lengths_used = sim_lengths[sim_used]
        sim_classes_used = sim_classes[sim_used]
        sim_areas_used = sim_areas[sim_used]
        sim_radii_used = sim_radii[sim_used] if sim_radii is not None else None
        
        # Determine weighting
        weights = sim_areas_used if center_weighting == "proposal_area" else None
        
        # 시뮬레이션으로 모델 확률표(결합/길이/등급)를 만들고, 관측 카운트와 로그우도 계산.
        prob_joint, n_model_used = probability_table(sim_lengths_used, sim_classes_used, edges, weights=weights)
        prob_length = marginal_length_probability(prob_joint)
        prob_class = marginal_class_probability(prob_joint)

        loglik_joint = loglik_from_tables(obs_counts, prob_joint)
        loglik_length = loglik_from_marginal_counts(obs_length_counts, prob_length)
        loglik_class = loglik_from_marginal_counts(obs_class_counts, prob_class)
        loglik = score_candidate(
            loglik_joint,
            loglik_length,
            loglik_class,
            likelihood_component,
            class_likelihood_weight,
        )

        # 관측/모델의 검열등급 비율 차이(L1)와 길이 분위수비(q90,q95)로 적합도 진단.
        obs_unc, obs_one, obs_two = fraction_by_class(obs_classes)
        mod_unc, mod_one, mod_two = fraction_by_class(sim_classes_used)
        class_l1 = float(abs(obs_unc - mod_unc) + abs(obs_one - mod_one) + abs(obs_two - mod_two))
        q90_obs = float(np.percentile(obs_lengths, 90)) if len(obs_lengths) else float("nan")
        q95_obs = float(np.percentile(obs_lengths, 95)) if len(obs_lengths) else float("nan")
        q90_mod = float(np.percentile(sim_lengths_used, 90)) if len(sim_lengths_used) else float("nan")
        q95_mod = float(np.percentile(sim_lengths_used, 95)) if len(sim_lengths_used) else float("nan")
        q90_ratio = q90_mod / q90_obs if q90_obs > 0.0 else float("nan")
        q95_ratio = q95_mod / q95_obs if q95_obs > 0.0 else float("nan")

        candidate_rows.append(
            {
                "kr": float(kr),
                "loglik_joint": loglik_joint,
                "loglik_length_only": loglik_length,
                "loglik_class_only": loglik_class,
                "loglik": loglik,
                "n_model_used": n_model_used,
                "sim_lengths": sim_lengths_used,
                "sim_classes": sim_classes_used,
                "sim_areas": sim_areas_used,
                "sim_radii": sim_radii_used,
                "q90_ratio": q90_ratio,
                "q95_ratio": q95_ratio,
                "class_l1": class_l1,
                "obs_unc": obs_unc,
                "obs_one": obs_one,
                "obs_two": obs_two,
                "mod_unc": mod_unc,
                "mod_one": mod_one,
                "mod_two": mod_two,
                "prob_joint": prob_joint,
                "prob_length": prob_length,
                "prob_class": prob_class,
            }
        )
        grid_prob_summaries.append(
            {
                "joint": prob_joint,
                "length_only": prob_length,
                "class_only": prob_class,
            }
        )

        profile_rows.append(
            {
                "set_id": set_id,
                "lmin_fit": float(lmin_fit),
                "kr_window_mc": float(kr),
                "loglik": loglik,
                "loglik_joint": loglik_joint,
                "loglik_length_only": loglik_length,
                "loglik_class_only": loglik_class,
                "mc_accepted_used": n_model_used,
            }
        )
        # 현재까지 최고 로그우도 후보를 best로 갱신.
        if best is None or loglik > best["loglik"]:
            best = dict(candidate_rows[-1])

    # 프로파일 우도 요약(최대값/신뢰폭/약식별성)과 각 행의 Δloglik 기록.
    assert best is not None
    profile_summary = summarize_profile(profile_rows, float(kr_grid[0]), float(kr_grid[-1]))
    for row in profile_rows:
        row["max_loglik"] = profile_summary["max_loglik"]
        row["delta_loglik"] = profile_summary["max_loglik"] - float(row["loglik"])

    # Compute proposal area metrics on the best model's simulation
    best_areas = best["sim_areas"]
    if len(best_areas) > 0:
        mean_area = float(np.mean(best_areas))
        p50_area = float(np.percentile(best_areas, 50))
        p90_area = float(np.percentile(best_areas, 90))
        # Correlation between simulated visible length and proposal area
        if len(best_areas) > 1 and np.std(best["sim_lengths"]) > 1e-8 and np.std(best_areas) > 1e-8:
            area_len_corr = float(np.corrcoef(best["sim_lengths"], best_areas)[0, 1])
        else:
            area_len_corr = 0.0
    else:
        mean_area, p50_area, p90_area, area_len_corr = float("nan"), float("nan"), float("nan"), float("nan")

    # 최적(best) 후보 기준으로 관측/모델 등급비율, 분위수, 상태 판정을 재계산.
    obs_unc = float(best["obs_unc"])
    obs_one = float(best["obs_one"])
    obs_two = float(best["obs_two"])
    mod_unc = float(best["mod_unc"])
    mod_one = float(best["mod_one"])
    mod_two = float(best["mod_two"])
    class_l1 = float(abs(obs_unc - mod_unc) + abs(obs_one - mod_one) + abs(obs_two - mod_two))
    q50_obs = float(np.percentile(obs_lengths, 50)) if len(obs_lengths) else float("nan")
    q90_obs = float(np.percentile(obs_lengths, 90)) if len(obs_lengths) else float("nan")
    q95_obs = float(np.percentile(obs_lengths, 95)) if len(obs_lengths) else float("nan")
    q50_mod = float(np.percentile(best["sim_lengths"], 50)) if len(best["sim_lengths"]) else float("nan")
    q90_mod = float(np.percentile(best["sim_lengths"], 90)) if len(best["sim_lengths"]) else float("nan")
    q95_mod = float(np.percentile(best["sim_lengths"], 95)) if len(best["sim_lengths"]) else float("nan")
    q90_ratio = q90_mod / q90_obs if q90_obs > 0.0 else float("nan")
    q95_ratio = q95_mod / q95_obs if q95_obs > 0.0 else float("nan")
    status, rejected, reason, warnings = determine_status(
        set_id,
        int(best["n_model_used"]),
        q90_ratio,
        q95_ratio,
        class_l1,
        profile_summary["weak_identifiability_flag"],
        window_mode,
    )
    # 등급별 (모델-관측) 차이 중 가장 큰 항으로 지배적 등급 오차 유형을 라벨링.
    class_diffs = {
        "uncensored": mod_unc - obs_unc,
        "one_end": mod_one - obs_one,
        "two_end": mod_two - obs_two,
    }
    dominant_key = max(class_diffs, key=lambda key: abs(class_diffs[key]))
    if abs(class_diffs[dominant_key]) < 0.10:
        dominant_class_error = "mixed"
    elif dominant_key == "uncensored":
        dominant_class_error = "uncensored_overpredicted" if class_diffs[dominant_key] > 0.0 else "uncensored_underpredicted"
    elif dominant_key == "one_end":
        dominant_class_error = "one_end_overpredicted" if class_diffs[dominant_key] > 0.0 else "one_end_underpredicted"
    else:
        dominant_class_error = "two_end_overpredicted" if class_diffs[dominant_key] > 0.0 else "two_end_underpredicted"

    # 사후예측(posterior predictive) 점검용: 등급별 관측/모델 비율 행 기록.
    for cls, label in [(0, "uncensored"), (1, "one_end_censored"), (2, "two_end_censored")]:
        pp_rows.append(
            {
                "set_id": set_id,
                "lmin_fit": float(lmin_fit),
                "kr_window_mc": best["kr"],
                "class": label,
                "observed_fraction": [obs_unc, obs_one, obs_two][cls],
                "model_fraction": [mod_unc, mod_one, mod_two][cls],
            }
        )

    n_unc = int(np.sum(obs_classes == 0))
    n_one = int(np.sum(obs_classes == 1))
    n_two = int(np.sum(obs_classes == 2))
    
    # Bootstrap CI calculation
    kr_boot_mean, kr_boot_std, kr_ci_low, kr_ci_high = float("nan"), float("nan"), float("nan"), float("nan")
    boundary_fraction = float("nan")
    recovery_ci_status = "unknown"
    
    # 부트스트랩: 관측 트레이스를 복원추출하며 각 재표본에서 최적 kr을 재선택해 CI를 추정.
    # (격자별 확률표는 미리 계산된 grid_prob_summaries를 재사용하므로 재시뮬레이션 불필요)
    if run_bootstrap and len(obs_lengths) > 0:
        boot_krs = []
        boot_rng = np.random.default_rng(12345 + set_id)
        for _ in range(n_bootstrap):
            idx_resampled = boot_rng.choice(len(obs_lengths), size=len(obs_lengths), replace=True)
            boot_lengths = obs_lengths[idx_resampled]
            boot_classes = obs_classes[idx_resampled]
            boot_counts = binned_counts(boot_lengths, boot_classes, edges)
            
            boot_best_lik = -np.inf
            boot_best_kr = kr_grid[0]
            boot_length_counts = np.sum(boot_counts, axis=1)
            boot_class_counts = np.sum(boot_counts, axis=0)
            for g_idx, g_probs in enumerate(grid_prob_summaries):
                lik = score_candidate(
                    loglik_from_tables(boot_counts, g_probs["joint"]),
                    loglik_from_marginal_counts(boot_length_counts, g_probs["length_only"]),
                    loglik_from_marginal_counts(boot_class_counts, g_probs["class_only"]),
                    likelihood_component,
                    class_likelihood_weight,
                )
                if lik > boot_best_lik:
                    boot_best_lik = lik
                    boot_best_kr = kr_grid[g_idx]
            boot_krs.append(boot_best_kr)
        
        # 부트스트랩 kr 분포에서 평균/표준편차/2.5-97.5% CI와 경계 히트 비율을 산출.
        boot_krs = np.array(boot_krs, dtype=np.float64)
        kr_boot_mean = float(np.mean(boot_krs))
        kr_boot_std = float(np.std(boot_krs))
        kr_ci_low = float(np.percentile(boot_krs, 2.5))
        kr_ci_high = float(np.percentile(boot_krs, 97.5))
        boundary_fraction = float(np.mean((boot_krs <= kr_grid[0] + 1e-5) | (boot_krs >= kr_grid[-1] - 1e-5)))
        
        if kr_true is not None and np.isfinite(kr_true):
            if kr_ci_low <= kr_true <= kr_ci_high:
                recovery_ci_status = "recovery_uncertain"
            else:
                recovery_ci_status = "systematic_bias"
    
    # 회복/채택 상태 판정, 그리고 우도 성분별(결합/길이/등급) 최적 후보를 분해용으로 추출.
    recovery_status = determine_recovery_status(float(best["kr"]), kr_true)
    adoption_status = determine_adoption_status(status, recovery_status)
    best_joint = max(candidate_rows, key=lambda row: row["loglik_joint"])
    best_length = max(candidate_rows, key=lambda row: row["loglik_length_only"])
    best_class = max(candidate_rows, key=lambda row: row["loglik_class_only"])
    # 모든 추정치/진단지표/상태를 담은 최종 요약 행 구성.
    fit_row = {
        "set_id": set_id,
        "model": f"window_mc_{window_mode}_v4_1",
        "window_mode": window_mode,
        "direction_mode": direction_mode,
        "center_weighting": center_weighting,
        "center_weighting_status": center_weighting_status(center_weighting),
        "likelihood_component": likelihood_component,
        "class_likelihood_weight": float(class_likelihood_weight),
        "main_class_likelihood_weight": float(class_likelihood_weight),
        "oracle_radius_mode": oracle_radius_mode,
        "weighted_probability_used": bool(center_weighting == "proposal_area"),
        "lmin_fit": float(lmin_fit),
        "rmin": float(rmin),
        "rmax": float(rmax),
        "kr_window_mc_hat": best["kr"],
        "kr_true": kr_true if kr_true is not None else float("nan"),
        "loglik": best["loglik"],
        "n_total": len(set_rows),
        "n_used": len(obs_lengths),
        "n_uncensored_used": n_unc,
        "n_one_end_censored_used": n_one,
        "n_two_end_censored_used": n_two,
        "censoring_ratio_used": (n_one + n_two) / len(obs_lengths) if len(obs_lengths) else float("nan"),
        "two_end_censoring_ratio_used": n_two / len(obs_lengths) if len(obs_lengths) else float("nan"),
        "mc_accepted_count": int(best["n_model_used"]),
        "q50_observed": q50_obs,
        "q90_observed": q90_obs,
        "q95_observed": q95_obs,
        "q50_model": q50_mod,
        "q90_model": q90_mod,
        "q95_model": q95_mod,
        "q90_ratio_model_observed": q90_ratio,
        "q95_ratio_model_observed": q95_ratio,
        "observed_uncensored_fraction": obs_unc,
        "model_uncensored_fraction": mod_unc,
        "observed_one_end_fraction": obs_one,
        "model_one_end_fraction": mod_one,
        "observed_two_end_fraction": obs_two,
        "model_two_end_fraction": mod_two,
        "class_fraction_l1_error": class_l1,
        "dominant_class_error": dominant_class_error,
        "profile_width_delta2": profile_summary["profile_width_delta2"],
        "weak_identifiability_flag": profile_summary["weak_identifiability_flag"],
        "mean_proposal_area": mean_area,
        "p50_proposal_area": p50_area,
        "p90_proposal_area": p90_area,
        "proposal_area_length_corr": area_len_corr,
        "kr_boot_mean": kr_boot_mean,
        "kr_boot_std": kr_boot_std,
        "kr_ci_low": kr_ci_low,
        "kr_ci_high": kr_ci_high,
        "bootstrap_boundary_fraction": boundary_fraction,
        "recovery_ci_status": recovery_ci_status,
        "fit_status": status,           # posterior predictive criterion
        "final_status": status,         # backward-compat alias
        "recovery_status": recovery_status,  # kr_true recovery criterion
        "adoption_status": adoption_status,  # combined decision
        "model_rejected": bool(rejected),
        "rejection_reason": reason,
        "kr_hat_length_only": float(best_length["kr"]),
        "kr_hat_class_only": float(best_class["kr"]),
        "kr_hat_joint": float(best_joint["kr"]),
        "kr_abs_error_length_only": abs(float(best_length["kr"]) - kr_true) if kr_true is not None and np.isfinite(kr_true) else float("nan"),
        "kr_abs_error_class_only": abs(float(best_class["kr"]) - kr_true) if kr_true is not None and np.isfinite(kr_true) else float("nan"),
        "kr_abs_error_joint": abs(float(best_joint["kr"]) - kr_true) if kr_true is not None and np.isfinite(kr_true) else float("nan"),
        "q90_ratio_length_only": float(best_length["q90_ratio"]),
        "q95_ratio_length_only": float(best_length["q95_ratio"]),
        "class_l1_joint": float(best_joint["class_l1"]),
        "fit_status_joint": determine_status(
            set_id,
            int(best_joint["n_model_used"]),
            float(best_joint["q90_ratio"]),
            float(best_joint["q95_ratio"]),
            float(best_joint["class_l1"]),
            profile_summary["weak_identifiability_flag"],
            window_mode,
        )[0],
        "recovery_status_joint": determine_recovery_status(float(best_joint["kr"]), kr_true),
        "bias_source": determine_bias_source(
            kr_true,
            float(best_length["kr"]),
            float(best_class["kr"]),
            float(best_joint["kr"]),
        ),
        "warning": "; ".join(warnings),
    }
    return fit_row, profile_rows, pp_rows, candidate_rows



# bbox 창과 polygon 창의 적합 결과를 나란히 비교하는 행(dict)을 만든다.
# 인자: bbox_row, polygon_row 각 적합 요약. 반환: kr 차이 등 비교 지표 dict.
def build_bbox_polygon_comparison_row(bbox_row: dict, polygon_row: dict) -> dict:
    return {
        "set_id": polygon_row["set_id"],
        "lmin_fit": polygon_row["lmin_fit"],
        "kr_bbox": bbox_row["kr_window_mc_hat"],
        "kr_polygon": polygon_row["kr_window_mc_hat"],
        "delta_kr": polygon_row["kr_window_mc_hat"] - bbox_row["kr_window_mc_hat"],
        "class_l1_bbox": bbox_row["class_fraction_l1_error"],
        "class_l1_polygon": polygon_row["class_fraction_l1_error"],
        "q90_ratio_bbox": bbox_row["q90_ratio_model_observed"],
        "q90_ratio_polygon": polygon_row["q90_ratio_model_observed"],
        "q95_ratio_bbox": bbox_row["q95_ratio_model_observed"],
        "q95_ratio_polygon": polygon_row["q95_ratio_model_observed"],
        "status_bbox": bbox_row["final_status"],
        "status_polygon": polygon_row["final_status"],
    }


# dict 행 리스트를 CSV로 기록한다(첫 행 키를 헤더로 사용). 행이 없으면 에러.
# 인자: rows, path 출력 경로.
def write_csv(rows: Sequence[dict], path: str) -> None:
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


# 길이 표본의 경험적 생존곡선 S(x)=P(L>x)를 계산한다.
# 인자: lengths. 반환: (정렬된 고유 길이 grid, 각 지점 생존확률).
def empirical_survival_curve(lengths: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    lengths = np.atleast_1d(np.asarray(lengths, dtype=np.float64))
    if len(lengths) == 0:
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)
    grid = np.unique(np.sort(lengths))
    survival = np.asarray([np.mean(lengths > value) for value in grid], dtype=np.float64)
    return grid, survival


# 검열을 고려한 Kaplan-Meier 생존곡선을 계산한다(등급 0=비검열을 사건으로 취급).
# 인자: lengths, censoring_classes. 반환: (길이 grid, KM 생존확률).
def kaplan_meier_survival_curve(lengths: np.ndarray, censoring_classes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    lengths = np.atleast_1d(np.asarray(lengths, dtype=np.float64))
    censoring_classes = np.atleast_1d(np.asarray(censoring_classes, dtype=np.int32))
    if len(lengths) == 0:
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)
    grid = np.unique(np.sort(lengths))
    survival = np.empty(len(grid), dtype=np.float64)
    # 각 길이 지점에서 위험집합(at_risk)과 사건수(비검열 정확일치)로 생존확률을 곱연산 갱신.
    current_survival = 1.0
    for idx, value in enumerate(grid):
        at_risk = lengths >= value - 1e-12
        exact = np.abs(lengths - value) <= 1e-12
        n_i = int(np.sum(at_risk))
        d_i = int(np.sum(exact & (censoring_classes == 0)))
        if n_i > 0 and d_i > 0:
            current_survival *= (1.0 - d_i / n_i)
        survival[idx] = current_survival
    return grid, survival


# [엔트리포인트] CLI 인자를 파싱해 입력을 로드하고, set×lmin_fit 조합마다 fit_set_lmin을
#   호출하여 kr을 추정한 뒤, 적합/프로파일/사후예측/비교/생존곡선 등 결과를 CSV/JSON으로 저장한다.
def main() -> None:
    # 인자 정의: 입력(h5/csv), 반경/커널 지지구간, kr 격자, MC/빈 설정, 창/방향 모드,
    # 가중/우도성분/부트스트랩/오라클 등 진단 옵션, 출력 디렉터리.
    parser = argparse.ArgumentParser(description="Window-aware Monte Carlo likelihood for radius power-law candidates.")
    parser.add_argument("--trace-h5", help="Input trace HDF5")
    parser.add_argument("--trace-csv", help="Input trace CSV")
    parser.add_argument("--target-set", nargs="+", type=int)
    parser.add_argument("--dfn-model", default="", help="DFN model/site label, e.g. forsmark or laxemar.")
    parser.add_argument("--rmin", type=float, default=0.5, help="Estimation rmin")
    parser.add_argument("--rmax", type=float, default=250.0, help="Estimation rmax")
    parser.add_argument(
        "--set-rmin-mode",
        choices=["global", "effective_generation", "table_r0"],
        default="effective_generation",
        help="Per-set lower-bound mode for radius support: global, effective_generation, or table_r0.",
    )
    parser.add_argument("--generation-rmin", type=float, default=0.5, help="Radius lower bound used during DFN generation.")
    parser.add_argument("--generation-rmax", type=float, default=250.0, help="Radius upper bound used during DFN generation.")
    parser.add_argument("--p32-label", default="P32_r_ge_0p5m", help="P32 population label, e.g. P32_r_ge_0p5m.")
    parser.add_argument("--kr-min", type=float, default=1.5)
    parser.add_argument("--kr-max", type=float, default=5.5)
    parser.add_argument("--lmin-fit-values", nargs="+", type=float, default=[0.1, 0.2, 0.3, 0.5, 0.75])
    parser.add_argument("--allow-rmin-mismatch", action="store_true", help="Allow estimation rmin to differ from DFN generation rmin.")
    parser.add_argument("--profile-grid-size", type=int, default=81)
    parser.add_argument("--mc-samples-per-grid", type=int, default=50000)
    parser.add_argument("--length-bin-count", type=int, default=40)
    parser.add_argument("--length-bin-mode", choices=["log", "linear"], default="log")
    parser.add_argument("--window-mode", choices=["polygon", "bbox"], default="polygon")
    parser.add_argument(
        "--direction-mode",
        choices=["empirical_trace", "orientation_conditioned"],
        default="empirical_trace",
        help="Trace direction generation mode: empirical_trace (default) or orientation_conditioned.",
    )
    parser.add_argument(
        "--site",
        choices=["forsmark", "laxemar"],
        default="",
        help="Site name for Fisher param lookup (required when --direction-mode=orientation_conditioned).",
    )
    parser.add_argument(
        "--kr-true-map",
        nargs="+",
        metavar="SET_ID:KR_TRUE",
        help="Known true kr values per set (e.g. 1:2.88 2:3.02). Used to compute recovery_status.",
    )
    parser.add_argument(
        "--center-weighting",
        choices=["unweighted", "proposal_area"],
        default="unweighted",
        help="Weighting mode for MC accepted samples: unweighted (default) or proposal_area.",
    )
    parser.add_argument(
        "--likelihood-component",
        choices=["joint", "length_only", "class_only"],
        default="joint",
        help="Likelihood objective for kr selection: joint (default), length_only, or class_only.",
    )
    parser.add_argument(
        "--class-likelihood-weight",
        nargs="+",
        type=float,
        default=[1.0],
        help="Class likelihood weights for sensitivity analysis. The first value is used for the main fit if needed.",
    )
    parser.add_argument(
        "--oracle-radius-mode",
        choices=["none", "observed_trace_radii"],
        default="none",
        help="Diagnostic-only radius source override. observed_trace_radii resamples radius_m from observed traces instead of sampling by kr.",
    )
    parser.add_argument(
        "--run-bootstrap",
        action="store_true",
        help="Run bootstrap resampling on observed traces to estimate kr confidence intervals.",
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=100,
        help="Number of bootstrap iterations.",
    )
    parser.add_argument(
        "--export-predicted-survival",
        action="store_true",
        help="Export MC-predicted trace-length survival curves for the fitted kr_hat values.",
    )
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()

    # 입력은 h5/csv 중 정확히 하나여야 하며, 생성 rmin/rmax와 P32 라벨 기본값을 확정.
    if bool(args.trace_h5) == bool(args.trace_csv):
        raise ValueError("Provide exactly one of --trace-h5 or --trace-csv.")
    generation_rmin = float(args.generation_rmin) if args.generation_rmin is not None else float(args.rmin)
    generation_rmax = float(args.generation_rmax) if args.generation_rmax is not None else float(args.rmax)
    p32_label = args.p32_label or f"P32_r_ge_{str(args.rmin).replace('.', 'p')}m"

    # rmin consistency guard
    h5_generation_rmin = None
    trace_rmin_metadata = {}
    if args.trace_h5:
        try:
            trace_rmin_metadata = load_trace_rmin_metadata_from_h5(args.trace_h5)
            if trace_rmin_metadata.get("generation_rmin") is not None:
                h5_generation_rmin = float(trace_rmin_metadata["generation_rmin"])
        except Exception as e:
            print(f"[WARNING] Failed to read generation_rmin from trace HDF5: {e}")

    # 추정 rmin과 DFN 생성 rmin의 일관성 점검(불일치 시 --allow-rmin-mismatch 없으면 에러).
    gen_rmin_to_check = generation_rmin
    if h5_generation_rmin is not None:
        gen_rmin_to_check = h5_generation_rmin

    rmin_consistency_status = "matched"
    diagnostic_only = False
    warning_msg = None

    if gen_rmin_to_check is not None:
        if abs(gen_rmin_to_check - args.rmin) > 1e-5:
            if not args.allow_rmin_mismatch:
                raise ValueError(
                    f"mismatch: DFN generation_rmin ({gen_rmin_to_check}) does not match "
                    f"estimation rmin ({args.rmin}). Use --allow-rmin-mismatch to bypass."
                )
            else:
                rmin_consistency_status = "mismatch_diagnostic_only"
                diagnostic_only = True
                warning_msg = "generation_rmin and estimator rmin are inconsistent; kr recovery should not be interpreted"
                print(f"[WARNING] {warning_msg}")

    # kr 참값 지도 구성: 사이트 내장표를 기본으로 하고 --kr-true-map CLI로 덮어씀(회복평가용).
    # Build kr_true_map
    _site_to_kr_true: Dict[str, Dict[int, float]] = {
        "forsmark": {1: 2.88, 2: 3.02, 3: 2.81, 4: 2.95, 5: 2.92},
        "laxemar":  {1: 2.85, 2: 3.04, 3: 3.01, 5: 3.60},
    }
    site = getattr(args, "site", "") or ""
    direction_mode = getattr(args, "direction_mode", "empirical_trace")
    kr_true_map: Dict[int, float] = {}
    # Populate from --site built-in table
    if site:
        kr_true_map = dict(_site_to_kr_true.get(site, {}))
    # Override / supplement from --kr-true-map CLI
    if getattr(args, "kr_true_map", None):
        for token in args.kr_true_map:
            try:
                sid_str, kt_str = token.split(":")
                kr_true_map[int(sid_str)] = float(kt_str)
            except ValueError:
                print(f"[WARNING] Cannot parse --kr-true-map token: {token}")

    if direction_mode == "orientation_conditioned" and not site:
        raise ValueError("--site {forsmark,laxemar} is required when --direction-mode=orientation_conditioned")

    # 트레이스/폴리곤 로드, set별 rmin 조회표 구성, target set 그룹화, kr 격자 생성.
    rows, polygon_yz = load_trace_data_from_h5(args.trace_h5) if args.trace_h5 else load_trace_data_from_csv(args.trace_csv)
    if polygon_yz is None:
        raise ValueError("Window polygon is required for window MC. Use --trace-h5 with /meta/tunnel_poly_yz.")
    set_rmin_lookup = build_set_rmin_lookup(site, float(gen_rmin_to_check), trace_rmin_metadata)

    target_sets = set(args.target_set) if args.target_set else None
    grouped = group_rows_by_set(rows, target_sets)
    if not grouped:
        raise ValueError("No matching rows for target sets.")

    kr_grid = np.linspace(args.kr_min, args.kr_max, args.profile_grid_size, dtype=np.float64)
    fit_rows: List[dict] = []
    profile_rows: List[dict] = []
    pp_rows: List[dict] = []
    comparison_rows: List[dict] = []
    weighting_comp_rows: List[dict] = []
    decomposition_rows: List[dict] = []
    class_weight_rows: List[dict] = []
    survival_rows: List[dict] = []
    main_class_weight = float(args.class_likelihood_weight[0])
    
    # 메인 루프: 각 set과 각 lmin_fit에 대해 set별 우도 rmin을 확정하고 적합을 수행.
    for set_id, set_rows in grouped.items():
        for lmin_fit in args.lmin_fit_values:
            kr_true = kr_true_map.get(set_id)
            set_likelihood_rmin, set_effective_generation_rmin, set_table_r0, set_support_status = resolve_set_likelihood_rmin(
                set_id,
                args.set_rmin_mode,
                float(args.rmin),
                set_rmin_lookup,
            )
            fit_row, profile, pp, candidate_rows = fit_set_lmin(
                set_id,
                set_rows,
                polygon_yz,
                kr_grid,
                set_likelihood_rmin,
                args.rmax,
                float(lmin_fit),
                args.mc_samples_per_grid,
                args.length_bin_count,
                args.length_bin_mode,
                args.window_mode,
                direction_mode=direction_mode,
                site=site,
                kr_true=kr_true,
                center_weighting=args.center_weighting,
                likelihood_component=args.likelihood_component,
                class_likelihood_weight=main_class_weight,
                oracle_radius_mode=args.oracle_radius_mode,
                run_bootstrap=args.run_bootstrap,
                n_bootstrap=args.n_bootstrap,
            )
            # 적합 결과에 rmin 일관성/진단전용 플래그와 메타데이터를 부착하고 채택 상태를 재판정.
            fit_row["rmin_consistency_status"] = rmin_consistency_status
            fit_row["diagnostic_only"] = bool(diagnostic_only)
            if warning_msg:
                fit_row["warning"] = (fit_row["warning"] + "; " if fit_row["warning"] else "") + warning_msg
            metadata = {
                "dfn_model": args.dfn_model,
                "generation_rmin": generation_rmin,
                "generation_rmax": generation_rmax,
                "estimation_rmin": float(args.rmin),
                "estimation_rmax": float(args.rmax),
                "likelihood_rmin": float(set_likelihood_rmin),
                "likelihood_rmax": float(args.rmax),
                "set_likelihood_rmin": float(set_likelihood_rmin),
                "set_effective_generation_rmin": float(set_effective_generation_rmin),
                "set_table_r0": float(set_table_r0) if np.isfinite(set_table_r0) else float("nan"),
                "set_rmin_mode": args.set_rmin_mode,
                "rmin_support_status": set_support_status,
                "lmin_fit_values": " ".join(str(v) for v in args.lmin_fit_values),
                "p32_label": p32_label,
            }
            fit_row.update(metadata)
            fit_row["adoption_status"] = determine_adoption_status(
                str(fit_row["fit_status"]),
                str(fit_row["recovery_status"]),
                set_support_status,
            )
            for row in profile:
                row.update(metadata)
            for row in pp:
                row.update(metadata)
            fit_rows.append(fit_row)
            profile_rows.extend(profile)
            pp_rows.extend(pp)
            # (옵션) 적합된 kr_hat에서 MC 예측 생존곡선(가시/KM/참현) 3종을 export.
            if args.export_predicted_survival:
                best_candidate = max(candidate_rows, key=lambda row: row["loglik"])
                sim_lengths_best = np.asarray(best_candidate["sim_lengths"], dtype=np.float64)
                sim_classes_best = np.asarray(best_candidate["sim_classes"], dtype=np.int32)
                visible_lengths, visible_survival = empirical_survival_curve(sim_lengths_best)
                km_lengths, km_survival = kaplan_meier_survival_curve(sim_lengths_best, sim_classes_best)
                true_radii = np.asarray(best_candidate["sim_radii"], dtype=np.float64)
                true_chords = 2.0 * true_radii
                chord_lengths, chord_survival = empirical_survival_curve(true_chords)

                for length_value, surv in zip(visible_lengths, visible_survival):
                    survival_rows.append(
                        {
                            "site": site or args.dfn_model,
                            "set_id": set_id,
                            "lmin_fit": float(lmin_fit),
                            "length": float(length_value),
                            "mc_survival": float(surv),
                            "survival_mode": "mc_observed_visible_survival",
                            "kr_hat": float(fit_row["kr_window_mc_hat"]),
                            "set_likelihood_rmin": float(set_likelihood_rmin),
                            "set_effective_generation_rmin": float(set_effective_generation_rmin),
                            "center_weighting": args.center_weighting,
                            "window_mode": args.window_mode,
                            "direction_mode": direction_mode,
                            "n_mc_samples": int(args.mc_samples_per_grid),
                            "length_bin_count": int(args.length_bin_count),
                            "notes": "predicted_survival_from_fitted_window_mc_visible_lengths",
                        }
                    )
                for length_value, surv in zip(km_lengths, km_survival):
                    survival_rows.append(
                        {
                            "site": site or args.dfn_model,
                            "set_id": set_id,
                            "lmin_fit": float(lmin_fit),
                            "length": float(length_value),
                            "mc_survival": float(surv),
                            "survival_mode": "mc_km_emulated_survival",
                            "kr_hat": float(fit_row["kr_window_mc_hat"]),
                            "set_likelihood_rmin": float(set_likelihood_rmin),
                            "set_effective_generation_rmin": float(set_effective_generation_rmin),
                            "center_weighting": args.center_weighting,
                            "window_mode": args.window_mode,
                            "direction_mode": direction_mode,
                            "n_mc_samples": int(args.mc_samples_per_grid),
                            "length_bin_count": int(args.length_bin_count),
                            "notes": "kaplan_meier_applied_to_simulated_observed_lengths_and_censoring_classes",
                        }
                    )
                for length_value, surv in zip(chord_lengths, chord_survival):
                    survival_rows.append(
                        {
                            "site": site or args.dfn_model,
                            "set_id": set_id,
                            "lmin_fit": float(lmin_fit),
                            "length": float(length_value),
                            "mc_survival": float(surv),
                            "survival_mode": "mc_true_chord_survival",
                            "kr_hat": float(fit_row["kr_window_mc_hat"]),
                            "set_likelihood_rmin": float(set_likelihood_rmin),
                            "set_effective_generation_rmin": float(set_effective_generation_rmin),
                            "center_weighting": args.center_weighting,
                            "window_mode": args.window_mode,
                            "direction_mode": direction_mode,
                            "n_mc_samples": int(args.mc_samples_per_grid),
                            "length_bin_count": int(args.length_bin_count),
                            "notes": "diagnostic_true_chord_survival_before_window_clipping",
                        }
                    )
            # 우도 성분 분해(길이/등급/결합) 및 편향 원인 진단 행 기록.
            decomposition_rows.append(
                {
                    "site": site or args.dfn_model,
                    "set_id": set_id,
                    "lmin_fit": float(lmin_fit),
                    "center_weighting": args.center_weighting,
                    "center_weighting_status": center_weighting_status(args.center_weighting),
                    "direction_mode": direction_mode,
                    "kr_true": kr_true if kr_true is not None else float("nan"),
                    "kr_hat_length_only": fit_row["kr_hat_length_only"],
                    "kr_hat_class_only": fit_row["kr_hat_class_only"],
                    "kr_hat_joint": fit_row["kr_hat_joint"],
                    "kr_abs_error_length_only": fit_row["kr_abs_error_length_only"],
                    "kr_abs_error_class_only": fit_row["kr_abs_error_class_only"],
                    "kr_abs_error_joint": fit_row["kr_abs_error_joint"],
                    "q90_ratio_length_only": fit_row["q90_ratio_length_only"],
                    "q95_ratio_length_only": fit_row["q95_ratio_length_only"],
                    "class_l1_joint": fit_row["class_l1_joint"],
                    "fit_status_joint": fit_row["fit_status_joint"],
                    "recovery_status_joint": fit_row["recovery_status_joint"],
                    "bias_source": fit_row["bias_source"],
                }
            )
            # class-weight 민감도 분석: 등급우도 가중치를 바꿔가며 kr_hat 변화를 기록.
            for class_weight in args.class_likelihood_weight:
                best_weighted = max(
                    candidate_rows,
                    key=lambda row: row["loglik_length_only"] + float(class_weight) * row["loglik_class_only"],
                )
                best_weighted_status, _, _, _ = determine_status(
                    set_id,
                    int(best_weighted["n_model_used"]),
                    float(best_weighted["q90_ratio"]),
                    float(best_weighted["q95_ratio"]),
                    float(best_weighted["class_l1"]),
                    fit_row["weak_identifiability_flag"],
                    args.window_mode,
                )
                weight_summary = build_component_row(
                    {
                        **best_weighted,
                        "fit_status": best_weighted_status,
                    },
                    kr_true,
                )
                class_weight_rows.append(
                    {
                        "site": site or args.dfn_model,
                        "set_id": set_id,
                        "lmin_fit": float(lmin_fit),
                        "class_weight": float(class_weight),
                        "kr_true": kr_true if kr_true is not None else float("nan"),
                        "kr_hat": weight_summary["kr_hat"],
                        "kr_abs_error": weight_summary["kr_abs_error"],
                        "q90_ratio": weight_summary["q90_ratio"],
                        "q95_ratio": weight_summary["q95_ratio"],
                        "class_l1": weight_summary["class_l1"],
                        "fit_status": weight_summary["fit_status"],
                        "recovery_status": weight_summary["recovery_status"],
                        "adoption_status": weight_summary["adoption_status"],
                    }
                )
            
            # If comparing center weighting, run the alternative and write comparison
            if args.center_weighting == "proposal_area":
                # Run the unweighted model as the counterpart for comparison
                unw_fit_row, _, _, _ = fit_set_lmin(
                    set_id,
                    set_rows,
                    polygon_yz,
                    kr_grid,
                    set_likelihood_rmin,
                    args.rmax,
                    float(lmin_fit),
                    args.mc_samples_per_grid,
                    args.length_bin_count,
                    args.length_bin_mode,
                    args.window_mode,
                    direction_mode=direction_mode,
                    site=site,
                    kr_true=kr_true,
                    center_weighting="unweighted",
                    likelihood_component=args.likelihood_component,
                    class_likelihood_weight=main_class_weight,
                    oracle_radius_mode=args.oracle_radius_mode,
                    run_bootstrap=False,
                )
                
                # Append both to comparison rows
                for r, weighting_type in [(unw_fit_row, "unweighted"), (fit_row, "proposal_area")]:
                    weighting_comp_rows.append({
                        "site": site or args.dfn_model,
                        "set_id": set_id,
                        "lmin_fit": float(lmin_fit),
                        "direction_mode": direction_mode,
                        "center_weighting": weighting_type,
                        "center_weighting_status": center_weighting_status(weighting_type),
                        "kr_true": kr_true if kr_true is not None else float("nan"),
                        "kr_hat": r["kr_window_mc_hat"],
                        "kr_abs_error": abs(r["kr_window_mc_hat"] - kr_true) if kr_true is not None else float("nan"),
                        "q90_ratio": r["q90_ratio_model_observed"],
                        "q95_ratio": r["q95_ratio_model_observed"],
                        "class_l1": r["class_fraction_l1_error"],
                        "fit_status": r["fit_status"],
                        "recovery_status": r["recovery_status"],
                        "adoption_status": r["adoption_status"],
                    })
            
            # polygon 모드일 때는 bbox 모드로도 적합해 두 창 방식의 결과를 비교 기록.
            if args.window_mode == "polygon":
                bbox_fit_row, _, _, _ = fit_set_lmin(
                    set_id,
                    set_rows,
                    polygon_yz,
                    kr_grid,
                    set_likelihood_rmin,
                    args.rmax,
                    float(lmin_fit),
                    args.mc_samples_per_grid,
                    args.length_bin_count,
                    args.length_bin_mode,
                    "bbox",
                    direction_mode=direction_mode,
                    site=site,
                    kr_true=kr_true,
                    center_weighting=args.center_weighting,
                    likelihood_component=args.likelihood_component,
                    class_likelihood_weight=main_class_weight,
                    oracle_radius_mode=args.oracle_radius_mode,
                    run_bootstrap=False,
                )
                comparison_rows.append(build_bbox_polygon_comparison_row(bbox_fit_row, fit_row))

    # 출력 디렉터리 생성 및 각 결과 CSV/JSON 경로 지정.
    os.makedirs(args.outdir, exist_ok=True)
    fit_csv = os.path.join(args.outdir, "window_mc_fit_by_set.csv")
    fit_json = os.path.join(args.outdir, "window_mc_fit_by_set.json")
    profile_csv = os.path.join(args.outdir, "window_mc_profile_likelihood.csv")
    pp_csv = os.path.join(args.outdir, "window_mc_posterior_predictive.csv")
    comparison_csv = os.path.join(args.outdir, "window_mc_bbox_vs_polygon_comparison.csv")
    weighting_comparison_csv = os.path.join(args.outdir, "window_mc_center_weighting_comparison.csv")
    decomposition_csv = os.path.join(args.outdir, "window_mc_likelihood_decomposition.csv")
    class_weight_csv = os.path.join(args.outdir, "window_mc_class_weight_sensitivity.csv")
    predicted_survival_csv = os.path.join(args.outdir, "window_mc_predicted_survival_curve.csv")
    
    # 필수 결과와 (생성된 경우) 부가 진단 결과들을 CSV로 기록.
    write_csv(fit_rows, fit_csv)
    write_csv(profile_rows, profile_csv)
    write_csv(pp_rows, pp_csv)
    if comparison_rows:
        write_csv(comparison_rows, comparison_csv)
    if weighting_comp_rows:
        write_csv(weighting_comp_rows, weighting_comparison_csv)
    if decomposition_rows:
        write_csv(decomposition_rows, decomposition_csv)
    if class_weight_rows:
        write_csv(class_weight_rows, class_weight_csv)
    if survival_rows:
        write_csv(survival_rows, predicted_survival_csv)
        
    # 실행 재현용 입력요약(input_summary)과 적합 행들을 JSON으로 저장.
    with open(fit_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "input_summary": {
                    "trace_h5": args.trace_h5,
                    "trace_csv": args.trace_csv,
                    "target_set": args.target_set,
                    "dfn_model": args.dfn_model,
                    "generation_rmin": generation_rmin,
                    "generation_rmax": generation_rmax,
                    "estimation_rmin": float(args.rmin),
                    "estimation_rmax": float(args.rmax),
                    "likelihood_rmin": float(args.rmin),
                    "likelihood_rmax": float(args.rmax),
                    "set_rmin_mode": args.set_rmin_mode,
                    "lmin_fit_values": args.lmin_fit_values,
                    "p32_label": p32_label,
                    "rmin": args.rmin,
                    "rmax": args.rmax,
                    "kr_min": args.kr_min,
                    "kr_max": args.kr_max,
                    "profile_grid_size": args.profile_grid_size,
                    "mc_samples_per_grid": args.mc_samples_per_grid,
                    "length_bin_count": args.length_bin_count,
                    "length_bin_mode": args.length_bin_mode,
                    "window_mode": args.window_mode,
                    "center_weighting": args.center_weighting,
                    "likelihood_component": args.likelihood_component,
                    "class_likelihood_weight": args.class_likelihood_weight,
                    "main_class_likelihood_weight": main_class_weight,
                    "sensitivity_class_likelihood_weights": " ".join(str(v) for v in args.class_likelihood_weight),
                    "oracle_radius_mode": args.oracle_radius_mode,
                    "run_bootstrap": args.run_bootstrap,
                    "n_bootstrap": args.n_bootstrap,
                    "warning": WINDOW_WARNING_POLYGON if args.window_mode == "polygon" else WINDOW_WARNING_BBOX,
                },
                "fit_rows": fit_rows,
            },
            f,
            indent=2,
        )

    # 콘솔에 set별 요약(kr_hat, 참값, 부트스트랩 CI, 상태, MC 사용수, class_l1)을 출력.
    print("[*] Window MC fit summary")
    for row in fit_rows:
        kr_true_val = row.get("kr_true")
        kr_true_str = f", kr_true={kr_true_val:.2f}" if kr_true_val is not None and np.isfinite(float(kr_true_val)) else ""
        boot_str = ""
        if args.run_bootstrap:
            boot_str = f", boot_mean={row['kr_boot_mean']:.3f} CI=[{row['kr_ci_low']:.2f}, {row['kr_ci_high']:.2f}]"
        print(
            f"    - Set {row['set_id']}, lmin={row['lmin_fit']:.3f}: "
            f"kr={row['kr_window_mc_hat']:.3f}{kr_true_str}{boot_str}, "
            f"fit={row['fit_status']}, recovery={row['recovery_status']}, adoption={row['adoption_status']}, "
            f"mc_used={row['mc_accepted_count']}, class_l1={row['class_fraction_l1_error']:.3f}"
        )
    print(f"[*] Fit CSV written to: {fit_csv}")
    print(f"[*] Fit JSON written to: {fit_json}")
    print(f"[*] Profile CSV written to: {profile_csv}")
    print(f"[*] Posterior predictive CSV written to: {pp_csv}")
    if comparison_rows:
        print(f"[*] Bbox-vs-polygon comparison CSV written to: {comparison_csv}")
    if weighting_comp_rows:
        print(f"[*] Center weighting comparison CSV written to: {weighting_comparison_csv}")
    if decomposition_rows:
        print(f"[*] Likelihood decomposition CSV written to: {decomposition_csv}")
    if class_weight_rows:
        print(f"[*] Class-weight sensitivity CSV written to: {class_weight_csv}")


# 스크립트로 직접 실행될 때 main() 진입.
if __name__ == "__main__":
    main()
