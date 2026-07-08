# =====================================================================
# 이 파일의 역할:
#   관측된 트레이스 강도(P21)로부터 3D 균열 면적강도 P32를 추정하는 최종 추정기.
#   핵심은 unit_p32_forward_mc 보정 방식으로, "단위 P32(=P32 1)"를 가정한
#   순방향 몬테카를로(Monte Carlo) 시뮬레이션을 돌려 보정계수 C(=P21/P32)를 구한다.
#   그 뒤 P32_hat = 관측 P21 / C 로 P32를 역산한다.
#
# 주요 입력(주로 CLI 인자 및 HDF5/CSV):
#   - --trace-h5   : export_setwise_3d_traces.py 가 만든 트레이스 HDF5 (관측 트레이스, 터널 폴리곤 등)
#   - --dfn-h5     : 배향(orientation) 계수 추정용 DFN export HDF5
#   - --kr-summary-csv / --bootstrap-csv : 반경 멱함수 지수 kr 추정치와 신뢰구간(CI)
#   - --rough-mesh-h5 : 관측 면적(P21 분모) 계산용 러프 페이스 컬렉션 HDF5
#   - --site / --target-set : 대상 사이트(forsmark/laxemar) 및 세트 ID
#
# 주요 출력:
#   - --outcsv 로 지정된 요약 CSV. 세트별로 보정계수 C, P32_hat 및 CI,
#     참조값(P32_reference)과의 오차, 채택/상태 플래그(p32_status) 등을 기록.
#
# 핵심 처리 흐름(main):
#   1) kr 요약/부트스트랩 CSV, 트레이스/러프메시 HDF5, 배향계수 로드.
#   2) 세트별로 보정계수 C 추정:
#        - unit 모드: estimate_unit_p32_forward_mc (순방향 MC, 최종 방식)
#        - proxy 모드: estimate_calibration_factor (해석적 근사 스캐폴드)
#      kr의 CI 하/상한으로도 C를 각각 계산해 P32 CI 전파.
#   3) P32_hat = 관측 P21 / C 로 역산하고, 참조 P32와의 오차/상태 분류 후 CSV 기록.
# =====================================================================
import argparse
import csv
import h5py
import math
import os
import sys
from typing import Dict, List, Sequence, Tuple

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dfn_analysis.build_p32_pilot_summary import SITE_SET_CONFIG, support_scaled_p32
from dfn_analysis.estimate_radius_powerlaw_window_mc import (
    SITE_FISHER_PARAMS,
    clip_segments_to_bbox_vectorized,
    clip_segments_to_convex_polygon_vectorized,
    empirical_trace_directions_yz,
    load_trace_data_from_h5,
    mean_pole_from_trend_plunge,
    normals_to_trace_directions_yz,
    orientation_conditioned_trace_directions_yz,
    sample_fisher_normals,
    sample_size_biased_radius,
    sample_true_chords,
)
from dfn_analysis.export_setwise_3d_traces import load_hdf5_dfn
from dfn_analysis.summarize_setwise_trace_statistics import (
    build_summary_rows,
    compute_total_observation_area,
    load_rough_face_collection_from_h5,
    load_trace_rows_from_h5,
)


# 사이트별 기본 DFN export HDF5 경로 (--dfn-h5 미지정 시 사용).
DEFAULT_DFN_H5 = {
    "forsmark": "storage/output/dfn_forsmark_rmin0p5/dfn_export_for_python.h5",
    "laxemar": "storage/output/dfn_laxemar_rmin0p5/dfn_export_for_python.h5",
}
# 보정계수 산출 모드 식별자.
#  - PROXY: 해석적 근사 기반 스캐폴드(빠르지만 근사) 모드.
#  - UNIT : 단위 P32 순방향 MC 기반 최종 보정 모드(권장/최종).
CALIBRATION_FACTOR_MODE_PROXY = "conditional_visible_trace_proxy"
CALIBRATION_FACTOR_MODE_UNIT = "unit_p32_forward_mc"


# CSV 파일을 읽어 각 행을 dict로 담은 리스트로 반환.
#  - 인자 path: 읽을 CSV 경로. 반환: dict 리스트.
def read_csv(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


# dict 리스트를 CSV로 기록. 첫 행의 key를 헤더로 사용.
#  - 인자 rows: 기록할 행들, path: 출력 경로. 반환: 없음(빈 rows면 예외).
def write_csv(rows: Sequence[dict], path: str) -> None:
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


# dict의 특정 key 값을 float로 안전 변환. 변환 실패 시 NaN 반환.
#  - 인자 row: 행 dict, key: 조회 키. 반환: float 또는 NaN.
def to_float(row: dict, key: str) -> float:
    value = row.get(key, "")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


# 절단(truncated) 멱함수 반경 분포의 order차 모멘트 E[r^order]를 해석적으로 계산.
#  - 인자 kr: 멱함수 지수, rmin/rmax: 반경 절단 범위, order: 모멘트 차수.
#  - 반환: order차 모멘트 값.
def powerlaw_moment(kr: float, rmin: float, rmax: float, order: float) -> float:
    # 정규화 상수 계산. 지수가 0에 가까우면 로그 형태로 극한 처리.
    exponent = -kr
    if abs(exponent) < 1e-12:
        norm = 1.0 / math.log(rmax / rmin)
    else:
        norm = exponent / (rmax**exponent - rmin**exponent)

    # 모멘트 적분항 계산. 지수(order-kr)가 0에 가까우면 로그 형태로 극한 처리.
    moment_power = order - kr
    if abs(moment_power) < 1e-12:
        integral = math.log(rmax / rmin)
    else:
        integral = (rmax**moment_power - rmin**moment_power) / moment_power
    return norm * integral


# 지수분포(비정규화) 원시 모멘트 적분의 부정적분 값을 반환(1차/2차만 지원).
#  - 절단 지수분포 모멘트 계산(exponential_moment)에서 상/하한 대입용 헬퍼.
#  - 인자 rate: 지수분포율(1/r0), radius: 대입 반경, order: 차수(1 또는 2).
def exponential_raw_moment(rate: float, radius: float, order: int) -> float:
    if order == 1:
        return -math.exp(-rate * radius) * (radius / rate + 1.0 / (rate * rate))
    if order == 2:
        return -math.exp(-rate * radius) * (
            radius * radius / rate + 2.0 * radius / (rate * rate) + 2.0 / (rate**3)
        )
    raise ValueError(f"Unsupported exponential moment order: {order}")


# [rmin, rmax]로 절단된 지수분포 반경의 order차 모멘트를 해석적으로 계산.
#  - 인자 r0: 평균 반경 스케일, rmin/rmax: 절단 범위, order: 차수(1 또는 2).
#  - 반환: 절단 지수분포의 order차 모멘트.
def exponential_moment(r0: float, rmin: float, rmax: float, order: int) -> float:
    # 절단 구간의 정규화 상수(z)와 상/하한 원시 모멘트 차이로 모멘트 계산.
    rate = 1.0 / r0
    z = math.exp(-rate * rmin) - math.exp(-rate * rmax)
    integral = exponential_raw_moment(rate, rmax, order) - exponential_raw_moment(rate, rmin, order)
    return (rate / z) * integral


# 사이트/세트의 반경 분포 유형에 따라 1차·2차 모멘트(E[r], E[r^2])를 반환.
#  - 인자 site/set_id: 분포 설정 조회 키, kr/rmin/rmax: 분포 파라미터.
#  - 반환: (평균 반경, 평균 반경 제곱). 설정 없거나 미지원 분포면 (NaN, NaN).
def radius_moments(site: str, set_id: int, kr: float, rmin: float, rmax: float) -> Tuple[float, float]:
    # 사이트/세트별 분포 설정을 조회하고 분포 유형에 맞는 모멘트 함수 호출.
    cfg = SITE_SET_CONFIG.get(site, {}).get(set_id)
    if cfg is None:
        return float("nan"), float("nan")
    dist_type = str(cfg["dist_type"])
    if dist_type == "powerlaw":
        return powerlaw_moment(kr, rmin, rmax, 1.0), powerlaw_moment(kr, rmin, rmax, 2.0)
    if dist_type == "exponential":
        r0 = float(cfg["r0"])
        return exponential_moment(r0, rmin, rmax, 1), exponential_moment(r0, rmin, rmax, 2)
    return float("nan"), float("nan")


# 모집단(생성) 반경 분포에서 size개의 반경을 역변환 샘플링(inverse-CDF)으로 추출.
#  - 크기 편향 없는 "실제 모집단" 반경 표본으로, unit_p32 순방향 MC에서 사용.
#  - 인자 site/set_id: 분포 설정, kr/rmin/rmax: 파라미터, size: 표본수, rng: 난수 생성기.
#  - 반환: 반경 배열(shape=(size,)).
def sample_population_radius(site: str, set_id: int, kr: float, rmin: float, rmax: float, size: int, rng: np.random.Generator) -> np.ndarray:
    # 분포 설정 조회 후 [0,1) 균등난수 u로부터 역CDF 변환.
    cfg = SITE_SET_CONFIG.get(site, {}).get(set_id)
    if cfg is None:
        raise ValueError(f"Missing SITE_SET_CONFIG for site={site}, set_id={set_id}")
    dist_type = str(cfg["dist_type"])
    u = rng.uniform(0.0, 1.0, size=size)
    # 멱함수 분포: 지수 alpha에 따라 로그(극한) 또는 일반 역변환식 적용.
    if dist_type == "powerlaw":
        alpha = kr + 1.0
        if abs(alpha - 1.0) < 1e-12:
            return rmin * np.exp(u * np.log(rmax / rmin))
        return (rmin ** (1.0 - alpha) + u * (rmax ** (1.0 - alpha) - rmin ** (1.0 - alpha))) ** (1.0 / (1.0 - alpha))
    # 지수분포: 절단 구간 [rmin, rmax]에 대한 역CDF 변환.
    if dist_type == "exponential":
        lam = 1.0 / float(cfg["r0"])
        a = math.exp(-lam * rmin)
        b = math.exp(-lam * rmax)
        return -(1.0 / lam) * np.log(a - u * (a - b))
    raise ValueError(f"Unsupported distribution type for unit_p32_forward_mc: {dist_type}")


# 무작위 평면 절단 시 기대 교차 현(chord) 길이 E[chord] = (pi/2)*E[r^2]/E[r] 계산.
#  - 인자 site/set_id/kr/rmin/rmax: 반경 분포 파라미터. 반환: 평균 현 길이(무효 시 NaN).
def mean_intersection_chord_length(site: str, set_id: int, kr: float, rmin: float, rmax: float) -> float:
    mean_r, mean_r2 = radius_moments(site, set_id, kr, rmin, rmax)
    if not np.isfinite(mean_r) or not np.isfinite(mean_r2) or mean_r <= 0.0:
        return float("nan")
    return (math.pi / 2.0) * (mean_r2 / mean_r)


# 관측 P21 요약과 총 관측 면적을 로드/계산.
#  - 트레이스 행과 러프 페이스 컬렉션을 읽어 세트별 P21 요약 행을 구성.
#  - 인자 trace_h5: 트레이스 HDF5, rough_mesh_h5: 관측면 메시 HDF5.
#  - 반환: (set_id -> 요약행 dict, 총 관측 면적 m^2).
def load_p21_summary(trace_h5: str, rough_mesh_h5: str) -> Tuple[Dict[int, dict], float]:
    rows = load_trace_rows_from_h5(trace_h5)
    rough_faces = load_rough_face_collection_from_h5(rough_mesh_h5)
    observation_area_m2 = compute_total_observation_area(rough_faces)
    summary_rows = build_summary_rows(rows, observation_area_m2)
    return {int(row["set_id"]): row for row in summary_rows}, float(observation_area_m2)


# 세트별 배향(orientation) 계수 g = 평균 sin(theta) 를 DFN 법선으로부터 계산.
#  - g는 균열 법선의 x성분(터널 진행축)으로부터 면 교차 강도 보정에 쓰이는 기하 인자.
#  - 인자 dfn_h5: DFN export HDF5 경로. 반환: set_id -> 평균 g 계수(dict).
def load_orientation_factors(dfn_h5: str) -> Dict[int, float]:
    # DFN에서 세트 ID와 법선 벡터를 로드.
    data = load_hdf5_dfn(dfn_h5)
    out: Dict[int, float] = {}
    set_ids = np.asarray(data["set_ids"]).astype(np.int32).ravel()
    normals = np.asarray(data["normals"], dtype=np.float64)
    # 각 세트별로 g = sqrt(1 - nx^2) = sin(theta)의 평균을 계산.
    for set_id in sorted({int(v) for v in set_ids.tolist()}):
        mask = set_ids == set_id
        set_normals = normals[mask]
        if len(set_normals) == 0:
            out[set_id] = float("nan")
            continue
        g_factors = np.sqrt(np.clip(1.0 - set_normals[:, 0] ** 2, 0.0, 1.0))
        out[set_id] = float(np.mean(g_factors))
    return out


# 관측 면(excavation face)들의 x위치(터널 진행축 좌표) 배열을 트레이스 HDF5에서 로드.
#  - /meta/face_x_positions_m 우선, 없으면 /traces/face_x_m 의 고유값 사용.
#  - 인자 trace_h5: 트레이스 HDF5 경로. 반환: face x위치 배열(없으면 예외).
def load_face_x_positions(trace_h5: str) -> np.ndarray:
    with h5py.File(trace_h5, "r") as f:
        if "meta" in f and "face_x_positions_m" in f["meta"]:
            return np.asarray(f["meta"]["face_x_positions_m"][:], dtype=np.float64).ravel()
        if "traces" in f and "face_x_m" in f["traces"]:
            return np.unique(np.asarray(f["traces"]["face_x_m"][:], dtype=np.float64).ravel())
    raise ValueError(f"Could not find face_x_positions_m in trace HDF5: {trace_h5}")


# [PROXY 모드 보조] 주어진 반경 표본에 대해 관측창 내 "가시 트레이스 길이"를 MC로 시뮬레이션.
#  - 반경별 실제 현 길이를 뽑고, 무작위 방향/중심으로 배치한 뒤 관측창(bbox/polygon)으로 클리핑.
#  - 인자: polygon_yz(관측 폴리곤, YZ 로컬좌표), directions_yz(방향 풀), radii(반경),
#          rng, window_mode(bbox/polygon), direction_mode, set_id, site.
#  - 반환: (가시 길이 배열, 제안 면적 배열). 제안 면적은 중요도표본 가중치 정규화용.
def simulate_visible_lengths_all(
    polygon_yz: np.ndarray,
    directions_yz: np.ndarray,
    radii: np.ndarray,
    rng: np.random.Generator,
    window_mode: str,
    direction_mode: str,
    set_id: int,
    site: str,
) -> Tuple[np.ndarray, np.ndarray]:
    # 관측 폴리곤의 경계상자(bbox)와 폭/높이 계산, 반경별 실제 현 길이 샘플링.
    bbox_min = np.min(polygon_yz, axis=0)
    bbox_max = np.max(polygon_yz, axis=0)
    bbox_w = bbox_max[0] - bbox_min[0]
    bbox_h = bbox_max[1] - bbox_min[1]
    true_lengths = sample_true_chords(radii, rng)
    n_samples = len(radii)

    # 트레이스 방향 표본 추출: 배향조건부 풀에서 뽑거나(empirical) 경험적 방향 배열에서 복원추출.
    if direction_mode == "orientation_conditioned":
        dir_pool = orientation_conditioned_trace_directions_yz(set_id, site, n_samples * 3, rng)
        direction_idx = rng.integers(0, len(dir_pool), size=n_samples)
        directions = dir_pool[direction_idx]
    else:
        direction_idx = rng.integers(0, len(directions_yz), size=n_samples)
        directions = directions_yz[direction_idx]

    # 제안 면적(경계상자를 현 길이만큼 확장한 영역)과 무작위 중심 위치 생성.
    proposal_areas = (bbox_w + true_lengths * np.abs(directions[:, 0])) * (bbox_h + true_lengths * np.abs(directions[:, 1]))
    expand = 0.5 * true_lengths[:, None]
    centers = rng.uniform(bbox_min - expand, bbox_max + expand)

    # 배치된 선분을 관측창(bbox 또는 볼록 폴리곤)으로 클리핑하여 가시 길이 산출.
    if window_mode == "bbox":
        visible_lengths, _ = clip_segments_to_bbox_vectorized(centers, directions, true_lengths, bbox_min, bbox_max)
    elif window_mode == "polygon":
        visible_lengths, _ = clip_segments_to_convex_polygon_vectorized(centers, directions, true_lengths, polygon_yz)
    else:
        raise ValueError(f"Unsupported window_mode: {window_mode}")

    return visible_lengths, proposal_areas


# [PROXY 모드] 해석적 근사 기반 보정계수 C(=P21/P32)를 추정하는 스캐폴드.
#  - 교차강도(해석식)와 MC로 얻은 가시길이 기대값을 곱해 C를 산출(빠른 근사).
#  - 인자: site/set_id/kr/rmin/rmax(분포), polygon_yz(관측창), directions_yz(방향 풀),
#          orientation_factor(g), mc_samples, rng_seed, window_mode, direction_mode.
#  - 반환: 보정계수 C(무효 입력 시 NaN).
def estimate_calibration_factor(
    site: str,
    set_id: int,
    kr: float,
    rmin: float,
    rmax: float,
    polygon_yz: np.ndarray,
    directions_yz: np.ndarray,
    orientation_factor: float,
    mc_samples: int,
    rng_seed: int,
    window_mode: str,
    direction_mode: str,
) -> float:
    # 크기 편향(size-biased) 반경을 샘플링하고 가시 길이/제안 면적을 MC로 산출.
    rng = np.random.default_rng(rng_seed)
    radii = sample_size_biased_radius(kr, rmin, rmax, mc_samples, rng)
    visible_lengths, proposal_areas = simulate_visible_lengths_all(
        polygon_yz=polygon_yz,
        directions_yz=directions_yz,
        radii=radii,
        rng=rng,
        window_mode=window_mode,
        direction_mode=direction_mode,
        set_id=set_id,
        site=site,
    )

    # 관측 폴리곤 면적을 신발끈(shoelace) 공식으로 계산(0 이하면 무효).
    observation_area = abs(0.5 * float(np.dot(polygon_yz[:, 0], np.roll(polygon_yz[:, 1], -1)) - np.dot(polygon_yz[:, 1], np.roll(polygon_yz[:, 0], -1))))
    if observation_area <= 0.0:
        return float("nan")

    # 반경 모멘트와 배향계수 유효성 확인.
    mean_r, mean_r2 = radius_moments(site, set_id, kr, rmin, rmax)
    if not np.isfinite(mean_r) or not np.isfinite(mean_r2) or mean_r2 <= 0.0:
        return float("nan")
    if not np.isfinite(orientation_factor) or orientation_factor <= 0.0:
        return float("nan")

    # 단위 면적당 교차 강도(해석식)와 교차당 평균 가시길이(MC)를 곱해 보정계수 C 산출.
    intersection_intensity_per_area = (2.0 * orientation_factor * mean_r) / (math.pi * mean_r2)
    visible_length_per_intersection = float(np.mean((proposal_areas * visible_lengths) / observation_area))
    return intersection_intensity_per_area * visible_length_per_intersection


# 기본 보정계수 모드 식별자를 반환하는 헬퍼(현재는 PROXY 반환).
def infer_calibration_factor_mode() -> str:
    return CALIBRATION_FACTOR_MODE_PROXY


# [UNIT 모드 - 최종 방식] 단위 P32(=1)를 가정한 순방향 MC로 보정계수 C(=P21/P32)를 추정.
#  - 절차: 모집단 반경 + Fisher 배향 법선을 샘플링해 각 관측면 x위치에서 균열을 무작위 배치하고,
#          평면-면 교차 현을 계산해 관측창으로 클리핑, 가시 트레이스 길이를 중요도표본 가중합.
#          이를 총 관측 면적으로 나누면 "단위 P32당 P21"=보정계수 C가 된다.
#  - mc_replicates회 반복하여 C의 평균/표준편차/95% CI를 산출.
#  - 인자: site/set_id/kr/rmin/rmax(분포), polygon_yz(관측창), face_x_positions(관측면 x위치),
#          total_observation_area, mc_samples(면당 표본수), mc_replicates(반복수), rng_seed, window_mode.
#  - 반환: 보정계수 통계와 평균 균열면적/수밀도 등을 담은 dict.
def estimate_unit_p32_forward_mc(
    site: str,
    set_id: int,
    kr: float,
    rmin: float,
    rmax: float,
    polygon_yz: np.ndarray,
    face_x_positions: np.ndarray,
    total_observation_area: float,
    mc_samples: int,
    mc_replicates: int,
    rng_seed: int,
    window_mode: str,
) -> dict:
    # 관측 면적이 0 이하이면 계산 불가 -> 모든 결과를 NaN으로 반환.
    if total_observation_area <= 0.0:
        return {
            "calibration_factor_C": float("nan"),
            "calibration_factor_std": float("nan"),
            "calibration_factor_ci_low": float("nan"),
            "calibration_factor_ci_high": float("nan"),
            "unit_p32_mc_volume": float("nan"),
            "mean_fracture_area": float("nan"),
            "fracture_number_density_for_unit_p32": float("nan"),
        }

    # 세트의 Fisher 배향 파라미터(경향/경사/집중도 kappa)를 조회하고 평균 극(법선) 벡터 계산.
    params = SITE_FISHER_PARAMS.get(site, {}).get(set_id)
    if params is None:
        raise ValueError(f"Missing Fisher parameters for unit_p32_forward_mc: site={site}, set_id={set_id}")
    trend, plunge, kappa = params
    mean_pole = mean_pole_from_trend_plunge(trend, plunge)

    # 관측 폴리곤의 경계상자와 폭/높이 계산.
    bbox_min = np.min(polygon_yz, axis=0)
    bbox_max = np.max(polygon_yz, axis=0)
    bbox_w = float(bbox_max[0] - bbox_min[0])
    bbox_h = float(bbox_max[1] - bbox_min[1])

    # 평균 균열 면적(pi*E[r^2])과 P32=1을 만족시키는 균열 수밀도(1/평균면적) 계산.
    #  - P32 = 수밀도 * 평균면적 = 1 이 되도록 수밀도를 정의(단위 P32 가정의 핵심).
    _, mean_r2 = radius_moments(site, set_id, kr, rmin, rmax)
    mean_fracture_area = math.pi * mean_r2 if np.isfinite(mean_r2) else float("nan")
    fracture_number_density = 1.0 / mean_fracture_area if np.isfinite(mean_fracture_area) and mean_fracture_area > 0.0 else float("nan")

    # 반복(replicate)별 C 추정치를 모을 리스트와 제안 부피 누적자 초기화.
    replicate_values: List[float] = []
    mean_proposal_volume_accum = 0.0
    face_x_positions = np.asarray(face_x_positions, dtype=np.float64)

    # MC 반복 루프: 각 반복마다 독립 난수로 전체 관측면에 대한 가중 가시길이를 누적.
    for rep in range(mc_replicates):
        rng = np.random.default_rng(rng_seed + rep)
        total_weighted_length = 0.0
        proposal_volume_sum = 0.0

        # 각 관측면(x위치)마다 균열을 샘플링/배치하여 면 교차 가시길이를 계산.
        for face_x in face_x_positions:
            # 모집단 반경과 Fisher 법선을 샘플링하고, 법선→트레이스 방향(YZ) 변환(유효한 것만 유지).
            radii = sample_population_radius(site, set_id, kr, rmin, rmax, mc_samples, rng)
            normals = sample_fisher_normals(mean_pole, kappa, mc_samples, rng)
            directions_yz, valid = normals_to_trace_directions_yz(normals)
            if not np.any(valid):
                continue

            radii = radii[valid]
            normals = normals[valid]
            directions_yz = directions_yz[valid]

            # sin(theta)=면과의 경사 성분. 0에 가까운(면과 평행) 균열은 교차 불가로 제외.
            sin_theta = np.sqrt(np.clip(1.0 - normals[:, 0] ** 2, 0.0, 1.0))
            valid_theta = sin_theta > 1e-8
            if not np.any(valid_theta):
                continue

            radii = radii[valid_theta]
            normals = normals[valid_theta]
            directions_yz = directions_yz[valid_theta]
            sin_theta = sin_theta[valid_theta]

            # 균열 중심이 면과 교차 가능한 x-반폭(x_half)과 제안 부피(중요도표본 영역) 계산.
            x_half = radii * sin_theta
            proposal_w = bbox_w + 4.0 * radii
            proposal_h = bbox_h + 4.0 * radii
            proposal_volume = 2.0 * x_half * proposal_w * proposal_h
            proposal_volume_sum += float(np.mean(proposal_volume))

            # 제안 부피 내에서 균열 중심 (x, y, z)을 무작위 배치.
            center_x = rng.uniform(face_x - x_half, face_x + x_half)
            center_y = rng.uniform(bbox_min[0] - 2.0 * radii, bbox_max[0] + 2.0 * radii)
            center_z = rng.uniform(bbox_min[1] - 2.0 * radii, bbox_max[1] + 2.0 * radii)
            centers_yz = np.column_stack([center_y, center_z])

            # 면(x=face_x)까지의 축방향 거리로부터 디스크 중심-교차선 오프셋을 구하고,
            # 오프셋이 반경 이내인(=면과 실제로 교차하는) 균열만 채택.
            dx = face_x - center_x
            chord_offsets = np.abs(dx) / sin_theta
            valid_intersections = chord_offsets <= radii + 1e-10
            if not np.any(valid_intersections):
                continue

            normals = normals[valid_intersections]
            directions_yz = directions_yz[valid_intersections]
            radii = radii[valid_intersections]
            sin_theta = sin_theta[valid_intersections]
            chord_offsets = chord_offsets[valid_intersections]
            centers_yz = centers_yz[valid_intersections]
            proposal_volume = proposal_volume[valid_intersections]
            dx = dx[valid_intersections]

            # 면 위에 생기는 교차 현(chord)의 실제 길이와, 현 중점의 면-로컬(YZ) 좌표 계산.
            chord_lengths = 2.0 * np.sqrt(np.maximum(radii * radii - chord_offsets * chord_offsets, 0.0))
            t = dx / np.maximum(sin_theta * sin_theta, 1e-12)
            line_midpoints_yz = centers_yz + np.column_stack(
                [-t * normals[:, 0] * normals[:, 1], -t * normals[:, 0] * normals[:, 2]]
            )

            # 교차 현을 관측창(bbox 또는 폴리곤)으로 클리핑하여 가시 길이/분류를 산출.
            if window_mode == "bbox":
                visible_lengths, classes = clip_segments_to_bbox_vectorized(
                    line_midpoints_yz,
                    directions_yz,
                    chord_lengths,
                    bbox_min,
                    bbox_max,
                )
            elif window_mode == "polygon":
                visible_lengths, classes = clip_segments_to_convex_polygon_vectorized(
                    line_midpoints_yz,
                    directions_yz,
                    chord_lengths,
                    polygon_yz,
                )
            else:
                raise ValueError(f"Unsupported window mode: {window_mode}")

            # 관측창 내부에 실제로 보이는(가시길이>0) 교차만 채택.
            accepted = (classes >= 0) & (visible_lengths > 0.0)
            if not np.any(accepted):
                continue

            # 중요도표본 가중치(수밀도*제안부피/표본수)로 가시 길이를 가중 합산.
            weights = fracture_number_density * proposal_volume[accepted] / mc_samples
            total_weighted_length += float(np.sum(weights * visible_lengths[accepted]))

        # 이번 반복의 C = (총 가중 가시길이)/(총 관측면적) = 단위 P32당 기대 P21.
        replicate_values.append(total_weighted_length / total_observation_area)
        mean_proposal_volume_accum += proposal_volume_sum / max(len(face_x_positions), 1)

    # 반복별 C 추정치들로 평균/표준편차/95% CI 및 부가 통계를 집계해 반환.
    values = np.asarray(replicate_values, dtype=np.float64)
    return {
        "calibration_factor_C": float(np.mean(values)) if len(values) else float("nan"),
        "calibration_factor_std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        "calibration_factor_ci_low": float(np.percentile(values, 2.5)) if len(values) else float("nan"),
        "calibration_factor_ci_high": float(np.percentile(values, 97.5)) if len(values) else float("nan"),
        "unit_p32_mc_volume": float(mean_proposal_volume_accum / max(mc_replicates, 1)),
        "mean_fracture_area": mean_fracture_area,
        "fracture_number_density_for_unit_p32": fracture_number_density,
    }


# P32 추정 결과의 신뢰/채택 상태 문자열을 규칙에 따라 분류.
#  - 보정 모드(UNIT 여부), kr 채택 상태, 회복 CI 상태, kr CI 상대폭 등을 종합.
#  - 인자: adoption_status/recovery_ci_status(kr 채택·회복 상태), kr_used/kr_ci_low/kr_ci_high,
#          calibration_factor_mode. 반환: 상태 라벨 문자열(예: p32_mc_pilot_candidate 등).
def classify_p32_status(
    adoption_status: str,
    recovery_ci_status: str,
    kr_used: float,
    kr_ci_low: float,
    kr_ci_high: float,
    calibration_factor_mode: str,
) -> str:
    # UNIT(최종) 모드가 아니면 스캐폴드 전용 상태로 처리.
    if calibration_factor_mode != CALIBRATION_FACTOR_MODE_UNIT:
        return "p32_scaffold_only"
    # kr이 기각되면 보류(hold). 잠정채택 + 계통편향이면 별도 라벨.
    if adoption_status == "rejected":
        return "p32_hold"
    if adoption_status == "provisional_accepted" and recovery_ci_status == "systematic_bias":
        return "p32_mc_provisional_systematic_bias"
    # kr 채택 시: kr CI 상대폭이 크면 불확실성 라벨, 아니면 파일럿 후보 라벨.
    if adoption_status == "accepted":
        if np.isfinite(kr_used) and np.isfinite(kr_ci_low) and np.isfinite(kr_ci_high):
            rel_width = (kr_ci_high - kr_ci_low) / max(abs(kr_used), 1e-12)
            if rel_width > 1.0:
                return "p32_mc_candidate_with_uncertainty"
        return "p32_mc_pilot_candidate"
    # 잠정 채택은 잠정 라벨, 그 외 모든 경우는 보류.
    if adoption_status == "provisional_accepted":
        return "p32_mc_provisional"
    return "p32_hold"


# 출력 CSV의 notes 필드 문자열을 생성(보정 모드/방법/계수/P32 산출법 등 기록).
#  - 인자: base_notes(기존 메모), calibration_factor(C), p32_hat, calibration_mode.
#  - 반환: 세미콜론으로 연결된 메모 문자열.
def build_notes(base_notes: str, calibration_factor: float, p32_hat: float, calibration_mode: str) -> str:
    notes = [base_notes] if base_notes else []
    notes.append(f"calibration_factor_mode={calibration_mode}")
    if np.isfinite(calibration_factor):
        notes.append(f"calibration_method={calibration_mode}")
        notes.append(f"calibration_factor_C={calibration_factor:.6f}")
    if np.isfinite(p32_hat):
        notes.append("p32_method=observed_P21_over_calibration_factor")
    notes.append("p32_ci_method=kr_only_calibration_propagation")
    return "; ".join(note for note in notes if note)


# CLI 진입점: 인자 파싱 -> 입력 로드 -> 세트별 보정계수 C 및 P32 추정 -> 결과 CSV 기록.
#  - 반환값 없음. 최종 결과는 --outcsv 경로에 저장.
def main() -> None:
    # ---- CLI 인자 정의 (입력 경로/사이트/세트/보정 모드/MC 설정/출력 경로 등) ----
    parser = argparse.ArgumentParser(description="Estimate MC-calibrated P32 from observed trace intensity under the effective-rmin window MC model.")
    parser.add_argument("--trace-h5", required=True, help="Trace HDF5 created by export_setwise_3d_traces.py")
    parser.add_argument("--dfn-h5", help="DFN export HDF5 for orientation factor estimation.")
    parser.add_argument(
        "--kr-summary-csv",
        default="storage/output/final_kr_recovery_summary_effective_rmin.csv",
        help="Final effective-rmin kr recovery summary CSV.",
    )
    parser.add_argument(
        "--bootstrap-csv",
        default="storage/output/final_kr_bootstrap_effective_rmin/final_kr_bootstrap_summary_effective_rmin.csv",
        help="Bootstrap summary CSV.",
    )
    parser.add_argument("--site", choices=["forsmark", "laxemar"], required=True)
    parser.add_argument("--target-set", nargs="+", type=int, required=True)
    parser.add_argument("--p32-label", default="P32_r_ge_0p5m")
    parser.add_argument("--set-rmin-mode", default="effective_generation")
    parser.add_argument(
        "--rough-mesh-h5",
        default="storage/output/rough_face_mesh_collection/synthetic_rough_face_collection.h5",
        help="Rough face collection HDF5 used to compute observed P21.",
    )
    parser.add_argument("--window-mode", choices=["polygon", "bbox"], default="polygon")
    parser.add_argument("--direction-mode", choices=["empirical_trace", "orientation_conditioned"], default="empirical_trace")
    parser.add_argument("--mc-samples", type=int, default=50000)
    parser.add_argument(
        "--calibration-factor-mode",
        choices=[CALIBRATION_FACTOR_MODE_PROXY, CALIBRATION_FACTOR_MODE_UNIT],
        default=CALIBRATION_FACTOR_MODE_PROXY,
        help="Proxy scaffold mode or unit-P32 forward MC calibration mode.",
    )
    parser.add_argument("--unit-p32-mc-replicates", type=int, default=32, help="Number of MC replicates for unit_p32_forward_mc mode.")
    parser.add_argument(
        "--outcsv",
        default="storage/output/p32_mc_calibrated_effective_rmin/p32_mc_calibrated_summary.csv",
        help="Output CSV path.",
    )
    args = parser.parse_args()

    # ---- kr 요약/부트스트랩 CSV 로드 및 대상 사이트·세트로 필터링 ----
    dfn_h5 = args.dfn_h5 or DEFAULT_DFN_H5[args.site]
    kr_rows = [row for row in read_csv(args.kr_summary_csv) if str(row["site"]) == args.site and int(row["set_id"]) in set(args.target_set)]
    bootstrap_map = {
        (str(row["site"]), int(row["set_id"])): row
        for row in read_csv(args.bootstrap_csv)
        if str(row["site"]) == args.site and int(row["set_id"]) in set(args.target_set)
    }
    if not kr_rows:
        raise ValueError("No matching rows found in kr summary CSV for the requested site/sets.")

    # ---- 트레이스 HDF5에서 트레이스 행과 터널 관측 폴리곤(YZ) 로드, 세트별로 그룹화 ----
    trace_rows, polygon_yz = load_trace_data_from_h5(args.trace_h5)
    if polygon_yz is None:
        raise ValueError("trace_h5 must contain /meta/tunnel_poly_yz for MC-calibrated P32 estimation.")
    grouped_rows: Dict[int, List[dict]] = {}
    for row in trace_rows:
        set_id = int(row["set_id"])
        if set_id in set(args.target_set):
            grouped_rows.setdefault(set_id, []).append(row)

    # ---- 관측 P21 요약/관측 면적, 배향계수, 관측면 x위치 로드 ----
    p21_summary_map, total_observation_area = load_p21_summary(args.trace_h5, args.rough_mesh_h5)
    orientation_map = load_orientation_factors(dfn_h5)
    face_x_positions = load_face_x_positions(args.trace_h5)

    # ---- 세트별 메인 루프: 보정계수 C 추정 -> P32 역산 -> 결과 행 구성 ----
    out_rows: List[dict] = []
    for idx, row in enumerate(sorted(kr_rows, key=lambda r: int(r["set_id"]))):
        # 이 세트에 필요한 관측값/파라미터(관측 P21, 배향계수, rmin, kr와 CI 등) 수집.
        set_id = int(row["set_id"])
        boot_row = bootstrap_map.get((args.site, set_id), {})
        observed_p21 = to_float(p21_summary_map.get(set_id, {}), "P21_observed")
        orientation_factor = float(orientation_map.get(set_id, float("nan")))
        set_likelihood_rmin = to_float(row, "set_likelihood_rmin")
        set_effective_rmin = to_float(row, "set_effective_generation_rmin")
        kr_used = to_float(row, "kr_hat")
        kr_ci_low = to_float(boot_row, "kr_ci_low")
        kr_ci_high = to_float(boot_row, "kr_ci_high")
        kr_true = to_float(boot_row, "kr_true")
        recovery_ci_status = str(boot_row.get("recovery_ci_status", ""))
        adoption_status = str(row.get("adoption_status", ""))
        directions_yz = empirical_trace_directions_yz(grouped_rows.get(set_id, []))
        calibration_mode = args.calibration_factor_mode

        # 재현 가능한 난수 시드 구성(세트/인덱스별로 분리).
        base_seed = 260000 + set_id * 100 + idx * 10000
        # ---- UNIT(최종) 모드: 순방향 MC로 C를 추정하고, kr CI 하/상한으로 C의 CI도 계산 ----
        if calibration_mode == CALIBRATION_FACTOR_MODE_UNIT:
            unit_main = estimate_unit_p32_forward_mc(
                site=args.site,
                set_id=set_id,
                kr=kr_used,
                rmin=set_likelihood_rmin,
                rmax=250.0,
                polygon_yz=polygon_yz,
                face_x_positions=face_x_positions,
                total_observation_area=total_observation_area,
                mc_samples=args.mc_samples,
                mc_replicates=args.unit_p32_mc_replicates,
                rng_seed=base_seed,
                window_mode=args.window_mode,
            )
            # 중심 kr에 대한 C 및 부가 통계 추출.
            c_hat = float(unit_main["calibration_factor_C"])
            c_std = float(unit_main["calibration_factor_std"])
            c_ci_low = float(unit_main["calibration_factor_ci_low"])
            c_ci_high = float(unit_main["calibration_factor_ci_high"])
            unit_volume = float(unit_main["unit_p32_mc_volume"])
            mean_fracture_area = float(unit_main["mean_fracture_area"])
            fracture_number_density = float(unit_main["fracture_number_density_for_unit_p32"])
            # kr CI 하한으로 C_low 계산(kr CI가 유한할 때만).
            c_low = estimate_unit_p32_forward_mc(
                site=args.site,
                set_id=set_id,
                kr=kr_ci_low,
                rmin=set_likelihood_rmin,
                rmax=250.0,
                polygon_yz=polygon_yz,
                face_x_positions=face_x_positions,
                total_observation_area=total_observation_area,
                mc_samples=args.mc_samples,
                mc_replicates=args.unit_p32_mc_replicates,
                rng_seed=base_seed + 1,
                window_mode=args.window_mode,
            )["calibration_factor_C"] if np.isfinite(kr_ci_low) else float("nan")
            # kr CI 상한으로 C_high 계산(kr CI가 유한할 때만).
            c_high = estimate_unit_p32_forward_mc(
                site=args.site,
                set_id=set_id,
                kr=kr_ci_high,
                rmin=set_likelihood_rmin,
                rmax=250.0,
                polygon_yz=polygon_yz,
                face_x_positions=face_x_positions,
                total_observation_area=total_observation_area,
                mc_samples=args.mc_samples,
                mc_replicates=args.unit_p32_mc_replicates,
                rng_seed=base_seed + 2,
                window_mode=args.window_mode,
            )["calibration_factor_C"] if np.isfinite(kr_ci_high) else float("nan")
        # ---- PROXY 모드: 해석적 근사 스캐폴드로 C(중심/하한/상한)를 추정 ----
        else:
            c_hat = estimate_calibration_factor(
                site=args.site,
                set_id=set_id,
                kr=kr_used,
                rmin=set_likelihood_rmin,
                rmax=250.0,
                polygon_yz=polygon_yz,
                directions_yz=directions_yz,
                orientation_factor=orientation_factor,
                mc_samples=args.mc_samples,
                rng_seed=base_seed,
                window_mode=args.window_mode,
                direction_mode=args.direction_mode,
            )
            c_low = estimate_calibration_factor(
                site=args.site,
                set_id=set_id,
                kr=kr_ci_low,
                rmin=set_likelihood_rmin,
                rmax=250.0,
                polygon_yz=polygon_yz,
                directions_yz=directions_yz,
                orientation_factor=orientation_factor,
                mc_samples=args.mc_samples,
                rng_seed=base_seed + 1,
                window_mode=args.window_mode,
                direction_mode=args.direction_mode,
            ) if np.isfinite(kr_ci_low) else float("nan")
            c_high = estimate_calibration_factor(
                site=args.site,
                set_id=set_id,
                kr=kr_ci_high,
                rmin=set_likelihood_rmin,
                rmax=250.0,
                polygon_yz=polygon_yz,
                directions_yz=directions_yz,
                orientation_factor=orientation_factor,
                mc_samples=args.mc_samples,
                rng_seed=base_seed + 2,
                window_mode=args.window_mode,
                direction_mode=args.direction_mode,
            ) if np.isfinite(kr_ci_high) else float("nan")
            # PROXY 모드는 MC 통계(표준편차/CI/부피)가 없으므로 NaN, 균열면적/수밀도는 해석식으로 계산.
            c_std = float("nan")
            c_ci_low = float("nan")
            c_ci_high = float("nan")
            unit_volume = float("nan")
            mean_fracture_area = math.pi * radius_moments(args.site, set_id, kr_used, set_likelihood_rmin, 250.0)[1]
            fracture_number_density = 1.0 / mean_fracture_area if np.isfinite(mean_fracture_area) and mean_fracture_area > 0.0 else float("nan")

        # ---- P32 역산: P32 = 관측 P21 / C. C의 하/상한으로 P32 CI를 산출 ----
        p32_hat = observed_p21 / c_hat if np.isfinite(observed_p21) and np.isfinite(c_hat) and c_hat > 0.0 else float("nan")
        p32_from_low = observed_p21 / c_low if np.isfinite(observed_p21) and np.isfinite(c_low) and c_low > 0.0 else float("nan")
        p32_from_high = observed_p21 / c_high if np.isfinite(observed_p21) and np.isfinite(c_high) and c_high > 0.0 else float("nan")
        # C의 하/상한에서 나온 P32들 중 최소/최대를 P32 CI 경계로 사용.
        p32_candidates = [value for value in (p32_from_low, p32_from_high) if np.isfinite(value)]
        if p32_candidates:
            p32_ci_low = min(p32_candidates)
            p32_ci_high = max(p32_candidates)
        else:
            p32_ci_low = float("nan")
            p32_ci_high = float("nan")

        # ---- 참조 P32(진값 기반)와 비교하여 절대/상대 오차 계산 ----
        p32_reference = support_scaled_p32(args.site, set_id, kr_true, set_effective_rmin, 250.0)
        p32_abs_error = abs(p32_hat - p32_reference) if np.isfinite(p32_hat) and np.isfinite(p32_reference) else float("nan")
        p32_rel_error = 100.0 * p32_abs_error / abs(p32_reference) if np.isfinite(p32_abs_error) and abs(p32_reference) > 0.0 else float("nan")

        # ---- 이 세트의 모든 결과(입력/보정계수/P32/오차/상태/메모)를 출력 행으로 축적 ----
        out_rows.append(
            {
                "site": args.site,
                "set_id": set_id,
                "p32_label": args.p32_label,
                "set_effective_generation_rmin": set_effective_rmin,
                "set_likelihood_rmin": set_likelihood_rmin,
                "kr_used": kr_used,
                "kr_ci_low": kr_ci_low,
                "kr_ci_high": kr_ci_high,
                "observed_P21": observed_p21,
                "calibration_factor_C": c_hat,
                "calibration_factor_mode": calibration_mode,
                "unit_p32_mc_replicates": args.unit_p32_mc_replicates if calibration_mode == CALIBRATION_FACTOR_MODE_UNIT else 0,
                "unit_p32_mc_volume": unit_volume,
                "mean_fracture_area": mean_fracture_area,
                "fracture_number_density_for_unit_p32": fracture_number_density,
                "calibration_factor_std": c_std,
                "calibration_factor_ci_low": c_ci_low,
                "calibration_factor_ci_high": c_ci_high,
                "P32_hat": p32_hat,
                "P32_ci_low": p32_ci_low,
                "P32_ci_high": p32_ci_high,
                "P32_reference": p32_reference,
                "P32_abs_error": p32_abs_error,
                "P32_relative_error_percent": p32_rel_error,
                "kr_adoption_status": adoption_status,
                "p32_status": classify_p32_status(adoption_status, recovery_ci_status, kr_used, kr_ci_low, kr_ci_high, calibration_mode),
                "notes": build_notes(str(row.get("notes", "")), c_hat, p32_hat, calibration_mode),
            }
        )

    # ---- 출력 디렉터리 생성 후 요약 CSV 기록 및 완료 로그 출력 ----
    os.makedirs(os.path.dirname(args.outcsv) or ".", exist_ok=True)
    write_csv(out_rows, args.outcsv)
    print(f"[*] MC-calibrated P32 summary written to: {args.outcsv}")


# 스크립트로 직접 실행될 때만 main() 호출.
if __name__ == "__main__":
    main()
