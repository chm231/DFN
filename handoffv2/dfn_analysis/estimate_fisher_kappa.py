# =============================================================================
# 파일 역할: (1) 3개 이상의 트레이스(trace) 점들로부터 절리면의 법선을 추정하고,
#           (2) 법선 집합의 Fisher 분포 집중도(kappa)를 추정한다.
# 주요 입력:
#   - estimate_trace_normal_3pt: 3D 트레이스 점 배열 (N x 3)
#   - estimate_fisher_k_axial: 법선 벡터 배열 (N x 3)
# 주요 출력:
#   - 트레이스 법선(단위 벡터)과 비공선성 품질(quality) 정보 dict
#   - Fisher kappa(집중도), 평균 법선, 합벡터 길이 정보 dict
# 핵심 처리 흐름:
#   1) point_line_distance: 점~직선 수직 거리(비공선성 판단용)
#   2) estimate_trace_normal_3pt: 양 끝점 현(chord)과 최대 이탈점으로 평면 법선 산출
#   3) estimate_fisher_k_axial: 반구 정렬 합벡터 → R-bar → kappa 근사식
# =============================================================================
from typing import Optional, Tuple

import numpy as np

from dfn_analysis.estimate_mean_orientation import estimate_mean_normal_axial, normalize


# 점 point_xyz 로부터 직선(a_xyz-b_xyz)까지의 수직 거리를 계산한다.
#   원리: |(point-a) x (b-a)| / |b-a| (외적 크기 = 평행사변형 넓이)
#   인자: point_xyz(대상 점), a_xyz/b_xyz(직선 두 점), eps(0 판정)
#   반환: 수직 거리(float). 직선 길이가 0이면 0.0
def point_line_distance(point_xyz: np.ndarray, a_xyz: np.ndarray, b_xyz: np.ndarray, eps: float = 1e-12) -> float:
    # 입력을 float64 배열로 변환
    point_xyz = np.asarray(point_xyz, dtype=np.float64)
    a_xyz = np.asarray(a_xyz, dtype=np.float64)
    b_xyz = np.asarray(b_xyz, dtype=np.float64)

    # 직선 방향 벡터와 그 길이 계산. 길이가 0이면 거리 0으로 처리
    ab_xyz = b_xyz - a_xyz
    ab_norm = float(np.linalg.norm(ab_xyz))
    if ab_norm < eps:
        return 0.0
    # 외적 크기(평행사변형 넓이)를 밑변 길이로 나누어 수직 거리(높이) 산출
    return float(np.linalg.norm(np.cross(point_xyz - a_xyz, ab_xyz)) / ab_norm)


# 3D 트레이스 점들로부터 절리면(평면)의 법선을 추정한다.
#   방법: 양 끝점이 이루는 현(chord)에서 가장 멀리 떨어진 점(pm)을 세 번째 점으로 삼아
#         두 벡터의 외적으로 평면 법선을 구한다.
#   인자: points_xyz(N x 3 트레이스 점), min_quality(최소 비공선성 임계값), eps(0 판정)
#   반환: valid/normal/quality/reason 등을 담은 dict
def estimate_trace_normal_3pt(
    points_xyz: np.ndarray,
    min_quality: float = 0.005,
    eps: float = 1e-12,
) -> dict:
    # 입력 검증: (N x 3) 형태이며 점이 3개 이상이어야 평면 정의 가능
    pts_xyz = np.asarray(points_xyz, dtype=np.float64)
    if pts_xyz.ndim != 2 or pts_xyz.shape[0] < 3 or pts_xyz.shape[1] != 3:
        return {"valid": False, "normal": None, "quality": 0.0, "reason": "not_enough_points"}

    # 첫 점과 끝 점을 현(chord)의 양 끝으로 설정하고 현 길이 계산
    p0_xyz = pts_xyz[0]
    p1_xyz = pts_xyz[-1]
    chord_length = float(np.linalg.norm(p1_xyz - p0_xyz))
    # 현 길이가 0이면 방향 정의 불가 → 무효 반환
    if chord_length < eps:
        return {"valid": False, "normal": None, "quality": 0.0, "reason": "zero_chord_length"}

    # 각 점의 현(직선)까지 수직 거리를 계산하여 최대 이탈점(pm)을 탐색
    distances = np.array(
        [point_line_distance(point_xyz, p0_xyz, p1_xyz, eps=eps) for point_xyz in pts_xyz],
        dtype=np.float64,
    )
    max_index = int(np.argmax(distances))
    pm_xyz = pts_xyz[max_index]
    max_deviation = float(distances[max_index])
    # 품질(quality) = 최대 이탈거리 / 현 길이 → 점들이 얼마나 비공선(휘어짐)인지 척도
    quality = max_deviation / chord_length

    # 현 벡터(p1-p0)와 (pm-p0)의 외적으로 평면 법선 산출 후 정규화
    normal_xyz = normalize(np.cross(p1_xyz - p0_xyz, pm_xyz - p0_xyz), eps=eps)
    # 외적이 0에 가까우면 세 점이 공선(collinear)이라 평면 정의 불가
    if normal_xyz is None:
        return {"valid": False, "normal": None, "quality": quality, "reason": "collinear_points"}
    # 비공선성이 임계값 미만이면 법선은 있으나 신뢰도가 낮아 무효 처리
    if quality < min_quality:
        return {"valid": False, "normal": normal_xyz, "quality": quality, "reason": "low_noncollinearity"}

    # 유효한 법선과 진단 정보(끝점/최대이탈점/현길이/이탈거리)를 함께 반환
    return {
        "valid": True,
        "normal": normal_xyz,
        "quality": quality,
        "reason": "ok",
        "p0_xyz": p0_xyz,
        "p1_xyz": p1_xyz,
        "pm_xyz": pm_xyz,
        "chord_length": chord_length,
        "max_deviation": max_deviation,
    }


# 축(axial) 법선 집합에 대해 Fisher 분포의 집중도(kappa)를 추정한다.
#   kappa가 클수록 법선들이 평균 방향 주위로 강하게 모여 있음을 의미.
#   인자: normals_xyz(N x 3 법선 배열), eps(0 판정)
#   반환: valid/kappa/mean_normal/n/resultant_length 를 담은 dict
def estimate_fisher_k_axial(normals_xyz: np.ndarray, eps: float = 1e-12) -> dict:
    # 입력 검증: (N x 3) 형태가 아니면 무효 dict 반환
    normals_xyz = np.asarray(normals_xyz, dtype=np.float64)
    if normals_xyz.ndim != 2 or normals_xyz.shape[0] == 0 or normals_xyz.shape[1] != 3:
        return {"valid": False, "kappa": np.nan, "mean_normal": None, "n": 0, "resultant_length": 0.0}

    # 반구 정렬 기반 평균 법선 산출. 유효 표본이 2개 미만이면 kappa 추정 불가
    mean_normal = estimate_mean_normal_axial(normals_xyz, eps=eps)
    n_normals = len(normals_xyz)
    if mean_normal is None or n_normals < 2:
        return {
            "valid": False,
            "kappa": np.nan,
            "mean_normal": mean_normal,
            "n": n_normals,
            "resultant_length": 0.0,
        }

    # 평균 법선 기준으로 모든 법선을 같은 반구로 정렬(부호 통일)
    # Calculate resultant length using aligned vectors
    aligned = normals_xyz.copy()
    for idx in range(len(aligned)):
        if float(np.dot(aligned[idx], mean_normal)) < 0.0:
            aligned[idx] *= -1.0
    # 정렬된 법선들의 합벡터와 그 길이(R) 계산 → 집중도의 핵심 통계량
    resultant_xyz = np.sum(aligned, axis=0)
    resultant_length = float(np.linalg.norm(resultant_xyz))

    # 합벡터 길이가 0에 가까우면(완전 분산) kappa 추정 불가
    if resultant_length < eps:
        return {
            "valid": False,
            "kappa": np.nan,
            "mean_normal": mean_normal,
            "n": n_normals,
            "resultant_length": resultant_length,
        }

    # 평균 합벡터 길이 R-bar = R / n. R-bar이 1에 가까우면 완전 집중 → kappa=무한대
    r_bar = resultant_length / n_normals
    if r_bar >= 1.0 - 1e-12:
        kappa = np.inf
    else:
        # 3D(구면) Fisher kappa 근사식: kappa ≈ (3*R̄ - R̄^3) / (1 - R̄^2)
        kappa = (3.0 * r_bar - r_bar**3) / max(1e-12, 1.0 - r_bar**2)

    # 추정된 kappa와 평균 법선, 표본 수, 합벡터 길이를 반환
    return {
        "valid": True,
        "kappa": kappa,
        "mean_normal": mean_normal,
        "n": n_normals,
        "resultant_length": resultant_length,
    }
