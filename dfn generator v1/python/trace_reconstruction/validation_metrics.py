"""
[Direction B: Inverse Reconstruction]
A 방향 3D Ground Truth 데이터 파일과, B 방향 역산(Reconstruction) 파이프라인 결과를 비교하여
평가(Validation)를 수행하는 도구 모음입니다.
"""
import numpy as np
from .trace_types import ReconstructedPlane
from typing import List

def evaluate_plane_orientation_error(
    reconstructed_planes: List[ReconstructedPlane], 
    true_normals: np.ndarray
) -> List[float]:
    """
    모든 도출된 평면과 원본 GT 3D fracture normal들 간의 
    최소 각도 차이(Orientation Error)를 도출합니다.
    """
    errors = []
    for p in reconstructed_planes:
        n_recon = np.array([p.normal_x, p.normal_y, p.normal_z])
        # n_recon 과 일치하는 것을 고르기 위한 dot product max
        # (방향 고려 abs 취함)
        dots = np.abs(np.dot(true_normals, n_recon))
        best_match_idx = np.argmax(dots)
        best_angle = np.arccos(min(1.0, dots[best_match_idx]))
        errors.append(float(np.degrees(best_angle)))
    return errors

def evaluate_plane_position_error(
    reconstructed_planes: List[ReconstructedPlane], 
    true_centers: np.ndarray,
    # 필요하다면 추가 정보
) -> List[float]:
    """
    평면의 중심 좌표 혹은 평면 방정식 상수(D)간의 거리 편차 계산
    """
    # Placeholder
    return [np.nan for _ in reconstructed_planes]
