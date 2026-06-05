"""
radius_estimator.py
===================
복원된 글로벌 평면(ReconstructedPlane)의 3D 반경을 기하학적으로 추정하는 모듈.

추정 방법:
1. 관통 길이(Penetration Length) 기반: 평면이 관통하는 Slab들의 X축 범위와
   법선벡터의 경사각(θ)을 이용하여 원판의 최소 반경을 추정.
2. 포인트 분포(Extent) 기반: 클러스터 포인트들의 YZ 평면 상 범위로 하한선 보장.
3. 단일 Slab 평면: 관통 길이를 사용할 수 없으므로 YZ extent만 사용.
"""

import numpy as np
from typing import List
from .slab_types import ReconstructedPlane


def estimate_radius_penetration(plane: ReconstructedPlane, slab_spacing: float = 3.0) -> float:
    """
    관통 길이(Penetration Length) 기반 반경 추정.
    
    원판이 여러 Slab을 관통할 때, X축 방향의 관통 범위(x_span)와
    법선벡터의 X축에 대한 각도(θ)를 이용하여 반경을 추정합니다.
    
    R >= x_span / (2 * sin(θ))
    
    여기서 θ = arccos(|n_x|) 는 법선벡터와 X축 사이의 각도이며,
    sin(θ) = sqrt(n_y^2 + n_z^2)
    """
    pts = plane.points
    if len(pts) < 2:
        return 0.5  # 최소 기본값
    
    # X축 관통 범위
    x_min, x_max = np.min(pts[:, 0]), np.max(pts[:, 0])
    x_span = x_max - x_min
    
    # 법선벡터의 X축에 대한 sin(θ) 계산
    n = plane.normal
    sin_theta = np.sqrt(n[1]**2 + n[2]**2)
    
    if sin_theta < 0.01:
        # 법선이 X축에 거의 평행 → 디스크가 면에 수직으로 관통
        # 이 경우 YZ extent만 사용 가능
        return estimate_radius_extent(plane)
    
    # 다중 Slab을 관통하는 경우
    n_slabs = len(plane.source_slab_indices)
    if n_slabs >= 2:
        # 관통 길이 기반 반경 추정 (양 끝 slab의 중심간 거리 + 반 slab 보정)
        effective_span = x_span + slab_spacing * 0.5
        R_penetration = effective_span / (2.0 * sin_theta)
    else:
        # 단일 Slab인 경우 YZ extent 기반
        R_penetration = estimate_radius_extent(plane)
    
    return max(R_penetration, 0.5)  # 최소 0.5m 보장


def estimate_radius_extent(plane: ReconstructedPlane) -> float:
    """
    포인트 분포(Extent) 기반 반경 추정.
    
    클러스터 포인트들의 YZ 평면 상의 최대 범위를 원판 지름의 근사치로 사용합니다.
    """
    pts = plane.points
    if len(pts) < 2:
        return 0.5
    
    # YZ 평면 상의 범위
    yz_pts = pts[:, 1:]  # (N, 2)
    yz_min = np.min(yz_pts, axis=0)
    yz_max = np.max(yz_pts, axis=0)
    yz_extent = np.linalg.norm(yz_max - yz_min)
    
    # 3D 전체 extent도 고려
    pts_min = np.min(pts, axis=0)
    pts_max = np.max(pts, axis=0)
    full_extent = np.linalg.norm(pts_max - pts_min)
    
    # YZ extent와 full extent의 평균을 반지름으로
    diameter_est = max(yz_extent, full_extent * 0.7)
    
    return float(max(diameter_est / 2.0, 0.5))


def estimate_radii_for_planes(
    planes: List[ReconstructedPlane],
    slab_spacing: float = 3.0
) -> List[ReconstructedPlane]:
    """
    모든 복원된 평면에 대해 반경을 추정하고 결과를 기록합니다.
    
    Args:
        planes: 복원된 평면 리스트
        slab_spacing: Slab 간격 (m)
    
    Returns:
        반경이 추정된 평면 리스트 (in-place 수정됨)
    """
    for plane in planes:
        R_pen = estimate_radius_penetration(plane, slab_spacing)
        R_ext = estimate_radius_extent(plane)
        
        # 관통 기반과 분포 기반의 max를 최종 추정값으로
        R_final = max(R_pen, R_ext)
        
        plane.estimated_radius = R_final
        plane.radius = R_final
    
    return planes
