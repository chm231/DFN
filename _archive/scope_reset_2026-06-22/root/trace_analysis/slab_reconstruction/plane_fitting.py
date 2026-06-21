import numpy as np
from typing import Tuple

def fit_plane_svd(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    SVD를 이용한 3D 평면 피팅
    
    Returns:
        normal: (3,) unit vector
        centroid: (3,) array
        residual: RMSE (평균 제곱근 오차)
    """
    if len(points) < 3:
        return np.array([0, 0, 1]), np.mean(points, axis=0), 0.0
        
    centroid = np.mean(points, axis=0)
    pts_centered = points - centroid
    
    # SVD 수행 (pts_centered = U * S * Vt)
    # Vt의 마지막 행(U의 마지막 열)이 가장 작은 분산을 가지는 방향(Normal)
    _, _, vt = np.linalg.svd(pts_centered)
    normal = vt[2, :]
    
    # Normal 방향 보정 (항상 X축 양의 방향 또는 Z축 양의 방향 등 일관성 유지)
    # 여기서는 터널 진행 방향인 X축의 부호로 일관성 부여 (N[0]가 우세하게)
    if normal[0] < 0:
        normal = -normal
        
    # Residual (거리에 대한 RMSE) 계산
    # d = |(P - C) . n|
    distances = np.abs(np.dot(pts_centered, normal))
    residual = float(np.sqrt(np.mean(distances**2)))
    
    return normal, centroid, residual

def get_points_extent(points: np.ndarray) -> float:
    """포인트 클라우드의 최대 공간적 범위(최대 거리) 반환"""
    if len(points) < 2:
        return 0.0
    # 간단히 AABB 대각선 또는 분산 기반 추정
    mins = np.min(points, axis=0)
    maxs = np.max(points, axis=0)
    return float(np.linalg.norm(maxs - mins))
