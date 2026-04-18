import numpy as np
from scipy.optimize import minimize
from typing import List, Optional
from .data_models import Trace, FractureSet, FractureHypothesis

class PlaneReconstructor:
    """제약 조건 기반 3차원 평면 및 원판 복원 엔진"""
    def __init__(self, set_lambda: float = 0.5):
        self.set_lambda = set_lambda

    def fit_plane_constrained(self, traces: List[Trace], 
                              assigned_set: Optional[FractureSet] = None) -> FractureHypothesis:
        """선분 데이터와 절리군 정보를 결합하여 최적 평면 산출"""
        
        # 모든 선분의 점 수집 (6~n개 점)
        points = []
        for t in traces:
            points.append(t.endpoints_3d[0])
            points.append(t.endpoints_3d[1])
        points = np.array(points)
        
        # 1. 초기값 계산 (SVD)
        centroid = np.mean(points, axis=0)
        centered_points = points - centroid
        
        if len(centered_points) < 3:
            initial_normal = np.array([1.0, 0.0, 0.0])
            s = [0.0]
        else:
            u, s, vh = np.linalg.svd(centered_points)
            initial_normal = vh[-1]
        
        # 2. 절리군 제약 조건이 있는 경우 수치 최적화 수행
        if assigned_set:
            def objective(n_spherical):
                # 구면 좌표계 [theta, phi] -> 카테시안 [nx, ny, nz]
                theta, phi = n_spherical
                n = np.array([
                    np.sin(theta) * np.cos(phi),
                    np.sin(theta) * np.sin(phi),
                    np.cos(theta)
                ])
                
                # (1) 점들과의 거리 오차
                dist_error = np.sum(np.dot(centered_points, n)**2)
                
                # (2) 절리군 대표 법선과의 배향 오차
                set_normal = assigned_set.representative_normal
                orient_error = 1.0 - np.abs(np.dot(n, set_normal))
                
                return dist_error + self.set_lambda * orient_error
            
            # 초기값 변환
            init_theta = np.arccos(initial_normal[2])
            init_phi = np.arctan2(initial_normal[1], initial_normal[0])
            
            res = minimize(objective, [init_theta, init_phi], method='Nelder-Mead')
            
            # 최종 법선 산출
            final_theta, final_phi = res.x
            final_normal = np.array([
                np.sin(final_theta) * np.cos(final_phi),
                np.sin(final_theta) * np.sin(final_phi),
                np.cos(final_theta)
            ])
            error = res.fun
        else:
            final_normal = initial_normal
            error = s[-1] if len(s) > 2 else 0.0

        # 중심점 및 반경 추정 (Disc estimation baseline)
        center, radius = self._estimate_disc_geometry(points, final_normal)

        return FractureHypothesis(
            hypothesis_id=0,
            assigned_trace_ids=[t.trace_id for t in traces],
            set_id=assigned_set.set_id if assigned_set else None,
            normal=final_normal,
            center=center,
            radius=radius,
            fit_error=float(error)
        )

    def _estimate_disc_geometry(self, points: np.ndarray, normal: np.ndarray):
        """평면 점군으로부터 원판의 중심과 반경 추산"""
        center = np.mean(points, axis=0)
        # 모든 점들로부터 중심까지의 거리 중 최대값 (Baseline)
        dists = np.linalg.norm(points - center, axis=1)
        radius = np.max(dists) if len(dists) > 0 else 1.0
        return center, radius
