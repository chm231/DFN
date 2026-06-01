"""
evaluator.py
=============
복원 결과를 Ground Truth 대비 평가하는 모듈.
반경 오차 및 세트별 분리 평가를 지원합니다.
"""

import numpy as np
from typing import List, Dict, Tuple
from .slab_types import ReconstructedPlane, EvaluationResult


def find_best_truth_match(
    recon: ReconstructedPlane, 
    truth_centers: np.ndarray, 
    truth_normals: np.ndarray,
    truth_radii: np.ndarray = None
) -> Tuple[int, float, float, float, float]:
    """
    단일 복원된 평면에 대해 가장 잘 맞는 원본 평면 인덱스와 오차 반환
    (Angle + Distance 가중치 오차 최소화)
    
    Returns:
        best_idx: 매칭된 Ground Truth 인덱스
        angle_err: 각도 오차 (degrees)
        dist_err: 중심 거리 오차 (m)
        p2p_dist: 평면-평면 거리 (m)
        radius_err: 반경 오차 (m) 또는 -1.0
    """
    # 1. Angle Diff (dot product)
    dot_prods = np.abs(np.clip(np.dot(truth_normals, recon.normal), -1.0, 1.0))
    angle_diffs = np.arccos(dot_prods) * (180.0 / np.pi) # degrees
    
    # 2. Centroid Distance
    dist_centroids = np.linalg.norm(truth_centers - recon.centroid, axis=1)
    
    # 3. Combined Score (간단히 가중치 합)
    score = angle_diffs * 0.5 + dist_centroids * 1.0
    
    best_idx = int(np.argmin(score))
    
    # 평면-평면 거리 (Offset)
    p2p_dist = np.abs(np.dot(truth_centers[best_idx] - recon.centroid, truth_normals[best_idx]))
    
    # 반경 오차
    radius_err = -1.0
    if truth_radii is not None and recon.estimated_radius > 0:
        radius_err = abs(recon.estimated_radius - truth_radii[best_idx])
    
    return best_idx, angle_diffs[best_idx], dist_centroids[best_idx], p2p_dist, radius_err


def evaluate_reconstruction_performance(
    reconstructed_list: List[ReconstructedPlane],
    truth_centers: np.ndarray,
    truth_normals: np.ndarray,
    truth_radii: np.ndarray = None,
    angle_threshold: float = 10.0,
    dist_threshold: float = 3.0
) -> EvaluationResult:
    """
    전체 복원 결과 요약 및 통계 산출
    """
    total_truth = len(truth_centers)
    total_recon = len(reconstructed_list)
    
    if total_recon == 0:
        return EvaluationResult(total_truth, 0, 0, 0.0, 0.0, 0.0, 0.0)
        
    angle_errors = []
    dist_errors = []
    radius_errors = []
    matched_indices = set()
    
    for recon in reconstructed_list:
        idx, angle_err, dist_err, p2p_dist, radius_err = find_best_truth_match(
            recon, truth_centers, truth_normals, truth_radii
        )
        
        # 임계치 이내면 매칭 성공으로 간주
        if angle_err < angle_threshold and dist_err < dist_threshold:
            angle_errors.append(angle_err)
            dist_errors.append(dist_err)
            if radius_err >= 0:
                radius_errors.append(radius_err)
            matched_indices.add(idx)
            
            # 매칭 결과 기록
            recon.truth_match_id = idx
            recon.angle_error = angle_err
            recon.dist_error = dist_err
            recon.radius_error = radius_err
            
    matched_count = len(matched_indices)
    
    return EvaluationResult(
        total_truth=total_truth,
        total_reconstructed=total_recon,
        matched_count=matched_count,
        avg_angle_error=float(np.mean(angle_errors)) if angle_errors else 0.0,
        avg_dist_error=float(np.mean(dist_errors)) if dist_errors else 0.0,
        avg_radius_error=float(np.mean(radius_errors)) if radius_errors else 0.0,
        success_rate=float(matched_count / total_truth * 100.0) if total_truth > 0 else 0.0
    )


def evaluate_per_set(
    reconstructed_list: List[ReconstructedPlane],
    truth_centers: np.ndarray,
    truth_normals: np.ndarray,
    truth_radii: np.ndarray = None,
    truth_set_ids: np.ndarray = None,
    angle_threshold: float = 10.0,
    dist_threshold: float = 3.0
) -> Dict[int, EvaluationResult]:
    """
    세트별 분리 평가를 수행합니다.
    
    Returns:
        {set_id: EvaluationResult} 딕셔너리
    """
    # 세트별 복원 평면 분류
    set_planes = {}
    for p in reconstructed_list:
        sid = p.set_id if p.set_id > 0 else 0
        if sid not in set_planes:
            set_planes[sid] = []
        set_planes[sid].append(p)
    
    results = {}
    for sid, planes in set_planes.items():
        eval_res = evaluate_reconstruction_performance(
            planes, truth_centers, truth_normals, truth_radii,
            angle_threshold, dist_threshold
        )
        results[sid] = eval_res
    
    return results
