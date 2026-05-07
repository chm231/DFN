import numpy as np
from typing import List, Dict, Tuple
from .slab_types import ReconstructedPlane, EvaluationResult

def find_best_truth_match(
    recon: ReconstructedPlane, 
    truth_centers: np.ndarray, 
    truth_normals: np.ndarray
) -> Tuple[int, float, float, float]:
    """
    단일 복원된 평면에 대해 가장 잘 맞는 원본 평면 인덱스와 오차 반환
    (Angle + Distance 가중치 오차 최소화)
    """
    # 1. Angle Diff (dot product)
    # n_r . n_t
    dot_prods = np.abs(np.clip(np.dot(truth_normals, recon.normal), -1.0, 1.0))
    angle_diffs = np.arccos(dot_prods) * (180.0 / np.pi) # degrees
    
    # 2. Centroid Distance
    dist_centroids = np.linalg.norm(truth_centers - recon.centroid, axis=1)
    
    # 3. Combined Score (간단히 가중치 합)
    # angle_error < 15도, dist < 5m 이내에서 탐색 선호
    score = angle_diffs * 0.5 + dist_centroids * 1.0
    
    best_idx = int(np.argmin(score))
    
    # 평면-평면 거리 (Offset)
    p2p_dist = np.abs(np.dot(truth_centers[best_idx] - recon.centroid, truth_normals[best_idx]))
    
    return best_idx, angle_diffs[best_idx], dist_centroids[best_idx], p2p_dist

def evaluate_reconstruction_performance(
    reconstructed_list: List[ReconstructedPlane],
    truth_centers: np.ndarray,
    truth_normals: np.ndarray,
    angle_threshold: float = 10.0,
    dist_threshold: float = 3.0
) -> EvaluationResult:
    """
    전체 복원 결과 요약 및 통계 산출
    """
    total_truth = len(truth_centers)
    total_recon = len(reconstructed_list)
    
    if total_recon == 0:
        return EvaluationResult(total_truth, 0, 0, 0, 0, 0)
        
    angle_errors = []
    dist_errors = []
    matched_indices = set()
    
    for recon in reconstructed_list:
        idx, angle_err, dist_err, p2p_dist = find_best_truth_match(recon, truth_centers, truth_normals)
        
        # 임계치 이내면 매칭 성공으로 간주
        if angle_err < angle_threshold and dist_err < dist_threshold:
            angle_errors.append(angle_err)
            dist_errors.append(dist_err)
            matched_indices.add(idx)
            
            # 매칭 결과 기록
            recon.truth_match_id = idx
            recon.angle_error = angle_err
            recon.dist_error = dist_err
            
    matched_count = len(matched_indices)
    
    return EvaluationResult(
        total_truth=total_truth,
        total_reconstructed=total_recon,
        matched_count=matched_count,
        avg_angle_error=float(np.mean(angle_errors)) if angle_errors else 0.0,
        avg_dist_error=float(np.mean(dist_errors)) if dist_errors else 0.0,
        success_rate=float(matched_count / total_truth * 100.0) if total_truth > 0 else 0.0
    )
