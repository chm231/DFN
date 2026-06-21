import numpy as np
from scipy.optimize import linear_sum_assignment
from typing import List, Tuple
from .slab_types import LocalCandidate, SlabLink

def calculate_match_score(
    cand_A: LocalCandidate, 
    cand_B: LocalCandidate, 
    angle_weight: float = 1.0,
    dist_weight: float = 1.0
) -> float:
    """
    두 로컬 평면 후보 간의 매칭 점수 (낮을수록 좋음/유사함)
    """
    # 1. Normal Angle Difference (dot product)
    cos_theta = np.abs(np.clip(np.dot(cand_A.normal, cand_B.normal), -1.0, 1.0))
    angle_diff = np.arccos(cos_theta) # radians
    
    # 2. Distance: Point-to-Plane (Centroid B from Plane A)
    # d = |(Cb - Ca) . Na|
    dist_plane = np.abs(np.dot(cand_B.centroid - cand_A.centroid, cand_A.normal))
    
    # 3. Centroid Euclidean distance (보조지표)
    dist_centroid = np.linalg.norm(cand_A.centroid - cand_B.centroid)

    # 단순 선형 결합 (필요 시 정교화 가능)
    # angle_weight = 10.0 (각도 차이에 민감), dist_weight = 1.0 (거리 보정)
    score = (angle_diff * 10.0 * angle_weight) + (dist_plane * 2.0 * dist_weight) + (dist_centroid * 0.1)
    
    return float(score)

def link_adjacent_slabs(
    candidates_A: List[LocalCandidate], 
    candidates_B: List[LocalCandidate],
    max_score_threshold: float = 2.0
) -> List[SlabLink]:
    """
    Slab A와 Slab B의 후보(clasters) 간의 1:1 매칭 (Hungarian Algorithm)
    """
    if not candidates_A or not candidates_B:
        return []
        
    nA = len(candidates_A)
    nB = len(candidates_B)
    
    # Cost Matrix 생성
    cost_matrix = np.zeros((nA, nB))
    for i in range(nA):
        for j in range(nB):
            cost_matrix[i, j] = calculate_match_score(candidates_A[i], candidates_B[j])
            
    # 최적 이분 매칭 (Bipartite Matching)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    links = []
    for r, c in zip(row_ind, col_ind):
        score = cost_matrix[r, c]
        if score < max_score_threshold:
            links.append(SlabLink(
                slab_idx_A=candidates_A[r].slab_index,
                slab_idx_B=candidates_B[c].slab_index,
                id_A=candidates_A[r].candidate_id,
                id_B=candidates_B[c].candidate_id,
                score=score
            ))
            
    return links
