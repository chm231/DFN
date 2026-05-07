import numpy as np
from typing import List
from scipy.spatial import KDTree
from collections import deque

def cluster_slab_points(points: np.ndarray, eps: float = 0.5, min_samples: int = 3) -> List[np.ndarray]:
    """
    Slab 내 포인트들을 거리 기반으로 클러스터링합니다. (DBSCAN 유사 방식)
    포인트 자체가 아닌 '인덱스'의 리스트를 반환합니다.
    """
    if len(points) < min_samples:
        return []
        
    tree = KDTree(points)
    n = len(points)
    visited = np.zeros(n, dtype=bool)
    clusters_indices = []
    
    for i in range(n):
        if visited[i]:
            continue
            
        # 새로운 클러스터 시작
        neighbors = tree.query_ball_point(points[i], eps)
        if len(neighbors) < min_samples:
            continue
            
        current_cluster_idx = []
        queue = deque(neighbors)
        visited[i] = True
        
        while queue:
            idx = queue.popleft()
            if not visited[idx]:
                visited[idx] = True
                current_cluster_idx.append(idx)
                
                # 주변 이웃 확장
                new_neighbors = tree.query_ball_point(points[idx], eps)
                if len(new_neighbors) >= min_samples:
                    for nn in new_neighbors:
                        if not visited[nn]:
                            queue.append(nn)
            elif idx == i: # 첫 원소 처리
                current_cluster_idx.append(idx)

        if len(current_cluster_idx) >= min_samples:
            clusters_indices.append(np.array(current_cluster_idx))
            
    return clusters_indices

def get_major_truth_id(truth_ids: np.ndarray) -> int:
    """클러스터 내에서 가장 빈도가 높은 원본 Fracture ID 반환"""
    if len(truth_ids) == 0:
        return -1
    counts = np.bincount(truth_ids)
    return int(np.argmax(counts))
