import numpy as np
from typing import List, Tuple

class SlabTrace3D:
    """Slab 내부의 개별 3차원 선분(Trace) 객체"""
    def __init__(self, segment_id: int, p0: np.ndarray, p1: np.ndarray, parent_id: int = -1):
        self.segment_id = segment_id
        self.p0 = np.array(p0, dtype=np.float64)
        self.p1 = np.array(p1, dtype=np.float64)
        self.parent_id = parent_id
        
        # 선분 정보 계산
        self.vector = self.p1 - self.p0
        self.length = np.linalg.norm(self.vector)
        self.direction = self.vector / (self.length + 1e-12)
        self.midpoint = (self.p0 + self.p1) / 2.0

def segment_to_segment_distance(s1: SlabTrace3D, s2: SlabTrace3D) -> float:
    """
    공간 상에 정의된 두 3차원 선분 s1, s2 간의 최소 거리를 구하는 기하학적 알고리즘.
    (Lumelsky, 1985 / Real-Time Collision Detection 기법 구현)
    """
    u = s1.vector
    v = s2.vector
    w = s1.p0 - s2.p0
    
    a = np.dot(u, u)  # 항상 >= 0
    b = np.dot(u, v)
    c = np.dot(v, v)  # 항상 >= 0
    d = np.dot(u, w)
    e = np.dot(v, w)
    
    D = a * c - b * b  # 행렬식
    
    sc, sN, sD = 0.0, 0.0, D  # sc = sN / sD, 기본적으로 sD = D
    tc, tN, tD = 0.0, 0.0, D  # tc = tN / tD, 기본적으로 tD = D
    
    # 두 선분이 평행한지 확인
    if D < 1e-12:
        sN = 0.0
        sD = 1.0
        tN = e
        tD = c
    else:
        # 평행하지 않은 경우 매개변수 sc, tc 계산
        sN = (b * e - c * d)
        tN = (a * e - b * d)
        
        if sN < 0.0:
            sN = 0.0
            tN = e
            tD = c
        elif sN > sD:
            sN = sD
            tN = e + b
            tD = c
            
    # tc 범위 제한 [0, 1]
    if tN < 0.0:
        tN = 0.0
        # s 조정
        if -d < 0.0:
            sN = 0.0
        elif -d > a:
            sN = sD
        else:
            sN = -d
            sD = a
    elif tN > tD:
        tN = tD
        # s 조정
        if (-d + b) < 0.0:
            sN = 0.0
        elif (-d + b) > a:
            sN = sD
        else:
            sN = (-d + b)
            sD = a
            
    # 최종 매개변수 sc, tc 계산
    sc = 0.0 if abs(sN) < 1e-12 else sN / sD
    tc = 0.0 if abs(tN) < 1e-12 else tN / tD
    
    # 두 선분 상의 가장 가까운 점 계산
    dP = w + (sc * u) - (tc * v)
    return float(np.linalg.norm(dP))

def cluster_3d_segments(
    traces: List[SlabTrace3D],
    dist_threshold: float = 1.5,
    angle_penalty_threshold: float = 0.8,
    min_samples: int = 2
) -> List[List[int]]:
    """
    3차원 선분 리스트를 공간 거리와 방향 정렬도를 결합하여 클러스터링합니다.
    Scipy의 Hierarchical clustering (single linkage)을 Precomputed Distance Matrix에 적용합니다.
    
    Returns:
        List of cluster indices list (e.g., [[0, 2], [1, 3, 4]])
    """
    n = len(traces)
    if n == 0:
        return []
    if n == 1:
        return [[0]]
        
    # 1. 3D 선분 간 거리 행렬 계산
    dist_matrix = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            d = segment_to_segment_distance(traces[i], traces[j])
            
            # 방향 벡터 유사도(Cosine Similarity)를 패널티로 적용
            cos_sim = abs(np.dot(traces[i].direction, traces[j].direction))
            if cos_sim < angle_penalty_threshold:
                # 방향이 다를 경우 거리를 가상으로 팽창시켜 다른 클러스터로 가도록 유도
                d += (1.0 - cos_sim) * 10.0
                
            dist_matrix[i, j] = d
            dist_matrix[j, i] = d
            
    # 2. Scipy Hierarchical Clustering 수행
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform
    
    # 대칭 행렬을 1D condensed distance vector로 변환
    condensed_dist = squareform(dist_matrix)
    
    # 단일 연결법(single linkage) 적용 (선분끼리 하나라도 가까우면 하나의 균열면으로 연결 가능)
    Z = linkage(condensed_dist, method='single')
    
    # 임계값 기준으로 평탄 클러스터 레이블 획득
    labels = fcluster(Z, t=dist_threshold, criterion='distance')
    
    # 3. 레이블 그룹화
    clusters = {}
    for idx, label in enumerate(labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(idx)
        
    # 최소 크기 조건 필터링 적용
    final_clusters = []
    for c_idx_list in clusters.values():
        if len(c_idx_list) >= min_samples:
            final_clusters.append(c_idx_list)
        else:
            # 낱개 선분도 단독 클러스터로 유입 허용 (필요 시)
            final_clusters.append(c_idx_list)
            
    return final_clusters
