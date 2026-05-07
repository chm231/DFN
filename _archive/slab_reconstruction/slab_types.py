from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np

@dataclass
class Slab:
    """터널 X축 방향의 특정 두께를 가진 슬래브 구역"""
    index: int
    x_center: float
    x_min: float
    x_max: float
    thickness: float = 0.2

@dataclass
class LocalCandidate:
    """Slab 내부의 개별 클러스터에 대해 추정된 로컬 평면 후보"""
    slab_index: int
    candidate_id: int
    points: np.ndarray  # (N, 3) 
    normal: np.ndarray  # (3,)
    centroid: np.ndarray # (3,)
    residual: float     # Plane fitting RMSE
    extent: float       # 클러스터 포인트들의 최대 범위 (지름 유사치)
    
    # 평가용 (Synthetic 데이터인 경우)
    truth_fracture_ids: List[int] = field(default_factory=list) # 클러스터 내 우세 ID
    major_truth_id: int = -1

@dataclass
class SlabLink:
    """인접 Slab 간의 로컬 후보 연결 정보"""
    slab_idx_A: int
    slab_idx_B: int
    id_A: int
    id_B: int
    score: float

@dataclass
class ReconstructedPlane:
    """여러 Slab의 후보들이 연결되어 최종 복원된 평면"""
    plane_id: int
    points: np.ndarray   # 모든 Slab의 기여 포인트들
    normal: np.ndarray
    centroid: np.ndarray
    residual: float
    source_slab_indices: List[int] = field(default_factory=list)
    
    # 정밀 복원용 (이번 단계는 Placeholder)
    radius: float = 0.0
    
    # 평가 결과
    truth_match_id: int = -1
    angle_error: float = -1.0
    dist_error: float = -1.0

@dataclass
class EvaluationResult:
    """복원 품질 평가 지표"""
    total_truth: int
    total_reconstructed: int
    matched_count: int
    avg_angle_error: float
    avg_dist_error: float
    success_rate: float
