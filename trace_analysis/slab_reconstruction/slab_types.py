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
    
    # 반경 추정 결과
    radius: float = 0.0
    estimated_radius: float = 0.0
    
    # 세트 분류 결과
    set_id: int = -1
    
    # 지질공학적 방향성
    dip: float = 0.0
    dip_direction: float = 0.0
    
    # 평가 결과
    truth_match_id: int = -1
    angle_error: float = -1.0
    dist_error: float = -1.0
    radius_error: float = -1.0

@dataclass
class EvaluationResult:
    """복원 품질 평가 지표"""
    total_truth: int
    total_reconstructed: int
    matched_count: int
    avg_angle_error: float
    avg_dist_error: float
    avg_radius_error: float
    success_rate: float

@dataclass
class DFNSetResult:
    """단일 균열 세트의 DFN 파라미터 추출 결과"""
    set_id: int
    n_planes: int
    mean_normal: np.ndarray
    dip: float
    dip_direction: float
    kappa: float
    mean_radius: float
    radii: np.ndarray
    alpha_R: float            # Pareto shape parameter (Power-law exponent)
    r_min: float              # Pareto scale parameter
    P30: float                # 체적 개수 밀도 (1/m³)
    P32: float                # 체적 면적 밀도 (m²/m³)
    P32_terzaghi: float       # Terzaghi 보정 P32
    mean_sin_theta: float     # 평균 sin(θ) (방향 편향 보정 인자)

@dataclass
class DFNParameterResult:
    """전체 DFN 파라미터 추출 결과 (모든 세트 종합)"""
    n_sets: int
    domain_volume: float
    total_reconstructed: int
    set_results: dict = field(default_factory=dict)  # set_id -> DFNSetResult
