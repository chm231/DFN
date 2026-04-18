from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import numpy as np

@dataclass
class Trace:
    """터널 막장면에서 관측된 단일 선분(Trace) 데이터"""
    trace_id: str
    face_id: int
    endpoints_3d: np.ndarray  # Shape (2, 3)
    midpoint_3d: Optional[np.ndarray] = None
    length: float = 0.0
    direction: Optional[np.ndarray] = None  # unit vector
    confidence: float = 1.0
    raw_polyline: Optional[np.ndarray] = None  # Optional raw points

@dataclass
class Face:
    """터널 굴착 막장면 정의"""
    face_id: int
    plane_point: np.ndarray   # [x, y, z]
    plane_normal: np.ndarray  # [nx, ny, nz]
    traces: List[Trace] = field(default_factory=list)
    excavation_step: int = 0

@dataclass
class FractureSet:
    """추론된 절리군 정보"""
    set_id: int
    representative_normal: np.ndarray  # [nx, ny, nz]
    dispersion: float = 0.0            # Fisher k or similar
    membership_stats: dict = field(default_factory=dict)

@dataclass
class FractureHypothesis:
    """3차원 균열 복원 가설"""
    hypothesis_id: int
    assigned_trace_ids: List[str] = field(default_factory=list)
    set_id: Optional[int] = None
    
    # 기하학적 파라미터 [Best Fit]
    normal: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0]))
    center: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))
    radius: float = 1.0
    offset: float = 0.0
    
    # 평가 지표
    fit_error: float = 0.0
    prior_score: float = 0.0
    confidence: float = 0.0
