"""
Core Data Structures for Bayesian 3D DFN Inverse Reconstruction.
Defines clean, type-hinted dataclasses to maintain type safety throughout the pipeline.
"""
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np


@dataclass
class ExcavationFace:
    """터널 굴착 막장면 (x=const 평면에서의 터널 단면 정보)"""
    face_id: int
    x_face: float
    tunnel_polygon_yz: np.ndarray  # Shape (N, 2) array of [y, z] coordinates
    advance_step: float            # 이전 face로부터의 굴진 거리


@dataclass
class FaceTrace:
    """터널 굴착 막장면(polygon 내부)에서 관측된 단일 fracture trace segment"""
    face_id: int
    trace_id: int
    x_face: float
    p0_y: float
    p0_z: float
    p1_y: float
    p1_z: float
    confidence: float = 1.0
    parent_fracture_id: Optional[int] = None
    
    # Pre-calculated geometric fields
    midpoint_y: float = field(init=False)
    midpoint_z: float = field(init=False)
    length: float = field(init=False)
    orientation_2d: float = field(init=False)  # 2D orientation angle in [-pi/2, pi/2] (axial)
    censoring_class: int = 0                    # 0: Contained, 1: One-end clipped, 2: Both-end clipped
    set_id: Optional[int] = None                # Clustering set identifier

    def __post_init__(self):
        self.midpoint_y = (self.p0_y + self.p1_y) / 2.0
        self.midpoint_z = (self.p0_z + self.p1_z) / 2.0
        self.length = float(np.sqrt((self.p1_y - self.p0_y)**2 + (self.p1_z - self.p0_z)**2))
        
        # Calculate orientation 2d: angle of line segment bounded to [-pi/2, pi/2]
        dy = self.p1_y - self.p0_y
        dz = self.p1_z - self.p0_z
        angle = np.arctan2(dz, dy)
        
        # Wrap angle to axial range [-pi/2, pi/2]
        if angle > np.pi / 2.0:
            angle -= np.pi
        elif angle < -np.pi / 2.0:
            angle += np.pi
        self.orientation_2d = float(angle)


@dataclass
class TraceMatch:
    """인접한 두 face 간의 trace 매칭 결과 및 Bayes Factor 정보"""
    face_id_prev: int
    face_id_curr: int
    trace_id_prev: int
    trace_id_curr: int
    log_bayes_factor: float
    accepted: bool = False


@dataclass
class ReconstructedPlane:
    """연속 매칭된 trace(들)로부터 3차원 공간 상에 역산된 평면 (MAP)"""
    plane_id: int
    point_x: float
    point_y: float
    point_z: float
    normal_x: float
    normal_y: float
    normal_z: float
    radius: float
    source_trace_ids: List[int] = field(default_factory=list)
    confidence: float = 1.0  # Posterior Inclusion Probability (PIP)
    covariance: Optional[np.ndarray] = None  # 3x3 covariance matrix from Laplace approximation
    set_id: Optional[int] = None
    is_single_face_candidate: bool = False  # True if generated as a probabilistic sample of a single-face trace


@dataclass
class StochasticFracture:
    """Stochastic DFN 생성을 위한 3D 균열 원판 구조체"""
    fracture_id: int
    center_x: float
    center_y: float
    center_z: float
    normal_x: float
    normal_y: float
    normal_z: float
    radius: float
    set_id: int
