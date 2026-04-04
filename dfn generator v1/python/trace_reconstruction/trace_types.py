"""
[Direction B: Inverse Reconstruction]
B 방향 연구에서 다루어지는 핵심 데이터 구조체(Dataclass) 정의 모듈입니다.

주의: 
여기 정의된 FaceTrace는 "crop box 전체 단면의 trace"가 아니라,
터널 굴착면(excavation face) 폴리곤 내부에서 측정 가능한 segment만을 의미합니다.
"""
from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np


@dataclass
class ExcavationFace:
    """터널 굴착 막장면 (x=const 평면에서의 터널 단면 정보)"""
    face_id: int
    x_face: float
    tunnel_polygon_yz: np.ndarray  # (N, 2) array of [y, z] coordinates
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
    midpoint_y: float = field(init=False)
    midpoint_z: float = field(init=False)
    length: float = field(init=False)
    local_orientation_2d: float = field(init=False)  # 2D dip angle equivalent
    confidence: float = 1.0

    def __post_init__(self):
        self.midpoint_y = (self.p0_y + self.p1_y) / 2.0
        self.midpoint_z = (self.p0_z + self.p1_z) / 2.0
        self.length = np.sqrt((self.p1_y - self.p0_y)**2 + (self.p1_z - self.p0_z)**2)
        dy = self.p1_y - self.p0_y
        dz = self.p1_z - self.p0_z
        self.local_orientation_2d = float(np.arctan2(dz, dy))


@dataclass
class TraceMatch:
    """인접한 두 face 간의 trace 매칭 결과"""
    face_id_prev: int
    face_id_curr: int
    trace_id_prev: int
    trace_id_curr: int
    score: float
    accepted: bool = False


@dataclass
class ReconstructedPlane:
    """연속 매칭된 trace(들)로부터 3차원 공간 상에 역산된 평면"""
    plane_id: int
    point_x: float
    point_y: float
    point_z: float
    normal_x: float
    normal_y: float
    normal_z: float
    source_trace_ids: List[int] = field(default_factory=list)
    confidence: float = 1.0


@dataclass
class ReconstructedBlock:
    """역산된 평면들과 터널 경계를 조합(Intersection)하여 만들어낸 폴리헤드론 블록"""
    block_id: int
    vertices: np.ndarray  # (N, 3)
    faces: List[List[int]] # List of vertex index lists for each face
    centroid: tuple
    volume: float
