"""
[Direction B: Inverse Reconstruction]
매칭된 여러 막장면에서의 Trace 데이터 혹은 1쌍의 Trace를 조합하여
실제 3차원 공간 상의 절리면(Plane)을 복원(역산)하는 모듈입니다.
"""
import numpy as np
from typing import List, Optional
from .trace_types import FaceTrace, ReconstructedPlane

def lift_face_trace_to_3d(trace: FaceTrace) -> tuple:
    """단일 Face Trace를 3차원 공간 상의 2개의 점(Point) 좌표로 변환"""
    p0 = np.array([trace.x_face, trace.p0_y, trace.p0_z])
    p1 = np.array([trace.x_face, trace.p1_y, trace.p1_z])
    return p0, p1

def reconstruct_plane_from_trace_pair(trace_prev: FaceTrace, trace_curr: FaceTrace, plane_id: int) -> Optional[ReconstructedPlane]:
    """
    이전 막장면(Face n-1)과 현재 막장면(Face n)에 있는 2개의 매칭된 Trace 선분으로부터
    두 선분을 모두 통과하는(최소자승 허용) 3D 평면을 복원합니다.
    """
    # 2개의 trace가 이루는 4개의 끝점을 추출
    pts_A = lift_face_trace_to_3d(trace_prev)
    pts_B = lift_face_trace_to_3d(trace_curr)
    
    points = np.vstack([pts_A, pts_B])  # (4, 3) D matrix
    
    # 평면의 기하학적 중심
    centroid = np.mean(points, axis=0)
    
    # SVD를 통한 평면 법선 벡터 추정 (최소자승법)
    centered = points - centroid
    _, _, vh = np.linalg.svd(centered)
    normal = vh[-1, :] # 가장 변동성이 적은 방향이 평면의 법선
    
    # 방향 일관성 (가급적 법선이 x방향 + 혹은 - 로 일관성 갖도록 조정)
    if normal[0] < 0:
        normal = -normal
        
    return ReconstructedPlane(
        plane_id=plane_id,
        point_x=float(centroid[0]),
        point_y=float(centroid[1]),
        point_z=float(centroid[2]),
        normal_x=float(normal[0]),
        normal_y=float(normal[1]),
        normal_z=float(normal[2]),
        source_trace_ids=[trace_prev.trace_id, trace_curr.trace_id],
        confidence=1.0  # SVD 핏 오차를 이용해 Confidence 도출 가능
    )

def fit_plane_from_trace_track(trace_track: List[FaceTrace], plane_id: int) -> Optional[ReconstructedPlane]:
    """
    3개 이상의 face에서 연속 매칭된 막대한 트랙에 대한 평면 복원
    """
    if len(trace_track) < 2:
        return None
    # TODO: 3개 이상의 trace points에 대한 SVD 피팅 로직
    return reconstruct_plane_from_trace_pair(trace_track[0], trace_track[-1], plane_id)

def evaluate_plane_fit(plane: ReconstructedPlane, trace_track: List[FaceTrace]) -> float:
    """복원된 평면이 실제 입력 trace 들과 얼마나 거리가 먼지 오차(Tolerance) 평가. 미구현 시 NaN 허용"""
    return np.nan
