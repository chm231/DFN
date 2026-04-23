import numpy as np
from typing import List, Optional
from .data_models import Trace, CensoringType

def detect_trace_censoring(trace: Trace, face_polygon: Optional[np.ndarray], tolerance: float = 0.1) -> Trace:
    """
    트레이스의 끝점이 굴착면 경계에 닿아 있는지 확인하여 Censoring 상태 부여.
    
    Args:
        trace: 대상 트레이스 객체
        face_polygon: 굴착면 외곽 폴리곤 좌표 (N, 3)
        tolerance: 경계로 간주할 임계 거리 (m)
    """
    if face_polygon is None or len(face_polygon) == 0:
        trace.censoring = CensoringType.UNKNOWN
        return trace
    
    pts = trace.endpoints_3d
    is_clipped = [False, False]
    
    for i in range(2):
        pt = pts[i]
        # 폴리곤의 각 변과의 거리 중 최소값 찾기
        min_dist = float('inf')
        for j in range(len(face_polygon)):
            p1 = face_polygon[j]
            p2 = face_polygon[(j + 1) % len(face_polygon)]
            
            # 점과 선분(p1-p2) 사이의 거리 계산
            dist = _point_to_segment_distance(pt, p1, p2)
            min_dist = min(min_dist, dist)
            
        if min_dist <= tolerance:
            is_clipped[i] = True
            
    # 상태 분류
    num_clipped = sum(is_clipped)
    if num_clipped == 0:
        trace.censoring = CensoringType.VISIBLE
    elif num_clipped == 1:
        trace.censoring = CensoringType.ONE_END_CLIPPED
    else:
        trace.censoring = CensoringType.BOTH_END_CLIPPED
        
    return trace

def _point_to_segment_distance(p, a, b):
    """3D 공간에서 점 p와 선분 ab 사이의 최단 거리 계산"""
    ab = b - a
    ap = p - a
    if np.allclose(a, b):
        return np.linalg.norm(ap)
    
    t = np.dot(ap, ab) / np.dot(ab, ab)
    t = max(0, min(1, t))
    closest = a + t * ab
    return np.linalg.norm(p - closest)

def compute_trace_properties(trace: Trace) -> Trace:
    """선분의 중점, 길이, 단위 방향 벡터를 계산하여 업데이트"""
    pts = trace.endpoints_3d
    v = pts[1] - pts[0]
    trace.length = np.linalg.norm(v)
    if trace.length > 0:
        trace.direction = v / trace.length
    else:
        trace.direction = np.array([0.0, 0.0, 0.0])
    
    trace.midpoint_3d = np.mean(pts, axis=0)
    return trace

def preprocess_traces(traces: List[Trace], min_length: float = 0.1) -> List[Trace]:
    """선분 리스트 전처리 및 노이즈 필터링"""
    processed = []
    for t in traces:
        t = compute_trace_properties(t)
        if t.length >= min_length:
            processed.append(t)
    
    print(f" [Preprocess] {len(traces)} traces -> {len(processed)} active traces (min_len={min_length}m)")
    return processed

def merge_collinear_traces(traces: List[Trace], dist_tol: float = 0.05, angle_tol_deg: float = 5.0) -> List[Trace]:
    """동일 평면상에서 거의 일직선상에 있는 파편화된 선분들을 병합 (TODO: Implementation)"""
    # 초기 버전에서는 병합 없이 필터링만 수행
    return traces
