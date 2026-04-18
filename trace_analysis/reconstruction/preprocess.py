import numpy as np
from typing import List
from .data_models import Trace

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
