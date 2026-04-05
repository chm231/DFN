"""
[Direction B: Inverse Reconstruction]
연속된 굴착 막장면(face n-1, face n) 상에서 관측된 Face Trace들을 엮는 매칭 모듈입니다.
추출된 trace를 이용해 "이 trace가 이전 면에서 본 그 절리의 trace인가?"를 Heuristic으로 판단합니다.
"""
import numpy as np
from typing import List, Dict
from .trace_types import FaceTrace, TraceMatch

def compute_trace_match_score(trace_a: FaceTrace, trace_b: FaceTrace, params: dict = None) -> float:
    """
    Face n-1의 Trace A와 Face n의 Trace B의 유사성을 점수화 (낮을수록 일치 확률 높음/Cost 기반)
    고려요소: 중심점 거리 차이, 스케일(길이) 변동성, 2D 회전(방향) 변위.
    """
    dy = trace_a.midpoint_y - trace_b.midpoint_y
    dz = trace_a.midpoint_z - trace_b.midpoint_z
    dist_sq = dy**2 + dz**2
    
    len_diff = abs(trace_a.length - trace_b.length)
    ang_diff = abs(np.sin(trace_a.local_orientation_2d - trace_b.local_orientation_2d))
    
    # Heuristic cost (파라미터화 필요)
    cost = dist_sq + (len_diff**2 * 0.5) + (ang_diff * 1.0)
    return float(cost)


def match_traces_between_faces(
    face_prev_traces: List[FaceTrace], 
    face_curr_traces: List[FaceTrace], 
    params: dict = None
) -> List[TraceMatch]:
    """
    이전 막장면과 현재 막장면의 trace 집합을 서로 매칭(예: nearest neighbor / bipartite).
    기준 임계치(tol)를 넘어가는 경우 false로 둠.
    """
    matches = []
    if not face_prev_traces or not face_curr_traces:
        return matches
        
    for idx_b, tb in enumerate(face_curr_traces):
        best_score = float('inf')
        best_ta = None
        
        for ta in face_prev_traces:
            score = compute_trace_match_score(ta, tb, params)
            if score < best_score:
                best_score = score
                best_ta = ta
                
        # threshold(비용 제한) 통과 여부 검사 (임시값 5.0)
        accepted = True if best_score < 10.0 else False
        if best_ta is not None:
             matches.append(TraceMatch(
                 face_id_prev=best_ta.face_id,
                 face_id_curr=tb.face_id,
                 trace_id_prev=best_ta.trace_id,
                 trace_id_curr=tb.trace_id,
                 score=best_score,
                 accepted=accepted
             ))
             
    return matches

def build_trace_tracks(grouped_traces: Dict[int, List[FaceTrace]], params: dict = None) -> List[List[FaceTrace]]:
    """
    매칭 정보를 종합하여, face 0 부터 n까지 연속적으로 이어지는 
    trace들의 궤적(Trace Track) 리스트를 구성하여 반환합니다.
    이 궤적 1개가 1개의 3D 절리면(Plane) 재구성 후보가 됨.
    """
    # TODO: 연결 리스트 형태의 궤적 빌더 구현
    return []
