import numpy as np
from typing import List, Dict
from .trace_types import FaceTrace, TraceMatch, CensoringType

def compute_trace_match_score(trace_a: FaceTrace, trace_b: FaceTrace, params: dict = None) -> float:
    """
    Face n-1의 Trace A와 Face n의 Trace B의 유사성을 점수화 (낮을수록 일치 확률 높음)
    Zhang & Einstein (2000) 권고에 의거:
    - 2D Trace 길이는 Bias가 크므로 매칭 비중을 낮춤.
    - 방향성(Orientation) 및 기하학적 연속성(Midpoint)을 핵심 지표로 사용.
    - Censoring 상태에 따른 가중치 시스템 도입.
    """
    # 1. 기하학적 중심 거리 비용 (Y, Z 평면)
    dy = trace_a.midpoint_y - trace_b.midpoint_y
    dz = trace_a.midpoint_z - trace_b.midpoint_z
    dist_cost = (dy**2 + dz**2) * 2.0  # 위치 연속성 가중치
    
    # 2. 방향성(Orientation) 일치도 비용 - 매우 중요
    # sin 차이를 사용하여 각도 차이가 클수록 급격하게 페널티 부여
    ang_diff = abs(np.sin(trace_a.local_orientation_2d - trace_b.local_orientation_2d))
    ang_cost = ang_diff * 10.0  # 방향 일치도 가중치 대폭 강화
    
    # 3. 길이 차이 비용 (축소 - Clipping에 의한 편향 때문)
    # 절대 차이 대신 비율 차이를 사용해 스케일 영향 최소화
    len_diff_ratio = abs(trace_a.length - trace_b.length) / (max(trace_a.length, trace_b.length) + 1e-6)
    len_cost = len_diff_ratio * 1.0  
    
    # 4. Censoring 데이터 품질 보정
    # 둘 다 온전하게 보이는(Visible) 경우 매칭 신뢰도 대폭 상승 (Cost 차감)
    quality_bonus = 0.0
    if trace_a.censoring == CensoringType.VISIBLE and trace_b.censoring == CensoringType.VISIBLE:
        quality_bonus = -1.0  # Quality Bonus
    
    # 둘 중 하나라도 양 끝이 잘린 경우(Both-end clipped) 불확실성 페널티 부여
    elif (trace_a.censoring == CensoringType.BOTH_END_CLIPPED or 
          trace_b.censoring == CensoringType.BOTH_END_CLIPPED):
        quality_bonus = 2.0   # Uncertainty Penalty

    cost = dist_cost + ang_cost + len_cost + quality_bonus
    return float(max(0.1, cost))

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
                
        # threshold(비용 제한) 통과 여부 검사 (임시값 5.0 -> 10.0)
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

def build_trace_tracks(grouped_traces: Dict[int, List[FaceTrace]], params: dict = None, min_faces: int = 2) -> List[List[FaceTrace]]:
    """
    여러 막장면에 걸쳐 연속적으로 매칭되는 플래그먼트들을 'Track'단위로 묶습니다.
    반환값: 각 트랙(리스트)들의 리스트
    """
    face_ids = sorted(grouped_traces.keys())
    if len(face_ids) < 2: return []
        
    finished_tracks = []
    active_tracks = []
    
    for i in range(1, len(face_ids)):
        prev_traces = grouped_traces[face_ids[i - 1]]
        curr_traces = grouped_traces[face_ids[i]]
        
        matches = match_traces_between_faces(prev_traces, curr_traces, params)
        
        match_dict = {}
        for m in matches:
            if m.accepted: match_dict[m.trace_id_prev] = m
                
        new_active_tracks = []
        
        # 이전 트랙 이어나가기
        for track in active_tracks:
            last_trace = track[-1]
            if last_trace.trace_id in match_dict:
                m = match_dict[last_trace.trace_id]
                t_curr = next(t for t in curr_traces if t.trace_id == m.trace_id_curr)
                track.append(t_curr)
                new_active_tracks.append(track)
                del match_dict[last_trace.trace_id]
            else:
                finished_tracks.append(track)
                
        # 새로 시작하는 트랙 (이전에 매칭 안된 애들 중 이번에 매칭 성공한 애들)
        for prev_id, m in match_dict.items():
            t_prev = next(t for t in prev_traces if t.trace_id == m.trace_id_prev)
            t_curr = next(t for t in curr_traces if t.trace_id == m.trace_id_curr)
            new_active_tracks.append([t_prev, t_curr])
            
        active_tracks = new_active_tracks
        
    finished_tracks.extend(active_tracks)
    
    # 최소 관측 횟수를 만족하는 트랙만 필터링
    valid_tracks = [t for t in finished_tracks if len(t) >= min_faces]
    return valid_tracks
