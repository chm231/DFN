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

def build_trace_tracks(grouped_traces: Dict[int, List[FaceTrace]], params: dict = None, min_faces: int = 3) -> List[List[FaceTrace]]:
    """
    매칭 정보를 종합하여, face 0 부터 n까지 연속적으로 이어지는 
    trace들의 궤적(Trace Track) 리스트를 구성하여 반환합니다.
    이 궤적 1개가 1개의 3D 절리면(Plane) 재구성 후보가 됨.
    최소 min_faces 이상 꼬리를 물고 이어진 트랙만 유효함!
    """
    face_ids = sorted(grouped_traces.keys())
    if len(face_ids) < 2:
        return []
        
    finished_tracks = []
    # active_tracks: List[List[FaceTrace]]
    active_tracks = []
    
    for i in range(1, len(face_ids)):
        prev_traces = grouped_traces[face_ids[i - 1]]
        curr_traces = grouped_traces[face_ids[i]]
        
        matches = match_traces_between_faces(prev_traces, curr_traces, params)
        
        # 이전 트레이스 ID -> 매칭 정보 딕셔너리 구성
        match_dict = {}
        for m in matches:
            if m.accepted:
                match_dict[m.trace_id_prev] = m
                
        new_active_tracks = []
        
        # 1. 생존 중인 트랙들 연장(Extension)
        for track in active_tracks:
            last_trace = track[-1]
            if last_trace.trace_id in match_dict:
                # 꼬리물기 연장 성공
                m = match_dict[last_trace.trace_id]
                t_curr = next(t for t in curr_traces if t.trace_id == m.trace_id_curr)
                track.append(t_curr)
                new_active_tracks.append(track)
                # 처리된 매칭은 목록에서 제거
                del match_dict[last_trace.trace_id]
            else:
                # 더 이상 연결되지 못해 궤적 종료
                finished_tracks.append(track)
                
        # 2. 이번 단계에서 새로 탄생하는 궤적들 시작
        for prev_id, m in match_dict.items():
            t_prev = next(t for t in prev_traces if t.trace_id == m.trace_id_prev)
            t_curr = next(t for t in curr_traces if t.trace_id == m.trace_id_curr)
            new_active_tracks.append([t_prev, t_curr])
            
        active_tracks = new_active_tracks
        
    # 루프가 끝나고 남은 진행 중인 트랙들은 모두 완료 처리
    finished_tracks.extend(active_tracks)
    
    # 3. 최소 막장면 수(min_faces) 기준 미달 트랙 폐기
    valid_tracks = [t for t in finished_tracks if len(t) >= min_faces]
    
    return valid_tracks
