"""
[Direction B: Inverse Reconstruction]
발굴면 Trace 집합을 바탕으로 Inverse 방향 전체 처리를 일괄 수행하는 파이프라인.
"""
from typing import List
from .trace_types import FaceTrace, ReconstructedPlane, ReconstructedBlock
from .face_trace_io import load_face_traces, group_traces_by_face
from .trace_matching import match_traces_between_faces, build_trace_tracks
from .plane_reconstruction import fit_plane_from_trace_track

def run_inverse_pipeline(
    csv_path: str, tunnel_poly_yz, params: dict = None
) -> tuple:
    """
    B 방향 파이프라인 전체 단계를 조율.
    1. CSV 로드
    2. Face 매칭 및 Track 구성
    3. Plane 복원
    4. Polyhedron 생성
    """
    print(f"\n[Reconstruction] Inverse Pipeline 시작")
    print(f" -> 입력 데이터: {csv_path}")
    
    # 1. 파일에서 Face 단위 내부 트레이스 로드
    all_traces = load_face_traces(csv_path)
    grouped = group_traces_by_face(all_traces)
    face_ids = sorted(grouped.keys())
    print(f" -> 인식된 굴착 막장면 수: {len(face_ids)} (총 {len(all_traces)} traces)")
    
    # 2. Trace 궤적 추적 (Matching)
    # 현재는 연속된 임의의 face들 간의 단순 리스트 생성으로 모사
    # TODO: 제대로 된 build_trace_tracks 사용
    dummy_tracks = []
    if len(face_ids) >= 2:
        prev = grouped[face_ids[0]]
        for i in range(1, len(face_ids)):
            curr = grouped[face_ids[i]]
            matches = match_traces_between_faces(prev, curr, params)
            # skeleton: 일대일 매칭이면 2개짜리 track으로 추가
            for m in matches:
                if m.accepted:
                    t_a = next(t for t in prev if t.trace_id == m.trace_id_prev)
                    t_b = next(t for t in curr if t.trace_id == m.trace_id_curr)
                    dummy_tracks.append([t_a, t_b])
            prev = curr
            
    print(f" -> 검출된 연속 Trace 매칭 궤적(Tracks) 수: {len(dummy_tracks)}")
    
    # 3. 3D 절리면 복원
    reconstructed_planes = []
    pid = 1
    for track in dummy_tracks:
        plane = fit_plane_from_trace_track(track, pid)
        if plane is not None:
            reconstructed_planes.append(plane)
            pid += 1
            
    print(f" -> 3차원 공간 복원 평면 수: {len(reconstructed_planes)}")
    
    # 4. 블록 추출
    # from .block_polyhedron import extract_closed_block_candidates
    # blocks = extract_closed_block_candidates(reconstructed_planes, tunnel_poly_yz)
    blocks = []
    
    # TODO: CSV 일괄 Export
    
    return reconstructed_planes, blocks
