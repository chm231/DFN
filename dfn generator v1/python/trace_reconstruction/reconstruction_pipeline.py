"""
[Direction B: Inverse Reconstruction]
발굴면 Trace 집합을 바탕으로 Inverse 방향 전체 처리를 일괄 수행하는 파이프라인.
"""
from typing import List
from .trace_types import FaceTrace, ReconstructedPlane, ReconstructedBlock
from .face_trace_io import load_face_traces, group_traces_by_face
from .trace_matching import match_traces_between_faces, build_trace_tracks
from .plane_reconstruction import fit_plane_from_trace_track
from .block_polyhedron import extract_reconstructed_blocks_voxel

def run_inverse_pipeline(
    csv_path: str, tunnel_poly_yz, start_x: float, end_x: float, params: dict = None
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
    # 최소 3장 이상의 터널 막장면에서 연속적으로 발견된 Trace들만 유효한 역산 후보로 삼습니다.
    valid_tracks = build_trace_tracks(grouped, params, min_faces=3)
            
    print(f" -> 검출된 연속 Trace 매칭 궤적(Tracks >= 3 faces) 수: {len(valid_tracks)}")
    
    # 3. 3D 절리면 복원
    reconstructed_planes = []
    pid = 1
    for track in valid_tracks:
        plane = fit_plane_from_trace_track(track, pid)
        if plane is not None:
            reconstructed_planes.append(plane)
            pid += 1
            
    print(f" -> 3차원 공간 복원 평면 수: {len(reconstructed_planes)}")
    
    # 4. 블록 추출 (Voxel GPU Engine 하이브리드 연동)
    # Reconstructed Discs를 이용해 실제 블록들을 뽑아냅니다.
    block_kwargs = params.get('block_kwargs', {}) if params else {}
    
    blocks, labels, grid_info = extract_reconstructed_blocks_voxel(
        reconstructed_planes, tunnel_poly_yz,
        start_x=start_x, end_x=end_x,
        voxel_size=block_kwargs.get('voxel_size', 0.5),
        halo_dist=block_kwargs.get('halo_dist', 3.0),
        tol_factor=block_kwargs.get('tol_factor', 0.6),
        connectivity=block_kwargs.get('connectivity', 26),
        min_voxels=block_kwargs.get('min_voxels', 8)
    )
    
    return reconstructed_planes, blocks, labels, grid_info
