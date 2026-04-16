"""
[Direction B: Inverse Reconstruction]
복원된 3D 평면 원판(Discs)들을 GPU Voxel Engine으로 전송하여 수학적 블록 판별을 수행하는 연동 모듈입니다.
"""
import os
import sys
import numpy as np
from typing import List
from .trace_types import ReconstructedPlane

def extract_reconstructed_blocks_voxel(
    planes: List[ReconstructedPlane], 
    tunnel_poly_yz: np.ndarray,
    start_x: float,
    end_x: float,
    voxel_size: float = 0.5,
    halo_dist: float = 3.0,
    tol_factor: float = 0.6,
    connectivity: int = 26,
    min_voxels: int = 8
) -> tuple:
    """
    복원된 3D Planes를 바탕으로 A 방향의 Voxel Engine 구동하여 블록 추출
    """
    if not planes or tunnel_poly_yz is None:
        return []
        
    # 부모 모듈 로드 설정 (공통 엔진 참조)
    _here = os.path.dirname(os.path.abspath(__file__))
    _parent = os.path.dirname(os.path.dirname(_here))
    _core_path = os.path.join(_parent, "dfn_analysis")
    
    if _core_path not in sys.path:
        sys.path.insert(0, _core_path)
        
    try:
        from tunnel_geometry import build_voxel_masks
        from block_detector import classify_voxels, run_cca, filter_and_stat_blocks
    except ImportError as e:
        print(f"[Error] Failed to import core engine from {_core_path}: {e}")
        return []

    print("\n" + "="*50)
    print(" [INFO] [Voxel Hybrid Engine] 역산 원판 기반 블록 판별 시작")
    print("="*50)
    
    # 1. 포맷 변환: Dataclass -> Numpy Arrays (float32)
    N = len(planes)
    centers = np.zeros((N, 3), dtype=np.float32)
    normals = np.zeros((N, 3), dtype=np.float32)
    radii = np.zeros(N, dtype=np.float32)
    
    for i, p in enumerate(planes):
        centers[i] = [p.point_x, p.point_y, p.point_z]
        normals[i] = [p.normal_x, p.normal_y, p.normal_z]
        radii[i] = p.radius
        
    # 2. 도메인 박스 구성
    ymin, ymax = np.min(tunnel_poly_yz[:, 0]), np.max(tunnel_poly_yz[:, 0])
    zmin, zmax = np.min(tunnel_poly_yz[:, 1]), np.max(tunnel_poly_yz[:, 1])
    margin = halo_dist + voxel_size * 2
    domain_box = np.array([
        start_x, end_x,
        ymin - margin, ymax + margin,
        zmin - margin, zmax + margin
    ])
    
    # 3. 마스크 생성 
    print(f" -> 도메인 구성 및 터널 마스크 생성 중...")
    poly_Y = tunnel_poly_yz[:, 0]
    poly_Z = tunnel_poly_yz[:, 1]
    
    voxel_centers, tunnel_mask, halo_mask, grid_info = build_voxel_masks(
        poly_Y, poly_Z, domain_box=domain_box, voxel_size=voxel_size, halo_dist=halo_dist,
        tunnel_xmin=start_x, tunnel_xmax=end_x
    )
    
    # CuPy 배열인 경우 classify_voxels (NumPy 기반)를 위해 변환
    if 'cupy' in str(type(tunnel_mask)):
        import cupy as cp
        tunnel_mask = cp.asnumpy(tunnel_mask)
        halo_mask = cp.asnumpy(halo_mask)
        
    # 4. 복셀 상태 분류 (ROCK, FRACTURE, TUNNEL)
    print(f" -> {N}개의 역산 디스크(Discs)를 복셀 파이프라인에 주입 중...")
    state, fracture_owner = classify_voxels(grid_info, centers, normals, radii, tunnel_mask, tol_factor=tol_factor)
    
    # 5. Connected Component Analysis
    labels, n_labels = run_cca(state, connectivity=connectivity)
    
    # 6. 블록 필터링 (CuPy 배열인 경우 NumPy로 변환)
    print(f" -> 형성된 역산 블록 후보들의 물리적 타당성(낙반 여부) 평가 중...")
    
    if 'cupy' in str(type(state)):
        import cupy as cp
        state_np = cp.asnumpy(state)
        labels_np = cp.asnumpy(labels)
    else:
        state_np = state
        labels_np = labels
        
    block_info = filter_and_stat_blocks(
        labels_np, n_labels, state_np, grid_info,
        min_voxels=min_voxels, connectivity=connectivity
    )
    
    print(f"\n  [INFO] Voxel Hybrid Engine 판별 완료: 최종 {len(block_info)}개의 블록 도출됨!")
    return block_info, labels_np, grid_info
