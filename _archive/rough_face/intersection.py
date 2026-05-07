import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from skimage import measure
from matplotlib.path import Path
from .generator import RoughFace

def extract_rough_traces(
    fracture_data: Dict[str, np.ndarray],
    rough_face: RoughFace,
    tunnel_poly_yz: Optional[np.ndarray] = None
) -> List[Dict[str, Any]]:
    """
    각 균열(Fracture)과 비평면 굴착면(RoughFace)의 교차선을 추출합니다.
    """
    centers = fracture_data['centers']
    normals = fracture_data['normals']
    radii = fracture_data['radii']
    
    # 터널 폴리곤 경로 생성 (클리핑용)
    tunnel_path = None
    if tunnel_poly_yz is not None:
        tunnel_path = Path(tunnel_poly_yz)
        
    all_traces = []
    
    # 0. 사전 필터링 (Bounding Box 기반)
    # 굴착면의 대략적인 범위 계산
    x_min_face, x_max_face = rough_face.base_x, rough_face.base_x + rough_face.amplitude
    y_min_face, y_max_face = rough_face.y_range
    z_min_face, z_max_face = rough_face.z_range
    
    # 균열의 Bounding Box가 굴착면 영역과 겹치는지 체크
    x_overlap = (centers[:, 0] - radii <= x_max_face) & (centers[:, 0] + radii >= x_min_face)
    y_overlap = (centers[:, 1] - radii <= y_max_face) & (centers[:, 1] + radii >= y_min_face)
    z_overlap = (centers[:, 2] - radii <= z_max_face) & (centers[:, 2] + radii >= z_min_face)
    
    potential_indices = np.where(x_overlap & y_overlap & z_overlap)[0]
    
    # 굴착면의 모든 점(3D) 준비
    # shape: (NZ, NY, 3)
    face_pts_3d = np.stack((rough_face.X, rough_face.Y, rough_face.Z), axis=-1)
    
    for i in potential_indices:
        C = centers[i]
        N = normals[i]
        R = radii[i]
        
        # 1. 굴착면의 각 점에서 균열 평면까지의 부호 있는 거리(Signed Distance) 계산
        # D = N . (P - C)
        # face_pts_3d - C: (NZ, NY, 3)
        diff = face_pts_3d - C
        dist_field = np.sum(diff * N, axis=-1)
        
        # 2. 등치선(Zero-crossing) 추출
        # find_contours는 그리드 인덱스 (row, col) 좌표를 반환함
        contours = measure.find_contours(dist_field, 0.0)
        
        for contour in contours:
            # contour shape: (P, 2) -> (row_idx, col_idx)
            # 선형 보간을 통해 정확한 Y, Z 좌표 계산
            # row -> Z, col -> Y
            
            # 인덱스를 실제 Y, Z 좌표로 변환
            cols = contour[:, 1]
            rows = contour[:, 0]
            
            y_coords = rough_face.y_range[0] + cols * rough_face.resolution
            z_coords = rough_face.z_range[0] + rows * rough_face.resolution
            
            # X 좌표는 굴착면(RoughFace) 높이 맵에서 보간 (여기서는 투영으로 단순 처리)
            # 좀 더 정확하려면 dist_field=0인 지점의 X를 찾아야 하지만, 
            # 해상도가 충분하다면 project_to_face로도 가깝게 도출됨.
            pts_yz = np.column_stack((y_coords, z_coords))
            pts_3d = rough_face.project_to_face(pts_yz)
            
            # 3. 클리핑 (디스크 내부 및 터널 내부)
            # 디스크 반지름 체크
            d_to_center = np.linalg.norm(pts_3d - C, axis=1)
            valid_mask = d_to_center <= R
            
            # 터널 단면 체크
            if tunnel_path is not None:
                in_tunnel = tunnel_path.contains_points(pts_yz)
                valid_mask = valid_mask & in_tunnel
            
            if not np.any(valid_mask):
                continue
                
            # 기록 (유효한 포인트들만)
            # 조각난 선분들이 생길 수 있으므로 연속된 부분만 추출하거나 통째로 관리
            # 여기서는 단순화를 위해 유효한 포인트들을 순서대로 담되, 끊긴 부분은 무시(또는 분리 가능)
            
            # 실제로는 valid_mask가 [F, T, T, F, T, T]처럼 나올 수 있음 -> split 필요
            sub_polylines = split_mask_to_polylines(pts_3d, valid_mask)
            
            for poly in sub_polylines:
                if len(poly) < 2: continue
                all_traces.append({
                    'fracture_id': i,
                    'points': poly,
                    'length': np.sum(np.linalg.norm(np.diff(poly, axis=0), axis=1))
                })
                
    return all_traces

def split_mask_to_polylines(points: np.ndarray, mask: np.ndarray) -> List[np.ndarray]:
    """
    마스크에 따라 연속된 유효 포인트들을 별개의 폴리라인으로 분리합니다.
    """
    polylines = []
    current_poly = []
    
    for i in range(len(mask)):
        if mask[i]:
            current_poly.append(points[i])
        else:
            if current_poly:
                polylines.append(np.array(current_poly))
                current_poly = []
    
    if current_poly:
        polylines.append(np.array(current_poly))
        
    return polylines
