"""
[Direction B: Inverse Reconstruction]
Synthetic Benchmark (Direction A)에서 알려진 3D DFN 정보를 바탕으로 
B 방향 연구를 위한 "가상의 굴착 막장면 Trace 데이터"를 역추출하는 전용 모듈입니다.

매우 중요한 철학: 
- 이 모듈이 출력하는 trace 데이터는 무조건 터널 폴리곤 안쪽으로 'Clip' 되어야 합니다.
- Crop Box 전체 교차선이 아닙니다.
"""
import numpy as np
from typing import List, Tuple
from shapely.geometry import LineString, Polygon
from .trace_types import ExcavationFace, FaceTrace

def clip_trace_to_tunnel_polygon(
    trace_line_yz: Tuple[Tuple[float, float], Tuple[float, float]], 
    tunnel_poly_yz: np.ndarray
) -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
    """
    무한/전체 단면 절리 교차선을 터널 단면(Polygon)에 대해 교집합 클리핑합니다.
    Shapely를 내부적으로 활용하여 터널 단면 내부에서만 존재하는 segment 리스트 반환
    """
    poly = Polygon(tunnel_poly_yz)
    line = LineString(trace_line_yz)
    
    if not poly.intersects(line):
        return []
    
    intersection = poly.intersection(line)
    
    segments = []
    if intersection.geom_type == 'LineString':
        coords = list(intersection.coords)
        if len(coords) >= 2:
            segments.append(((coords[0][0], coords[0][1]), (coords[-1][0], coords[-1][1])))
    elif intersection.geom_type == 'MultiLineString':
        for geom in intersection.geoms:
            coords = list(geom.coords)
            if len(coords) >= 2:
                segments.append(((coords[0][0], coords[0][1]), (coords[-1][0], coords[-1][1])))
    
    return segments


def extract_excavation_face_traces_from_truth(
    centers: np.ndarray, normals: np.ndarray, radii: np.ndarray,
    face: ExcavationFace
) -> List[FaceTrace]:
    """
    [A 방향(Ground Truth) -> B 방향(Synthetic Traces) 변환기]
    주어진 face의 x 좌표 단면과 만나는 Fracture(디스크)를 수학적으로 교차하여 구한 뒤,
    반드시 터널 내부 영역으로 잘라내어(clip) FaceTrace 구조체로 만들어 반환합니다.
    """
    traces = []
    
    Xf = face.x_face
    Cx = centers[:, 0]
    Nx = normals[:, 0]
    
    # 디스크 평면이 막장면과 거의 평행한 경우(수직 교차 불가) 제외
    valid_mask = np.abs(Nx) < 0.9999
    
    # 1. 막장면과 디스크 중심 사이의 X축 거리
    dx = Xf - Cx
    
    # 2. 디스크 평면 상에서 직선까지의 실제 최단 거리 d
    # d = |dx| / sin(alpha) = |dx| / sqrt(1 - Nx^2)
    sin_alpha = np.sqrt(1.0 - Nx**2)
    d = np.abs(dx) / (sin_alpha + 1e-12)
    
    # 교차 발생 조건: 중심거리가 반지름보다 작거나 같은 디스크만 유효
    intersect_mask = valid_mask & (d <= radii)
    
    valid_indices = np.where(intersect_mask)[0]
    
    trace_id_counter = 1
    tunnel_poly_yz = face.tunnel_polygon_yz
    has_tunnel = tunnel_poly_yz is not None and len(tunnel_poly_yz) > 2
    
    for idx in valid_indices:
        C = centers[idx]
        N = normals[idx]
        R = radii[idx]
        sa = sin_alpha[idx]
        
        # 교차선 방향 (Disc Normal과 X축(막장면 Normal)의 외적)
        # L = (1,0,0) x (Nx, Ny, Nz) = (0, -Nz, Ny)
        Ly, Lz = -N[2] / sa, N[1] / sa
        
        # 교차선의 중심점(Chord Center) 도출
        t = (Xf - C[0]) / (1.0 - N[0]**2)
        Pc_y = C[1] + t * (-N[0] * N[1])
        Pc_z = C[2] + t * (-N[0] * N[2])
        
        # Chord 절반 길이 (피타고라스)
        h = np.sqrt(R**2 - d[idx]**2)
        
        # 3D 공간 상의 교차선 양 끝점 (Y, Z 성분만 취함)
        p0_y = Pc_y + h * Ly
        p0_z = Pc_z + h * Lz
        p1_y = Pc_y - h * Ly
        p1_z = Pc_z - h * Lz
        
        tmp_line = ((p0_y, p0_z), (p1_y, p1_z))
        
        # 터널 단면 polygon 클리핑 수행
        if has_tunnel:
            clipped_segments = clip_trace_to_tunnel_polygon(tmp_line, tunnel_poly_yz)
        else:
            clipped_segments = [tmp_line] # 전체 도메인 테스트용

        for pt0, pt1 in clipped_segments:
            traces.append(FaceTrace(
                face_id=face.face_id,
                trace_id=trace_id_counter,
                x_face=Xf,
                p0_y=float(pt0[0]), p0_z=float(pt0[1]),
                p1_y=float(pt1[0]), p1_z=float(pt1[1])
            ))
            trace_id_counter += 1
            
    return traces
