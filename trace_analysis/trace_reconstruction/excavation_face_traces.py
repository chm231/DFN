"""
[Direction B: Inverse Reconstruction]
터널 단면에서 DFN 데이터(Truth)를 이용하여 관측 가능한 Trace 선분들을 추출합니다.
"""

import numpy as np
from typing import List, Tuple
from shapely.geometry import LineString, Polygon
from .trace_types import ExcavationFace, FaceTrace, CensoringType

def detect_trace_censoring_baseline(
    p0: Tuple[float, float], p1: Tuple[float, float], 
    tunnel_poly_yz: np.ndarray, tolerance: float = 0.05
) -> CensoringType:
    """터널 단면 경계와의 거리를 기반으로 Censoring 상태 판별"""
    if tunnel_poly_yz is None or len(tunnel_poly_yz) < 3:
        return CensoringType.UNKNOWN
        
    is_clipped = [False, False]
    pts = [np.array(p0), np.array(p1)]
    
    for i in range(2):
        pt = pts[i]
        min_dist = float('inf')
        for j in range(len(tunnel_poly_yz)):
            p_a = tunnel_poly_yz[j]
            p_b = tunnel_poly_yz[(j + 1) % len(tunnel_poly_yz)]
            
            # 2D 점-선분 거리
            dist = _point_to_segment_distance_2d(pt, p_a, p_b)
            min_dist = min(min_dist, dist)
        
        if min_dist <= tolerance:
            is_clipped[i] = True
            
    num_clipped = sum(is_clipped)
    if num_clipped == 0: return CensoringType.VISIBLE
    if num_clipped == 1: return CensoringType.ONE_END_CLIPPED
    return CensoringType.BOTH_END_CLIPPED

def _point_to_segment_distance_2d(p, a, b):
    """2D 공간에서 점 p와 선분 ab 사이의 최단 거리 계산"""
    ab = b - a
    ap = p - a
    if np.allclose(a, b): return np.linalg.norm(ap)
    
    t = np.dot(ap, ab) / np.dot(ab, ab)
    t = max(0, min(1, t))
    closest = a + t * ab
    return np.linalg.norm(p - closest)

def clip_trace_to_tunnel_polygon(
    trace_line_yz: Tuple[Tuple[float, float], Tuple[float, float]], 
    tunnel_poly_yz: np.ndarray
) -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
    """
    무한/전체 원판 길이의 2D 교선을 터널 다각형(polygon) 내부 구간으로 클리핑합니다.
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
    주어진 원본 DFN 정보에서 특정 막장면(face.x_face)과 교차하는 
    Fracture trace들을 추출하고, 터널 단면 내부로 클리핑하여 반환.
    """
    traces = []
    Xf = face.x_face
    
    if centers.shape[0] == 3 and centers.shape[0] < centers.shape[1]: centers = centers.T
    if normals.shape[0] == 3 and normals.shape[0] < normals.shape[1]: normals = normals.T

    # 막장면(YZ plane)과 교차하는 원판 선별 조건:
    # 중심 좌표 x_c 와 막장 x_f 의 거리가 원판 반지름 radius 보다 커야 함
    # (실제 거리는 법선 벡터 각도에 의해 추가 보정됨)
    Cx = centers[:, 0]
    Nx = normals[:, 0]
    
    # 평행인 경우 교차하지 않음
    valid_mask = np.abs(Nx) < 0.9999
    
    # 3D 거리 보정 (막장면 평면과 평행하지 않은 정도 고려)
    dx = Xf - Cx
    sin_alpha = np.sqrt(1.0 - Nx**2)
    d = np.abs(dx) / (sin_alpha + 1e-12)
    
    intersect_mask = valid_mask & (d <= radii)
    valid_indices = np.where(intersect_mask)[0]
    
    trace_id_counter = 1
    tunnel_poly_yz = face.tunnel_polygon_yz
    has_tunnel = tunnel_poly_yz is not None and len(tunnel_poly_yz) > 2
    
    for idx in valid_indices:
        C = centers[idx]
        N = normals[idx]
        R = radii[idx]
        
        # 교선의 2D (YZ) 방향 벡터 (식별자: Ly, Lz)
        sa = sin_alpha[idx]
        Ly, Lz = -N[2] / sa, N[1] / sa
        
        # 원판 중심에서 교선까지의 YZ 평면 상 투영 중심점 계산
        t = (Xf - C[0]) / (1.0 - N[0]**2)
        Pc_y = C[1] + t * (-N[0] * N[1])
        Pc_z = C[2] + t * (-N[0] * N[2])
        
        # 교선의 절반 길이 계산 h
        h = np.sqrt(R**2 - d[idx]**2)
        
        # 전체 교선(Unclipped) 양 끝점 (YZ 좌표)
        p0_y, p0_z = Pc_y + h * Ly, Pc_z + h * Lz
        p1_y, p1_z = Pc_y - h * Ly, Pc_z - h * Lz
        tmp_line = ((p0_y, p0_z), (p1_y, p1_z))
        
        # 터널 다각형 내부로 클리핑
        if has_tunnel:
            clipped_segments = clip_trace_to_tunnel_polygon(tmp_line, tunnel_poly_yz)
        else:
            clipped_segments = [tmp_line] # 전체 도메인 테스트용

        for pt0, pt1 in clipped_segments:
            censoring_val = detect_trace_censoring_baseline(pt0, pt1, tunnel_poly_yz)
            traces.append(FaceTrace(
                face_id=face.face_id,
                trace_id=trace_id_counter,
                x_face=Xf,
                p0_y=float(pt0[0]), p0_z=float(pt0[1]),
                p1_y=float(pt1[0]), p1_z=float(pt1[1]),
                censoring=censoring_val
            ))
            trace_id_counter += 1
            
    return traces
