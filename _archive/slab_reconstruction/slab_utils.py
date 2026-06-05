from __future__ import annotations
import os
import sys
import numpy as np
from typing import List, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from .slab_trace_bridge import SlabTrace3D
from shapely.geometry import LineString, Polygon

# Import from parent packages
_here = os.path.dirname(os.path.abspath(__file__))
_parent = os.path.dirname(os.path.dirname(_here)) # dfn_project root
_core_path = os.path.join(_parent, "dfn_analysis")

if _core_path not in sys.path:
    sys.path.insert(0, _core_path)

from .slab_types import Slab, LocalCandidate

def clip_line_to_polygon(p0, p1, poly_yz) -> List[Tuple[np.ndarray, np.ndarray]]:
    """터널 단면 폴리곤에 대해 2D 선분 클리핑 (Shapely)"""
    if poly_yz is None or len(poly_yz) < 3:
        return [ (p0, p1) ]
        
    poly = Polygon(poly_yz)
    line = LineString([p0, p1])
    
    if not poly.intersects(line):
        return []
        
    intersection = poly.intersection(line)
    
    segments = []
    if intersection.geom_type == 'LineString':
        coords = list(intersection.coords)
        if len(coords) >= 2:
            segments.append((np.array(coords[0]), np.array(coords[-1])))
    elif intersection.geom_type == 'MultiLineString':
        geoms = getattr(intersection, 'geoms', [])
        for geom in geoms:
            coords = list(geom.coords)
            if len(coords) >= 2:
                segments.append((np.array(coords[0]), np.array(coords[-1])))
    return segments

def extract_slab_points_from_truth(
    centers: np.ndarray, 
    normals: np.ndarray, 
    radii: np.ndarray,
    slab: Slab,
    tunnel_poly_yz: np.ndarray | None,
    sub_slice_count: int = 5
) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
    """
    Slab 두께 내에서 여러 개의 X-단면(sub-slices)을 샘플링하여 
    Trace 상의 포인트들과 선분들을 수집합니다.
    
    Returns:
        points: (N, 3) array [x, y, z]
        truth_ids: (N,) original fracture indexes
        segments: List of (2, 3) arrays, each representing a 3D line segment
    """
    all_pts = []
    all_ids = []
    all_segs = []
    
    x_steps = np.linspace(slab.x_min, slab.x_max, sub_slice_count)
    
    Cx = centers[:, 0]
    Nx = normals[:, 0]
    
    # 디스크 평면이 막장면과 거의 평행한 경우 제외
    valid_mask = np.abs(Nx) < 0.9999
    sin_alpha_all = np.sqrt(1.0 - Nx**2)
    
    # 사전 필터링
    x_overlap = (Cx - radii <= slab.x_max) & (Cx + radii >= slab.x_min)
    potential_indices = np.where(valid_mask & x_overlap)[0]
    
    if len(potential_indices) == 0:
        return np.empty((0, 3)), np.empty((0,), dtype=int), []
        
    p_centers = centers[potential_indices]
    p_normals = normals[potential_indices]
    p_radii = radii[potential_indices]
    p_sin_alpha = sin_alpha_all[potential_indices]
    
    for x_pos in x_steps:
        dx = x_pos - p_centers[:, 0]
        d = np.abs(dx) / (p_sin_alpha + 1e-12)
        
        intersect_mask = (d <= p_radii)
        valid_indices = np.where(intersect_mask)[0]
        
        for v_idx in valid_indices:
            idx = potential_indices[v_idx]
            C = p_centers[v_idx]
            N = p_normals[v_idx]
            R = p_radii[v_idx]
            sa = p_sin_alpha[v_idx]
            
            Ly, Lz = -N[2] / sa, N[1] / sa
            t = (x_pos - C[0]) / (1.0 - N[0]**2)
            Pc_y = C[1] + t * (-N[0] * N[1])
            Pc_z = C[2] + t * (-N[0] * N[2])
            
            h = np.sqrt(R**2 - d[v_idx]**2)
            p0 = np.array([Pc_y + h * Ly, Pc_z + h * Lz])
            p1 = np.array([Pc_y - h * Ly, Pc_z - h * Lz])
            
            clipped = clip_line_to_polygon(p0, p1, tunnel_poly_yz)
            
            for seg_p0, seg_p1 in clipped:
                s3d_0 = np.array([x_pos, seg_p0[0], seg_p0[1]])
                s3d_1 = np.array([x_pos, seg_p1[0], seg_p1[1]])
                all_segs.append(np.array([s3d_0, s3d_1]))
                
                # 포인트 샘플링
                pts = [seg_p0, (seg_p0 + seg_p1)/2, seg_p1]
                for p_yz in pts:
                    all_pts.append([x_pos, p_yz[0], p_yz[1]])
                    all_ids.append(idx)
                    
    if not all_pts:
        return np.empty((0, 3)), np.empty((0,), dtype=int), []
        
    return np.array(all_pts), np.array(all_ids), all_segs

def extract_slab_segments_from_truth(
    centers: np.ndarray, 
    normals: np.ndarray, 
    radii: np.ndarray,
    slab: Slab,
    tunnel_poly_yz: np.ndarray | None,
    sub_slice_count: int = 5
) -> List['SlabTrace3D']:
    """
    Slab 내부의 X-단면(sub-slices)에서 균열면과의 3D 교차선분(Traces)들을 SlabTrace3D 객체 리스트로 추출합니다.
    """
    from .slab_trace_bridge import SlabTrace3D
    
    traces_3d = []
    seg_idx = 0
    
    x_steps = np.linspace(slab.x_min, slab.x_max, sub_slice_count)
    Cx = centers[:, 0]
    Nx = normals[:, 0]
    
    valid_mask = np.abs(Nx) < 0.9999
    sin_alpha_all = np.sqrt(1.0 - Nx**2)
    
    x_overlap = (Cx - radii <= slab.x_max) & (Cx + radii >= slab.x_min)
    potential_indices = np.where(valid_mask & x_overlap)[0]
    
    if len(potential_indices) == 0:
        return []
        
    p_centers = centers[potential_indices]
    p_normals = normals[potential_indices]
    p_radii = radii[potential_indices]
    p_sin_alpha = sin_alpha_all[potential_indices]
    
    for x_pos in x_steps:
        dx = x_pos - p_centers[:, 0]
        d = np.abs(dx) / (p_sin_alpha + 1e-12)
        
        intersect_mask = (d <= p_radii)
        valid_indices = np.where(intersect_mask)[0]
        
        for v_idx in valid_indices:
            idx = potential_indices[v_idx]
            C = p_centers[v_idx]
            N = p_normals[v_idx]
            R = p_radii[v_idx]
            sa = p_sin_alpha[v_idx]
            
            Ly, Lz = -N[2] / sa, N[1] / sa
            t = (x_pos - C[0]) / (1.0 - N[0]**2)
            Pc_y = C[1] + t * (-N[0] * N[1])
            Pc_z = C[2] + t * (-N[0] * N[2])
            
            h = np.sqrt(R**2 - d[v_idx]**2)
            p0 = np.array([Pc_y + h * Ly, Pc_z + h * Lz])
            p1 = np.array([Pc_y - h * Ly, Pc_z - h * Lz])
            
            clipped = clip_line_to_polygon(p0, p1, tunnel_poly_yz)
            
            for seg_p0, seg_p1 in clipped:
                s3d_0 = np.array([x_pos, seg_p0[0], seg_p0[1]])
                s3d_1 = np.array([x_pos, seg_p1[0], seg_p1[1]])
                
                traces_3d.append(SlabTrace3D(
                    segment_id=seg_idx,
                    p0=s3d_0,
                    p1=s3d_1,
                    parent_id=idx
                ))
                seg_idx += 1
                
    return traces_3d

