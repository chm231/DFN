import numpy as np
from typing import List, Optional
from .trace_types import FaceTrace, ReconstructedPlane, CensoringType

def lift_face_trace_to_3d(trace: FaceTrace) -> tuple:
    p0 = np.array([trace.x_face, trace.p0_y, trace.p0_z])
    p1 = np.array([trace.x_face, trace.p1_y, trace.p1_z])
    return p0, p1

def reconstruct_plane_from_trace_pair(trace_prev: FaceTrace, trace_curr: FaceTrace, plane_id: int) -> Optional[ReconstructedPlane]:
    pts_A = lift_face_trace_to_3d(trace_prev)
    pts_B = lift_face_trace_to_3d(trace_curr)
    points = np.vstack([pts_A, pts_B])
    
    centroid = np.mean(points, axis=0)
    centered = points - centroid
    _, s, vh = np.linalg.svd(centered)
    normal = vh[-1, :]
    
    if normal[0] < 0: normal = -normal
        
    dists = np.linalg.norm(centered, axis=1)
    est_radius = float(np.max(dists) * 1.5)
    if est_radius < 1.0: est_radius = 5.0

    plane = ReconstructedPlane(
        plane_id=plane_id,
        point_x=float(centroid[0]), point_y=float(centroid[1]), point_z=float(centroid[2]),
        normal_x=float(normal[0]), normal_y=float(normal[1]), normal_z=float(normal[2]),
        radius=est_radius,
        source_trace_ids=[trace_prev.trace_id, trace_curr.trace_id]
    )
    
    plane.confidence = evaluate_plane_fit(plane, [trace_prev, trace_curr])
    return plane

def fit_plane_from_trace_track(trace_track: List[FaceTrace], plane_id: int) -> Optional[ReconstructedPlane]:
    if len(trace_track) < 2: return None
    if len(trace_track) == 2: return reconstruct_plane_from_trace_pair(trace_track[0], trace_track[1], plane_id)
        
    points = []
    for trace in trace_track:
        p0, p1 = lift_face_trace_to_3d(trace)
        points.extend([p0, p1])
    points = np.array(points)
    
    centroid = np.mean(points, axis=0)
    centered = points - centroid
    _, s, vh = np.linalg.svd(centered)
    normal = vh[-1, :]
    
    if normal[0] < 0: normal = -normal
        
    dists = np.linalg.norm(centered, axis=1)
    est_radius = float(np.max(dists) * 1.5)
    if est_radius < 1.0: est_radius = 5.0
        
    plane = ReconstructedPlane(
        plane_id=plane_id,
        point_x=float(centroid[0]), point_y=float(centroid[1]), point_z=float(centroid[2]),
        normal_x=float(normal[0]), normal_y=float(normal[1]), normal_z=float(normal[2]),
        radius=est_radius,
        source_trace_ids=[t.trace_id for t in trace_track]
    )
    
    plane.confidence = evaluate_plane_fit(plane, trace_track)
    return plane

def evaluate_plane_fit(plane: ReconstructedPlane, trace_track: List[FaceTrace]) -> float:
    if not trace_track: return 0.0

    points = []
    for trace in trace_track:
        p0, p1 = lift_face_trace_to_3d(trace)
        points.extend([p0, p1])
    points = np.array(points)
    
    p_origin = np.array([plane.point_x, plane.point_y, plane.point_z])
    p_normal = np.array([plane.normal_x, plane.normal_y, plane.normal_z])
    
    dists = np.abs(np.dot(points - p_origin, p_normal))
    rmse = np.sqrt(np.mean(dists**2))
    geo_score = np.exp(-rmse / 0.5) 

    q_weights = []
    for t in trace_track:
        if t.censoring == CensoringType.VISIBLE: q_weights.append(1.0)
        elif t.censoring == CensoringType.ONE_END_CLIPPED: q_weights.append(0.7)
        elif t.censoring == CensoringType.BOTH_END_CLIPPED: q_weights.append(0.4)
        else: q_weights.append(0.5)
    
    quality_score = np.mean(q_weights)
    final_conf = geo_score * quality_score
    return float(final_conf)
