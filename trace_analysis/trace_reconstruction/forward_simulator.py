"""
[Phase 7: Forward Simulator]
Implements Poisson Point Process stochastic DFN generation, spherical VMF orientation sampling,
exact analytical 3D disc to tunnel face line intersection, and polygon segment clipping.
"""
import numpy as np
from typing import List, Tuple, Optional, Dict
from .trace_types import FaceTrace, ExcavationFace, StochasticFracture


def is_point_inside_polygon(y: float, z: float, poly: np.ndarray) -> bool:
    """
    Ray-casting algorithm to determine if point (y, z) is inside a closed polygon.
    """
    n = len(poly)
    inside = False
    p1y, p1z = poly[0]
    for i in range(n + 1):
        p2y, p2z = poly[i % n]
        if z > min(p1z, p2z):
            if z <= max(p1z, p2z):
                if y <= max(p1y, p2y):
                    if p1z != p2z:
                        xinters = (z - p1z) * (p2y - p1y) / (p2z - p1z) + p1y
                    if p1y == p2y or y <= xinters:
                        inside = not inside
        p1y, p1z = p2y, p2z
    return inside


def clip_line_segment_to_polygon(
    p0: np.ndarray,
    p1: np.ndarray,
    poly: np.ndarray
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Clips a 2D line segment (p0, p1) to a closed polygon.
    Returns list of clipped segment endpoints [(sub_p0, sub_p1), ...] lying inside the polygon.
    """
    if len(poly) == 0:
        return []
        
    v_dir = p1 - p0
    v_len = np.linalg.norm(v_dir)
    if v_len < 1e-12:
        if is_point_inside_polygon(p0[0], p0[1], poly):
            return [(p0, p1)]
        return []
        
    v_u = v_dir / v_len
    
    # Collect all intersection t-values (0.0 <= t <= 1.0)
    t_vals = [0.0, 1.0]
    n_points = len(poly)
    
    for i in range(n_points):
        v0 = poly[i]
        v1 = poly[(i + 1) % n_points]
        
        # Intersect segment (p0 + t*v_dir) with polygon edge (v0 + s*(v1-v0))
        d_edge = v1 - v0
        denom = v_dir[0] * d_edge[1] - v_dir[1] * d_edge[0]
        
        if abs(denom) < 1e-12:
            continue
            
        t = (v0[0] * d_edge[1] - v0[1] * d_edge[0] - p0[0] * d_edge[1] + p0[1] * d_edge[0]) / denom
        s = (p0[0] * v_dir[1] - p0[1] * v_dir[0] - v0[0] * v_dir[1] + v0[1] * v_dir[0]) / (-denom)
        
        if 0.0 <= t <= 1.0 and 0.0 <= s <= 1.0:
            t_vals.append(t)
            
    # Sort t-values and check sub-segments
    t_vals = sorted(list(set(t_vals)))
    clipped_segments = []
    
    for i in range(len(t_vals) - 1):
        t_start, t_end = t_vals[i], t_vals[i + 1]
        if t_end - t_start < 1e-5:
            continue
            
        mid_t = 0.5 * (t_start + t_end)
        mid_pt = p0 + mid_t * v_dir
        
        if is_point_inside_polygon(mid_pt[0], mid_pt[1], poly):
            sub_p0 = p0 + t_start * v_dir
            sub_p1 = p0 + t_end * v_dir
            clipped_segments.append((sub_p0, sub_p1))
            
    return clipped_segments


def sample_vmf_normals(mean_normal: np.ndarray, kappa: float, n_samples: int = 1, seed: Optional[int] = None) -> np.ndarray:
    """Samples unit normals from a 3D von Mises-Fisher distribution."""
    rng = np.random.default_rng(seed)
    if abs(mean_normal[2]) < 0.9:
        ref = np.array([0.0, 0.0, 1.0])
    else:
        ref = np.array([1.0, 0.0, 0.0])
        
    e_u = np.cross(mean_normal, ref)
    e_u = e_u / np.linalg.norm(e_u)
    e_v = np.cross(mean_normal, e_u)
    e_v = e_v / np.linalg.norm(e_v)
    
    stdev = 1.0 / np.sqrt(kappa)
    thetas = rng.normal(0, stdev, n_samples)
    phis = rng.uniform(0, 2 * np.pi, n_samples)
    
    samples = []
    for theta, phi in zip(thetas, phis):
        n = mean_normal + theta * (np.cos(phi) * e_u + np.sin(phi) * e_v)
        n = n / np.linalg.norm(n)
        if n[0] < 0:
            n = -n
        samples.append(n)
        
    return np.array(samples)


def generate_stochastic_dfn(
    domain: Dict[str, float],  # {'xmin', 'xmax', 'ymin', 'ymax', 'zmin', 'zmax'}
    set_params: Dict[int, Dict[str, float]],  # set_id -> {'mu_s', 'sigma_s', 'P30'}
    set_stats: Dict[int, Tuple[np.ndarray, float]],  # set_id -> (mean_normal, kappa)
    start_id: int = 2000,
    seed: Optional[int] = 42
) -> List[StochasticFracture]:
    """
    Generates a stochastic 3D DFN using a Poisson Point Process.
    """
    rng = np.random.default_rng(seed)
    stoch_fractures = []
    fid = start_id
    
    vol = (domain['xmax'] - domain['xmin']) * (domain['ymax'] - domain['ymin']) * (domain['zmax'] - domain['zmin'])
    
    for set_id, params in set_params.items():
        P30 = params['P30']
        mu_s = params['mu_s']
        sigma_s = params['sigma_s']
        mean_normal, kappa = set_stats[set_id]
        
        # Poisson point count N ~ Poisson(P30 * Volume)
        n_frac = rng.poisson(P30 * vol)
        if n_frac <= 0:
            continue
            
        # Sample uniform centers
        cx = rng.uniform(domain['xmin'], domain['xmax'], n_frac)
        cy = rng.uniform(domain['ymin'], domain['ymax'], n_frac)
        cz = rng.uniform(domain['zmin'], domain['zmax'], n_frac)
        
        # Sample lognormal radii
        radii = rng.lognormal(mu_s, sigma_s, n_frac)
        
        # Sample VMF normals with derived seed
        set_seed = int(rng.integers(0, 1000000)) if seed is not None else None
        normals = sample_vmf_normals(mean_normal, kappa, n_frac, seed=set_seed)
        
        for i in range(n_frac):
            stoch_fractures.append(StochasticFracture(
                fracture_id=fid,
                center_x=float(cx[i]),
                center_y=float(cy[i]),
                center_z=float(cz[i]),
                normal_x=float(normals[i, 0]),
                normal_y=float(normals[i, 1]),
                normal_z=float(normals[i, 2]),
                radius=float(radii[i]),
                set_id=set_id
            ))
            fid += 1
            
    return stoch_fractures


def intersect_disc_with_face(
    c_x: float, c_y: float, c_z: float,
    n_x: float, n_y: float, n_z: float,
    radius: float,
    face: ExcavationFace,
    start_trace_id: int = 5000,
    set_id: int = 1
) -> List[FaceTrace]:
    """
    Analytically intersects a 3D disc with an excavation face plane (x = x_face),
    clipping the intersection segment to the tunnel polygon boundary.
    """
    x_f = face.x_face
    poly = face.tunnel_polygon_yz
    
    # 1. Plane intersection line equation in YZ: Ny * y + Nz * z = C_rhs
    ny, nz = n_y, n_z
    ny_z_sq = ny**2 + nz**2
    
    if ny_z_sq < 1e-12:
        return []  # Fracture plane is parallel to face
        
    C_rhs = n_x * (c_x - x_f) + ny * c_y + nz * c_z
    
    # 2. Shortest 3D distance from disc center to face intersection line
    # d = |x_face - c_x| / sqrt(ny^2 + nz^2)
    dist_to_line = abs(x_f - c_x) / np.sqrt(ny_z_sq)
    
    if dist_to_line >= radius:
        return []  # No intersection
        
    # 3. Chord midpoint in YZ plane
    factor = (ny * c_y + nz * c_z - C_rhs) / ny_z_sq
    y_mid = c_y - ny * factor
    z_mid = c_z - nz * factor
    mid_pt = np.array([y_mid, z_mid])
    
    # 4. Endpoints of full intersection chord
    chord_half_len = np.sqrt(radius**2 - dist_to_line**2)
    # Unit direction vector of the intersection line in YZ plane
    d_line = np.array([-nz, ny]) / np.sqrt(ny_z_sq)
    
    p0 = mid_pt - chord_half_len * d_line
    p1 = mid_pt + chord_half_len * d_line
    
    # 5. Clip segment to tunnel polygon boundary
    clipped = clip_line_segment_to_polygon(p0, p1, poly)
    
    traces = []
    tid = start_trace_id
    for cp0, cp1 in clipped:
        t = FaceTrace(
            face_id=face.face_id,
            trace_id=tid,
            x_face=x_f,
            p0_y=float(cp0[0]),
            p0_z=float(cp0[1]),
            p1_y=float(cp1[0]),
            p1_z=float(cp1[1]),
            confidence=1.0
        )
        t.set_id = set_id
        traces.append(t)
        tid += 1
        
    return traces
