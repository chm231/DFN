"""
[Phase 8: DFN Exporter]
Exports deterministic planes, single-face candidates, and stochastic fractures to a structured HDF5 file
fully compatible with the downstream GPU-accelerated 3D block-detector pipeline.
"""
import h5py
import numpy as np
from typing import List, Dict, Tuple
from .trace_types import ReconstructedPlane, StochasticFracture


def export_dfn_to_hdf5(
    file_path: str,
    det_planes: List[ReconstructedPlane],
    single_face_candidates: List[ReconstructedPlane],
    stoch_fractures: List[StochasticFracture],
    tunnel_poly_yz: np.ndarray,
    domain_box: np.ndarray,  # [xmin, xmax, ymin, ymax, zmin, zmax]
    x_start: float = 0.0,
    x_end: float = 6.0
):
    """
    Writes all fractures into a unified HDF5 file.
    Combines:
    1. Deterministic Multi-Face Planes
    2. Probabilistic Single-Face Candidates
    3. Stochastic PPP Fractures
    """
    centers = []
    normals = []
    radii = []
    set_ids = []
    sources = []  # 1: deterministic, 2: single-face candidate, 3: stochastic
    
    # 1. Deterministic Multi-Face Planes
    for p in det_planes:
        centers.append([p.point_x, p.point_y, p.point_z])
        normals.append([p.normal_x, p.normal_y, p.normal_z])
        radii.append(p.radius)
        set_ids.append(p.set_id or 1)
        sources.append(1)
        
    # 2. Single-Face Probabilistic Candidates
    for p in single_face_candidates:
        centers.append([p.point_x, p.point_y, p.point_z])
        normals.append([p.normal_x, p.normal_y, p.normal_z])
        radii.append(p.radius)
        set_ids.append(p.set_id or 1)
        sources.append(2)
        
    # 3. Stochastic PPP Fractures
    for sf in stoch_fractures:
        centers.append([sf.center_x, sf.center_y, sf.center_z])
        normals.append([sf.normal_x, sf.normal_y, sf.normal_z])
        radii.append(sf.radius)
        set_ids.append(sf.set_id)
        sources.append(3)
        
    centers = np.array(centers, dtype=np.float32)
    normals = np.array(normals, dtype=np.float32)
    radii = np.array(radii, dtype=np.float32)
    set_ids = np.array(set_ids, dtype=np.uint16)
    sources = np.array(sources, dtype=np.uint8)
    
    # Write to HDF5
    with h5py.File(file_path, 'w') as f:
        # Create groups
        grp_frac = f.create_group('fractures')
        grp_frac.create_dataset('centers', data=centers.T, compression='gzip') # Transposed to support Matlab/Python compatibility
        grp_frac.create_dataset('normals', data=normals.T, compression='gzip')
        grp_frac.create_dataset('radii', data=radii, compression='gzip')
        grp_frac.create_dataset('set_id', data=set_ids, compression='gzip')
        grp_frac.create_dataset('source_type', data=sources, compression='gzip')
        
        grp_tunnel = f.create_group('tunnel')
        grp_tunnel.create_dataset('poly_YZ', data=tunnel_poly_yz.T, compression='gzip')
        
        grp_meta = f.create_group('meta')
        grp_meta.create_dataset('domain_box', data=domain_box)
        grp_meta.create_dataset('crop_box', data=domain_box)
        grp_meta.create_dataset('x_start', data=x_start)
        grp_meta.create_dataset('x_end', data=x_end)
        
        # Write dataset attributes
        f.attrs['n_deterministic'] = len(det_planes)
        f.attrs['n_single_face'] = len(single_face_candidates)
        f.attrs['n_stochastic'] = len(stoch_fractures)
        f.attrs['n_total'] = len(radii)
        
    print(f"\n  [HDF5 Export Complete] Saved {len(radii)} fractures to: {file_path}")
    print(f"    - Multi-face Deterministic: {len(det_planes)}")
    print(f"    - Single-face Candidates  : {len(single_face_candidates)}")
    print(f"    - Volumetric Stochastic   : {len(stoch_fractures)}")
