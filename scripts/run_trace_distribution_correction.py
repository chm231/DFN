"""
Runner script for the TBTD Estimator (Tunnel-window Bias-corrected Trace
Distribution Estimator).

Usage:
    python scripts/run_trace_distribution_correction.py \
        --input storage/data/dfn_export_for_python.h5 \
        --tunnel-dat storage/data/단면_폴리곤.dat \
        --x-start 0 --x-end 9 --advance-step 3 \
        --output-dir trace_analysis/storage/output/tbtd_results

If no real data is available, the script generates a synthetic toy example.
"""
import os
import sys
import argparse
import numpy as np

# Resolve project paths
_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_here)
sys.path.insert(0, os.path.join(_root, 'trace_analysis'))
sys.path.insert(0, _root)


def generate_synthetic_example(output_dir: str):
    """Generate a synthetic tunnel + trace dataset for demonstration."""
    from trace_distribution_correction import (
        TraceRecord, run_tbtd_pipeline
    )

    print("\n[TBTD Runner] No real data provided -- generating synthetic example.\n")

    # Synthetic circular tunnel polygon (approximated as 36-gon, diameter=10m)
    n_pts = 36
    angles = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)
    radius = 5.0
    poly_y = radius * np.cos(angles)
    poly_z = radius * np.sin(angles)
    window_polygon = np.column_stack([poly_y, poly_z])

    # Generate synthetic true fracture traces
    rng = np.random.default_rng(123)
    n_true = 500

    # True lengths from lognormal distribution
    true_lengths = rng.lognormal(mean=0.8, sigma=0.5, size=n_true)
    true_angles = rng.uniform(0, np.pi, size=n_true)

    # Place midpoints randomly inside the tunnel face
    records = []
    from trace_reconstruction.forward_simulator import (
        clip_line_segment_to_polygon, is_point_inside_polygon
    )

    n_observed = 0
    for i in range(n_true):
        L = true_lengths[i]
        theta = true_angles[i]
        half = L / 2.0

        # Random midpoint in bounding box
        my = rng.uniform(-5.5, 5.5)
        mz = rng.uniform(-5.5, 5.5)

        p0 = np.array([my - half * np.cos(theta), mz - half * np.sin(theta)])
        p1 = np.array([my + half * np.cos(theta), mz + half * np.sin(theta)])

        clipped = clip_line_segment_to_polygon(p0, p1, window_polygon)
        for cp0, cp1 in clipped:
            obs_len = np.linalg.norm(cp1 - cp0)
            if obs_len < 0.15:
                continue

            from trace_distribution_correction import (
                _point_to_polygon_distance, compute_trace_angle
            )
            eps = 0.10
            t0_near = _point_to_polygon_distance(cp0, window_polygon) <= eps
            t1_near = _point_to_polygon_distance(cp1, window_polygon) <= eps

            if t0_near and t1_near:
                ct = 'both_end_clipped'
            elif t0_near or t1_near:
                ct = 'one_end_clipped'
            else:
                ct = 'complete'

            angle = np.arctan2(cp1[1] - cp0[1], cp1[0] - cp0[0]) % np.pi

            records.append(TraceRecord(
                p0=cp0, p1=cp1,
                observed_length=obs_len,
                observed_angle=angle,
                censoring_type=ct,
                face_id=1,
            ))
            n_observed += 1

    print(f"  Synthetic traces generated: {n_true} true, {n_observed} observed")

    result = run_tbtd_pipeline(
        traces=records,
        window_polygon=window_polygon,
        output_dir=output_dir,
        l_min=0.15,
        length_bin_max=12.0,
        n_length_bins=24,
        n_angle_bins=6,
        n_mc=5000,
        seed=42,
        prefix='synthetic_',
    )
    return result


def run_real_data(args):
    """Load real tunnel data and run TBTD pipeline."""
    import h5py
    from load_tunnel_dat import load_tunnel_polygon_from_dat
    from trace_reconstruction.trace_types import ExcavationFace, FaceTrace
    from trace_reconstruction.trace_preprocessor import classify_censoring as classify_cens_existing
    from trace_reconstruction.forward_simulator import intersect_disc_with_face
    from trace_distribution_correction import run_tbtd_pipeline

    print(f"\n[TBTD Runner] Loading real data...")
    print(f"  HDF5 input  : {args.input}")
    print(f"  Tunnel DAT  : {args.tunnel_dat}")

    # Load tunnel polygon
    poly_y, poly_z = load_tunnel_polygon_from_dat(args.tunnel_dat)
    poly_yz = np.column_stack([poly_y, poly_z])

    # Load DFN
    with h5py.File(args.input, 'r') as f:
        raw_c = f['/fractures/centers'][:]
        raw_n = f['/fractures/normals'][:]
        gt_radii = f['/fractures/radii'][:].ravel()
        gt_set_id = (f['/fractures/set_id'][:].ravel() if '/fractures/set_id' in f
                     else np.ones(len(gt_radii), dtype=np.uint16))
        gt_centers = raw_c.T if raw_c.shape[0] == 3 and raw_c.shape[0] < raw_c.shape[1] else raw_c
        gt_normals = raw_n.T if raw_n.shape[0] == 3 and raw_n.shape[0] < raw_n.shape[1] else raw_n

    print(f"  DFN fractures: {len(gt_radii):,}")

    # Build faces
    x_positions = np.arange(args.x_start, args.x_end + 1e-5, args.advance_step)
    faces = []
    for i, xp in enumerate(x_positions):
        faces.append(ExcavationFace(
            face_id=i + 1, x_face=float(xp),
            tunnel_polygon_yz=poly_yz,
            advance_step=args.advance_step if i > 0 else 0.0
        ))
    print(f"  Faces: {len(faces)} at x = {[f.x_face for f in faces]}")

    # Extract traces
    print(f"  Extracting 2D traces from 3D DFN...")
    obs_traces = []
    tid = 1
    for face in faces:
        for i in range(len(gt_radii)):
            ft = intersect_disc_with_face(
                gt_centers[i, 0], gt_centers[i, 1], gt_centers[i, 2],
                gt_normals[i, 0], gt_normals[i, 1], gt_normals[i, 2],
                gt_radii[i], face, start_trace_id=tid, set_id=int(gt_set_id[i]),
                parent_fracture_id=i
            )
            obs_traces.extend(ft)
            tid += len(ft)
    print(f"  Total observed traces: {len(obs_traces)}")

    # Classify censoring (existing method)
    for face in faces:
        classify_cens_existing(obs_traces, face, tolerance=0.10)

    # Filter by l_min
    obs_traces = [t for t in obs_traces if t.length >= args.l_min]
    print(f"  After truncation filter (l_min={args.l_min}m): {len(obs_traces)}")

    # Run TBTD pipeline
    result = run_tbtd_pipeline(
        traces=obs_traces,
        window_polygon=poly_yz,
        output_dir=args.output_dir,
        l_min=args.l_min,
        length_bin_max=args.length_bin_max,
        n_length_bins=args.n_length_bins,
        n_angle_bins=args.n_angle_bins,
        n_mc=args.n_mc,
        seed=42,
        prefix='real_',
    )
    return result


def main():
    parser = argparse.ArgumentParser(
        description='TBTD Estimator — Tunnel-window Bias-corrected Trace Distribution')
    parser.add_argument('--input', default=None,
                        help='Path to ground truth HDF5 DFN file')
    parser.add_argument('--tunnel-dat', default=None,
                        help='Path to tunnel polygon .dat file')
    parser.add_argument('--x-start', type=float, default=0.0)
    parser.add_argument('--x-end', type=float, default=9.0)
    parser.add_argument('--advance-step', type=float, default=3.0)
    parser.add_argument('--output-dir',
                        default='trace_analysis/storage/output/tbtd_results')
    parser.add_argument('--l-min', type=float, default=0.15,
                        help='Minimum detectable trace length (m)')
    parser.add_argument('--length-bin-max', type=float, default=12.0)
    parser.add_argument('--n-length-bins', type=int, default=24)
    parser.add_argument('--n-angle-bins', type=int, default=6)
    parser.add_argument('--n-mc', type=int, default=5000,
                        help='Monte Carlo samples per (L, theta) cell')
    parser.add_argument('--synthetic', action='store_true',
                        help='Force synthetic example even if data paths given')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.synthetic or args.input is None or args.tunnel_dat is None:
        # Check if default data paths exist
        default_h5 = os.path.join(_root, 'storage', 'data',
                                  'dfn_export_for_python.h5')
        default_dat = os.path.join(_root, 'storage', 'data', '단면_폴리곤.dat')

        if (not args.synthetic and args.input is None
                and os.path.exists(default_h5) and os.path.exists(default_dat)):
            print("[TBTD Runner] Found default data files, using real data.")
            args.input = default_h5
            args.tunnel_dat = default_dat
            run_real_data(args)
        else:
            generate_synthetic_example(args.output_dir)
    else:
        run_real_data(args)

    print(f"\n[TBTD Runner] All outputs saved to: {args.output_dir}")
    print("[TBTD Runner] Done.")


if __name__ == '__main__':
    main()
