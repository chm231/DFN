"""Verification script for dfnrec pipeline using actual/ground-truth tunnel face trace data."""
from __future__ import annotations

import os
import sys
import math
import numpy as np
import pandas as pd

# Set paths for local imports
_here = os.path.dirname(os.path.abspath(__file__))
_parent = os.path.dirname(_here)
sys.path.insert(0, _parent)

from dfnrec.models import (
    Face,
    Trace,
    CensorType,
    DomainGeometry,
    DomainModel,
)
from dfnrec.pipeline import run_pipeline


def classify_censoring(pt_uv: np.ndarray, poly_uv: np.ndarray, tol: float = 0.05) -> CensorType:
    """Classify endpoint censoring based on distance to the tunnel boundary polygon."""
    min_dist = float("inf")
    for i in range(len(poly_uv)):
        p_start = poly_uv[i]
        p_end = poly_uv[(i + 1) % len(poly_uv)]
        edge = p_end - p_start
        edge_len = np.linalg.norm(edge)
        if edge_len < 1e-12:
            dist = np.linalg.norm(pt_uv - p_start)
        else:
            t = np.clip(np.dot(pt_uv - p_start, edge) / (edge_len**2), 0.0, 1.0)
            proj = p_start + t * edge
            dist = np.linalg.norm(pt_uv - proj)
        if dist < min_dist:
            min_dist = dist
    return CensorType.CLIPPED if min_dist < tol else CensorType.NATURAL


def main():
    print("=" * 80)
    print(" Running DFN Reconstruction & Parameter Inversion Verification")
    print("=" * 80)

    # 1. Load data
    traces_csv_path = os.path.join(_parent, "storage", "output", "ground_truth_traces.csv")
    tunnel_csv_path = os.path.join(_parent, "storage", "output", "tunnel_polygon.csv")

    if not os.path.exists(traces_csv_path):
        print(f"[ERROR] Traces file not found: {traces_csv_path}")
        sys.exit(1)
    if not os.path.exists(tunnel_csv_path):
        print(f"[ERROR] Tunnel polygon file not found: {tunnel_csv_path}")
        sys.exit(1)

    print(f"[*] Loading traces from: {traces_csv_path}")
    df_traces = pd.read_csv(traces_csv_path)
    print(f"[*] Loading tunnel polygon from: {tunnel_csv_path}")
    df_tunnel = pd.read_csv(tunnel_csv_path)

    # Clean tunnel polygon points
    poly_uv = df_tunnel[["y", "z"]].values
    print(f"  -> Tunnel boundary nodes: {len(poly_uv)}")

    # 2. Setup Faces
    unique_face_ids = sorted(df_traces["face_id"].unique())
    faces = []
    
    # Calculate face positions
    face_x_coords = {}
    for f_id in unique_face_ids:
        # Mean x position of traces belonging to this face
        mean_x = float(df_traces[df_traces["face_id"] == f_id]["x_face"].mean())
        face_x_coords[f_id] = mean_x

    for idx, f_id in enumerate(unique_face_ids):
        x_pos = face_x_coords[f_id]
        faces.append(
            Face(
                face_id=str(f_id),
                order_index=idx,
                origin_xyz=[x_pos, 0.0, 0.0],
                normal_xyz=[1.0, 0.0, 0.0],
                axis_u_xyz=[0.0, 1.0, 0.0],
                axis_v_xyz=[0.0, 0.0, 1.0],
                observation_window_polygon_uv=poly_uv.tolist(),
                L_min=0.1,
            )
        )
    print(f"  -> Configured {len(faces)} faces: {[f.face_id for f in faces]}")
    for f in faces:
        print(f"     - Face {f.face_id}: x_face = {f.origin_xyz[0]:.2f}m")

    # 3. Setup Traces
    traces = []
    for _, row in df_traces.iterrows():
        p0_xyz = [float(row["p0_x"]), float(row["p0_y"]), float(row["p0_z"])]
        p1_xyz = [float(row["p1_x"]), float(row["p1_y"]), float(row["p1_z"])]
        
        # Determine censoring
        p0_uv = np.array([row["p0_y"], row["p0_z"]])
        p1_uv = np.array([row["p1_y"], row["p1_z"]])
        c0 = classify_censoring(p0_uv, poly_uv, tol=0.05)
        c1 = classify_censoring(p1_uv, poly_uv, tol=0.05)

        # Set IDs in ground truth CSV are integers, convert to set name like 'S1', 'S2', etc.
        set_id = f"S{int(row['set_id'])}"

        traces.append(
            Trace(
                trace_id=f"T{int(row['trace_id'])}",
                face_id=str(int(row["face_id"])),
                set_id=set_id,
                p0_xyz=p0_xyz,
                p1_xyz=p1_xyz,
                censor_p0=c0,
                censor_p1=c1,
                measurement_sigma=0.02,
            )
        )
    print(f"  -> Converted {len(traces)} traces with censoring classification.")

    # 4. Setup Domain Geometry
    # Define bounding box enclosing the tunnel face sequence with buffers
    x_min = min(face_x_coords.values()) - 2.0
    x_max = max(face_x_coords.values()) + 6.0
    y_min = float(np.min(poly_uv[:, 0])) - 2.0
    y_max = float(np.max(poly_uv[:, 0])) + 2.0
    z_min = float(np.min(poly_uv[:, 1])) - 2.0
    z_max = float(np.max(poly_uv[:, 1])) + 2.0

    domain_geom = DomainGeometry(
        x_min=x_min, x_max=x_max,
        y_min=y_min, y_max=y_max,
        z_min=z_min, z_max=z_max,
    )
    print(f"[*] Domain Box: x=[{x_min:.2f}, {x_max:.2f}], y=[{y_min:.2f}, {y_max:.2f}], z=[{z_min:.2f}, {z_max:.2f}]")
    print(f"  -> Volume: {domain_geom.volume_m3():.2f} m³")

    # 5. Run the Pipeline
    print("[*] Running run_pipeline...")
    # Using low SVD log BF threshold to allow robust track association
    domain_model = run_pipeline(
        faces=faces,
        traces=traces,
        domain_geom=domain_geom,
        seed=100,
        log_bf_threshold=-15.0,
        min_faces=1,
        r_min=0.5,
        r_max=15.0,
    )

    # 6. Print Report
    print("\n" + "=" * 50)
    print(" VERIFICATION PIPELINE REPORT")
    print("=" * 50)
    
    dp = domain_model.dfn_params
    print("[1] INVERTED DFN PARAMETERS")
    for sid in dp.set_ids():
        ori = dp.orientation[sid]
        si = dp.size_intensity[sid]
        print(f"  * Set {sid}:")
        print(f"    - Pole Orientation: Trend = {ori.mean_trend_deg:.2f} deg, Plunge = {ori.mean_plunge_deg:.2f} deg (kappa = {ori.kappa:.2f})")
        print(f"    - Radius CCDF Exponent k_r: {si.k_r:.3f} (alpha = {si.k_r+1:.3f})")
        print(f"    - Target total P32: {si.P32_total:.4f} m2/m3")
        print(f"    - Target total P30/n0: {si.n0:.6f} m-3")

    print("\n[2] COMPOSED DOMAIN FRACTURES")
    print(f"  * Reconstructed Observed Discs (Hard): {len(domain_model.observed_discs)}")
    print(f"  * Generated Hidden Fractures (Stochastic): {len(domain_model.hidden_fractures)}")
    print(f"  * Total domain fractures: {domain_model.all_fracture_count()}")

    print("\n[3] P32 INTENSITY COMPARISON")
    diag = domain_model.diagnostics
    for sid in sorted(diag.p32_target.keys()):
        target = diag.p32_target[sid]
        apparent = diag.p32_apparent[sid]
        err = diag.p32_relative_error[sid]
        print(f"  * Set {sid}: Target P32 = {target:.4f} | Apparent P32 = {apparent:.4f} | Rel Error = {err*100:+.1f}%")

    if diag.warnings:
        print("\n[4] DIAGNOSTIC WARNINGS")
        for w in diag.warnings:
            print(f"  [WARNING] {w}")

    # 7. Save results
    output_path = os.path.join(_parent, "storage", "output", "final_conditioned_domain.json")
    print(f"\n[*] Saving composed DomainModel to: {output_path}")
    with open(output_path, "w", encoding="utf-8") as f_out:
        f_out.write(domain_model.to_json(indent=2))
    print("[Done] Verification complete!")


if __name__ == "__main__":
    main()
