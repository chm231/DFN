import argparse
import csv
import os
import time
from typing import Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np

from dfn_analysis.estimate_mean_orientation import (
    estimate_mean_normal_axial,
    normal_to_trend_plunge_ned,
)
from dfn_analysis.estimate_radius_powerlaw_window_mc import (
    determine_recovery_status,
    fit_set_lmin,
)
from dfn_analysis.export_setwise_3d_traces import (
    build_rows_rough_faces,
    load_hdf5_dfn,
    load_rough_face_collection_from_h5,
    precompute_face_mesh,
)

# Ground truth kr values for Forsmark and Laxemar
FORSMARK_KR_TRUE = {1: 2.88, 2: 3.02, 3: 2.81, 4: 2.95, 5: 2.92}
LAXEMAR_KR_TRUE = {1: 2.85, 2: 3.04, 3: 3.01, 5: 3.6}


def get_rotation_matrix(direction: str) -> np.ndarray:
    """
    Returns the rotation matrix R that rotates the DFN such that the physical tunnel
    direction (in the original coordinate system) aligns with the simulated tunnel (X-axis).
    That is, R * v_phys = [1, 0, 0]^T.
    """
    if direction == "x":
        return np.eye(3)
    elif direction == "y":
        # Physical tunnel is along Y-axis.
        # R rotates Y-axis to X-axis: rotate around Z by -90 degrees.
        return np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    elif direction == "z":
        # Physical tunnel is along Z-axis.
        # R rotates Z-axis to X-axis: rotate around Y by 90 degrees.
        return np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
    elif direction.startswith("azimuth_"):
        # Physical tunnel has azimuth phi (in degrees) in the XY plane, plunge 0.
        # R rotates [cos(phi), sin(phi), 0] to [1, 0, 0]: rotate around Z by -phi.
        phi_deg = float(direction.split("_")[1])
        phi = np.radians(phi_deg)
        return np.array(
            [[np.cos(phi), np.sin(phi), 0.0], [-np.sin(phi), np.cos(phi), 0.0], [0.0, 0.0, 1.0]]
        )
    else:
        raise ValueError(f"Unknown tunnel direction: {direction}")


def rotate_dfn(data: dict, R: np.ndarray) -> dict:
    rotated = data.copy()
    centers = data["centers"]
    # Rotate around the DFN centroid to keep it in the same spatial region
    centroid = np.mean(centers, axis=0)
    rotated["centers"] = (centers - centroid) @ R.T + centroid
    rotated["normals"] = data["normals"] @ R.T
    return rotated


def compute_trace_direction_concentration(rows: List[dict]) -> float:
    """
    Computes the maximum eigenvalue of the 2D second-order orientation tensor
    in the YZ plane. Values close to 1.0 indicate highly concentrated trace directions.
    """
    directions = []
    for r in rows:
        dy = r["p1_y"] - r["p0_y"]
        dz = r["p1_z"] - r["p0_z"]
        length = np.hypot(dy, dz)
        if length > 1e-8:
            directions.append([dy / length, dz / length])

    if len(directions) < 2:
        return 1.0

    T = np.zeros((2, 2))
    for d in directions:
        T += np.outer(d, d)
    T /= len(directions)
    eigvals = np.linalg.eigvalsh(T)
    return float(eigvals[1])  # Largest eigenvalue (since eigvalsh returns sorted)


def compute_face_concentration_cv(rows: List[dict], face_ids: List[int]) -> float:
    """
    Computes the Coefficient of Variation (CV = std / mean) of trace counts per face.
    Higher CV indicates that traces are concentrated on specific faces.
    """
    counts = {fid: 0 for fid in face_ids}
    for r in rows:
        counts[r["face_id"]] = counts.get(r["face_id"], 0) + 1

    count_vals = np.array(list(counts.values()), dtype=np.float64)
    mean_val = np.mean(count_vals)
    if mean_val <= 0.0:
        return 0.0
    return float(np.std(count_vals) / mean_val)


def run_diagnostics_for_set(
    set_id: int,
    set_rows: List[dict],
    data: dict,
    face_ids: List[int],
    kr_true: float,
    kr_hat: float,
) -> dict:
    """
    Runs detailed diagnostics for a single set.
    """
    # 1. Orientation factors
    set_indices = np.where(data["set_ids"] == set_id)[0]
    if len(set_indices) == 0:
        return {}

    set_normals = data["normals"][set_indices]
    # Face normal is along X-axis: [1, 0, 0]
    g_factors = np.sqrt(1.0 - set_normals[:, 0] ** 2)

    g_mean = float(np.mean(g_factors))
    g_min = float(np.min(g_factors))
    g_max = float(np.max(g_factors))

    # Mean pole trend & plunge
    mean_normal = estimate_mean_normal_axial(set_normals)
    if mean_normal is not None:
        trend, plunge = normal_to_trend_plunge_ned(mean_normal)
    else:
        trend, plunge = None, None

    # 2. Trace statistics
    n_traces = len(set_rows)
    unique_faces = len(set(r["face_id"] for r in set_rows))

    lengths = np.array([r["observed_length_m"] for r in set_rows])
    mean_len = float(np.mean(lengths)) if len(lengths) else 0.0
    p90_len = float(np.percentile(lengths, 90)) if len(lengths) else 0.0
    censored_count = sum(1 for r in set_rows if r["censoring_class"] > 0)
    censoring_ratio = censored_count / n_traces if n_traces > 0 else 0.0

    # 3. Concentration analysis
    trace_concentration = compute_trace_direction_concentration(set_rows)
    face_cv = compute_face_concentration_cv(set_rows, face_ids)

    # 4. Long traces analysis (top 10% longest)
    if len(lengths) >= 5:
        threshold = np.percentile(lengths, 90)
        long_rows = [r for r in set_rows if r["observed_length_m"] >= threshold]
        long_trace_concentration = compute_trace_direction_concentration(long_rows)
        long_face_cv = compute_face_concentration_cv(long_rows, face_ids)
    else:
        long_trace_concentration = 1.0
        long_face_cv = 0.0

    kr_error = kr_hat - kr_true
    bias_flag = abs(kr_error) > 0.3

    return {
        "set_id": set_id,
        "mean_pole_trend": trend,
        "mean_pole_plunge": plunge,
        "orientation_factor_mean": g_mean,
        "orientation_factor_min": g_min,
        "orientation_factor_max": g_max,
        "n_traces": n_traces,
        "n_faces_with_traces": unique_faces,
        "mean_trace_length": mean_len,
        "p90_trace_length": p90_len,
        "censoring_ratio": censoring_ratio,
        "trace_concentration_eigenvalue": trace_concentration,
        "face_concentration_cv": face_cv,
        "long_trace_concentration_eigenvalue": long_trace_concentration,
        "long_face_concentration_cv": long_face_cv,
        "kr_true": kr_true,
        "kr_window_mc_hat": kr_hat,
        "kr_error": kr_error,
        "bias_flag": bias_flag,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diagnose tunnel direction and orientation sampling bias on power-law estimation."
    )
    parser.add_argument(
        "--input", default="storage/data/dfn_export_for_python.h5", help="Input HDF5 DFN file"
    )
    parser.add_argument(
        "--rough-mesh-h5",
        default="storage/output/rough_face_mesh_collection/synthetic_rough_face_collection.h5",
        help="HDF5 containing rough face collection",
    )
    parser.add_argument(
        "--outdir",
        default="storage/output/tunnel_direction_bias",
        help="Output directory for diagnostics",
    )
    parser.add_argument("--rmin", type=float, default=0.5, help="Estimation rmin")
    parser.add_argument("--rmax", type=float, default=250.0, help="Estimation rmax")
    parser.add_argument("--lmin-fit", type=float, default=0.3, help="Lmin fit threshold")
    parser.add_argument(
        "--mc-samples", type=int, default=10000, help="Number of MC samples per grid point"
    )
    parser.add_argument(
        "--grid-size", type=int, default=41, help="Grid size for kr profile likelihood"
    )
    parser.add_argument(
        "--target-sets", nargs="+", type=int, default=[1, 2, 3, 5], help="Target set IDs to analyze"
    )
    parser.add_argument(
        "--site",
        choices=["forsmark", "laxemar"],
        default="forsmark",
        help="Site name (for kr_true lookup)",
    )
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    print(f"[*] Loading DFN from: {args.input}")
    data = load_hdf5_dfn(args.input)
    print(f"[*] Loading rough face meshes from: {args.rough_mesh_h5}")
    rough_faces = load_rough_face_collection_from_h5(args.rough_mesh_h5)
    face_contexts = [precompute_face_mesh(face) for face in rough_faces]
    face_ids = [ctx["face_id"] for ctx in face_contexts]
    poly_yz = data["poly_yz"]

    kr_true_map = FORSMARK_KR_TRUE if args.site == "forsmark" else LAXEMAR_KR_TRUE
    kr_grid = np.linspace(1.5, 5.5, args.grid_size, dtype=np.float64)

    # 1. Run Original Configuration (X-axis)
    print("\n[1] Running original configuration (Tunnel along X-axis)...")
    orig_rows = build_rows_rough_faces(data, face_contexts)

    # Run Window MC for original configuration
    orig_kr_hats = {}
    for set_id in args.target_sets:
        set_rows = [r for r in orig_rows if r["set_id"] == set_id]
        if not set_rows:
            print(f"    - Set {set_id}: No traces found in original configuration.")
            orig_kr_hats[set_id] = float("nan")
            continue

        print(
            f"    - Set {set_id}: Running Window MC estimation (n={len(set_rows)} traces)...",
            end="",
            flush=True,
        )
        t_start = time.perf_counter()
        fit_row, _, _ = fit_set_lmin(
            set_id=set_id,
            set_rows=set_rows,
            polygon_yz=poly_yz,
            kr_grid=kr_grid,
            rmin=args.rmin,
            rmax=args.rmax,
            lmin_fit=args.lmin_fit,
            mc_samples_per_grid=args.mc_samples,
            bin_count=40,
            bin_mode="log",
            window_mode="polygon",
        )
        orig_kr_hats[set_id] = fit_row["kr_window_mc_hat"]
        print(f" Done ({time.perf_counter() - t_start:.2f}s) -> kr_hat={orig_kr_hats[set_id]:.3f}")

    # Run detailed diagnostics for original configuration
    diagnostics = []
    print("\n[2] Performing detailed diagnostics for original configuration...")
    for set_id in args.target_sets:
        set_rows = [r for r in orig_rows if r["set_id"] == set_id]
        if not set_rows:
            continue
        kr_true = kr_true_map.get(set_id, float("nan"))
        diag = run_diagnostics_for_set(
            set_id=set_id,
            set_rows=set_rows,
            data=data,
            face_ids=face_ids,
            kr_true=kr_true,
            kr_hat=orig_kr_hats[set_id],
        )
        if diag:
            diagnostics.append(diag)

    # Print Diagnostic Report
    print("\n" + "=" * 80)
    print("                      ORIGINAL CONFIGURATION DIAGNOSTICS")
    print("=" * 80)
    for d in diagnostics:
        print(f"\n[Set {d['set_id']}] (True kr = {d['kr_true']:.2f}, Estimated kr = {d['kr_window_mc_hat']:.2f})")
        print(f"  - Pole Trend/Plunge  : {d['mean_pole_trend']:05.1f}° / {d['mean_pole_plunge']:04.1f}°")
        print(f"  - Orientation Factor : mean={d['orientation_factor_mean']:.3f} (range: [{d['orientation_factor_min']:.3f}, {d['orientation_factor_max']:.3f}])")
        print(f"  - Traces             : count={d['n_traces']}, unique_faces={d['n_faces_with_traces']}/{len(face_ids)}")
        print(f"  - Trace Lengths      : mean={d['mean_trace_length']:.3f} m, p90={d['p90_trace_length']:.3f} m, censoring_ratio={d['censoring_ratio']:.2%}")
        print(f"  - YZ Direction Conc. : {d['trace_concentration_eigenvalue']:.3f} (1.0 = perfect alignment)")
        print(f"  - Face Count CV      : {d['face_concentration_cv']:.3f} (higher = clustered on fewer faces)")
        print(f"  - Long Traces (p90+) : YZ Conc.={d['long_trace_concentration_eigenvalue']:.3f}, Face CV={d['long_face_concentration_cv']:.3f}")
        print(f"  - Estimation Error   : {d['kr_error']:.3f} (Bias Flag: {d['bias_flag']})")
    print("=" * 80 + "\n")

    # 2. Run Sensitivity to Tunnel Direction
    directions = ["x", "y", "z", "azimuth_30", "azimuth_60"]
    summary_rows = []

    print("[3] Simulating alternative tunnel directions...")
    for dir_name in directions:
        print(f"\n  --- Tunnel Direction: {dir_name} ---")
        R = get_rotation_matrix(dir_name)
        rotated_data = rotate_dfn(data, R)

        print(f"    * Extracting traces...", end="", flush=True)
        t_start = time.perf_counter()
        rotated_rows = build_rows_rough_faces(rotated_data, face_contexts)
        print(f" Done ({time.perf_counter() - t_start:.2f}s) -> Total {len(rotated_rows)} traces extracted.")

        for set_id in args.target_sets:
            set_rows = [r for r in rotated_rows if r["set_id"] == set_id]
            if not set_rows:
                continue

            # Compute rotated orientation factor for this set
            set_indices = np.where(rotated_data["set_ids"] == set_id)[0]
            set_normals = rotated_data["normals"][set_indices]
            g_factors = np.sqrt(1.0 - set_normals[:, 0] ** 2)
            g_mean = float(np.mean(g_factors))

            censored_count = sum(1 for r in set_rows if r["censoring_class"] > 0)
            censoring_ratio = censored_count / len(set_rows) if set_rows else 0.0

            print(
                f"    * Set {set_id}: Running Window MC (n={len(set_rows)} traces, g_mean={g_mean:.3f})...",
                end="",
                flush=True,
            )
            t_est = time.perf_counter()
            fit_row, _, _ = fit_set_lmin(
                set_id=set_id,
                set_rows=set_rows,
                polygon_yz=poly_yz,
                kr_grid=kr_grid,
                rmin=args.rmin,
                rmax=args.rmax,
                lmin_fit=args.lmin_fit,
                mc_samples_per_grid=args.mc_samples,
                bin_count=40,
                bin_mode="log",
                window_mode="polygon",
            )
            kr_hat = fit_row["kr_window_mc_hat"]
            kr_true = kr_true_map.get(set_id, float("nan"))
            kr_abs_error = abs(kr_hat - kr_true) if np.isfinite(kr_true) else float("nan")
            fit_status = fit_row["fit_status"]
            recovery_status = determine_recovery_status(kr_hat, kr_true if np.isfinite(kr_true) else None)
            print(f" Done ({time.perf_counter() - t_est:.2f}s) -> kr_hat={kr_hat:.3f}")

            summary_rows.append(
                {
                    "site": args.site,
                    "rmin": args.rmin,
                    "set_id": set_id,
                    "tunnel_direction": dir_name,
                    "n_traces": len(set_rows),
                    "orientation_factor_mean": g_mean,
                    "censoring_ratio": censoring_ratio,
                    "kr_true": kr_true,
                    "kr_hat": kr_hat,
                    "kr_abs_error": kr_abs_error,
                    "q90_ratio": fit_row["q90_ratio_model_observed"],
                    "q95_ratio": fit_row["q95_ratio_model_observed"],
                    "class_l1": fit_row["class_fraction_l1_error"],
                    "fit_status": fit_status,
                    "recovery_status": recovery_status,
                }
            )

    # Save summary CSV
    rmin_tag = str(args.rmin).replace(".", "p")
    summary_csv_name = f"tunnel_direction_bias_summary_rmin{rmin_tag}.csv"
    summary_path = os.path.join(args.outdir, summary_csv_name)
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)

    print(f"\n[*] Summary table written to: {summary_path}")

    # 3. Print Final Conclusion
    print("\n" + "=" * 80)
    print("                               CONCLUSIONS")
    print("=" * 80)
    for set_id in args.target_sets:
        set_summaries = [s for s in summary_rows if s["set_id"] == set_id]
        if not set_summaries:
            continue
        print(f"\n[Set {set_id}] (True kr = {kr_true_map.get(set_id, float('nan')):.2f})")
        errors = [s["kr_abs_error"] for s in set_summaries]
        min_err_idx = np.argmin(errors)
        max_err_idx = np.argmax(errors)
        best_dir = set_summaries[min_err_idx]["tunnel_direction"]
        worst_dir = set_summaries[max_err_idx]["tunnel_direction"]

        print(f"  - Best estimation  : kr_hat={set_summaries[min_err_idx]['kr_hat']:.2f} in direction '{best_dir}' (error={errors[min_err_idx]:.2f}, g_mean={set_summaries[min_err_idx]['orientation_factor_mean']:.3f})")
        print(f"  - Worst estimation : kr_hat={set_summaries[max_err_idx]['kr_hat']:.2f} in direction '{worst_dir}' (error={errors[max_err_idx]:.2f}, g_mean={set_summaries[max_err_idx]['orientation_factor_mean']:.3f})")
        
        kr_range = max(s["kr_hat"] for s in set_summaries) - min(s["kr_hat"] for s in set_summaries)
        print(f"  - Estimation Range : {kr_range:.2f}")

        if kr_range > 0.4:
            print("  -> 판정: Tunnel Direction Bias가 매우 강하게 나타남. 터널 방향에 따라 3D kr 추정치가 크게 흔들림.")
        else:
            print("  -> 판정: Tunnel Direction Bias 영향이 크지 않거나 다른 요인(샘플 수 부족, 경계 윈도우 효과 등)이 지배적임.")
    print("=" * 80)


if __name__ == "__main__":
    main()
