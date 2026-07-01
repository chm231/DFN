import argparse
import math
import os
import sys
from typing import Dict, List, Tuple

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dfn_analysis.estimate_p32_mc_calibrated import (
    estimate_unit_p32_forward_mc,
    load_face_x_positions,
    load_p21_summary,
    read_csv,
    to_float,
    write_csv,
)
from dfn_analysis.estimate_p32_combined_bootstrap import (
    face_bootstrap_p21,
    load_face_areas,
    load_per_face_trace_lengths,
)
from dfn_analysis.estimate_radius_powerlaw_window_mc import load_trace_data_from_h5


DEFAULT_TRACE_H5 = {
    "forsmark": "storage/output/forsmark_rmin0p5_trace_dataset_collection/trace_dataset_3d.h5",
    "laxemar": "storage/output/laxemar_rmin0p5_trace_dataset_collection/trace_dataset_3d.h5",
}
DEFAULT_ROUGH_MESH_H5 = (
    "storage/output/rough_face_mesh_collection/synthetic_rough_face_collection.h5"
)
DEFAULT_KR_BOOT_CSV = (
    "storage/output/final_kr_bootstrap_effective_rmin/final_kr_bootstrap_summary_effective_rmin.csv"
)
DEFAULT_KR_RECOVERY_CSV = (
    "storage/output/final_kr_recovery_summary_effective_rmin.csv"
)
DEFAULT_OLD_SUMMARY_CSV = (
    "storage/output/p32_mc_calibrated_effective_rmin/p32_final_pilot/p32_combined_bootstrap_summary.csv"
)
DEFAULT_OUTDIR = (
    "storage/output/p32_mc_calibrated_effective_rmin/dense_c_interpolation_check"
)


def get_single_row(path: str, site: str, set_id: int) -> dict:
    for row in read_csv(path):
        if str(row.get("site", "")) == site and int(row.get("set_id", -1)) == set_id:
            return row
    raise ValueError(f"Missing row in {path} for site={site}, set_id={set_id}")


def interpolate_log_linear_dense(kr: float, kr_grid: np.ndarray, c_grid: np.ndarray) -> Tuple[float, bool]:
    valid = np.isfinite(kr_grid) & np.isfinite(c_grid) & (c_grid > 0.0)
    x = np.asarray(kr_grid[valid], dtype=np.float64)
    y = np.log(np.asarray(c_grid[valid], dtype=np.float64))
    if len(x) < 2:
        raise ValueError("Need at least two valid dense C(kr) points for interpolation.")
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    extrapolated = bool(kr < x[0] or kr > x[-1])
    log_c = float(np.interp(kr, x, y, left=y[0], right=y[-1]))
    return math.exp(log_c), extrapolated


def combined_bootstrap_dense(
    face_ids_all: np.ndarray,
    face_lengths: Dict[Tuple[int, int], List[float]],
    face_areas: Dict[int, float],
    set_id: int,
    kr_boot_mean: float,
    kr_boot_std: float,
    kr_ci_low: float,
    kr_ci_high: float,
    kr_grid: np.ndarray,
    c_grid: np.ndarray,
    n_combined: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, float, float]:
    n_faces = len(face_ids_all)
    p32_b = np.empty(n_combined, dtype=np.float64)
    n_kr_clipped = 0
    n_c_extrap = 0
    for b in range(n_combined):
        boot_faces = rng.choice(face_ids_all, size=n_faces, replace=True)
        total_length = 0.0
        boot_area = 0.0
        for fid in boot_faces:
            total_length += sum(face_lengths.get((int(fid), set_id), []))
            boot_area += face_areas.get(int(fid), 0.0)
        p21_b = total_length / boot_area if boot_area > 0.0 else float("nan")

        kr_raw = rng.normal(kr_boot_mean, kr_boot_std)
        kr_b = float(np.clip(kr_raw, kr_ci_low, kr_ci_high))
        if kr_raw != kr_b:
            n_kr_clipped += 1
        c_b, extrapolated = interpolate_log_linear_dense(kr_b, kr_grid, c_grid)
        if extrapolated:
            n_c_extrap += 1
        p32_b[b] = p21_b / c_b if (np.isfinite(p21_b) and c_b > 0.0) else float("nan")

    kr_clip_frac = n_kr_clipped / n_combined if n_combined > 0 else 0.0
    c_extrap_frac = n_c_extrap / n_combined if n_combined > 0 else 0.0
    return p32_b, kr_clip_frac, c_extrap_frac


def main() -> None:
    parser = argparse.ArgumentParser(description="Dense C(kr) interpolation check for unit-P32 combined bootstrap.")
    parser.add_argument("--site", choices=["forsmark", "laxemar"], default="forsmark")
    parser.add_argument("--set-id", type=int, default=5)
    parser.add_argument("--trace-h5", default=None)
    parser.add_argument("--rough-mesh-h5", default=DEFAULT_ROUGH_MESH_H5)
    parser.add_argument("--kr-bootstrap-csv", default=DEFAULT_KR_BOOT_CSV)
    parser.add_argument("--kr-recovery-csv", default=DEFAULT_KR_RECOVERY_CSV)
    parser.add_argument("--old-summary-csv", default=DEFAULT_OLD_SUMMARY_CSV)
    parser.add_argument("--mc-samples", type=int, default=50000)
    parser.add_argument("--unit-p32-mc-replicates", type=int, default=50)
    parser.add_argument("--n-combined", type=int, default=1000)
    parser.add_argument("--kr-grid-n", type=int, default=11)
    parser.add_argument("--window-mode", choices=["polygon", "bbox"], default="polygon")
    parser.add_argument("--rng-seed", type=int, default=991100)
    parser.add_argument("--outdir", default=DEFAULT_OUTDIR)
    args = parser.parse_args()

    trace_h5 = args.trace_h5 or DEFAULT_TRACE_H5[args.site]
    os.makedirs(args.outdir, exist_ok=True)

    old_row = get_single_row(args.old_summary_csv, args.site, args.set_id)
    kr_boot_row = get_single_row(args.kr_bootstrap_csv, args.site, args.set_id)
    kr_rec_row = get_single_row(args.kr_recovery_csv, args.site, args.set_id)

    _, polygon_yz = load_trace_data_from_h5(trace_h5)
    if polygon_yz is None:
        raise ValueError(f"Missing /meta/tunnel_poly_yz in {trace_h5}")
    face_x_positions = load_face_x_positions(trace_h5)
    _, total_observation_area = load_p21_summary(trace_h5, args.rough_mesh_h5)

    face_areas = load_face_areas(args.rough_mesh_h5)
    face_ids_all = np.array(sorted(face_areas.keys()), dtype=np.int32)
    face_lengths = load_per_face_trace_lengths(trace_h5)

    kr_boot_mean = to_float(kr_boot_row, "kr_boot_mean")
    kr_boot_std = to_float(kr_boot_row, "kr_boot_std")
    kr_ci_low = to_float(kr_boot_row, "kr_ci_low")
    kr_ci_high = to_float(kr_boot_row, "kr_ci_high")
    kr_used = to_float(kr_rec_row, "kr_hat")
    set_likelihood_rmin = to_float(kr_rec_row, "set_likelihood_rmin")
    observed_p21 = to_float(old_row, "observed_P21")
    p32_reference = to_float(old_row, "P32_reference")

    grid_min = kr_ci_low
    grid_max = kr_ci_high
    kr_grid = np.linspace(grid_min, grid_max, args.kr_grid_n, dtype=np.float64)

    dense_rows: List[dict] = []
    c_grid = np.empty_like(kr_grid)
    for idx, kr in enumerate(kr_grid):
        result = estimate_unit_p32_forward_mc(
            site=args.site,
            set_id=args.set_id,
            kr=float(kr),
            rmin=set_likelihood_rmin,
            rmax=250.0,
            polygon_yz=polygon_yz,
            face_x_positions=face_x_positions,
            total_observation_area=total_observation_area,
            mc_samples=args.mc_samples,
            mc_replicates=args.unit_p32_mc_replicates,
            rng_seed=args.rng_seed + idx * 100,
            window_mode=args.window_mode,
        )
        c_grid[idx] = float(result["calibration_factor_C"])
        dense_rows.append(
            {
                "site": args.site,
                "set_id": args.set_id,
                "kr_grid_value": float(kr),
                "set_likelihood_rmin": set_likelihood_rmin,
                "calibration_factor_mode": "unit_p32_forward_mc",
                "calibration_factor_C": c_grid[idx],
                "calibration_factor_std": float(result["calibration_factor_std"]),
                "calibration_factor_ci_low": float(result["calibration_factor_ci_low"]),
                "calibration_factor_ci_high": float(result["calibration_factor_ci_high"]),
                "unit_p32_mc_replicates": args.unit_p32_mc_replicates,
                "mc_samples": args.mc_samples,
            }
        )

    rng = np.random.default_rng(args.rng_seed + 777)
    p21_boot = face_bootstrap_p21(
        face_ids_all, face_lengths, face_areas, args.set_id, args.n_combined, rng
    )
    c_at_hat, _ = interpolate_log_linear_dense(kr_used, kr_grid, c_grid)
    p32_hat_new = observed_p21 / c_at_hat if np.isfinite(observed_p21) and c_at_hat > 0.0 else float("nan")
    p32_face_boot = p21_boot / c_at_hat if c_at_hat > 0.0 else np.full(len(p21_boot), float("nan"))
    p32_face_boot_std = float(np.std(p32_face_boot[np.isfinite(p32_face_boot)], ddof=1)) if np.any(np.isfinite(p32_face_boot)) else float("nan")

    p32_combined, kr_clip_frac, c_extrap_frac_new = combined_bootstrap_dense(
        face_ids_all=face_ids_all,
        face_lengths=face_lengths,
        face_areas=face_areas,
        set_id=args.set_id,
        kr_boot_mean=kr_boot_mean,
        kr_boot_std=kr_boot_std,
        kr_ci_low=kr_ci_low,
        kr_ci_high=kr_ci_high,
        kr_grid=kr_grid,
        c_grid=c_grid,
        n_combined=args.n_combined,
        rng=rng,
    )
    valid_combined = p32_combined[np.isfinite(p32_combined)]
    p32_combined_mean = float(np.mean(valid_combined)) if len(valid_combined) else float("nan")
    p32_combined_std = float(np.std(valid_combined, ddof=1)) if len(valid_combined) > 1 else float("nan")
    p32_ci_low_new = float(np.percentile(valid_combined, 2.5)) if len(valid_combined) else float("nan")
    p32_ci_high_new = float(np.percentile(valid_combined, 97.5)) if len(valid_combined) else float("nan")
    true_in_ci_new = bool(np.isfinite(p32_reference) and np.isfinite(p32_ci_low_new) and np.isfinite(p32_ci_high_new) and p32_ci_low_new <= p32_reference <= p32_ci_high_new)

    if true_in_ci_new:
        status_update = "p32_final_pilot_candidate"
    else:
        status_update = "p32_final_marginal_ci_miss"
        if c_extrap_frac_new <= 1e-6:
            status_update += "; residual_sampling_or_calibration_issue"

    comparison_row = {
        "site": args.site,
        "set_id": args.set_id,
        "kr_grid_min": grid_min,
        "kr_grid_max": grid_max,
        "kr_grid_n": args.kr_grid_n,
        "C_interpolation_mode": "log_linear_dense",
        "C_extrapolation_fraction_old": to_float(old_row, "C_extrapolation_fraction"),
        "C_extrapolation_fraction_new": c_extrap_frac_new,
        "P32_hat_old": to_float(old_row, "P32_hat"),
        "P32_ci_low_old": to_float(old_row, "P32_ci_low"),
        "P32_ci_high_old": to_float(old_row, "P32_ci_high"),
        "P32_hat_new": p32_hat_new,
        "P32_ci_low_new": p32_ci_low_new,
        "P32_ci_high_new": p32_ci_high_new,
        "P32_reference": p32_reference,
        "true_P32_inside_ci_new": true_in_ci_new,
        "status_update": status_update,
        "kr_boot_clip_fraction_new": kr_clip_frac,
        "P32_face_boot_std_new": p32_face_boot_std,
        "P32_combined_boot_mean_new": p32_combined_mean,
        "P32_combined_boot_std_new": p32_combined_std,
    }

    dense_csv = os.path.join(args.outdir, f"{args.site}_set{args.set_id}_dense_c_table.csv")
    summary_csv = os.path.join(args.outdir, f"{args.site}_set{args.set_id}_dense_c_interpolation_summary.csv")
    write_csv(dense_rows, dense_csv)
    write_csv([comparison_row], summary_csv)

    print(f"[*] Dense C(kr) table written to: {dense_csv}")
    print(f"[*] Dense interpolation summary written to: {summary_csv}")
    print(
        "[*] %s Set %d: old_C_extrap=%.3f new_C_extrap=%.3f old_CI=[%.4f, %.4f] new_CI=[%.4f, %.4f] ref=%.4f inside_new=%s"
        % (
            args.site,
            args.set_id,
            comparison_row["C_extrapolation_fraction_old"],
            comparison_row["C_extrapolation_fraction_new"],
            comparison_row["P32_ci_low_old"],
            comparison_row["P32_ci_high_old"],
            comparison_row["P32_ci_low_new"],
            comparison_row["P32_ci_high_new"],
            comparison_row["P32_reference"],
            comparison_row["true_P32_inside_ci_new"],
        )
    )


if __name__ == "__main__":
    main()
