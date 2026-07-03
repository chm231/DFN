import argparse
import math
import os
import sys
from typing import List

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dfn_analysis.estimate_p32_mc_calibrated import (
    CALIBRATION_FACTOR_MODE_UNIT,
    estimate_unit_p32_forward_mc,
    load_face_x_positions,
    load_p21_summary,
    read_csv,
    to_float,
    write_csv,
)
from dfn_analysis.estimate_radius_powerlaw_window_mc import load_trace_data_from_h5


DEFAULT_TRACE_H5 = {
    "forsmark": "storage/output/forsmark_rmin0p5_trace_dataset_collection/trace_dataset_3d.h5",
    "laxemar": "storage/output/laxemar_rmin0p5_trace_dataset_collection/trace_dataset_3d.h5",
}
DEFAULT_KR_BOOT_CSV = (
    "storage/output/final_kr_bootstrap_effective_rmin/final_kr_bootstrap_summary_effective_rmin.csv"
)
DEFAULT_KR_RECOVERY_CSV = (
    "storage/output/final_kr_recovery_summary_effective_rmin.csv"
)


def get_single_row(path: str, site: str, set_id: int) -> dict:
    for row in read_csv(path):
        if str(row.get("site", "")) == site and int(row.get("set_id", -1)) == set_id:
            return row
    raise ValueError(f"Missing row in {path} for site={site}, set_id={set_id}")


def resolve_grid_bounds(
    kr_boot_mean: float,
    kr_boot_std: float,
    kr_ci_low: float,
    kr_ci_high: float,
    grid_mode: str,
    estimator_grid_min: float,
    estimator_grid_max: float,
) -> tuple[float, float]:
    if grid_mode == "bootstrap_range":
        grid_min = kr_boot_mean - 3.0 * kr_boot_std
        grid_max = kr_boot_mean + 3.0 * kr_boot_std
    elif grid_mode == "ci_range":
        grid_min = kr_ci_low
        grid_max = kr_ci_high
    else:
        raise ValueError(f"Unsupported kr_grid_mode: {grid_mode}")

    if np.isfinite(estimator_grid_min):
        grid_min = max(grid_min, estimator_grid_min)
    if np.isfinite(estimator_grid_max):
        grid_max = min(grid_max, estimator_grid_max)
    return float(grid_min), float(grid_max)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build dense unit-P32 C(kr) table for a target set.")
    parser.add_argument("--trace-h5", default=None, help="Trace dataset HDF5.")
    parser.add_argument("--rough-mesh-h5", default="storage/output/rough_face_mesh_collection/synthetic_rough_face_collection.h5")
    parser.add_argument("--site", choices=["forsmark", "laxemar"], required=True)
    parser.add_argument("--target-set", type=int, nargs="+", required=True)
    parser.add_argument("--set-rmin-mode", default="effective_generation")
    parser.add_argument(
        "--calibration-factor-mode",
        choices=[CALIBRATION_FACTOR_MODE_UNIT],
        default=CALIBRATION_FACTOR_MODE_UNIT,
    )
    parser.add_argument("--kr-bootstrap-csv", default=DEFAULT_KR_BOOT_CSV)
    parser.add_argument("--kr-recovery-csv", default=DEFAULT_KR_RECOVERY_CSV)
    parser.add_argument("--kr-grid-mode", choices=["bootstrap_range", "ci_range"], default="bootstrap_range")
    parser.add_argument("--kr-grid-n", type=int, default=11)
    parser.add_argument("--estimator-grid-min", type=float, default=float("nan"))
    parser.add_argument("--estimator-grid-max", type=float, default=float("nan"))
    parser.add_argument("--mc-samples", type=int, default=50000)
    parser.add_argument("--unit-p32-mc-replicates", type=int, default=50)
    parser.add_argument("--window-mode", choices=["polygon", "bbox"], default="polygon")
    parser.add_argument("--rng-seed", type=int, default=992000)
    parser.add_argument("--outcsv", required=True)
    args = parser.parse_args()

    if len(args.target_set) != 1:
        raise ValueError("build_dense_ckr_table.py currently supports exactly one target set.")
    set_id = int(args.target_set[0])
    trace_h5 = args.trace_h5 or DEFAULT_TRACE_H5[args.site]

    _, polygon_yz = load_trace_data_from_h5(trace_h5)
    if polygon_yz is None:
        raise ValueError(f"Missing /meta/tunnel_poly_yz in {trace_h5}")
    face_x_positions = load_face_x_positions(trace_h5)
    _, total_observation_area = load_p21_summary(trace_h5, args.rough_mesh_h5)

    kr_boot_row = get_single_row(args.kr_bootstrap_csv, args.site, set_id)
    kr_rec_row = get_single_row(args.kr_recovery_csv, args.site, set_id)
    kr_boot_mean = to_float(kr_boot_row, "kr_boot_mean")
    kr_boot_std = to_float(kr_boot_row, "kr_boot_std")
    kr_ci_low = to_float(kr_boot_row, "kr_ci_low")
    kr_ci_high = to_float(kr_boot_row, "kr_ci_high")
    set_effective_rmin = to_float(kr_rec_row, "set_effective_generation_rmin")
    set_likelihood_rmin = to_float(kr_rec_row, "set_likelihood_rmin")

    grid_min, grid_max = resolve_grid_bounds(
        kr_boot_mean=kr_boot_mean,
        kr_boot_std=kr_boot_std,
        kr_ci_low=kr_ci_low,
        kr_ci_high=kr_ci_high,
        grid_mode=args.kr_grid_mode,
        estimator_grid_min=args.estimator_grid_min,
        estimator_grid_max=args.estimator_grid_max,
    )
    if not np.isfinite(grid_min) or not np.isfinite(grid_max) or grid_max <= grid_min:
        raise ValueError(f"Invalid kr grid bounds: [{grid_min}, {grid_max}]")

    kr_grid = np.linspace(grid_min, grid_max, args.kr_grid_n, dtype=np.float64)
    out_rows: List[dict] = []
    for idx, kr_value in enumerate(kr_grid):
        result = estimate_unit_p32_forward_mc(
            site=args.site,
            set_id=set_id,
            kr=float(kr_value),
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
        out_rows.append(
            {
                "site": args.site,
                "set_id": set_id,
                "kr_value": float(kr_value),
                "calibration_factor_C": float(result["calibration_factor_C"]),
                "calibration_factor_std": float(result["calibration_factor_std"]),
                "calibration_factor_ci_low": float(result["calibration_factor_ci_low"]),
                "calibration_factor_ci_high": float(result["calibration_factor_ci_high"]),
                "unit_p32_mc_replicates": args.unit_p32_mc_replicates,
                "mc_samples": args.mc_samples,
                "mean_fracture_area": float(result["mean_fracture_area"]),
                "fracture_number_density_for_unit_p32": float(result["fracture_number_density_for_unit_p32"]),
                "set_effective_generation_rmin": set_effective_rmin,
                "set_likelihood_rmin": set_likelihood_rmin,
                "calibration_factor_mode": args.calibration_factor_mode,
                "notes": (
                    f"kr_grid_mode={args.kr_grid_mode}; "
                    f"grid_bounds=[{grid_min:.6f},{grid_max:.6f}]; "
                    f"set_rmin_mode={args.set_rmin_mode}"
                ),
            }
        )

    os.makedirs(os.path.dirname(args.outcsv) or ".", exist_ok=True)
    write_csv(out_rows, args.outcsv)
    print(f"[*] Dense C(kr) table written to: {args.outcsv}")
    print(f"[*] kr_grid_range=[{grid_min:.4f}, {grid_max:.4f}]  n={args.kr_grid_n}")


if __name__ == "__main__":
    main()
