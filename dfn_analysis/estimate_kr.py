import argparse
import json
import math
import os
import sys
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dfn_analysis.estimate_radius_powerlaw_window_mc import (
    build_set_rmin_lookup,
    fit_set_lmin,
    load_trace_data_from_csv,
    load_trace_data_from_h5,
    load_trace_rmin_metadata_from_h5,
    resolve_set_likelihood_rmin,
    write_csv,
)
from dfn_analysis.export_setwise_3d_traces import load_tunnel_polygon_from_dat


ADOPTION_PRIORITY = {
    "accepted": 0,
    "provisional_accepted": 1,
    "rejected": 2,
    "diagnostic_only_rmin_support_mismatch": 3,
}

SITE_TO_KR_TRUE: Dict[str, Dict[int, float]] = {
    "forsmark": {1: 2.88, 2: 3.02, 3: 2.81, 4: 2.95, 5: 2.92},
    "laxemar": {1: 2.85, 2: 3.04, 3: 3.01, 5: 3.60},
}


def group_rows_by_set(rows: Sequence[dict], target_sets: Optional[Set[int]]) -> Dict[int, List[dict]]:
    grouped: Dict[int, List[dict]] = {}
    for row in rows:
        set_id = int(row["set_id"])
        if target_sets is not None and set_id not in target_sets:
            continue
        grouped.setdefault(set_id, []).append(row)
    return {set_id: grouped[set_id] for set_id in sorted(grouped)}


def parse_kr_true_map(site: str, tokens: Optional[Sequence[str]]) -> Dict[int, float]:
    kr_true_map = dict(SITE_TO_KR_TRUE.get(site, {})) if site else {}
    for token in tokens or []:
        sid_str, kr_str = token.split(":")
        kr_true_map[int(sid_str)] = float(kr_str)
    return kr_true_map


def to_float(row: dict, key: str) -> float:
    value = row.get(key, "")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def best_row(rows: Sequence[dict]) -> dict:
    def safe_abs_diff(value: float, ref: float) -> float:
        if not math.isfinite(value) or not math.isfinite(ref):
            return float("inf")
        return abs(value - ref)

    def sort_key(row: dict) -> Tuple[float, float, float, float]:
        adoption = str(row.get("adoption_status", "rejected"))
        return (
            ADOPTION_PRIORITY.get(adoption, 9),
            safe_abs_diff(to_float(row, "kr_window_mc_hat"), to_float(row, "kr_true")),
            to_float(row, "class_fraction_l1_error"),
            safe_abs_diff(to_float(row, "q90_ratio_model_observed"), 1.0),
        )

    return sorted(rows, key=sort_key)[0]


def build_summary_rows(fit_rows: Sequence[dict], site: str = "") -> List[dict]:
    grouped: Dict[int, List[dict]] = {}
    for row in fit_rows:
        grouped.setdefault(int(row["set_id"]), []).append(row)

    summary_rows: List[dict] = []
    for set_id, rows in sorted(grouped.items()):
        chosen = best_row(rows)
        summary_rows.append(
            {
                "site": site,
                "set_id": set_id,
                "best_lmin_fit": to_float(chosen, "lmin_fit"),
                "kr_hat": to_float(chosen, "kr_window_mc_hat"),
                "kr_true": to_float(chosen, "kr_true"),
                "kr_abs_error": safe_error(to_float(chosen, "kr_window_mc_hat"), to_float(chosen, "kr_true")),
                "set_effective_generation_rmin": to_float(chosen, "set_effective_generation_rmin"),
                "set_likelihood_rmin": to_float(chosen, "set_likelihood_rmin"),
                "generation_rmax": to_float(chosen, "generation_rmax"),
                "fit_status": str(chosen.get("fit_status", "")),
                "recovery_status": str(chosen.get("recovery_status", "")),
                "adoption_status": str(chosen.get("adoption_status", "")),
                "q90_ratio_model_observed": to_float(chosen, "q90_ratio_model_observed"),
                "q95_ratio_model_observed": to_float(chosen, "q95_ratio_model_observed"),
                "class_fraction_l1_error": to_float(chosen, "class_fraction_l1_error"),
                "kr_boot_mean": to_float(chosen, "kr_boot_mean"),
                "kr_boot_std": to_float(chosen, "kr_boot_std"),
                "kr_ci_low": to_float(chosen, "kr_ci_low"),
                "kr_ci_high": to_float(chosen, "kr_ci_high"),
                "bootstrap_boundary_fraction": to_float(chosen, "bootstrap_boundary_fraction"),
                "n_total": int(to_float(chosen, "n_total")) if math.isfinite(to_float(chosen, "n_total")) else "",
                "n_used": int(to_float(chosen, "n_used")) if math.isfinite(to_float(chosen, "n_used")) else "",
                "notes": str(chosen.get("warning", "")),
            }
        )
    return summary_rows


def safe_error(value: float, ref: float) -> float:
    if not math.isfinite(value) or not math.isfinite(ref):
        return float("nan")
    return abs(value - ref)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Integrated benchmark1 kr estimator using the active effective-rmin polygon window-MC likelihood."
    )
    parser.add_argument("--trace-h5", help="Input trace HDF5")
    parser.add_argument("--trace-csv", help="Input trace CSV")
    parser.add_argument("--tunnel-dat", help="Optional tunnel polygon .dat file if trace input does not contain /meta/tunnel_poly_yz.")
    parser.add_argument("--target-set", nargs="+", type=int)
    parser.add_argument("--site", choices=["forsmark", "laxemar"], default="")
    parser.add_argument("--dfn-model", default="", help="Site/model label written to outputs.")
    parser.add_argument("--rmin", type=float, default=0.5)
    parser.add_argument("--rmax", type=float, default=250.0)
    parser.add_argument(
        "--set-rmin-mode",
        choices=["global", "effective_generation", "table_r0"],
        default="effective_generation",
    )
    parser.add_argument("--generation-rmin", type=float, default=0.5)
    parser.add_argument("--generation-rmax", type=float, default=250.0)
    parser.add_argument("--p32-label", default="P32_r_ge_0p5m")
    parser.add_argument("--kr-min", type=float, default=1.5)
    parser.add_argument("--kr-max", type=float, default=5.5)
    parser.add_argument("--lmin-fit-values", nargs="+", type=float, default=[0.1, 0.2, 0.3, 0.5, 0.75])
    parser.add_argument("--allow-rmin-mismatch", action="store_true")
    parser.add_argument("--profile-grid-size", type=int, default=81)
    parser.add_argument("--mc-samples-per-grid", type=int, default=50000)
    parser.add_argument("--length-bin-count", type=int, default=40)
    parser.add_argument("--length-bin-mode", choices=["log", "linear"], default="log")
    parser.add_argument("--direction-mode", choices=["empirical_trace", "orientation_conditioned"], default="empirical_trace")
    parser.add_argument("--center-weighting", choices=["unweighted", "proposal_area"], default="proposal_area")
    parser.add_argument("--likelihood-component", choices=["joint", "length_only", "class_only"], default="joint")
    parser.add_argument("--class-likelihood-weight", type=float, default=1.0)
    parser.add_argument("--oracle-radius-mode", choices=["none", "observed_trace_radii"], default="none")
    parser.add_argument("--run-bootstrap", action="store_true")
    parser.add_argument("--n-bootstrap", type=int, default=100)
    parser.add_argument("--kr-true-map", nargs="+", metavar="SET_ID:KR_TRUE")
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()

    if bool(args.trace_h5) == bool(args.trace_csv):
        raise ValueError("Provide exactly one of --trace-h5 or --trace-csv.")

    rows, polygon_yz = load_trace_data_from_h5(args.trace_h5) if args.trace_h5 else load_trace_data_from_csv(args.trace_csv)
    if polygon_yz is None and args.tunnel_dat:
        polygon_yz = load_tunnel_polygon_from_dat(args.tunnel_dat)
    if polygon_yz is None:
        raise ValueError("Window polygon is required for the active kr estimator. Provide --trace-h5 with /meta/tunnel_poly_yz or pass --tunnel-dat.")
    target_sets = set(args.target_set) if args.target_set else None
    grouped = group_rows_by_set(rows, target_sets)
    if not grouped:
        raise ValueError("No matching rows for target sets.")

    generation_rmin = float(args.generation_rmin)
    generation_rmax = float(args.generation_rmax)
    trace_rmin_metadata = {}
    if args.trace_h5:
        trace_rmin_metadata = load_trace_rmin_metadata_from_h5(args.trace_h5)
        if trace_rmin_metadata.get("generation_rmin") is not None:
            generation_rmin = float(trace_rmin_metadata["generation_rmin"])
        if trace_rmin_metadata.get("generation_rmax") is not None:
            generation_rmax = float(trace_rmin_metadata["generation_rmax"])

    if abs(generation_rmin - float(args.rmin)) > 1e-6 and not args.allow_rmin_mismatch:
        raise ValueError(
            f"generation_rmin ({generation_rmin}) does not match estimator rmin ({args.rmin}). "
            "Use --allow-rmin-mismatch only for diagnostic runs."
        )

    site = args.site or args.dfn_model
    kr_true_map = parse_kr_true_map(site, args.kr_true_map)
    set_rmin_lookup = build_set_rmin_lookup(site, generation_rmin, trace_rmin_metadata)
    kr_grid = np.linspace(args.kr_min, args.kr_max, args.profile_grid_size, dtype=np.float64)

    fit_rows: List[dict] = []
    profile_rows: List[dict] = []
    pp_rows: List[dict] = []

    for set_id, set_rows in grouped.items():
        for lmin_fit in args.lmin_fit_values:
            set_likelihood_rmin, set_effective_generation_rmin, set_table_r0, set_support_status = resolve_set_likelihood_rmin(
                set_id,
                args.set_rmin_mode,
                float(args.rmin),
                set_rmin_lookup,
            )
            fit_row, profile, pp, _ = fit_set_lmin(
                set_id=set_id,
                set_rows=set_rows,
                polygon_yz=polygon_yz,
                kr_grid=kr_grid,
                rmin=set_likelihood_rmin,
                rmax=float(args.rmax),
                lmin_fit=float(lmin_fit),
                mc_samples_per_grid=int(args.mc_samples_per_grid),
                bin_count=int(args.length_bin_count),
                bin_mode=args.length_bin_mode,
                window_mode="polygon",
                direction_mode=args.direction_mode,
                site=site,
                kr_true=kr_true_map.get(set_id),
                center_weighting=args.center_weighting,
                likelihood_component=args.likelihood_component,
                class_likelihood_weight=float(args.class_likelihood_weight),
                oracle_radius_mode=args.oracle_radius_mode,
                run_bootstrap=bool(args.run_bootstrap),
                n_bootstrap=int(args.n_bootstrap),
            )
            metadata = {
                "dfn_model": site or args.dfn_model,
                "generation_rmin": generation_rmin,
                "generation_rmax": generation_rmax,
                "estimation_rmin": float(args.rmin),
                "estimation_rmax": float(args.rmax),
                "set_likelihood_rmin": float(set_likelihood_rmin),
                "set_effective_generation_rmin": float(set_effective_generation_rmin),
                "set_table_r0": float(set_table_r0) if math.isfinite(set_table_r0) else float("nan"),
                "set_rmin_mode": args.set_rmin_mode,
                "rmin_support_status": set_support_status,
                "p32_label": args.p32_label,
                "entrypoint_script": "dfn_analysis/estimate_kr.py",
            }
            fit_row.update(metadata)
            for row in profile:
                row.update(metadata)
            for row in pp:
                row.update(metadata)
            fit_rows.append(fit_row)
            profile_rows.extend(profile)
            pp_rows.extend(pp)

    os.makedirs(args.outdir, exist_ok=True)
    fit_csv = os.path.join(args.outdir, "kr_fit_by_lmin.csv")
    profile_csv = os.path.join(args.outdir, "kr_profile_likelihood.csv")
    pp_csv = os.path.join(args.outdir, "kr_posterior_predictive.csv")
    summary_csv = os.path.join(args.outdir, "kr_summary_by_set.csv")
    fit_json = os.path.join(args.outdir, "kr_fit_by_lmin.json")

    write_csv(fit_rows, fit_csv)
    write_csv(profile_rows, profile_csv)
    write_csv(pp_rows, pp_csv)
    summary_rows = build_summary_rows(fit_rows, site)
    write_csv(summary_rows, summary_csv)
    with open(fit_json, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "input_summary": {
                    "trace_h5": args.trace_h5,
                    "trace_csv": args.trace_csv,
                    "tunnel_dat": args.tunnel_dat,
                    "target_set": args.target_set,
                    "site": site,
                    "rmin": float(args.rmin),
                    "rmax": float(args.rmax),
                    "set_rmin_mode": args.set_rmin_mode,
                    "lmin_fit_values": [float(value) for value in args.lmin_fit_values],
                    "window_mode": "polygon",
                    "center_weighting": args.center_weighting,
                    "likelihood_component": args.likelihood_component,
                    "class_likelihood_weight": float(args.class_likelihood_weight),
                    "oracle_radius_mode": args.oracle_radius_mode,
                    "run_bootstrap": bool(args.run_bootstrap),
                    "n_bootstrap": int(args.n_bootstrap),
                    "truth_used_for_validation_only": bool(site or args.kr_true_map),
                },
                "fit_rows": fit_rows,
                "summary_rows": summary_rows,
            },
            handle,
            indent=2,
        )

    print("[*] estimate_kr.py summary")
    for row in summary_rows:
        kr_true = to_float(row, "kr_true")
        truth_text = f", kr_true={kr_true:.3f}" if math.isfinite(kr_true) else ""
        print(
            f"    - Set {row['set_id']}: best_lmin={float(row['best_lmin_fit']):.3f}, "
            f"kr_hat={float(row['kr_hat']):.3f}{truth_text}, status={row['fit_status']}, adoption={row['adoption_status']}"
        )
    print(f"[*] Fit CSV written to: {fit_csv}")
    print(f"[*] Profile CSV written to: {profile_csv}")
    print(f"[*] Posterior predictive CSV written to: {pp_csv}")
    print(f"[*] Summary CSV written to: {summary_csv}")
    print(f"[*] Fit JSON written to: {fit_json}")


if __name__ == "__main__":
    main()
