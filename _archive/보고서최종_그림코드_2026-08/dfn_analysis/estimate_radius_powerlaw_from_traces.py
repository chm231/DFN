import argparse
import csv
import json
import os
from typing import Dict, List, Optional, Sequence, Set, Tuple

import h5py
import numpy as np
from scipy.optimize import minimize_scalar


XMAX_NOTE = "radius likelihood v3 does not yet use face/window geometry directly"
TWO_END_NOTE = "two-end censored traces use survival approximation; window-aware likelihood recommended"


def load_trace_rows_from_h5(h5_path: str) -> List[dict]:
    rows: List[dict] = []
    with h5py.File(h5_path, "r") as f:
        if "traces" not in f:
            raise ValueError(f"Could not find /traces in: {h5_path}")
        grp = f["traces"]
        n_rows = len(grp["trace_id"])
        for idx in range(n_rows):
            rows.append(
                {
                    "set_id": int(grp["set_id"][idx]),
                    "observed_length_m": float(grp["observed_length_m"][idx]),
                    "censoring_class": int(grp["censoring_class"][idx]),
                }
            )
    return rows


def load_trace_rows_from_csv(csv_path: str) -> List[dict]:
    rows: List[dict] = []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"set_id", "observed_length_m", "censoring_class"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing required CSV fields: {sorted(missing)}")
        for row in reader:
            rows.append(
                {
                    "set_id": int(row["set_id"]),
                    "observed_length_m": float(row["observed_length_m"]),
                    "censoring_class": int(row["censoring_class"]),
                }
            )
    return rows


def group_rows_by_set(rows: Sequence[dict], target_sets: Optional[Set[int]]) -> Dict[int, List[dict]]:
    grouped: Dict[int, List[dict]] = {}
    for row in rows:
        set_id = int(row["set_id"])
        if target_sets is not None and set_id not in target_sets:
            continue
        grouped.setdefault(set_id, []).append(row)
    return {set_id: grouped[set_id] for set_id in sorted(grouped)}


def radius_powerlaw_pdf(radius: np.ndarray, kr: float, rmin: float, rmax: float) -> np.ndarray:
    radius = np.asarray(radius, dtype=np.float64)
    if kr == 1.0:
        norm = np.log(rmax / rmin)
    else:
        norm = (rmax ** (1.0 - kr) - rmin ** (1.0 - kr)) / (1.0 - kr)
    pdf = np.where((radius >= rmin) & (radius <= rmax), radius ** (-kr) / norm, 0.0)
    return pdf.astype(np.float64)


def intersected_radius_pdf(radius: np.ndarray, kr: float, rmin: float, rmax: float) -> np.ndarray:
    radius = np.asarray(radius, dtype=np.float64)
    exponent = 1.0 - kr
    if kr == 2.0:
        norm = np.log(rmax / rmin)
    else:
        norm = (rmax ** (2.0 - kr) - rmin ** (2.0 - kr)) / (2.0 - kr)
    pdf = np.where((radius >= rmin) & (radius <= rmax), radius ** exponent / norm, 0.0)
    return pdf.astype(np.float64)


def chord_pdf_given_radius(length: np.ndarray, radius: np.ndarray) -> np.ndarray:
    length = np.asarray(length, dtype=np.float64)
    radius = np.asarray(radius, dtype=np.float64)
    out = np.zeros(np.broadcast(length, radius).shape, dtype=np.float64)
    valid = (length > 0.0) & (radius > 0.0) & (length < 2.0 * radius)
    denom = 2.0 * radius * np.sqrt(np.maximum(4.0 * radius * radius - length * length, np.finfo(np.float64).tiny))
    np.divide(length, denom, out=out, where=valid)
    return out


def chord_survival_given_radius(length: np.ndarray, radius: np.ndarray) -> np.ndarray:
    length = np.asarray(length, dtype=np.float64)
    radius = np.asarray(radius, dtype=np.float64)
    out = np.zeros(np.broadcast(length, radius).shape, dtype=np.float64)
    below_zero = length <= 0.0
    valid = (length > 0.0) & (radius > 0.0) & (length < 2.0 * radius)
    out = np.where(below_zero, 1.0, out)
    survival = np.sqrt(np.maximum(radius * radius - 0.25 * length * length, 0.0)) / radius
    out = np.where(valid, survival, out)
    return out.astype(np.float64)


def _radius_grid(rmin: float, rmax: float, n_grid: int = 700) -> np.ndarray:
    return np.geomspace(rmin, rmax, n_grid).astype(np.float64)


def marginal_chord_density(
    lengths: np.ndarray,
    kr: float,
    rmin: float,
    rmax: float,
    radius_grid: Optional[np.ndarray] = None,
) -> np.ndarray:
    lengths = np.asarray(lengths, dtype=np.float64)
    radii = _radius_grid(rmin, rmax) if radius_grid is None else radius_grid
    weights = intersected_radius_pdf(radii, kr, rmin, rmax)
    values = chord_pdf_given_radius(lengths[:, None], radii[None, :]) * weights[None, :]
    return np.trapezoid(values, radii, axis=1)


def marginal_chord_survival(
    lengths: np.ndarray,
    kr: float,
    rmin: float,
    rmax: float,
    radius_grid: Optional[np.ndarray] = None,
) -> np.ndarray:
    lengths = np.asarray(lengths, dtype=np.float64)
    radii = _radius_grid(rmin, rmax) if radius_grid is None else radius_grid
    weights = intersected_radius_pdf(radii, kr, rmin, rmax)
    values = chord_survival_given_radius(lengths[:, None], radii[None, :]) * weights[None, :]
    return np.trapezoid(values, radii, axis=1)


def negative_loglik_radius_chord(
    kr: float,
    lengths: np.ndarray,
    censoring: np.ndarray,
    rmin: float,
    rmax: float,
    radius_grid: np.ndarray,
) -> float:
    if kr <= 0.0 or len(lengths) == 0:
        return float("inf")
    uncensored = censoring == 0
    censored = censoring > 0
    loglik = 0.0
    tiny = np.finfo(np.float64).tiny
    if np.any(uncensored):
        density = marginal_chord_density(lengths[uncensored], kr, rmin, rmax, radius_grid)
        loglik += float(np.sum(np.log(np.clip(density, tiny, None))))
    if np.any(censored):
        survival = marginal_chord_survival(lengths[censored], kr, rmin, rmax, radius_grid)
        loglik += float(np.sum(np.log(np.clip(survival, tiny, 1.0))))
    return -loglik


def fit_kr_radius(
    lengths: np.ndarray,
    censoring: np.ndarray,
    rmin: float,
    rmax: float,
    kr_min: float,
    kr_max: float,
    radius_grid: np.ndarray,
) -> dict:
    if len(lengths) == 0:
        return {"success": False, "kr_radius_hat": float("nan"), "loglik": float("nan"), "message": "no tail traces"}
    result = minimize_scalar(
        negative_loglik_radius_chord,
        bounds=(kr_min, kr_max),
        method="bounded",
        args=(lengths, censoring, rmin, rmax, radius_grid),
    )
    return {
        "success": bool(result.success),
        "kr_radius_hat": float(result.x) if result.success else float("nan"),
        "loglik": float(-result.fun) if result.success else float("nan"),
        "message": str(result.message),
    }


def search_interval_hit(kr_hat: float, kr_min: float, kr_max: float, tol: float = 1e-4) -> str:
    if np.isfinite(kr_hat) and abs(kr_hat - kr_min) <= tol:
        return "lower"
    if np.isfinite(kr_hat) and abs(kr_hat - kr_max) <= tol:
        return "upper"
    return "none"


def marginal_chord_tail_quantile(
    p: float,
    lmin_fit: float,
    kr: float,
    rmin: float,
    rmax: float,
    radius_grid: np.ndarray,
) -> float:
    if not np.isfinite(kr) or kr <= 0.0:
        return float("nan")
    lmax = 2.0 * rmax
    if lmin_fit >= lmax:
        return float("nan")
    length_grid = np.linspace(lmin_fit, lmax, 2000, dtype=np.float64)
    survival = marginal_chord_survival(length_grid, kr, rmin, rmax, radius_grid)
    survival_lmin = float(survival[0])
    if survival_lmin <= 0.0:
        return float("nan")
    target_survival = survival_lmin * (1.0 - p)
    survival_rev = survival[::-1]
    length_rev = length_grid[::-1]
    return float(np.interp(target_survival, survival_rev, length_rev))


def build_profile_rows(
    set_id: int,
    lmin_fit: float,
    lengths: np.ndarray,
    censoring: np.ndarray,
    rmin: float,
    rmax: float,
    kr_min: float,
    kr_max: float,
    profile_grid_size: int,
    radius_grid: np.ndarray,
) -> List[dict]:
    rows = []
    for kr in np.linspace(kr_min, kr_max, profile_grid_size):
        loglik = -negative_loglik_radius_chord(kr, lengths, censoring, rmin, rmax, radius_grid)
        rows.append({"set_id": set_id, "lmin_fit": lmin_fit, "kr_radius": float(kr), "loglik": float(loglik)})
    return rows


def summarize_profile_identifiability(profile_rows: Sequence[dict], kr_min: float, kr_max: float) -> dict:
    if not profile_rows:
        return {
            "max_loglik": float("nan"),
            "profile_width_delta2": float("nan"),
            "weak_identifiability_flag": True,
        }
    loglik = np.asarray([float(row["loglik"]) for row in profile_rows], dtype=np.float64)
    kr_values = np.asarray([float(row["kr_radius"]) for row in profile_rows], dtype=np.float64)
    max_loglik = float(np.max(loglik))
    for row in profile_rows:
        row["max_loglik"] = max_loglik
        row["delta_loglik"] = max_loglik - float(row["loglik"])
    inside = kr_values[(max_loglik - loglik) <= 2.0]
    profile_width = float(np.max(inside) - np.min(inside)) if len(inside) else float("nan")
    weak_flag = bool(np.isfinite(profile_width) and profile_width >= 0.5 * (kr_max - kr_min))
    return {
        "max_loglik": max_loglik,
        "profile_width_delta2": profile_width,
        "weak_identifiability_flag": weak_flag,
    }


def bootstrap_kr_radius(
    lengths: np.ndarray,
    censoring: np.ndarray,
    rmin: float,
    rmax: float,
    kr_min: float,
    kr_max: float,
    radius_grid: np.ndarray,
    n_bootstrap: int,
    seed: int,
) -> dict:
    if n_bootstrap <= 0 or len(lengths) == 0:
        return {"values": [], "mean": float("nan"), "std": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "boundary_fraction": float("nan")}
    rng = np.random.default_rng(seed)
    values: List[float] = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, len(lengths), size=len(lengths))
        fit = fit_kr_radius(lengths[idx], censoring[idx], rmin, rmax, kr_min, kr_max, radius_grid)
        if fit["success"] and np.isfinite(fit["kr_radius_hat"]):
            values.append(float(fit["kr_radius_hat"]))
    if not values:
        return {"values": [], "mean": float("nan"), "std": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "boundary_fraction": float("nan")}
    arr = np.asarray(values, dtype=np.float64)
    boundary_hits = np.sum((np.abs(arr - kr_min) <= 1e-4) | (np.abs(arr - kr_max) <= 1e-4))
    return {
        "values": values,
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        "ci_low": float(np.percentile(arr, 2.5)),
        "ci_high": float(np.percentile(arr, 97.5)),
        "boundary_fraction": float(boundary_hits / len(arr)),
    }


def determine_status(
    fit_success: bool,
    interval_hit: str,
    n_used_tail: int,
    censoring_ratio_used: float,
    two_end_ratio_used: float,
    bootstrap_boundary_fraction: float,
    weak_identifiability_flag: bool,
    q90_ratio: float,
    q95_ratio: float,
    set_id: int,
) -> Tuple[str, bool, str, List[str]]:
    warnings: List[str] = [XMAX_NOTE]
    if two_end_ratio_used > 0.0:
        warnings.append(TWO_END_NOTE)
    if np.isfinite(bootstrap_boundary_fraction) and bootstrap_boundary_fraction >= 0.15:
        warnings.append("bootstrap estimates frequently hit search boundary")
    if not fit_success:
        return "fit_failed", True, "optimization failed", warnings
    if interval_hit != "none":
        return "boundary_solution", True, "kr_radius_hat hit search boundary", warnings
    if n_used_tail < 30:
        return "low_tail_sample", True, "n_used_tail < 30", warnings
    if set_id == 4:
        return "window_aware_required", True, "Set 4 is high-censoring; window-aware likelihood required", warnings
    if two_end_ratio_used >= 0.2:
        return "high_two_end_censoring", True, "two_end_censoring_ratio_tail >= 0.2", warnings
    if np.isfinite(q90_ratio) and (q90_ratio > 3.0 or q90_ratio < (1.0 / 3.0)):
        return "posterior_predictive_failed", True, "q90_model/q90_observed_used outside [1/3, 3]", warnings
    if np.isfinite(q95_ratio) and (q95_ratio > 3.0 or q95_ratio < (1.0 / 3.0)):
        return "posterior_predictive_failed", True, "q95_model/q95_observed_used outside [1/3, 3]", warnings
    if np.isfinite(bootstrap_boundary_fraction) and bootstrap_boundary_fraction >= 0.30:
        return "bootstrap_boundary_unstable", True, "bootstrap_boundary_fraction >= 0.30", warnings
    if weak_identifiability_flag:
        return "weak_identifiability", False, "profile likelihood delta<=2 interval is too wide", warnings
    if np.isfinite(bootstrap_boundary_fraction) and bootstrap_boundary_fraction >= 0.15:
        return "provisional_ok", False, "bootstrap_boundary_fraction >= 0.15", warnings
    if censoring_ratio_used >= 0.5:
        return "high_censoring", False, "censoring_ratio_tail >= 0.5; provisional interpretation", warnings
    return "ok", False, "", warnings


def build_fit_row(
    set_id: int,
    set_rows: Sequence[dict],
    lmin_fit: float,
    rmin: float,
    rmax: float,
    kr_min: float,
    kr_max: float,
    bootstrap: int,
    profile_grid_size: int,
) -> Tuple[dict, List[dict], List[float]]:
    lengths_all = np.asarray([float(row["observed_length_m"]) for row in set_rows], dtype=np.float64)
    censoring_all = np.asarray([int(row["censoring_class"]) for row in set_rows], dtype=np.int32)
    tail_mask = lengths_all >= lmin_fit
    lengths_tail = lengths_all[tail_mask]
    censoring_tail = censoring_all[tail_mask]
    radius_grid = _radius_grid(rmin, rmax)

    fit = fit_kr_radius(lengths_tail, censoring_tail, rmin, rmax, kr_min, kr_max, radius_grid)
    profile_rows = build_profile_rows(
        set_id, lmin_fit, lengths_tail, censoring_tail, rmin, rmax, kr_min, kr_max, profile_grid_size, radius_grid
    )
    profile_summary = summarize_profile_identifiability(profile_rows, kr_min, kr_max)
    boot = bootstrap_kr_radius(
        lengths_tail,
        censoring_tail,
        rmin,
        rmax,
        kr_min,
        kr_max,
        radius_grid,
        bootstrap,
        seed=20260701 + set_id * 1000 + int(round(lmin_fit * 1000.0)),
    )

    n_total = len(set_rows)
    n_used_tail = len(lengths_tail)
    n_below_lmin_fit = int(np.sum(~tail_mask))
    n_uncensored_tail = int(np.sum(censoring_tail == 0))
    n_censored_tail = int(np.sum(censoring_tail > 0))
    n_one_end_tail = int(np.sum(censoring_tail == 1))
    n_two_end_tail = int(np.sum(censoring_tail == 2))
    censoring_ratio_used = n_censored_tail / n_used_tail if n_used_tail else float("nan")
    two_end_ratio_used = n_two_end_tail / n_used_tail if n_used_tail else float("nan")
    interval_hit = search_interval_hit(fit["kr_radius_hat"], kr_min, kr_max)
    q50_observed = float(np.percentile(lengths_tail, 50)) if n_used_tail else float("nan")
    q90_observed = float(np.percentile(lengths_tail, 90)) if n_used_tail else float("nan")
    q95_observed = float(np.percentile(lengths_tail, 95)) if n_used_tail else float("nan")
    q50_model = marginal_chord_tail_quantile(0.50, lmin_fit, fit["kr_radius_hat"], rmin, rmax, radius_grid) if fit["success"] else float("nan")
    q90_model = marginal_chord_tail_quantile(0.90, lmin_fit, fit["kr_radius_hat"], rmin, rmax, radius_grid) if fit["success"] else float("nan")
    q95_model = marginal_chord_tail_quantile(0.95, lmin_fit, fit["kr_radius_hat"], rmin, rmax, radius_grid) if fit["success"] else float("nan")
    q90_ratio = q90_model / q90_observed if q90_observed > 0.0 else float("nan")
    q95_ratio = q95_model / q95_observed if q95_observed > 0.0 else float("nan")
    status, rejected, reason, warnings = determine_status(
        fit["success"],
        interval_hit,
        n_used_tail,
        censoring_ratio_used,
        two_end_ratio_used,
        boot["boundary_fraction"],
        bool(profile_summary["weak_identifiability_flag"]),
        q90_ratio,
        q95_ratio,
        set_id,
    )

    row = {
        "set_id": set_id,
        "lmin_fit": float(lmin_fit),
        "rmin": float(rmin),
        "rmax": float(rmax),
        "kr_radius_hat": fit["kr_radius_hat"],
        "loglik": fit["loglik"],
        "n_total": n_total,
        "n_used_tail": n_used_tail,
        "n_below_lmin_fit": n_below_lmin_fit,
        "n_uncensored_tail": n_uncensored_tail,
        "n_censored_tail": n_censored_tail,
        "n_one_end_censored_tail": n_one_end_tail,
        "n_two_end_censored_tail": n_two_end_tail,
        "censoring_ratio_tail": censoring_ratio_used,
        "censoring_ratio_used": censoring_ratio_used,
        "two_end_censoring_ratio_tail": two_end_ratio_used,
        "two_end_censoring_ratio_used": two_end_ratio_used,
        "kr_radius_boot_mean": boot["mean"],
        "kr_radius_boot_std": boot["std"],
        "kr_radius_ci_low": boot["ci_low"],
        "kr_radius_ci_high": boot["ci_high"],
        "kr_boot_mean": boot["mean"],
        "kr_boot_std": boot["std"],
        "kr_ci_low": boot["ci_low"],
        "kr_ci_high": boot["ci_high"],
        "bootstrap_boundary_fraction": boot["boundary_fraction"],
        "q50_observed_used": q50_observed,
        "q90_observed_used": q90_observed,
        "q95_observed_used": q95_observed,
        "q50_model": q50_model,
        "q90_model": q90_model,
        "q95_model": q95_model,
        "q90_ratio_model_observed": q90_ratio,
        "q95_ratio_model_observed": q95_ratio,
        "max_loglik": profile_summary["max_loglik"],
        "profile_width_delta2": profile_summary["profile_width_delta2"],
        "weak_identifiability_flag": bool(profile_summary["weak_identifiability_flag"]),
        "length_median_observed_tail": q50_observed,
        "length_p90_observed_tail": q90_observed,
        "radius_model_status": status,
        "length_model_status": status,
        "final_status": status,
        "model_rejected": bool(rejected),
        "rejection_reason": reason,
        "search_interval_hit": interval_hit,
        "warning": "; ".join(warnings),
        "fit_message": fit["message"],
    }
    return row, profile_rows, boot["values"]


def write_csv(rows: Sequence[dict], path: str, fieldnames: Sequence[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(fit_rows: Sequence[dict], path: str, input_summary: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"input_summary": input_summary, "fit_rows": list(fit_rows)}, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate radius power-law candidates from censored trace lengths.")
    parser.add_argument("--trace-h5", help="Input trace HDF5")
    parser.add_argument("--trace-csv", help="Input trace CSV")
    parser.add_argument("--target-set", nargs="+", type=int, help="Optional set_id values to fit")
    parser.add_argument("--rmin", type=float, default=0.5, help="Estimation rmin")
    parser.add_argument("--rmax", type=float, default=250.0, help="Estimation rmax")
    parser.add_argument("--generation-rmin", type=float, default=0.5, help="DFN generation rmin")
    parser.add_argument("--p32-label", default="P32_r_ge_0p5m", help="P32 label")
    parser.add_argument("--lmin-fit-values", nargs="+", type=float, default=[0.1, 0.2, 0.3, 0.5, 0.75], help="lmin fit thresholds")
    parser.add_argument("--allow-rmin-mismatch", action="store_true", help="Allow estimation rmin to differ from DFN generation rmin")
    parser.add_argument("--kr-min", type=float, default=1.5)
    parser.add_argument("--kr-max", type=float, default=5.5)
    parser.add_argument("--bootstrap", type=int, default=0)
    parser.add_argument("--profile-grid-size", type=int, default=161)
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()

    if bool(args.trace_h5) == bool(args.trace_csv):
        raise ValueError("Provide exactly one of --trace-h5 or --trace-csv.")
    if args.rmax <= args.rmin:
        raise ValueError("--rmax must be greater than --rmin.")
    if args.kr_max <= args.kr_min:
        raise ValueError("--kr-max must be greater than --kr-min.")

    # rmin consistency guard
    generation_rmin = None
    if args.trace_h5:
        try:
            with h5py.File(args.trace_h5, "r") as f_meta:
                if "meta" in f_meta and "generation_rmin" in f_meta["meta"]:
                    val = f_meta["meta/generation_rmin"][()]
                    if np.ndim(val) > 0:
                        generation_rmin = float(val.ravel()[0])
                    else:
                        generation_rmin = float(val)
        except Exception as e:
            print(f"[WARNING] Failed to read generation_rmin from trace HDF5: {e}")

    gen_rmin_to_check = args.generation_rmin
    if generation_rmin is not None:
        gen_rmin_to_check = generation_rmin

    rmin_consistency_status = "matched"
    diagnostic_only = False
    warning_msg = None

    if gen_rmin_to_check is not None:
        if abs(gen_rmin_to_check - args.rmin) > 1e-5:
            if not args.allow_rmin_mismatch:
                raise ValueError(
                    f"mismatch: DFN generation_rmin ({gen_rmin_to_check}) does not match "
                    f"estimation rmin ({args.rmin}). Use --allow-rmin-mismatch to bypass."
                )
            else:
                rmin_consistency_status = "mismatch_diagnostic_only"
                diagnostic_only = True
                warning_msg = "generation_rmin and estimator rmin are inconsistent; kr recovery should not be interpreted"
                print(f"[WARNING] {warning_msg}")

    rows = load_trace_rows_from_h5(args.trace_h5) if args.trace_h5 else load_trace_rows_from_csv(args.trace_csv)
    target_sets = set(args.target_set) if args.target_set else None
    grouped = group_rows_by_set(rows, target_sets)
    if not grouped:
        raise ValueError("No matching set rows found for the given input/target-set.")

    os.makedirs(args.outdir, exist_ok=True)
    fit_rows: List[dict] = []
    profile_rows: List[dict] = []
    bootstrap_rows: List[dict] = []

    for set_id, set_rows in grouped.items():
        for lmin_fit in args.lmin_fit_values:
            fit_row, profile, boot_values = build_fit_row(
                set_id=set_id,
                set_rows=set_rows,
                lmin_fit=float(lmin_fit),
                rmin=float(args.rmin),
                rmax=float(args.rmax),
                kr_min=float(args.kr_min),
                kr_max=float(args.kr_max),
                bootstrap=int(args.bootstrap),
                profile_grid_size=int(args.profile_grid_size),
            )
            fit_row["rmin_consistency_status"] = rmin_consistency_status
            fit_row["diagnostic_only"] = bool(diagnostic_only)
            if warning_msg:
                fit_row["warning"] = (fit_row["warning"] + "; " if fit_row["warning"] else "") + warning_msg

            fit_rows.append(fit_row)
            profile_rows.extend(profile)
            for idx, value in enumerate(boot_values, start=1):
                bootstrap_rows.append(
                    {
                        "set_id": set_id,
                        "lmin_fit": float(lmin_fit),
                        "bootstrap_iter": idx,
                        "kr_radius_hat": value,
                    }
                )

    fit_fieldnames = [
        "set_id",
        "lmin_fit",
        "rmin",
        "rmax",
        "kr_radius_hat",
        "loglik",
        "n_total",
        "n_used_tail",
        "n_below_lmin_fit",
        "n_uncensored_tail",
        "n_censored_tail",
        "n_one_end_censored_tail",
        "n_two_end_censored_tail",
        "censoring_ratio_tail",
        "censoring_ratio_used",
        "two_end_censoring_ratio_tail",
        "two_end_censoring_ratio_used",
        "kr_radius_boot_mean",
        "kr_radius_boot_std",
        "kr_radius_ci_low",
        "kr_radius_ci_high",
        "kr_boot_mean",
        "kr_boot_std",
        "kr_ci_low",
        "kr_ci_high",
        "bootstrap_boundary_fraction",
        "q50_observed_used",
        "q90_observed_used",
        "q95_observed_used",
        "q50_model",
        "q90_model",
        "q95_model",
        "q90_ratio_model_observed",
        "q95_ratio_model_observed",
        "max_loglik",
        "profile_width_delta2",
        "weak_identifiability_flag",
        "length_median_observed_tail",
        "length_p90_observed_tail",
        "radius_model_status",
        "length_model_status",
        "final_status",
        "model_rejected",
        "rejection_reason",
        "search_interval_hit",
        "warning",
        "fit_message",
        "rmin_consistency_status",
        "diagnostic_only",
    ]
    fit_csv = os.path.join(args.outdir, "radius_powerlaw_fit_by_set.csv")
    fit_json = os.path.join(args.outdir, "radius_powerlaw_fit_by_set.json")
    profile_csv = os.path.join(args.outdir, "radius_powerlaw_profile_likelihood.csv")
    write_csv(fit_rows, fit_csv, fit_fieldnames)
    write_csv(profile_rows, profile_csv, ["set_id", "lmin_fit", "kr_radius", "loglik", "max_loglik", "delta_loglik"])
    write_json(
        fit_rows,
        fit_json,
        {
            "trace_csv": args.trace_csv,
            "trace_h5": args.trace_h5,
            "target_set": args.target_set,
            "rmin": float(args.rmin),
            "rmax": float(args.rmax),
            "lmin_fit_values": [float(v) for v in args.lmin_fit_values],
            "kr_min": float(args.kr_min),
            "kr_max": float(args.kr_max),
            "bootstrap": int(args.bootstrap),
            "profile_grid_size": int(args.profile_grid_size),
            "model_note": "v3 estimates radius power-law kr candidates from size-biased radius-to-chord likelihood; P32 is not estimated.",
        },
    )
    if args.bootstrap > 0:
        boot_csv = os.path.join(args.outdir, "radius_powerlaw_bootstrap.csv")
        write_csv(bootstrap_rows, boot_csv, ["set_id", "lmin_fit", "bootstrap_iter", "kr_radius_hat"])
        print(f"[*] Bootstrap CSV written to: {boot_csv}")

    print("[*] Radius power-law v3 fit summary")
    for row in fit_rows:
        print(
            f"    - Set {row['set_id']}, lmin={row['lmin_fit']:.3f} m: "
            f"kr_radius_hat={row['kr_radius_hat']:.5f}, n_used_tail={row['n_used_tail']}, "
            f"status={row['radius_model_status']}, rejected={row['model_rejected']}, "
            f"reason={row['rejection_reason'] or 'none'}"
        )
    print(f"[*] Fit CSV written to: {fit_csv}")
    print(f"[*] Fit JSON written to: {fit_json}")
    print(f"[*] Profile likelihood CSV written to: {profile_csv}")


if __name__ == "__main__":
    main()
