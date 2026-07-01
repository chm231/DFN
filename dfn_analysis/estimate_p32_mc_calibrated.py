import argparse
import csv
import h5py
import math
import os
import sys
from typing import Dict, List, Sequence, Tuple

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dfn_analysis.build_p32_pilot_summary import SITE_SET_CONFIG, support_scaled_p32
from dfn_analysis.estimate_radius_powerlaw_window_mc import (
    SITE_FISHER_PARAMS,
    clip_segments_to_bbox_vectorized,
    clip_segments_to_convex_polygon_vectorized,
    empirical_trace_directions_yz,
    load_trace_data_from_h5,
    mean_pole_from_trend_plunge,
    normals_to_trace_directions_yz,
    orientation_conditioned_trace_directions_yz,
    sample_fisher_normals,
    sample_size_biased_radius,
    sample_true_chords,
)
from dfn_analysis.export_setwise_3d_traces import load_hdf5_dfn
from dfn_analysis.summarize_setwise_trace_statistics import (
    build_summary_rows,
    compute_total_observation_area,
    load_rough_face_collection_from_h5,
    load_trace_rows_from_h5,
)


DEFAULT_DFN_H5 = {
    "forsmark": "storage/output/dfn_forsmark_rmin0p5/dfn_export_for_python.h5",
    "laxemar": "storage/output/dfn_laxemar_rmin0p5/dfn_export_for_python.h5",
}
CALIBRATION_FACTOR_MODE_PROXY = "conditional_visible_trace_proxy"
CALIBRATION_FACTOR_MODE_UNIT = "unit_p32_forward_mc"


def read_csv(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(rows: Sequence[dict], path: str) -> None:
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def to_float(row: dict, key: str) -> float:
    value = row.get(key, "")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def powerlaw_moment(kr: float, rmin: float, rmax: float, order: float) -> float:
    exponent = -kr
    if abs(exponent) < 1e-12:
        norm = 1.0 / math.log(rmax / rmin)
    else:
        norm = exponent / (rmax**exponent - rmin**exponent)

    moment_power = order - kr
    if abs(moment_power) < 1e-12:
        integral = math.log(rmax / rmin)
    else:
        integral = (rmax**moment_power - rmin**moment_power) / moment_power
    return norm * integral


def exponential_raw_moment(rate: float, radius: float, order: int) -> float:
    if order == 1:
        return -math.exp(-rate * radius) * (radius / rate + 1.0 / (rate * rate))
    if order == 2:
        return -math.exp(-rate * radius) * (
            radius * radius / rate + 2.0 * radius / (rate * rate) + 2.0 / (rate**3)
        )
    raise ValueError(f"Unsupported exponential moment order: {order}")


def exponential_moment(r0: float, rmin: float, rmax: float, order: int) -> float:
    rate = 1.0 / r0
    z = math.exp(-rate * rmin) - math.exp(-rate * rmax)
    integral = exponential_raw_moment(rate, rmax, order) - exponential_raw_moment(rate, rmin, order)
    return (rate / z) * integral


def radius_moments(site: str, set_id: int, kr: float, rmin: float, rmax: float) -> Tuple[float, float]:
    cfg = SITE_SET_CONFIG.get(site, {}).get(set_id)
    if cfg is None:
        return float("nan"), float("nan")
    dist_type = str(cfg["dist_type"])
    if dist_type == "powerlaw":
        return powerlaw_moment(kr, rmin, rmax, 1.0), powerlaw_moment(kr, rmin, rmax, 2.0)
    if dist_type == "exponential":
        r0 = float(cfg["r0"])
        return exponential_moment(r0, rmin, rmax, 1), exponential_moment(r0, rmin, rmax, 2)
    return float("nan"), float("nan")


def sample_population_radius(site: str, set_id: int, kr: float, rmin: float, rmax: float, size: int, rng: np.random.Generator) -> np.ndarray:
    cfg = SITE_SET_CONFIG.get(site, {}).get(set_id)
    if cfg is None:
        raise ValueError(f"Missing SITE_SET_CONFIG for site={site}, set_id={set_id}")
    dist_type = str(cfg["dist_type"])
    u = rng.uniform(0.0, 1.0, size=size)
    if dist_type == "powerlaw":
        alpha = kr + 1.0
        if abs(alpha - 1.0) < 1e-12:
            return rmin * np.exp(u * np.log(rmax / rmin))
        return (rmin ** (1.0 - alpha) + u * (rmax ** (1.0 - alpha) - rmin ** (1.0 - alpha))) ** (1.0 / (1.0 - alpha))
    if dist_type == "exponential":
        lam = 1.0 / float(cfg["r0"])
        a = math.exp(-lam * rmin)
        b = math.exp(-lam * rmax)
        return -(1.0 / lam) * np.log(a - u * (a - b))
    raise ValueError(f"Unsupported distribution type for unit_p32_forward_mc: {dist_type}")


def mean_intersection_chord_length(site: str, set_id: int, kr: float, rmin: float, rmax: float) -> float:
    mean_r, mean_r2 = radius_moments(site, set_id, kr, rmin, rmax)
    if not np.isfinite(mean_r) or not np.isfinite(mean_r2) or mean_r <= 0.0:
        return float("nan")
    return (math.pi / 2.0) * (mean_r2 / mean_r)


def load_p21_summary(trace_h5: str, rough_mesh_h5: str) -> Tuple[Dict[int, dict], float]:
    rows = load_trace_rows_from_h5(trace_h5)
    rough_faces = load_rough_face_collection_from_h5(rough_mesh_h5)
    observation_area_m2 = compute_total_observation_area(rough_faces)
    summary_rows = build_summary_rows(rows, observation_area_m2)
    return {int(row["set_id"]): row for row in summary_rows}, float(observation_area_m2)


def load_orientation_factors(dfn_h5: str) -> Dict[int, float]:
    data = load_hdf5_dfn(dfn_h5)
    out: Dict[int, float] = {}
    set_ids = np.asarray(data["set_ids"]).astype(np.int32).ravel()
    normals = np.asarray(data["normals"], dtype=np.float64)
    for set_id in sorted({int(v) for v in set_ids.tolist()}):
        mask = set_ids == set_id
        set_normals = normals[mask]
        if len(set_normals) == 0:
            out[set_id] = float("nan")
            continue
        g_factors = np.sqrt(np.clip(1.0 - set_normals[:, 0] ** 2, 0.0, 1.0))
        out[set_id] = float(np.mean(g_factors))
    return out


def load_face_x_positions(trace_h5: str) -> np.ndarray:
    with h5py.File(trace_h5, "r") as f:
        if "meta" in f and "face_x_positions_m" in f["meta"]:
            return np.asarray(f["meta"]["face_x_positions_m"][:], dtype=np.float64).ravel()
        if "traces" in f and "face_x_m" in f["traces"]:
            return np.unique(np.asarray(f["traces"]["face_x_m"][:], dtype=np.float64).ravel())
    raise ValueError(f"Could not find face_x_positions_m in trace HDF5: {trace_h5}")


def simulate_visible_lengths_all(
    polygon_yz: np.ndarray,
    directions_yz: np.ndarray,
    radii: np.ndarray,
    rng: np.random.Generator,
    window_mode: str,
    direction_mode: str,
    set_id: int,
    site: str,
) -> Tuple[np.ndarray, np.ndarray]:
    bbox_min = np.min(polygon_yz, axis=0)
    bbox_max = np.max(polygon_yz, axis=0)
    bbox_w = bbox_max[0] - bbox_min[0]
    bbox_h = bbox_max[1] - bbox_min[1]
    true_lengths = sample_true_chords(radii, rng)
    n_samples = len(radii)

    if direction_mode == "orientation_conditioned":
        dir_pool = orientation_conditioned_trace_directions_yz(set_id, site, n_samples * 3, rng)
        direction_idx = rng.integers(0, len(dir_pool), size=n_samples)
        directions = dir_pool[direction_idx]
    else:
        direction_idx = rng.integers(0, len(directions_yz), size=n_samples)
        directions = directions_yz[direction_idx]

    proposal_areas = (bbox_w + true_lengths * np.abs(directions[:, 0])) * (bbox_h + true_lengths * np.abs(directions[:, 1]))
    expand = 0.5 * true_lengths[:, None]
    centers = rng.uniform(bbox_min - expand, bbox_max + expand)

    if window_mode == "bbox":
        visible_lengths, _ = clip_segments_to_bbox_vectorized(centers, directions, true_lengths, bbox_min, bbox_max)
    elif window_mode == "polygon":
        visible_lengths, _ = clip_segments_to_convex_polygon_vectorized(centers, directions, true_lengths, polygon_yz)
    else:
        raise ValueError(f"Unsupported window_mode: {window_mode}")

    return visible_lengths, proposal_areas


def estimate_calibration_factor(
    site: str,
    set_id: int,
    kr: float,
    rmin: float,
    rmax: float,
    polygon_yz: np.ndarray,
    directions_yz: np.ndarray,
    orientation_factor: float,
    mc_samples: int,
    rng_seed: int,
    window_mode: str,
    direction_mode: str,
) -> float:
    rng = np.random.default_rng(rng_seed)
    radii = sample_size_biased_radius(kr, rmin, rmax, mc_samples, rng)
    visible_lengths, proposal_areas = simulate_visible_lengths_all(
        polygon_yz=polygon_yz,
        directions_yz=directions_yz,
        radii=radii,
        rng=rng,
        window_mode=window_mode,
        direction_mode=direction_mode,
        set_id=set_id,
        site=site,
    )

    observation_area = abs(0.5 * float(np.dot(polygon_yz[:, 0], np.roll(polygon_yz[:, 1], -1)) - np.dot(polygon_yz[:, 1], np.roll(polygon_yz[:, 0], -1))))
    if observation_area <= 0.0:
        return float("nan")

    mean_r, mean_r2 = radius_moments(site, set_id, kr, rmin, rmax)
    if not np.isfinite(mean_r) or not np.isfinite(mean_r2) or mean_r2 <= 0.0:
        return float("nan")
    if not np.isfinite(orientation_factor) or orientation_factor <= 0.0:
        return float("nan")

    intersection_intensity_per_area = (2.0 * orientation_factor * mean_r) / (math.pi * mean_r2)
    visible_length_per_intersection = float(np.mean((proposal_areas * visible_lengths) / observation_area))
    return intersection_intensity_per_area * visible_length_per_intersection


def infer_calibration_factor_mode() -> str:
    return CALIBRATION_FACTOR_MODE_PROXY


def estimate_unit_p32_forward_mc(
    site: str,
    set_id: int,
    kr: float,
    rmin: float,
    rmax: float,
    polygon_yz: np.ndarray,
    face_x_positions: np.ndarray,
    total_observation_area: float,
    mc_samples: int,
    mc_replicates: int,
    rng_seed: int,
    window_mode: str,
) -> dict:
    if total_observation_area <= 0.0:
        return {
            "calibration_factor_C": float("nan"),
            "calibration_factor_std": float("nan"),
            "calibration_factor_ci_low": float("nan"),
            "calibration_factor_ci_high": float("nan"),
            "unit_p32_mc_volume": float("nan"),
            "mean_fracture_area": float("nan"),
            "fracture_number_density_for_unit_p32": float("nan"),
        }

    params = SITE_FISHER_PARAMS.get(site, {}).get(set_id)
    if params is None:
        raise ValueError(f"Missing Fisher parameters for unit_p32_forward_mc: site={site}, set_id={set_id}")
    trend, plunge, kappa = params
    mean_pole = mean_pole_from_trend_plunge(trend, plunge)

    bbox_min = np.min(polygon_yz, axis=0)
    bbox_max = np.max(polygon_yz, axis=0)
    bbox_w = float(bbox_max[0] - bbox_min[0])
    bbox_h = float(bbox_max[1] - bbox_min[1])

    _, mean_r2 = radius_moments(site, set_id, kr, rmin, rmax)
    mean_fracture_area = math.pi * mean_r2 if np.isfinite(mean_r2) else float("nan")
    fracture_number_density = 1.0 / mean_fracture_area if np.isfinite(mean_fracture_area) and mean_fracture_area > 0.0 else float("nan")

    replicate_values: List[float] = []
    mean_proposal_volume_accum = 0.0
    face_x_positions = np.asarray(face_x_positions, dtype=np.float64)

    for rep in range(mc_replicates):
        rng = np.random.default_rng(rng_seed + rep)
        total_weighted_length = 0.0
        proposal_volume_sum = 0.0

        for face_x in face_x_positions:
            radii = sample_population_radius(site, set_id, kr, rmin, rmax, mc_samples, rng)
            normals = sample_fisher_normals(mean_pole, kappa, mc_samples, rng)
            directions_yz, valid = normals_to_trace_directions_yz(normals)
            if not np.any(valid):
                continue

            radii = radii[valid]
            normals = normals[valid]
            directions_yz = directions_yz[valid]

            sin_theta = np.sqrt(np.clip(1.0 - normals[:, 0] ** 2, 0.0, 1.0))
            valid_theta = sin_theta > 1e-8
            if not np.any(valid_theta):
                continue

            radii = radii[valid_theta]
            normals = normals[valid_theta]
            directions_yz = directions_yz[valid_theta]
            sin_theta = sin_theta[valid_theta]

            x_half = radii * sin_theta
            proposal_w = bbox_w + 4.0 * radii
            proposal_h = bbox_h + 4.0 * radii
            proposal_volume = 2.0 * x_half * proposal_w * proposal_h
            proposal_volume_sum += float(np.mean(proposal_volume))

            center_x = rng.uniform(face_x - x_half, face_x + x_half)
            center_y = rng.uniform(bbox_min[0] - 2.0 * radii, bbox_max[0] + 2.0 * radii)
            center_z = rng.uniform(bbox_min[1] - 2.0 * radii, bbox_max[1] + 2.0 * radii)
            centers_yz = np.column_stack([center_y, center_z])

            dx = face_x - center_x
            chord_offsets = np.abs(dx) / sin_theta
            valid_intersections = chord_offsets <= radii + 1e-10
            if not np.any(valid_intersections):
                continue

            normals = normals[valid_intersections]
            directions_yz = directions_yz[valid_intersections]
            radii = radii[valid_intersections]
            sin_theta = sin_theta[valid_intersections]
            chord_offsets = chord_offsets[valid_intersections]
            centers_yz = centers_yz[valid_intersections]
            proposal_volume = proposal_volume[valid_intersections]
            dx = dx[valid_intersections]

            chord_lengths = 2.0 * np.sqrt(np.maximum(radii * radii - chord_offsets * chord_offsets, 0.0))
            t = dx / np.maximum(sin_theta * sin_theta, 1e-12)
            line_midpoints_yz = centers_yz + np.column_stack(
                [-t * normals[:, 0] * normals[:, 1], -t * normals[:, 0] * normals[:, 2]]
            )

            if window_mode == "bbox":
                visible_lengths, classes = clip_segments_to_bbox_vectorized(
                    line_midpoints_yz,
                    directions_yz,
                    chord_lengths,
                    bbox_min,
                    bbox_max,
                )
            elif window_mode == "polygon":
                visible_lengths, classes = clip_segments_to_convex_polygon_vectorized(
                    line_midpoints_yz,
                    directions_yz,
                    chord_lengths,
                    polygon_yz,
                )
            else:
                raise ValueError(f"Unsupported window mode: {window_mode}")

            accepted = (classes >= 0) & (visible_lengths > 0.0)
            if not np.any(accepted):
                continue

            weights = fracture_number_density * proposal_volume[accepted] / mc_samples
            total_weighted_length += float(np.sum(weights * visible_lengths[accepted]))

        replicate_values.append(total_weighted_length / total_observation_area)
        mean_proposal_volume_accum += proposal_volume_sum / max(len(face_x_positions), 1)

    values = np.asarray(replicate_values, dtype=np.float64)
    return {
        "calibration_factor_C": float(np.mean(values)) if len(values) else float("nan"),
        "calibration_factor_std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        "calibration_factor_ci_low": float(np.percentile(values, 2.5)) if len(values) else float("nan"),
        "calibration_factor_ci_high": float(np.percentile(values, 97.5)) if len(values) else float("nan"),
        "unit_p32_mc_volume": float(mean_proposal_volume_accum / max(mc_replicates, 1)),
        "mean_fracture_area": mean_fracture_area,
        "fracture_number_density_for_unit_p32": fracture_number_density,
    }


def classify_p32_status(
    adoption_status: str,
    recovery_ci_status: str,
    kr_used: float,
    kr_ci_low: float,
    kr_ci_high: float,
    calibration_factor_mode: str,
) -> str:
    if calibration_factor_mode != CALIBRATION_FACTOR_MODE_UNIT:
        return "p32_scaffold_only"
    if adoption_status == "rejected":
        return "p32_hold"
    if adoption_status == "provisional_accepted" and recovery_ci_status == "systematic_bias":
        return "p32_mc_provisional_systematic_bias"
    if adoption_status == "accepted":
        if np.isfinite(kr_used) and np.isfinite(kr_ci_low) and np.isfinite(kr_ci_high):
            rel_width = (kr_ci_high - kr_ci_low) / max(abs(kr_used), 1e-12)
            if rel_width > 1.0:
                return "p32_mc_candidate_with_uncertainty"
        return "p32_mc_pilot_candidate"
    if adoption_status == "provisional_accepted":
        return "p32_mc_provisional"
    return "p32_hold"


def build_notes(base_notes: str, calibration_factor: float, p32_hat: float, calibration_mode: str) -> str:
    notes = [base_notes] if base_notes else []
    notes.append(f"calibration_factor_mode={calibration_mode}")
    if np.isfinite(calibration_factor):
        notes.append(f"calibration_method={calibration_mode}")
        notes.append(f"calibration_factor_C={calibration_factor:.6f}")
    if np.isfinite(p32_hat):
        notes.append("p32_method=observed_P21_over_calibration_factor")
    notes.append("p32_ci_method=kr_only_calibration_propagation")
    return "; ".join(note for note in notes if note)


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate MC-calibrated P32 from observed trace intensity under the effective-rmin window MC model.")
    parser.add_argument("--trace-h5", required=True, help="Trace HDF5 created by export_setwise_3d_traces.py")
    parser.add_argument("--dfn-h5", help="DFN export HDF5 for orientation factor estimation.")
    parser.add_argument(
        "--kr-summary-csv",
        default="storage/output/final_kr_recovery_summary_effective_rmin.csv",
        help="Final effective-rmin kr recovery summary CSV.",
    )
    parser.add_argument(
        "--bootstrap-csv",
        default="storage/output/final_kr_bootstrap_effective_rmin/final_kr_bootstrap_summary_effective_rmin.csv",
        help="Bootstrap summary CSV.",
    )
    parser.add_argument("--site", choices=["forsmark", "laxemar"], required=True)
    parser.add_argument("--target-set", nargs="+", type=int, required=True)
    parser.add_argument("--p32-label", default="P32_r_ge_0p5m")
    parser.add_argument("--set-rmin-mode", default="effective_generation")
    parser.add_argument(
        "--rough-mesh-h5",
        default="storage/output/rough_face_mesh_collection/synthetic_rough_face_collection.h5",
        help="Rough face collection HDF5 used to compute observed P21.",
    )
    parser.add_argument("--window-mode", choices=["polygon", "bbox"], default="polygon")
    parser.add_argument("--direction-mode", choices=["empirical_trace", "orientation_conditioned"], default="empirical_trace")
    parser.add_argument("--mc-samples", type=int, default=50000)
    parser.add_argument(
        "--calibration-factor-mode",
        choices=[CALIBRATION_FACTOR_MODE_PROXY, CALIBRATION_FACTOR_MODE_UNIT],
        default=CALIBRATION_FACTOR_MODE_PROXY,
        help="Proxy scaffold mode or unit-P32 forward MC calibration mode.",
    )
    parser.add_argument("--unit-p32-mc-replicates", type=int, default=32, help="Number of MC replicates for unit_p32_forward_mc mode.")
    parser.add_argument(
        "--outcsv",
        default="storage/output/p32_mc_calibrated_effective_rmin/p32_mc_calibrated_summary.csv",
        help="Output CSV path.",
    )
    args = parser.parse_args()

    dfn_h5 = args.dfn_h5 or DEFAULT_DFN_H5[args.site]
    kr_rows = [row for row in read_csv(args.kr_summary_csv) if str(row["site"]) == args.site and int(row["set_id"]) in set(args.target_set)]
    bootstrap_map = {
        (str(row["site"]), int(row["set_id"])): row
        for row in read_csv(args.bootstrap_csv)
        if str(row["site"]) == args.site and int(row["set_id"]) in set(args.target_set)
    }
    if not kr_rows:
        raise ValueError("No matching rows found in kr summary CSV for the requested site/sets.")

    trace_rows, polygon_yz = load_trace_data_from_h5(args.trace_h5)
    if polygon_yz is None:
        raise ValueError("trace_h5 must contain /meta/tunnel_poly_yz for MC-calibrated P32 estimation.")
    grouped_rows: Dict[int, List[dict]] = {}
    for row in trace_rows:
        set_id = int(row["set_id"])
        if set_id in set(args.target_set):
            grouped_rows.setdefault(set_id, []).append(row)

    p21_summary_map, total_observation_area = load_p21_summary(args.trace_h5, args.rough_mesh_h5)
    orientation_map = load_orientation_factors(dfn_h5)
    face_x_positions = load_face_x_positions(args.trace_h5)

    out_rows: List[dict] = []
    for idx, row in enumerate(sorted(kr_rows, key=lambda r: int(r["set_id"]))):
        set_id = int(row["set_id"])
        boot_row = bootstrap_map.get((args.site, set_id), {})
        observed_p21 = to_float(p21_summary_map.get(set_id, {}), "P21_observed")
        orientation_factor = float(orientation_map.get(set_id, float("nan")))
        set_likelihood_rmin = to_float(row, "set_likelihood_rmin")
        set_effective_rmin = to_float(row, "set_effective_generation_rmin")
        kr_used = to_float(row, "kr_hat")
        kr_ci_low = to_float(boot_row, "kr_ci_low")
        kr_ci_high = to_float(boot_row, "kr_ci_high")
        kr_true = to_float(boot_row, "kr_true")
        recovery_ci_status = str(boot_row.get("recovery_ci_status", ""))
        adoption_status = str(row.get("adoption_status", ""))
        directions_yz = empirical_trace_directions_yz(grouped_rows.get(set_id, []))
        calibration_mode = args.calibration_factor_mode

        base_seed = 260000 + set_id * 100 + idx * 10000
        if calibration_mode == CALIBRATION_FACTOR_MODE_UNIT:
            unit_main = estimate_unit_p32_forward_mc(
                site=args.site,
                set_id=set_id,
                kr=kr_used,
                rmin=set_likelihood_rmin,
                rmax=250.0,
                polygon_yz=polygon_yz,
                face_x_positions=face_x_positions,
                total_observation_area=total_observation_area,
                mc_samples=args.mc_samples,
                mc_replicates=args.unit_p32_mc_replicates,
                rng_seed=base_seed,
                window_mode=args.window_mode,
            )
            c_hat = float(unit_main["calibration_factor_C"])
            c_std = float(unit_main["calibration_factor_std"])
            c_ci_low = float(unit_main["calibration_factor_ci_low"])
            c_ci_high = float(unit_main["calibration_factor_ci_high"])
            unit_volume = float(unit_main["unit_p32_mc_volume"])
            mean_fracture_area = float(unit_main["mean_fracture_area"])
            fracture_number_density = float(unit_main["fracture_number_density_for_unit_p32"])
            c_low = estimate_unit_p32_forward_mc(
                site=args.site,
                set_id=set_id,
                kr=kr_ci_low,
                rmin=set_likelihood_rmin,
                rmax=250.0,
                polygon_yz=polygon_yz,
                face_x_positions=face_x_positions,
                total_observation_area=total_observation_area,
                mc_samples=args.mc_samples,
                mc_replicates=args.unit_p32_mc_replicates,
                rng_seed=base_seed + 1,
                window_mode=args.window_mode,
            )["calibration_factor_C"] if np.isfinite(kr_ci_low) else float("nan")
            c_high = estimate_unit_p32_forward_mc(
                site=args.site,
                set_id=set_id,
                kr=kr_ci_high,
                rmin=set_likelihood_rmin,
                rmax=250.0,
                polygon_yz=polygon_yz,
                face_x_positions=face_x_positions,
                total_observation_area=total_observation_area,
                mc_samples=args.mc_samples,
                mc_replicates=args.unit_p32_mc_replicates,
                rng_seed=base_seed + 2,
                window_mode=args.window_mode,
            )["calibration_factor_C"] if np.isfinite(kr_ci_high) else float("nan")
        else:
            c_hat = estimate_calibration_factor(
                site=args.site,
                set_id=set_id,
                kr=kr_used,
                rmin=set_likelihood_rmin,
                rmax=250.0,
                polygon_yz=polygon_yz,
                directions_yz=directions_yz,
                orientation_factor=orientation_factor,
                mc_samples=args.mc_samples,
                rng_seed=base_seed,
                window_mode=args.window_mode,
                direction_mode=args.direction_mode,
            )
            c_low = estimate_calibration_factor(
                site=args.site,
                set_id=set_id,
                kr=kr_ci_low,
                rmin=set_likelihood_rmin,
                rmax=250.0,
                polygon_yz=polygon_yz,
                directions_yz=directions_yz,
                orientation_factor=orientation_factor,
                mc_samples=args.mc_samples,
                rng_seed=base_seed + 1,
                window_mode=args.window_mode,
                direction_mode=args.direction_mode,
            ) if np.isfinite(kr_ci_low) else float("nan")
            c_high = estimate_calibration_factor(
                site=args.site,
                set_id=set_id,
                kr=kr_ci_high,
                rmin=set_likelihood_rmin,
                rmax=250.0,
                polygon_yz=polygon_yz,
                directions_yz=directions_yz,
                orientation_factor=orientation_factor,
                mc_samples=args.mc_samples,
                rng_seed=base_seed + 2,
                window_mode=args.window_mode,
                direction_mode=args.direction_mode,
            ) if np.isfinite(kr_ci_high) else float("nan")
            c_std = float("nan")
            c_ci_low = float("nan")
            c_ci_high = float("nan")
            unit_volume = float("nan")
            mean_fracture_area = math.pi * radius_moments(args.site, set_id, kr_used, set_likelihood_rmin, 250.0)[1]
            fracture_number_density = 1.0 / mean_fracture_area if np.isfinite(mean_fracture_area) and mean_fracture_area > 0.0 else float("nan")

        p32_hat = observed_p21 / c_hat if np.isfinite(observed_p21) and np.isfinite(c_hat) and c_hat > 0.0 else float("nan")
        p32_from_low = observed_p21 / c_low if np.isfinite(observed_p21) and np.isfinite(c_low) and c_low > 0.0 else float("nan")
        p32_from_high = observed_p21 / c_high if np.isfinite(observed_p21) and np.isfinite(c_high) and c_high > 0.0 else float("nan")
        p32_candidates = [value for value in (p32_from_low, p32_from_high) if np.isfinite(value)]
        if p32_candidates:
            p32_ci_low = min(p32_candidates)
            p32_ci_high = max(p32_candidates)
        else:
            p32_ci_low = float("nan")
            p32_ci_high = float("nan")

        p32_reference = support_scaled_p32(args.site, set_id, kr_true, set_effective_rmin, 250.0)
        p32_abs_error = abs(p32_hat - p32_reference) if np.isfinite(p32_hat) and np.isfinite(p32_reference) else float("nan")
        p32_rel_error = 100.0 * p32_abs_error / abs(p32_reference) if np.isfinite(p32_abs_error) and abs(p32_reference) > 0.0 else float("nan")

        out_rows.append(
            {
                "site": args.site,
                "set_id": set_id,
                "p32_label": args.p32_label,
                "set_effective_generation_rmin": set_effective_rmin,
                "set_likelihood_rmin": set_likelihood_rmin,
                "kr_used": kr_used,
                "kr_ci_low": kr_ci_low,
                "kr_ci_high": kr_ci_high,
                "observed_P21": observed_p21,
                "calibration_factor_C": c_hat,
                "calibration_factor_mode": calibration_mode,
                "unit_p32_mc_replicates": args.unit_p32_mc_replicates if calibration_mode == CALIBRATION_FACTOR_MODE_UNIT else 0,
                "unit_p32_mc_volume": unit_volume,
                "mean_fracture_area": mean_fracture_area,
                "fracture_number_density_for_unit_p32": fracture_number_density,
                "calibration_factor_std": c_std,
                "calibration_factor_ci_low": c_ci_low,
                "calibration_factor_ci_high": c_ci_high,
                "P32_hat": p32_hat,
                "P32_ci_low": p32_ci_low,
                "P32_ci_high": p32_ci_high,
                "P32_reference": p32_reference,
                "P32_abs_error": p32_abs_error,
                "P32_relative_error_percent": p32_rel_error,
                "kr_adoption_status": adoption_status,
                "p32_status": classify_p32_status(adoption_status, recovery_ci_status, kr_used, kr_ci_low, kr_ci_high, calibration_mode),
                "notes": build_notes(str(row.get("notes", "")), c_hat, p32_hat, calibration_mode),
            }
        )

    os.makedirs(os.path.dirname(args.outcsv) or ".", exist_ok=True)
    write_csv(out_rows, args.outcsv)
    print(f"[*] MC-calibrated P32 summary written to: {args.outcsv}")


if __name__ == "__main__":
    main()
