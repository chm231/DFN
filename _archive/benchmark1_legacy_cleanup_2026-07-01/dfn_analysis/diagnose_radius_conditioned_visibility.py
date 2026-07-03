import argparse
import os
from typing import Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np

from dfn_analysis.estimate_radius_powerlaw_window_mc import (
    clip_segments_to_convex_polygon_vectorized,
    empirical_trace_directions_yz,
    sample_true_chords,
)
from dfn_analysis.export_setwise_3d_traces import load_hdf5_dfn


def decode_bytes(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def write_csv(rows: Sequence[dict], path: str) -> None:
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    import csv

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def load_trace_dataset(trace_h5: str) -> tuple[List[dict], np.ndarray, np.ndarray]:
    rows: List[dict] = []
    with h5py.File(trace_h5, "r") as f:
        grp = f["traces"]
        p0 = grp["p0_xyz"][:].astype(np.float64)
        p1 = grp["p1_xyz"][:].astype(np.float64)
        radius_m = grp["radius_m"][:].astype(np.float64) if "radius_m" in grp else None
        face_x_positions = f["meta/face_x_positions_m"][:].astype(np.float64)
        polygon_yz = f["meta/tunnel_poly_yz"][:].astype(np.float64)
        for idx in range(len(grp["trace_id"])):
            rows.append(
                {
                    "trace_id": int(grp["trace_id"][idx]),
                    "fracture_id": int(grp["fracture_id"][idx]),
                    "radius_m": float(radius_m[idx]) if radius_m is not None else float("nan"),
                    "set_id": int(grp["set_id"][idx]),
                    "face_id": int(grp["face_id"][idx]),
                    "face_x_m": float(grp["face_x_m"][idx]),
                    "observed_length_m": float(grp["observed_length_m"][idx]),
                    "censoring_class": int(grp["censoring_class"][idx]),
                    "p0_y": float(p0[idx, 1]),
                    "p0_z": float(p0[idx, 2]),
                    "p1_y": float(p1[idx, 1]),
                    "p1_z": float(p1[idx, 2]),
                    "p0_endpoint_type": decode_bytes(grp["p0_endpoint_type"][idx]),
                    "p1_endpoint_type": decode_bytes(grp["p1_endpoint_type"][idx]),
                }
            )
    return rows, polygon_yz, face_x_positions


def normalize_rows_with_radius(rows: Sequence[dict], dfn_data: dict) -> List[dict]:
    radii = dfn_data["radii"]
    normalized: List[dict] = []
    for row in rows:
        new_row = dict(row)
        if not np.isfinite(float(new_row["radius_m"])):
            fracture_id = int(new_row["fracture_id"])
            new_row["radius_m"] = float(radii[fracture_id])
        normalized.append(new_row)
    return normalized


def build_face_id_to_x(face_x_positions: np.ndarray) -> Dict[int, float]:
    return {idx + 1: float(x) for idx, x in enumerate(face_x_positions)}


def fracture_face_intersection_possible(center_xyz: np.ndarray, normal_xyz: np.ndarray, radius: float, face_x: float) -> bool:
    nrm = float(np.linalg.norm(normal_xyz))
    if nrm <= 1e-12:
        return False
    nx = float(normal_xyz[0] / nrm)
    max_x_extent = float(radius * np.sqrt(max(1.0 - nx * nx, 0.0)))
    return abs(float(center_xyz[0]) - face_x) <= max_x_extent + 1e-12


def build_face_opportunities(
    dfn_data: dict,
    face_id_to_x: Dict[int, float],
    target_sets: Optional[set[int]],
    rmin: float,
    rmax: float,
) -> List[dict]:
    opportunities: List[dict] = []
    for fracture_id, (center_xyz, normal_xyz, radius_m, set_id) in enumerate(
        zip(dfn_data["centers"], dfn_data["normals"], dfn_data["radii"], dfn_data["set_ids"])
    ):
        set_id_int = int(set_id)
        radius_float = float(radius_m)
        if target_sets is not None and set_id_int not in target_sets:
            continue
        if radius_float < rmin or radius_float > rmax:
            continue
        for face_id, face_x in face_id_to_x.items():
            if fracture_face_intersection_possible(center_xyz, normal_xyz, radius_float, face_x):
                opportunities.append(
                    {
                        "fracture_id": int(fracture_id),
                        "set_id": set_id_int,
                        "face_id": int(face_id),
                        "face_x_m": float(face_x),
                        "radius_m": radius_float,
                    }
                )
    return opportunities


def make_radius_edges(rmin: float, rmax: float, bin_count: int) -> np.ndarray:
    if rmin <= 0.0 or rmax <= rmin:
        raise ValueError("Require 0 < rmin < rmax for radius bins.")
    return np.geomspace(rmin, rmax, bin_count + 1, dtype=np.float64)


def radius_bin_index(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    idx = np.digitize(values, edges, right=False) - 1
    return np.clip(idx, 0, len(edges) - 2)


def fraction_by_class(classes: np.ndarray) -> tuple[float, float, float]:
    if len(classes) == 0:
        return float("nan"), float("nan"), float("nan")
    return tuple(float(np.mean(classes == cls)) for cls in (0, 1, 2))


def simulate_visible_traces_for_radii(
    radii: np.ndarray,
    polygon_yz: np.ndarray,
    directions_yz: np.ndarray,
    rng: np.random.Generator,
    chunk_size: int = 250000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    bbox_min = np.min(polygon_yz, axis=0)
    bbox_max = np.max(polygon_yz, axis=0)
    visible_lengths_parts: List[np.ndarray] = []
    visible_classes_parts: List[np.ndarray] = []
    visible_mask_parts: List[np.ndarray] = []

    for start in range(0, len(radii), chunk_size):
        stop = min(start + chunk_size, len(radii))
        radii_chunk = radii[start:stop]
        true_lengths = sample_true_chords(radii_chunk, rng)
        direction_idx = rng.integers(0, len(directions_yz), size=len(radii_chunk))
        directions = directions_yz[direction_idx]
        expand = 0.5 * true_lengths[:, None]
        centers = rng.uniform(bbox_min - expand, bbox_max + expand)
        visible_lengths, classes = clip_segments_to_convex_polygon_vectorized(centers, directions, true_lengths, polygon_yz)
        visible_mask = classes >= 0
        visible_lengths_parts.append(visible_lengths.astype(np.float64))
        visible_classes_parts.append(classes.astype(np.int32))
        visible_mask_parts.append(visible_mask.astype(bool))

    return (
        np.concatenate(visible_lengths_parts) if visible_lengths_parts else np.zeros((0,), dtype=np.float64),
        np.concatenate(visible_classes_parts) if visible_classes_parts else np.zeros((0,), dtype=np.int32),
        np.concatenate(visible_mask_parts) if visible_mask_parts else np.zeros((0,), dtype=bool),
    )


def dominant_mismatch_label(length_ratio_p90: float, class_l1: float, observed_prob: float, model_prob: float) -> str:
    if np.isfinite(length_ratio_p90) and length_ratio_p90 < 0.85:
        return "mc_underpredicts_long_traces"
    if np.isfinite(length_ratio_p90) and length_ratio_p90 > 1.15:
        return "mc_overpredicts_long_traces"
    if np.isfinite(observed_prob) and np.isfinite(model_prob) and abs(model_prob - observed_prob) > 0.20:
        return "visibility_probability_mismatch"
    if np.isfinite(class_l1) and class_l1 > 0.30:
        return "class_fraction_mismatch"
    return "balanced_or_mild_mismatch"


def quantile_or_nan(values: np.ndarray, q: float) -> float:
    if len(values) == 0:
        return float("nan")
    return float(np.percentile(values, q))


def build_trace_summary_rows(rows: Sequence[dict], edges: np.ndarray, site: str) -> List[dict]:
    output: List[dict] = []
    radii = np.asarray([float(row["radius_m"]) for row in rows], dtype=np.float64)
    bin_idx = radius_bin_index(radii, edges)
    pair_counts: Dict[Tuple[int, int], int] = {}
    for row in rows:
        key = (int(row["fracture_id"]), int(row["face_id"]))
        pair_counts[key] = pair_counts.get(key, 0) + 1
    for row, idx in zip(rows, bin_idx):
        output.append(
            {
                "site": site,
                "trace_id": int(row["trace_id"]),
                "fracture_id": int(row["fracture_id"]),
                "face_id": int(row["face_id"]),
                "set_id": int(row["set_id"]),
                "radius_m": float(row["radius_m"]),
                "radius_bin_low": float(edges[idx]),
                "radius_bin_high": float(edges[idx + 1]),
                "observed_length_m": float(row["observed_length_m"]),
                "censoring_class": int(row["censoring_class"]),
                "face_x_m": float(row["face_x_m"]),
                "fracture_face_trace_count": int(pair_counts[(int(row["fracture_id"]), int(row["face_id"]))]),
                "p0_endpoint_type": row["p0_endpoint_type"],
                "p1_endpoint_type": row["p1_endpoint_type"],
            }
        )
    return output


def summarize_by_radius_bin(
    site: str,
    set_id: int,
    trace_rows: Sequence[dict],
    opportunities: Sequence[dict],
    polygon_yz: np.ndarray,
    directions_yz: np.ndarray,
    edges: np.ndarray,
    mc_samples_per_opportunity: int,
    seed: int,
) -> tuple[List[dict], List[dict]]:
    visibility_rows: List[dict] = []
    comparison_rows: List[dict] = []
    if not opportunities:
        return visibility_rows, comparison_rows

    trace_radii = np.asarray([float(row["radius_m"]) for row in trace_rows], dtype=np.float64) if trace_rows else np.zeros((0,), dtype=np.float64)
    opportunity_radii = np.asarray([float(row["radius_m"]) for row in opportunities], dtype=np.float64)
    observed_pair_keys = {(int(row["fracture_id"]), int(row["face_id"])) for row in trace_rows}
    rng = np.random.default_rng(seed + set_id * 1000)

    for bin_idx in range(len(edges) - 1):
        radius_low = float(edges[bin_idx])
        radius_high = float(edges[bin_idx + 1])
        if bin_idx == len(edges) - 2:
            opp_mask = (opportunity_radii >= radius_low) & (opportunity_radii <= radius_high)
            trace_mask = (trace_radii >= radius_low) & (trace_radii <= radius_high)
        else:
            opp_mask = (opportunity_radii >= radius_low) & (opportunity_radii < radius_high)
            trace_mask = (trace_radii >= radius_low) & (trace_radii < radius_high)

        bin_opportunities = [row for row, keep in zip(opportunities, opp_mask) if keep]
        bin_traces = [row for row, keep in zip(trace_rows, trace_mask) if keep]
        if not bin_opportunities and not bin_traces:
            continue

        unique_fractures_generated = len({int(row["fracture_id"]) for row in bin_opportunities})
        opportunity_pairs = {(int(row["fracture_id"]), int(row["face_id"])) for row in bin_opportunities}
        observed_pairs_in_bin = observed_pair_keys.intersection(opportunity_pairs)

        obs_lengths = np.asarray([float(row["observed_length_m"]) for row in bin_traces], dtype=np.float64)
        obs_classes = np.asarray([int(row["censoring_class"]) for row in bin_traces], dtype=np.int32)
        obs_unc, obs_one, obs_two = fraction_by_class(obs_classes)

        repeated_radii = np.repeat(np.asarray([float(row["radius_m"]) for row in bin_opportunities], dtype=np.float64), mc_samples_per_opportunity)
        model_visible_lengths, model_classes, model_visible_mask = simulate_visible_traces_for_radii(
            repeated_radii,
            polygon_yz,
            directions_yz,
            rng,
        )
        model_visible_lengths = model_visible_lengths[model_visible_mask]
        model_visible_classes = model_classes[model_visible_mask]
        mod_unc, mod_one, mod_two = fraction_by_class(model_visible_classes)

        observed_trace_probability = (
            float(len(observed_pairs_in_bin)) / float(len(opportunity_pairs)) if len(opportunity_pairs) else float("nan")
        )
        model_trace_probability = float(np.mean(model_visible_mask)) if len(model_visible_mask) else float("nan")
        class_l1 = float(abs(obs_unc - mod_unc) + abs(obs_one - mod_one) + abs(obs_two - mod_two))
        observed_p90 = quantile_or_nan(obs_lengths, 90)
        model_p90 = quantile_or_nan(model_visible_lengths, 90)
        length_ratio_p90 = model_p90 / observed_p90 if observed_p90 > 0.0 else float("nan")

        visibility_rows.append(
            {
                "site": site,
                "set_id": set_id,
                "radius_bin_low": radius_low,
                "radius_bin_high": radius_high,
                "n_fractures_generated": unique_fractures_generated,
                "n_face_opportunities_generated": len(opportunity_pairs),
                "n_traces_observed": len(bin_traces),
                "n_observed_face_hits": len(observed_pairs_in_bin),
                "trace_probability": observed_trace_probability,
                "mean_observed_length": float(np.mean(obs_lengths)) if len(obs_lengths) else float("nan"),
                "p50_observed_length": quantile_or_nan(obs_lengths, 50),
                "p90_observed_length": observed_p90,
                "p95_observed_length": quantile_or_nan(obs_lengths, 95),
                "uncensored_fraction": obs_unc,
                "one_end_fraction": obs_one,
                "two_end_fraction": obs_two,
            }
        )
        comparison_rows.append(
            {
                "site": site,
                "set_id": set_id,
                "radius_bin": f"[{radius_low:.3f}, {radius_high:.3f}]",
                "radius_bin_low": radius_low,
                "radius_bin_high": radius_high,
                "observed_mean_length": float(np.mean(obs_lengths)) if len(obs_lengths) else float("nan"),
                "mc_model_mean_length": float(np.mean(model_visible_lengths)) if len(model_visible_lengths) else float("nan"),
                "observed_p90_length": observed_p90,
                "mc_model_p90_length": model_p90,
                "observed_uncensored_fraction": obs_unc,
                "mc_model_uncensored_fraction": mod_unc,
                "observed_trace_probability": observed_trace_probability,
                "mc_model_trace_probability": model_trace_probability,
                "length_ratio_p90": length_ratio_p90,
                "trace_probability_delta": model_trace_probability - observed_trace_probability
                if np.isfinite(model_trace_probability) and np.isfinite(observed_trace_probability)
                else float("nan"),
                "class_l1": class_l1,
                "dominant_mismatch": dominant_mismatch_label(
                    length_ratio_p90,
                    class_l1,
                    observed_trace_probability,
                    model_trace_probability,
                ),
            }
        )

    return visibility_rows, comparison_rows


def build_report_markdown(
    site: str,
    target_sets: Sequence[int],
    comparison_rows: Sequence[dict],
    visibility_rows: Sequence[dict],
    direction_mode: str,
    mc_samples_per_opportunity: int,
) -> str:
    lines = [
        "# Radius-Conditioned Visibility Report",
        "",
        f"- site: `{site}`",
        f"- target_sets: `{' '.join(str(v) for v in target_sets)}`",
        f"- direction_mode_under_test: `{direction_mode}`",
        f"- mc_samples_per_opportunity: `{mc_samples_per_opportunity}`",
        "",
        "## Interpretation Guide",
        "",
        "- `length_ratio_p90 < 1`: MC underpredicts long visible traces for that radius bin.",
        "- `trace_probability_delta < 0`: MC underpredicts visibility probability.",
        "- `class_l1` large with length ratios near 1 suggests class-process mismatch more than length mismatch.",
        "",
        "## Key Bin Mismatches",
        "",
    ]

    for set_id in target_sets:
        set_rows = [row for row in comparison_rows if int(row["set_id"]) == int(set_id)]
        if not set_rows:
            lines.extend([f"### Set {set_id}", "", "No comparison rows.", ""])
            continue
        ranked = sorted(
            set_rows,
            key=lambda row: (
                abs(np.log(max(float(row["length_ratio_p90"]), 1e-12))) if np.isfinite(float(row["length_ratio_p90"])) else -1.0,
                abs(float(row["trace_probability_delta"])) if np.isfinite(float(row["trace_probability_delta"])) else -1.0,
            ),
            reverse=True,
        )
        lines.append(f"### Set {set_id}")
        lines.append("")
        for row in ranked[:3]:
            lines.append(
                f"- radius_bin {row['radius_bin']}: p90 model/obs = {float(row['length_ratio_p90']):.3f}, "
                f"trace_prob_delta = {float(row['trace_probability_delta']):+.3f}, "
                f"class_l1 = {float(row['class_l1']):.3f}, mismatch = `{row['dominant_mismatch']}`"
            )
        set_visibility = [row for row in visibility_rows if int(row["set_id"]) == int(set_id)]
        if set_visibility:
            max_bin = max(set_visibility, key=lambda row: float(row["radius_bin_high"]))
            lines.append(
                f"- largest-radius bin observed p90 = {float(max_bin['p90_observed_length']):.3f} m, "
                f"observed trace_probability = {float(max_bin['trace_probability']):.3f}"
            )
        lines.append("")

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Oracle diagnostic for radius-conditioned visibility and trace length bias.")
    parser.add_argument("--dfn-h5", required=True, help="Input DFN HDF5 with fractures/centers, normals, radii, set_id.")
    parser.add_argument("--trace-h5", required=True, help="Trace dataset HDF5 with fracture_id and observed trace rows.")
    parser.add_argument("--site", required=True, help="Site label for outputs.")
    parser.add_argument("--target-set", nargs="+", type=int, required=True, help="Target fracture sets.")
    parser.add_argument("--rmin", type=float, default=0.5)
    parser.add_argument("--rmax", type=float, default=250.0)
    parser.add_argument("--radius-bin-count", type=int, default=8)
    parser.add_argument("--mc-samples-per-opportunity", type=int, default=128)
    parser.add_argument("--direction-mode", choices=["empirical_trace"], default="empirical_trace")
    parser.add_argument("--seed", type=int, default=20260701)
    parser.add_argument(
        "--outdir",
        default="storage/output/radius_conditioned_visibility_diagnostics",
        help="Output directory for diagnostic CSV and markdown files.",
    )
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    target_sets = set(args.target_set)

    dfn_data = load_hdf5_dfn(args.dfn_h5)
    trace_rows_raw, polygon_yz, face_x_positions = load_trace_dataset(args.trace_h5)
    trace_rows = normalize_rows_with_radius(trace_rows_raw, dfn_data)
    trace_rows = [row for row in trace_rows if int(row["set_id"]) in target_sets and args.rmin <= float(row["radius_m"]) <= args.rmax]
    if not trace_rows:
        raise ValueError("No trace rows remain after target-set/radius filtering.")

    face_id_to_x = build_face_id_to_x(face_x_positions)
    opportunities = build_face_opportunities(dfn_data, face_id_to_x, target_sets, args.rmin, args.rmax)
    if not opportunities:
        raise ValueError("No fracture-face opportunities remain after filtering.")

    radius_edges = make_radius_edges(args.rmin, args.rmax, args.radius_bin_count)
    directions_by_set = {
        set_id: empirical_trace_directions_yz([row for row in trace_rows if int(row["set_id"]) == set_id])
        for set_id in sorted(target_sets)
    }

    trace_summary_rows = build_trace_summary_rows(trace_rows, radius_edges, args.site)
    visibility_rows: List[dict] = []
    comparison_rows: List[dict] = []
    for set_id in sorted(target_sets):
        set_trace_rows = [row for row in trace_rows if int(row["set_id"]) == set_id]
        set_opportunities = [row for row in opportunities if int(row["set_id"]) == set_id]
        set_visibility, set_comparison = summarize_by_radius_bin(
            args.site,
            set_id,
            set_trace_rows,
            set_opportunities,
            polygon_yz,
            directions_by_set[set_id],
            radius_edges,
            args.mc_samples_per_opportunity,
            args.seed,
        )
        visibility_rows.extend(set_visibility)
        comparison_rows.extend(set_comparison)

    trace_summary_csv = os.path.join(args.outdir, "radius_conditioned_trace_summary.csv")
    visibility_csv = os.path.join(args.outdir, "radius_bin_visibility_summary.csv")
    comparison_csv = os.path.join(args.outdir, "radius_length_model_comparison.csv")
    report_md = os.path.join(args.outdir, "radius_conditioned_visibility_report.md")

    write_csv(trace_summary_rows, trace_summary_csv)
    write_csv(visibility_rows, visibility_csv)
    write_csv(comparison_rows, comparison_csv)
    with open(report_md, "w", encoding="utf-8") as f:
        f.write(
            build_report_markdown(
                args.site,
                sorted(target_sets),
                comparison_rows,
                visibility_rows,
                args.direction_mode,
                args.mc_samples_per_opportunity,
            )
        )

    print("[*] Radius-conditioned visibility diagnostic complete")
    print(f"[*] Trace summary CSV written to: {trace_summary_csv}")
    print(f"[*] Visibility summary CSV written to: {visibility_csv}")
    print(f"[*] Model comparison CSV written to: {comparison_csv}")
    print(f"[*] Report written to: {report_md}")


if __name__ == "__main__":
    main()
