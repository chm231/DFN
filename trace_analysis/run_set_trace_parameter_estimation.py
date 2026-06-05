"""Run set-wise trace QC, correction, radius estimation, and intensity estimation."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

from trace_analysis.censoring import append_censoring_columns, load_tunnel_polygon
from trace_analysis.intensity_estimator import estimate_intensity_parameters
from trace_analysis.load_measured_traces import load_measured_traces
from trace_analysis.radius_distribution_estimator import estimate_radius_distributions
from trace_analysis.set_trace_statistics import compute_set_observed_statistics
from trace_analysis.trace_distribution_correction import correct_trace_distributions
from trace_analysis.trace_qc import build_trace_qc_dataframe


def _to_jsonable(value):
    if isinstance(value, dict):
        return {key: _to_jsonable(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(val) for val in value]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _build_set_dfn_params(
    traces_csv: str,
    correction_summary: Dict[int, Dict[str, object]],
    radius_distributions: Dict[int, Dict[str, object]],
    intensity_parameters: Dict[int, Dict[str, object]],
    set_stats_df: pd.DataFrame,
) -> Dict[str, object]:
    stats_map = {int(row["set_id"]): row for _, row in set_stats_df.iterrows()}
    sets = []
    for set_id in sorted(intensity_parameters):
        stats_row = stats_map[set_id]
        correction = correction_summary[set_id]
        intensity = intensity_parameters[set_id]
        radius = radius_distributions[set_id]
        sets.append(
            {
                "set_id": set_id,
                "n_traces": int(stats_row["n_traces"]),
                "observation": {
                    "p21_observed": intensity["observed_P21"],
                    "p21_corrected": intensity["corrected_P21"],
                    "censored_ratio": float(stats_row["censored_ratio"]),
                    "length_min": float(stats_row["min_length_yz"]) if np.isfinite(stats_row["min_length_yz"]) else None,
                    "length_mean": float(stats_row["mean_length_yz"]) if np.isfinite(stats_row["mean_length_yz"]) else None,
                    "length_median": float(stats_row["median_length_yz"]) if np.isfinite(stats_row["median_length_yz"]) else None,
                    "length_max": float(stats_row["max_length_yz"]) if np.isfinite(stats_row["max_length_yz"]) else None,
                },
                "orientation": {
                    "mean_normal": correction.get("mean_normal"),
                    "orientation_bias_factor": correction.get("q_set"),
                    "has_orientation_data": correction.get("mean_normal") is not None,
                },
                "radius_distribution": radius["radius_distribution"],
                "intensity": {
                    "P21": intensity["orientation_corrected_P21"],
                    "P30": intensity["estimated_P30"],
                    "P32": intensity["estimated_P32"],
                    "method": intensity["method"],
                    "warnings": intensity["warnings"],
                },
                "qc": {
                    "confidence": "baseline",
                    "warnings": correction.get("warnings", []),
                },
            }
        )
    return {
        "metadata": {
            "source_traces_csv": traces_csv,
            "coordinate_system": {
                "x": "tunnel axis",
                "y": "horizontal transverse on tunnel face",
                "z": "vertical on tunnel face",
                "face_model": "X = x_face_nominal ± dx",
            },
            "created_by": "run_set_trace_parameter_estimation.py",
        },
        "sets": sets,
    }


def _write_run_summary(
    output_path: str,
    qc_df: pd.DataFrame,
    set_stats_df: pd.DataFrame,
    radius_distributions: Dict[int, Dict[str, object]],
    intensity_parameters: Dict[int, Dict[str, object]],
    warnings: List[str],
    validation_summary_path: str | None,
) -> None:
    lines = ["# Set Trace Parameter Estimation Run Summary", ""]
    lines.append(f"- Total traces: {len(qc_df)}")
    lines.append(f"- Valid traces: {int(qc_df['valid_length'].sum()) if not qc_df.empty else 0}")
    lines.append("")
    lines.append("## Set Summary")
    for _, row in set_stats_df.iterrows():
        set_id = int(row["set_id"])
        intensity = intensity_parameters.get(set_id, {})
        radius = radius_distributions.get(set_id, {})
        lines.append(f"- Set {set_id}:")
        lines.append(f"  trace count = {int(row['n_traces'])}")
        lines.append(f"  P21 observed = {row['observed_P21']}")
        lines.append(f"  radius distribution = {radius.get('radius_distribution', {}).get('type')}")
        lines.append(f"  P30 = {intensity.get('estimated_P30')}")
        lines.append(f"  P32 = {intensity.get('estimated_P32')}")
        lines.append(f"  censoring ratio = {row['censored_ratio']}")
        lines.append(
            f"  orientation correction applied = {intensity.get('method') == 'orientation_corrected_p21_proxy'}"
        )
    if warnings:
        lines.append("")
        lines.append("## Loader Warnings")
        for warning in warnings:
            lines.append(f"- {warning}")
    if validation_summary_path:
        lines.append("")
        lines.append("## Validation")
        lines.append(f"- Parent fracture validation summary: {validation_summary_path}")
    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def _write_validation_summary(qc_df: pd.DataFrame, output_path: str) -> None:
    validation_df = (
        qc_df.dropna(subset=["parent_fracture_id"])
        .groupby(["set_id", "parent_fracture_id"], as_index=False)
        .agg(segment_count=("trace_id", "count"))
    )
    if validation_df.empty:
        return
    summary = (
        validation_df.groupby("set_id", as_index=False)
        .agg(
            n_parent_fractures=("parent_fracture_id", "count"),
            trace_segment_count=("segment_count", "sum"),
        )
    )
    summary["segments_per_parent_mean"] = (
        summary["trace_segment_count"] / summary["n_parent_fractures"]
    )
    summary.to_csv(output_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate set-wise DFN parameters from measured traces.")
    parser.add_argument("--traces-csv", required=True, help="Input measured traces CSV.")
    parser.add_argument("--face-stations-csv", help="Optional face station table CSV.")
    parser.add_argument("--tunnel-polygon", help="Tunnel polygon CSV or JSON in Y-Z coordinates.")
    parser.add_argument("--output-dir", required=True, help="Output directory.")
    parser.add_argument(
        "--x-face-mode",
        choices=["nominal", "mid_rough"],
        default="nominal",
        help="Interpretation mode for x_face when x_face_nominal is absent.",
    )
    parser.add_argument(
        "--min-trace-length",
        type=float,
        default=0.0,
        help="Minimum projected trace length used in correction and statistics.",
    )
    parser.add_argument(
        "--boundary-tol",
        type=float,
        default=0.05,
        help="Boundary tolerance for censoring classification in Y-Z.",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    traces, trace_meta, original_3d_points, loader_warnings = load_measured_traces(
        traces_csv=args.traces_csv,
        x_face_mode=args.x_face_mode,
        face_stations_csv=args.face_stations_csv,
        min_trace_length=max(args.min_trace_length, 1e-9),
    )
    qc_df = build_trace_qc_dataframe(
        traces=traces,
        trace_meta=trace_meta,
        original_3d_points=original_3d_points,
        min_trace_length=args.min_trace_length,
    )

    tunnel_polygon_yz = None
    if args.tunnel_polygon:
        tunnel_polygon_yz = load_tunnel_polygon(args.tunnel_polygon)
        qc_df = append_censoring_columns(
            qc_df=qc_df,
            tunnel_polygon_yz=tunnel_polygon_yz,
            boundary_tolerance=args.boundary_tol,
        )

    set_stats_df = compute_set_observed_statistics(qc_df=qc_df, tunnel_polygon_yz=tunnel_polygon_yz)
    corrected_df, correction_summary = correct_trace_distributions(
        qc_df=qc_df,
        min_trace_length=args.min_trace_length,
    )
    radius_distributions = estimate_radius_distributions(corrected_df)
    intensity_parameters = estimate_intensity_parameters(
        corrected_df=corrected_df,
        set_stats_df=set_stats_df,
        correction_summary=correction_summary,
        radius_distributions=radius_distributions,
    )
    set_dfn_params = _build_set_dfn_params(
        traces_csv=args.traces_csv,
        correction_summary=correction_summary,
        radius_distributions=radius_distributions,
        intensity_parameters=intensity_parameters,
        set_stats_df=set_stats_df,
    )

    qc_path = os.path.join(args.output_dir, "trace_qc.csv")
    stats_path = os.path.join(args.output_dir, "set_observed_statistics.csv")
    corrected_path = os.path.join(args.output_dir, "set_corrected_trace_distribution.csv")
    radius_path = os.path.join(args.output_dir, "set_radius_distribution.json")
    intensity_path = os.path.join(args.output_dir, "set_intensity_parameters.json")
    params_path = os.path.join(args.output_dir, "set_dfn_params.json")
    summary_path = os.path.join(args.output_dir, "run_summary.md")

    qc_df.to_csv(qc_path, index=False)
    set_stats_df.to_csv(stats_path, index=False)
    corrected_df.to_csv(corrected_path, index=False)
    with open(radius_path, "w", encoding="utf-8") as handle:
        json.dump(_to_jsonable(radius_distributions), handle, indent=2)
    with open(intensity_path, "w", encoding="utf-8") as handle:
        json.dump(_to_jsonable(intensity_parameters), handle, indent=2)
    with open(params_path, "w", encoding="utf-8") as handle:
        json.dump(_to_jsonable(set_dfn_params), handle, indent=2)

    validation_summary_path = None
    if "parent_fracture_id" in qc_df.columns and qc_df["parent_fracture_id"].notna().any():
        validation_summary_path = os.path.join(
            args.output_dir,
            "validation_parent_fracture_summary.csv",
        )
        _write_validation_summary(qc_df, validation_summary_path)

    _write_run_summary(
        output_path=summary_path,
        qc_df=qc_df,
        set_stats_df=set_stats_df,
        radius_distributions=radius_distributions,
        intensity_parameters=intensity_parameters,
        warnings=loader_warnings,
        validation_summary_path=validation_summary_path,
    )


if __name__ == "__main__":
    main()
