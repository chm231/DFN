import argparse
import math
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dfn_analysis.estimate_p32_mc_calibrated import read_csv, write_csv
from dfn_analysis.estimate_radius_powerlaw_window_mc import load_trace_data_from_h5


REQUIRED_REPORT_STATEMENT = (
    "Kaplan-Meier correction is used only as a non-parametric diagnostic for trace length "
    "censoring. It is not used as the final estimator for the 3D fracture radius distribution. "
    "The final kr inversion remains based on the effective-rmin window-aware Monte Carlo likelihood."
)


def load_trace_rows(trace_h5: str) -> List[dict]:
    rows, _ = load_trace_data_from_h5(trace_h5)
    return rows


def classify_event_indicators(censoring_classes: np.ndarray) -> np.ndarray:
    classes = np.asarray(censoring_classes, dtype=np.int32)
    return (classes == 0).astype(np.int32)


def empirical_survival(lengths: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    lengths = np.asarray(lengths, dtype=np.float64)
    if len(lengths) == 0:
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)
    grid = np.unique(np.sort(lengths))
    survival = np.asarray([np.mean(lengths > value) for value in grid], dtype=np.float64)
    return grid, survival


def kaplan_meier_curve(lengths: np.ndarray, event_indicators: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    lengths = np.asarray(lengths, dtype=np.float64)
    events = np.asarray(event_indicators, dtype=np.int32)
    if len(lengths) == 0:
        empty = np.asarray([], dtype=np.float64)
        return empty, empty, empty.astype(np.int32), empty.astype(np.int32), empty.astype(np.int32)

    grid = np.unique(np.sort(lengths))
    survival = np.empty(len(grid), dtype=np.float64)
    n_at_risk = np.empty(len(grid), dtype=np.int32)
    n_events = np.empty(len(grid), dtype=np.int32)
    n_censored = np.empty(len(grid), dtype=np.int32)

    current_survival = 1.0
    for idx, value in enumerate(grid):
        at_risk = lengths >= value - 1e-12
        exact = np.abs(lengths - value) <= 1e-12
        d_i = int(np.sum(exact & (events == 1)))
        c_i = int(np.sum(exact & (events == 0)))
        n_i = int(np.sum(at_risk))
        if n_i > 0 and d_i > 0:
            current_survival *= (1.0 - d_i / n_i)
        survival[idx] = current_survival
        n_at_risk[idx] = n_i
        n_events[idx] = d_i
        n_censored[idx] = c_i

    return grid, survival, n_at_risk, n_events, n_censored


def percentile_from_survival(length_grid: np.ndarray, survival: np.ndarray, percentile: float) -> float:
    if len(length_grid) == 0 or len(survival) == 0:
        return float("nan")
    threshold = 1.0 - percentile
    indices = np.where(survival <= threshold)[0]
    if len(indices) == 0:
        return float(length_grid[-1])
    return float(length_grid[indices[0]])


def evaluate_step_survival(length_grid: np.ndarray, survival: np.ndarray, query_grid: np.ndarray) -> np.ndarray:
    if len(length_grid) == 0:
        return np.full(len(query_grid), float("nan"), dtype=np.float64)
    out = np.empty(len(query_grid), dtype=np.float64)
    for idx, value in enumerate(query_grid):
        pos = np.searchsorted(length_grid, value, side="right") - 1
        out[idx] = 1.0 if pos < 0 else float(survival[pos])
    return out


def summarize_subset(rows: Sequence[dict], site: str, set_id: int, lmin_fit: float) -> Tuple[List[dict], dict]:
    subset = [
        row for row in rows
        if int(row["set_id"]) == set_id and float(row["observed_length_m"]) >= lmin_fit
    ]
    lengths = np.asarray([float(row["observed_length_m"]) for row in subset], dtype=np.float64)
    censoring = np.asarray([int(row["censoring_class"]) for row in subset], dtype=np.int32)
    event_indicators = classify_event_indicators(censoring)
    n_traces = len(subset)
    n_exact = int(np.sum(event_indicators == 1))
    n_censored = int(np.sum(event_indicators == 0))
    censoring_fraction = (n_censored / n_traces) if n_traces else float("nan")

    raw_grid, raw_survival = empirical_survival(lengths)
    km_grid, km_survival, n_at_risk, n_events, n_censored_curve = kaplan_meier_curve(lengths, event_indicators)
    curve_grid = np.unique(np.concatenate([raw_grid, km_grid])) if len(raw_grid) or len(km_grid) else np.asarray([], dtype=np.float64)
    raw_curve = evaluate_step_survival(raw_grid, raw_survival, curve_grid) if len(curve_grid) else np.asarray([], dtype=np.float64)
    km_curve = evaluate_step_survival(km_grid, km_survival, curve_grid) if len(curve_grid) else np.asarray([], dtype=np.float64)
    at_risk_curve = np.asarray([
        int(n_at_risk[np.searchsorted(km_grid, value, side="left")]) if len(km_grid) and value in set(km_grid.tolist()) else (
            int(np.sum(lengths >= value - 1e-12)) if n_traces else 0
        )
        for value in curve_grid
    ], dtype=np.int32) if len(curve_grid) else np.asarray([], dtype=np.int32)
    event_curve = np.asarray([
        int(n_events[np.where(np.abs(km_grid - value) <= 1e-12)[0][0]]) if np.any(np.abs(km_grid - value) <= 1e-12) else 0
        for value in curve_grid
    ], dtype=np.int32) if len(curve_grid) else np.asarray([], dtype=np.int32)
    censored_curve = np.asarray([
        int(n_censored_curve[np.where(np.abs(km_grid - value) <= 1e-12)[0][0]]) if np.any(np.abs(km_grid - value) <= 1e-12) else 0
        for value in curve_grid
    ], dtype=np.int32) if len(curve_grid) else np.asarray([], dtype=np.int32)

    raw_p50 = float(np.percentile(lengths, 50)) if n_traces else float("nan")
    raw_p90 = float(np.percentile(lengths, 90)) if n_traces else float("nan")
    raw_p95 = float(np.percentile(lengths, 95)) if n_traces else float("nan")
    km_p50 = percentile_from_survival(km_grid, km_survival, 0.50)
    km_p90 = percentile_from_survival(km_grid, km_survival, 0.90)
    km_p95 = percentile_from_survival(km_grid, km_survival, 0.95)

    p50_shift = km_p50 - raw_p50 if np.isfinite(km_p50) and np.isfinite(raw_p50) else float("nan")
    p90_shift = km_p90 - raw_p90 if np.isfinite(km_p90) and np.isfinite(raw_p90) else float("nan")
    p95_shift = km_p95 - raw_p95 if np.isfinite(km_p95) and np.isfinite(raw_p95) else float("nan")
    p90_ratio = (km_p90 / raw_p90) if np.isfinite(km_p90) and np.isfinite(raw_p90) and raw_p90 > 0.0 else float("nan")
    p95_ratio = (km_p95 / raw_p95) if np.isfinite(km_p95) and np.isfinite(raw_p95) and raw_p95 > 0.0 else float("nan")

    if n_traces == 0:
        km_status = "no_traces"
    elif np.isfinite(censoring_fraction) and censoring_fraction < 0.15 and np.isfinite(p90_ratio) and p90_ratio < 1.10:
        km_status = "minor_censoring_effect"
    elif np.isfinite(p90_ratio) and p90_ratio <= 1.30:
        km_status = "moderate_censoring_effect"
    else:
        km_status = "strong_censoring_effect"

    curve_rows = [
        {
            "site": site,
            "set_id": set_id,
            "lmin_fit": lmin_fit,
            "length": float(curve_grid[idx]),
            "raw_survival": float(raw_curve[idx]),
            "km_survival": float(km_curve[idx]),
            "n_at_risk": int(at_risk_curve[idx]),
            "n_events": int(event_curve[idx]),
            "n_censored": int(censored_curve[idx]),
            "censoring_fraction": censoring_fraction,
        }
        for idx in range(len(curve_grid))
    ]
    summary_row = {
        "site": site,
        "set_id": set_id,
        "lmin_fit": lmin_fit,
        "n_traces": n_traces,
        "n_exact": n_exact,
        "n_censored": n_censored,
        "censoring_fraction": censoring_fraction,
        "raw_p50": raw_p50,
        "raw_p90": raw_p90,
        "raw_p95": raw_p95,
        "km_p50": km_p50,
        "km_p90": km_p90,
        "km_p95": km_p95,
        "p50_shift_km_vs_raw": p50_shift,
        "p90_shift_km_vs_raw": p90_shift,
        "p95_shift_km_vs_raw": p95_shift,
        "p90_ratio_km_vs_raw": p90_ratio,
        "p95_ratio_km_vs_raw": p95_ratio,
        "km_status": km_status,
        "notes": "",
    }
    return curve_rows, summary_row


def build_lmin_sensitivity(summary_rows: Sequence[dict]) -> List[dict]:
    out: List[dict] = []
    grouped: Dict[Tuple[str, int], List[dict]] = {}
    for row in summary_rows:
        grouped.setdefault((str(row["site"]), int(row["set_id"])), []).append(row)

    for (site, set_id), rows in sorted(grouped.items()):
        valid_p90 = [float(row["km_p90"]) for row in rows if np.isfinite(float(row["km_p90"]))]
        if len(valid_p90) >= 2 and min(valid_p90) > 0.0 and (max(valid_p90) / min(valid_p90)) <= 1.2:
            status = "km_tail_stable"
        elif len(valid_p90) >= 2:
            status = "km_tail_sensitive_to_lmin"
        else:
            status = "insufficient_km_tail_data"
        for row in sorted(rows, key=lambda item: float(item["lmin_fit"])):
            out.append(
                {
                    "site": site,
                    "set_id": set_id,
                    "lmin_fit": row["lmin_fit"],
                    "km_p50": row["km_p50"],
                    "km_p90": row["km_p90"],
                    "km_p95": row["km_p95"],
                    "n_traces": row["n_traces"],
                    "n_censored": row["n_censored"],
                    "censoring_fraction": row["censoring_fraction"],
                    "km_tail_stability_status": status,
                }
            )
    return out


def load_mc_survival_rows(path: str) -> List[dict]:
    return read_csv(path)


def compare_with_mc(
    curve_rows: Sequence[dict],
    mc_rows: Sequence[dict],
    mc_survival_mode: str,
) -> Tuple[List[dict], List[dict]]:
    if not mc_rows:
        return list(curve_rows), []
    curve_out: List[dict] = []
    comparison_rows: List[dict] = []
    curve_grouped: Dict[Tuple[str, int, float], List[dict]] = {}
    for row in curve_rows:
        curve_grouped.setdefault((str(row["site"]), int(row["set_id"]), float(row["lmin_fit"])), []).append(row)
    mc_grouped: Dict[Tuple[str, int, float], List[dict]] = {}
    for row in mc_rows:
        if str(row.get("survival_mode", "")) != mc_survival_mode:
            continue
        mc_grouped.setdefault((str(row["site"]), int(row["set_id"]), float(row["lmin_fit"])), []).append(row)

    for key, rows in curve_grouped.items():
        mc_subset = mc_grouped.get(key, [])
        if not mc_subset:
            curve_out.extend(rows)
            continue
        site, set_id, lmin_fit = key
        km_grid = np.asarray([float(row["length"]) for row in rows], dtype=np.float64)
        km_survival = np.asarray([float(row["km_survival"]) for row in rows], dtype=np.float64)
        mc_grid = np.asarray([float(row["length"]) for row in mc_subset], dtype=np.float64)
        mc_survival = np.asarray([float(row["mc_survival"]) for row in mc_subset], dtype=np.float64)
        combined_grid = np.unique(np.concatenate([km_grid, mc_grid]))
        km_eval = evaluate_step_survival(km_grid, km_survival, combined_grid)
        mc_eval = evaluate_step_survival(mc_grid, mc_survival, combined_grid)
        diff = km_eval - mc_eval
        l1 = float(np.mean(np.abs(diff))) if len(diff) else float("nan")
        l2 = float(math.sqrt(np.mean(diff * diff))) if len(diff) else float("nan")
        km_p90 = percentile_from_survival(km_grid, km_survival, 0.90)
        km_p95 = percentile_from_survival(km_grid, km_survival, 0.95)
        mc_p90 = percentile_from_survival(mc_grid, mc_survival, 0.90)
        mc_p95 = percentile_from_survival(mc_grid, mc_survival, 0.95)
        p90_ratio = (mc_p90 / km_p90) if np.isfinite(mc_p90) and np.isfinite(km_p90) and km_p90 > 0.0 else float("nan")
        p95_ratio = (mc_p95 / km_p95) if np.isfinite(mc_p95) and np.isfinite(km_p95) and km_p95 > 0.0 else float("nan")
        if np.isfinite(p90_ratio) and np.isfinite(p95_ratio) and 0.8 <= p90_ratio <= 1.25 and 0.8 <= p95_ratio <= 1.25:
            status = "mc_consistent_with_km"
        else:
            status = "mc_km_tail_mismatch"
        for row in rows:
            value = float(row["length"])
            idx = int(np.where(np.abs(combined_grid - value) <= 1e-12)[0][0])
            updated = dict(row)
            updated["mc_survival"] = float(mc_eval[idx])
            updated["km_minus_mc"] = float(diff[idx])
            curve_out.append(updated)
        comparison_rows.append(
            {
                "site": site,
                "set_id": set_id,
                "lmin_fit": lmin_fit,
                "comparison_mode": f"observed_km_vs_{mc_survival_mode}",
                "mc_survival_mode": mc_survival_mode,
                "survival_l1_distance": l1,
                "survival_l2_distance": l2,
                "km_p90": km_p90,
                "mc_p90": mc_p90,
                "km_p95": km_p95,
                "mc_p95": mc_p95,
                "p90_ratio_mc_vs_km": p90_ratio,
                "p95_ratio_mc_vs_km": p95_ratio,
                "mc_km_consistency_status": status,
            }
        )
    return curve_out, comparison_rows


def build_markdown_report(
    site: str,
    summary_rows: Sequence[dict],
    sensitivity_rows: Sequence[dict],
    comparison_rows: Sequence[dict],
) -> str:
    strong_sets = [row for row in summary_rows if str(row["km_status"]) == "strong_censoring_effect"]
    sensitive_sets = [row for row in sensitivity_rows if str(row["km_tail_stability_status"]) == "km_tail_sensitive_to_lmin"]
    mismatch_sets = [row for row in comparison_rows if str(row["mc_km_consistency_status"]) == "mc_km_tail_mismatch"]
    strong_text = ", ".join([f"Set {row['set_id']} (lmin={row['lmin_fit']:.2f})" for row in strong_sets]) if strong_sets else "none"
    sensitive_text = ", ".join([f"Set {row['set_id']}" for row in sensitive_sets]) if sensitive_sets else "none"
    mismatch_text = ", ".join([f"Set {row['set_id']} (lmin={row['lmin_fit']:.2f})" for row in mismatch_sets]) if mismatch_sets else "none"
    lines = [
        "# Trace Length KM Diagnostic Report",
        "",
        f"- site: `{site}`",
        "",
        "## Purpose",
        "Kaplan-Meier is used here as a non-parametric diagnostic to assess censoring effects in observed trace lengths.",
        REQUIRED_REPORT_STATEMENT,
        "The direct comparison between observed KM survival and MC visible-length survival is diagnostic only, because KM estimates a censoring-adjusted survival curve whereas MC visible-length survival represents clipped observed lengths. The primary consistency check therefore uses MC KM-emulated survival, where the same Kaplan-Meier procedure is applied to simulated observed lengths and simulated censoring classes.",
        "",
        "## Set Summary",
        "",
        "| set_id | lmin_fit | n_traces | censoring_fraction | raw_p90 | km_p90 | raw_p95 | km_p95 | km_status |",
        "|--------|----------|----------|--------------------|---------|--------|---------|--------|-----------|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['set_id']} | {row['lmin_fit']:.2f} | {row['n_traces']} | "
            f"{row['censoring_fraction']:.3f} | {row['raw_p90']:.3f} | {row['km_p90']:.3f} | "
            f"{row['raw_p95']:.3f} | {row['km_p95']:.3f} | {row['km_status']} |"
        )
    lines += [
        "",
        "## Interpretation",
        "- KM is not a replacement for the final kr inversion.",
        "- KM only diagnoses how much right-censoring shifts the observed trace-length tail.",
        "",
        "## Flags",
        f"- strong censoring effect sets: {strong_text}",
        f"- lmin-sensitive sets: {sensitive_text}",
        f"- MC/KM mismatch sets: {mismatch_text}",
        "",
        "## Follow-up",
        "- Use KM outputs as supporting diagnostics for censoring/tail interpretation.",
        "- Keep final kr inversion on the effective-rmin window-aware MC likelihood.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Kaplan-Meier diagnostic for trace length censoring.")
    parser.add_argument("--trace-h5", required=True)
    parser.add_argument("--site", choices=["forsmark", "laxemar"], required=True)
    parser.add_argument("--target-set", type=int, nargs="+", required=True)
    parser.add_argument("--lmin-fit-values", type=float, nargs="+", required=True)
    parser.add_argument("--set-rmin-mode", default="effective_generation")
    parser.add_argument("--compare-mc-prediction", action="store_true")
    parser.add_argument("--mc-prediction-csv", default=None)
    parser.add_argument(
        "--mc-survival-mode",
        choices=["mc_observed_visible_survival", "mc_km_emulated_survival", "mc_true_chord_survival"],
        default="mc_km_emulated_survival",
    )
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()

    rows = load_trace_rows(args.trace_h5)
    os.makedirs(args.outdir, exist_ok=True)

    curve_rows: List[dict] = []
    summary_rows: List[dict] = []
    for set_id in args.target_set:
        for lmin_fit in args.lmin_fit_values:
            subset_curve_rows, summary_row = summarize_subset(rows, args.site, int(set_id), float(lmin_fit))
            curve_rows.extend(subset_curve_rows)
            summary_rows.append(summary_row)

    sensitivity_rows = build_lmin_sensitivity(summary_rows)
    mc_rows = load_mc_survival_rows(args.mc_prediction_csv) if args.compare_mc_prediction and args.mc_prediction_csv else []
    curve_rows, comparison_rows = compare_with_mc(curve_rows, mc_rows, args.mc_survival_mode)

    curve_csv = os.path.join(args.outdir, "km_survival_curve.csv")
    summary_csv = os.path.join(args.outdir, "km_summary_by_set.csv")
    sensitivity_csv = os.path.join(args.outdir, "km_lmin_sensitivity.csv")
    comparison_csv = os.path.join(args.outdir, "km_mc_comparison.csv")
    report_md = os.path.join(args.outdir, "km_diagnostic_report.md")

    if not curve_rows:
        curve_rows = [{
            "site": args.site,
            "set_id": int(args.target_set[0]),
            "lmin_fit": float(args.lmin_fit_values[0]),
            "length": float("nan"),
            "raw_survival": float("nan"),
            "km_survival": float("nan"),
            "n_at_risk": 0,
            "n_events": 0,
            "n_censored": 0,
            "censoring_fraction": float("nan"),
        }]
    write_csv(curve_rows, curve_csv)
    write_csv(summary_rows, summary_csv)
    write_csv(sensitivity_rows, sensitivity_csv)
    if comparison_rows:
        write_csv(comparison_rows, comparison_csv)
    with open(report_md, "w", encoding="utf-8") as f:
        f.write(build_markdown_report(args.site, summary_rows, sensitivity_rows, comparison_rows))

    print(f"[*] KM survival curve written to: {curve_csv}")
    print(f"[*] KM summary written to: {summary_csv}")
    print(f"[*] KM lmin sensitivity written to: {sensitivity_csv}")
    if comparison_rows:
        print(f"[*] KM/MC comparison written to: {comparison_csv}")
    print(f"[*] KM report written to: {report_md}")


if __name__ == "__main__":
    main()
