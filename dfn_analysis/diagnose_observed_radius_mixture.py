import argparse
import os
from typing import Dict, List, Optional, Sequence

import numpy as np

from dfn_analysis.diagnose_radius_conditioned_visibility import (
    load_trace_dataset,
    normalize_rows_with_radius,
    write_csv,
)
from dfn_analysis.estimate_radius_powerlaw_window_mc import (
    build_set_rmin_lookup,
    empirical_trace_directions_yz,
    load_trace_rmin_metadata_from_h5,
    resolve_set_likelihood_rmin,
    sample_size_biased_radius,
    simulate_window_samples,
)
from dfn_analysis.export_setwise_3d_traces import load_hdf5_dfn


_SITE_TO_KR_TRUE: Dict[str, Dict[int, float]] = {
    "forsmark": {1: 2.88, 2: 3.02, 3: 2.81, 4: 2.95, 5: 2.92},
    "laxemar": {1: 2.85, 2: 3.04, 3: 3.01, 5: 3.60},
}


def quantile_or_nan(values: np.ndarray, q: float) -> float:
    if len(values) == 0:
        return float("nan")
    return float(np.percentile(values, q))


def make_radius_edges(rmin: float, rmax: float, bin_count: int) -> np.ndarray:
    return np.geomspace(rmin, rmax, bin_count + 1, dtype=np.float64)


def fraction_in_bins(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    if len(values) == 0:
        return np.zeros(len(edges) - 1, dtype=np.float64)
    counts, _ = np.histogram(values, bins=edges)
    total = float(np.sum(counts))
    if total <= 0.0:
        return np.zeros(len(edges) - 1, dtype=np.float64)
    return counts.astype(np.float64) / total


def estimate_naive_kr_from_visible_radii(radii: np.ndarray, rmin: float, rmax: float, kr_grid: np.ndarray) -> float:
    if len(radii) == 0:
        return float("nan")
    best_kr = float(kr_grid[0])
    best_loglik = -np.inf
    log_r = np.log(radii)
    for kr in kr_grid:
        exponent = 1.0 - float(kr)
        if abs(exponent) < 1e-12:
            norm = np.log(rmax / rmin)
            log_pdf = -log_r - np.log(norm)
        else:
            z = (rmax**exponent - rmin**exponent) / exponent
            log_pdf = -float(kr) * log_r - np.log(z)
        loglik = float(np.sum(log_pdf))
        if loglik > best_loglik:
            best_loglik = loglik
            best_kr = float(kr)
    return best_kr


def mismatch_label(observed_fraction: float, mc_fraction: float) -> str:
    delta = observed_fraction - mc_fraction
    if delta > 0.05:
        return "observed_overrepresents_this_radius_bin_relative_to_mc"
    if delta < -0.05:
        return "mc_overrepresents_this_radius_bin_relative_to_observed"
    return "balanced_this_radius_bin"


def bias_interpretation(
    observed_p90: float,
    mc_p90: float,
    observed_p95: float,
    mc_p95: float,
) -> str:
    ratio90 = observed_p90 / mc_p90 if np.isfinite(observed_p90) and np.isfinite(mc_p90) and mc_p90 > 0.0 else float("nan")
    ratio95 = observed_p95 / mc_p95 if np.isfinite(observed_p95) and np.isfinite(mc_p95) and mc_p95 > 0.0 else float("nan")
    if np.isfinite(ratio90) and np.isfinite(ratio95) and ratio90 > 1.10 and ratio95 > 1.10:
        return "observed_traces_are_biased_to_larger_radii_than_mc_visible_samples"
    if np.isfinite(ratio90) and np.isfinite(ratio95) and ratio90 < 0.90 and ratio95 < 0.90:
        return "mc_visible_samples_are_biased_to_larger_radii_than_observed_traces"
    return "radius_mixture_difference_is_mild_or_mixed"


def build_report(
    site: str,
    target_sets: Sequence[int],
    summary_rows: Sequence[dict],
    bin_rows: Sequence[dict],
) -> str:
    lines = [
        "# Observed Radius Mixture Report",
        "",
        f"- site: `{site}`",
        f"- target_sets: `{' '.join(str(v) for v in target_sets)}`",
        "",
        "## Interpretation Guide",
        "",
        "- `observed_vs_mc_radius_p90_ratio > 1`: observed traces come from larger radii than MC visible samples.",
        "- `observed_minus_mc_fraction > 0`: observed traces are overrepresented in that radius bin relative to MC visible samples.",
        "",
        "## Set Summaries",
        "",
    ]
    for set_id in target_sets:
        row = next((r for r in summary_rows if int(r["set_id"]) == int(set_id)), None)
        if row is None:
            continue
        lines.append(f"### Set {set_id}")
        lines.append("")
        lines.append(
            f"- observed/mc radius p90 ratio = {float(row['observed_vs_mc_radius_p90_ratio']):.3f}, "
            f"p95 ratio = {float(row['observed_vs_mc_radius_p95_ratio']):.3f}, "
            f"naive_kr_from_observed_trace_radii = {float(row['naive_kr_from_observed_trace_radii']):.3f}"
        )
        lines.append(f"- interpretation: `{row['bias_interpretation']}`")
        top_bins = sorted(
            [r for r in bin_rows if int(r["set_id"]) == int(set_id)],
            key=lambda r: abs(float(r["observed_minus_mc_fraction"])),
            reverse=True,
        )[:3]
        for bin_row in top_bins:
            lines.append(
                f"- radius_bin [{float(bin_row['radius_bin_low']):.3f}, {float(bin_row['radius_bin_high']):.3f}]: "
                f"generated={float(bin_row['generated_fraction']):.3f}, "
                f"observed={float(bin_row['observed_trace_fraction']):.3f}, "
                f"mc_visible={float(bin_row['mc_visible_fraction']):.3f}, "
                f"delta={float(bin_row['observed_minus_mc_fraction']):+.3f}, "
                f"mismatch=`{bin_row['dominant_radius_mismatch']}`"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare observed trace radius mixture against generated DFN and MC visible radius mixture.")
    parser.add_argument("--dfn-h5", required=True)
    parser.add_argument("--trace-h5", required=True)
    parser.add_argument("--site", required=True)
    parser.add_argument("--target-set", nargs="+", type=int, required=True)
    parser.add_argument("--rmin", type=float, default=0.5)
    parser.add_argument("--rmax", type=float, default=250.0)
    parser.add_argument("--radius-bin-count", type=int, default=10)
    parser.add_argument("--mc-visible-samples", type=int, default=300000)
    parser.add_argument("--kr-min", type=float, default=1.5)
    parser.add_argument("--kr-max", type=float, default=5.5)
    parser.add_argument("--kr-grid-size", type=int, default=81)
    parser.add_argument(
        "--set-rmin-mode",
        choices=["global", "effective_generation", "table_r0"],
        default="effective_generation",
        help="Per-set lower-bound mode for MC visible radius sampling.",
    )
    parser.add_argument("--seed", type=int, default=20260701)
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    dfn_data = load_hdf5_dfn(args.dfn_h5)
    trace_rows_raw, polygon_yz, _ = load_trace_dataset(args.trace_h5)
    trace_rows = normalize_rows_with_radius(trace_rows_raw, dfn_data)
    target_sets = sorted(set(args.target_set))
    edges = make_radius_edges(args.rmin, args.rmax, args.radius_bin_count)
    kr_grid = np.linspace(args.kr_min, args.kr_max, args.kr_grid_size, dtype=np.float64)
    trace_rmin_metadata = load_trace_rmin_metadata_from_h5(args.trace_h5)
    global_generation_rmin = float(trace_rmin_metadata.get("generation_rmin") or dfn_data.get("generation_rmin") or args.rmin)
    set_rmin_lookup = build_set_rmin_lookup(args.site, global_generation_rmin, trace_rmin_metadata)

    summary_rows: List[dict] = []
    bin_rows: List[dict] = []

    for set_id in target_sets:
        set_trace_rows = [
            row for row in trace_rows
            if int(row["set_id"]) == int(set_id) and args.rmin <= float(row["radius_m"]) <= args.rmax
        ]
        generated_radii = dfn_data["radii"][
            (dfn_data["set_ids"].astype(int) == int(set_id))
            & (dfn_data["radii"] >= args.rmin)
            & (dfn_data["radii"] <= args.rmax)
        ].astype(np.float64)
        observed_radii = np.asarray([float(row["radius_m"]) for row in set_trace_rows], dtype=np.float64)
        directions = empirical_trace_directions_yz(set_trace_rows)
        kr_true = _SITE_TO_KR_TRUE.get(args.site, {}).get(int(set_id))
        if kr_true is None:
            raise ValueError(f"No built-in kr_true for site={args.site}, set_id={set_id}")
        set_likelihood_rmin, set_effective_generation_rmin, _, _ = resolve_set_likelihood_rmin(
            int(set_id),
            args.set_rmin_mode,
            float(args.rmin),
            set_rmin_lookup,
        )
        rng = np.random.default_rng(args.seed + int(set_id) * 1000)
        mc_radii = sample_size_biased_radius(
            float(kr_true),
            float(set_likelihood_rmin),
            float(args.rmax),
            int(args.mc_visible_samples),
            rng,
        )
        _, _, _, mc_visible_radii = simulate_window_samples(
            polygon_yz,
            directions,
            mc_radii,
            rng,
            window_mode="polygon",
            direction_mode="empirical_trace",
            set_id=int(set_id),
            site=args.site,
        )

        generated_fraction = fraction_in_bins(generated_radii, edges)
        observed_fraction = fraction_in_bins(observed_radii, edges)
        mc_visible_fraction = fraction_in_bins(mc_visible_radii, edges)

        for idx in range(len(edges) - 1):
            bin_rows.append(
                {
                    "site": args.site,
                    "set_id": int(set_id),
                    "radius_bin_low": float(edges[idx]),
                    "radius_bin_high": float(edges[idx + 1]),
                    "generated_fraction": float(generated_fraction[idx]),
                    "observed_trace_fraction": float(observed_fraction[idx]),
                    "mc_visible_fraction": float(mc_visible_fraction[idx]),
                    "observed_minus_mc_fraction": float(observed_fraction[idx] - mc_visible_fraction[idx]),
                    "dominant_radius_mismatch": mismatch_label(float(observed_fraction[idx]), float(mc_visible_fraction[idx])),
                }
            )

        observed_p90 = quantile_or_nan(observed_radii, 90)
        observed_p95 = quantile_or_nan(observed_radii, 95)
        mc_p90 = quantile_or_nan(mc_visible_radii, 90)
        mc_p95 = quantile_or_nan(mc_visible_radii, 95)
        summary_rows.append(
            {
                "site": args.site,
                "set_id": int(set_id),
                "kr_true": float(kr_true),
                "n_generated_fractures": int(len(generated_radii)),
                "n_observed_traces": int(len(observed_radii)),
                "n_mc_visible_samples": int(len(mc_visible_radii)),
                "set_likelihood_rmin": float(set_likelihood_rmin),
                "set_effective_generation_rmin": float(set_effective_generation_rmin),
                "set_rmin_mode": args.set_rmin_mode,
                "generated_radius_p50": quantile_or_nan(generated_radii, 50),
                "generated_radius_p90": quantile_or_nan(generated_radii, 90),
                "generated_radius_p95": quantile_or_nan(generated_radii, 95),
                "observed_trace_radius_p50": quantile_or_nan(observed_radii, 50),
                "observed_trace_radius_p90": observed_p90,
                "observed_trace_radius_p95": observed_p95,
                "mc_visible_radius_p50": quantile_or_nan(mc_visible_radii, 50),
                "mc_visible_radius_p90": mc_p90,
                "mc_visible_radius_p95": mc_p95,
                "observed_vs_mc_radius_p90_ratio": observed_p90 / mc_p90 if np.isfinite(observed_p90) and np.isfinite(mc_p90) and mc_p90 > 0.0 else float("nan"),
                "observed_vs_mc_radius_p95_ratio": observed_p95 / mc_p95 if np.isfinite(observed_p95) and np.isfinite(mc_p95) and mc_p95 > 0.0 else float("nan"),
                "naive_kr_from_observed_trace_radii": estimate_naive_kr_from_visible_radii(observed_radii, float(args.rmin), float(args.rmax), kr_grid),
                "bias_interpretation": bias_interpretation(observed_p90, mc_p90, observed_p95, mc_p95),
            }
        )

    summary_csv = os.path.join(args.outdir, "observed_radius_mixture_summary.csv")
    bins_csv = os.path.join(args.outdir, "observed_vs_mc_visible_radius_bins.csv")
    report_md = os.path.join(args.outdir, "observed_radius_mixture_report.md")
    write_csv(summary_rows, summary_csv)
    write_csv(bin_rows, bins_csv)
    with open(report_md, "w", encoding="utf-8") as f:
        f.write(build_report(args.site, target_sets, summary_rows, bin_rows))

    print("[*] Observed radius mixture diagnostic complete")
    print(f"[*] Summary CSV written to: {summary_csv}")
    print(f"[*] Bin comparison CSV written to: {bins_csv}")
    print(f"[*] Report written to: {report_md}")


if __name__ == "__main__":
    main()
