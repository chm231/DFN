import csv
import os
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(ROOT, "storage", "output", "benchmark1_parsimonious")
# Self-contained curated input snapshot for the parsimonious baseline report.
# The report reads these first so the legacy experiment output folders can be
# archived without breaking the baseline. If a curated file is absent, the
# original experiment path is used as a regeneration fallback.
INPUTS_DIR = os.path.join(OUTPUT_DIR, "inputs")
LEGACY_OUTPUT_DIR = os.path.join(ROOT, "storage", "output")


# --- Inlined CSV helpers (previously imported from estimate_p32_mc_calibrated) ---
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


def curated_input(curated_name: str, *legacy_parts: str) -> str:
    """Prefer the curated benchmark1_parsimonious/inputs snapshot; fall back to
    the legacy experiment output path if the curated file is not present."""
    curated = os.path.join(INPUTS_DIR, curated_name)
    if os.path.exists(curated):
        return curated
    return os.path.join(LEGACY_OUTPUT_DIR, *legacy_parts)


KR_SUMMARY_CSV = curated_input(
    "final_kr_recovery_summary_effective_rmin.csv",
    "final_kr_recovery_summary_effective_rmin.csv",
)
KR_BOOTSTRAP_CSV = curated_input(
    "final_kr_bootstrap_summary_effective_rmin.csv",
    "final_kr_bootstrap_effective_rmin",
    "final_kr_bootstrap_summary_effective_rmin.csv",
)
P32_FINAL_CSV = curated_input(
    "p32_combined_bootstrap_summary.csv",
    "p32_mc_calibrated_effective_rmin",
    "p32_final_pilot",
    "p32_combined_bootstrap_summary.csv",
)
KM_MC_CSV = curated_input(
    "km_mc_final_comparison_summary.csv",
    "trace_length_km_diagnostics",
    "km_mc_final_comparison_summary.csv",
)
WINDOW_MC_LAXEMAR_CSV = curated_input(
    "window_mc_fit_by_set_laxemar.csv",
    "final_kr_bootstrap_effective_rmin",
    "laxemar_sets_1_2_3_5",
    "window_mc_fit_by_set.csv",
)
WINDOW_MC_FORSMARK_CSV = curated_input(
    "window_mc_fit_by_set_forsmark.csv",
    "final_kr_bootstrap_effective_rmin",
    "forsmark_sets_1_2_5",
    "window_mc_fit_by_set.csv",
)
# Canonical active trace dataset (README step 3 output); kept in the active tree.
TRACE_STATS_CSVS = {
    "laxemar": os.path.join(ROOT, "storage", "output", "trace_dataset_collection", "setwise_trace_statistics.csv"),
    "forsmark": os.path.join(ROOT, "storage", "output", "trace_dataset_collection_forsmark", "setwise_trace_statistics.csv"),
}
TRACE_DATASET_CSVS = {
    "laxemar": os.path.join(ROOT, "storage", "output", "trace_dataset_collection", "trace_dataset_3d.csv"),
    "forsmark": os.path.join(ROOT, "storage", "output", "trace_dataset_collection_forsmark", "trace_dataset_3d.csv"),
}
TRACE_CENSORING_CSVS = {
    "laxemar": curated_input("trace_censoring_by_set_laxemar.csv", "trace_censoring_diagnostics", "trace_censoring_by_set.csv"),
    "forsmark": curated_input("trace_censoring_by_set_forsmark.csv", "trace_censoring_diagnostics_forsmark", "trace_censoring_by_set.csv"),
}
WINDOW_CLIPPING_CSVS = {
    "laxemar": curated_input("window_clipping_summary_by_set_laxemar.csv", "window_clipping_diagnostics", "window_clipping_summary_by_set.csv"),
    "forsmark": curated_input("window_clipping_summary_by_set_forsmark.csv", "window_clipping_diagnostics_forsmark", "window_clipping_summary_by_set.csv"),
}
ORIENTATION_CSVS = {
    "laxemar": curated_input("tunnel_direction_bias_summary_laxemar.csv", "tunnel_direction_bias_rmin0p5", "laxemar", "tunnel_direction_bias_summary_rmin0p5.csv"),
    "forsmark": curated_input("tunnel_direction_bias_summary_forsmark.csv", "tunnel_direction_bias_rmin0p5", "forsmark", "tunnel_direction_bias_summary_rmin0p5.csv"),
}

METHODOLOGY_TEXT = """# Parsimonious Estimator Baseline Summary

## Common Estimator
- Estimation targets: fracture-set-wise `kr` and `P32` over the declared support range.
- Observation basis: 2D tunnel face traces with observed lengths, censoring classes, and finite face geometry.
- Shared components:
  1. trace observation model
  2. power-law radius likelihood
  3. finite window / face clipping treatment
  4. size-biased observation effect
  5. `unit_p32_forward_mc` calibration for `P32`

## Truth Policy
Benchmark truth values are used only for validation, not for estimator design or calibration.
A single parsimonious estimator is applied consistently to all fracture sets.
Remaining discrepancies are interpreted through observation bias, finite-window clipping, censoring, finite-sample variability, orientation-dependent intersection probability, and limited identifiability of 3D DFN parameters from 2D trace observations.

## Diagnostic Flag Rules
- `finite_window_effect_flag`: `true` when the clipping diagnostic marks `window_aware_priority = high`, otherwise `false`.
- `size_bias_expected_flag`: `true` for every set because larger fractures have higher tunnel-face intersection opportunity by construction.
- `low_sample_size_flag`: `true` when `n_observed_traces < 30`.
- `orientation_effect_flag`: `true` when `orientation_factor_mean <= 0.5`, otherwise `false`.
- `identifiability_issue_flag`: `true` when low sample size, censoring >= 0.5, or KM-vs-MC mismatch remains.
- Unknown or unavailable diagnostics are left as `not_evaluated`.
"""


def keyed_rows(rows: Iterable[dict]) -> Dict[Tuple[str, int], dict]:
    out: Dict[Tuple[str, int], dict] = {}
    for row in rows:
        out[(str(row["site"]), int(row["set_id"]))] = row
    return out


def load_window_metadata() -> Dict[Tuple[str, int], dict]:
    out: Dict[Tuple[str, int], dict] = {}
    for site, path in (("laxemar", WINDOW_MC_LAXEMAR_CSV), ("forsmark", WINDOW_MC_FORSMARK_CSV)):
        for row in read_csv(path):
            key = (site, int(row["set_id"]))
            out.setdefault(key, row)
    return out


def load_site_set_rows(paths: Dict[str, str]) -> Dict[Tuple[str, int], dict]:
    out: Dict[Tuple[str, int], dict] = {}
    for site, path in paths.items():
        for row in read_csv(path):
            out[(site, int(row["set_id"]))] = row
    return out


def load_trace_max_lengths() -> Dict[Tuple[str, int], float]:
    out: Dict[Tuple[str, int], float] = {}
    for site, path in TRACE_DATASET_CSVS.items():
        with open(path, "r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                key = (site, int(row["set_id"]))
                length = float(row["observed_length_m"])
                out[key] = max(length, out.get(key, 0.0))
    return out


def truth_stage_rows() -> Tuple[Dict[Tuple[str, int], dict], Dict[Tuple[str, int], dict]]:
    return keyed_rows(read_csv(KR_SUMMARY_CSV)), keyed_rows(read_csv(P32_FINAL_CSV))


def bool_text(value: Optional[bool]) -> str:
    if value is None:
        return "not_evaluated"
    return "true" if value else "false"


def build_main_possible_source(
    low_sample: Optional[bool],
    finite_window: Optional[bool],
    orientation: Optional[bool],
    identifiability: Optional[bool],
    mismatch_status: str,
    clipping_row: Optional[dict],
) -> str:
    if low_sample:
        return "finite_sample_variability"
    if finite_window:
        return "finite_window_clipping"
    if orientation:
        return "orientation_dependent_intersection_probability"
    if clipping_row is not None and "censor" in str(clipping_row.get("dominant_failure_mode", "")):
        return "trace_censoring"
    if mismatch_status == "mc_km_tail_mismatch":
        return "limited_identifiability_from_2d_traces"
    if identifiability:
        return "limited_identifiability_from_2d_traces"
    return "no_single_dominant_source_identified"


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    kr_rows = keyed_rows(read_csv(KR_SUMMARY_CSV))
    kr_boot_rows = keyed_rows(read_csv(KR_BOOTSTRAP_CSV))
    p32_rows = keyed_rows(read_csv(P32_FINAL_CSV))
    km_rows = keyed_rows(read_csv(KM_MC_CSV))
    window_rows = load_window_metadata()
    trace_stats_rows = load_site_set_rows(TRACE_STATS_CSVS)
    censor_rows = load_site_set_rows(TRACE_CENSORING_CSVS)
    clipping_rows = load_site_set_rows(WINDOW_CLIPPING_CSVS)
    orientation_rows = load_site_set_rows(ORIENTATION_CSVS)
    trace_max_lengths = load_trace_max_lengths()

    keys = sorted(set(kr_rows) | set(p32_rows))
    common_rows: List[dict] = []
    for key in keys:
        site, set_id = key
        kr_row = kr_rows.get(key, {})
        kr_boot_row = kr_boot_rows.get(key, {})
        p32_row = p32_rows.get(key, {})
        trace_row = trace_stats_rows.get(key, {})
        window_row = window_rows.get(key, {})
        common_rows.append(
            {
                "site": site,
                "set_id": set_id,
                "n_observed_traces": int(float(trace_row.get("n_total", 0))) if trace_row else "",
                "radius_range_min": to_float(kr_row, "set_effective_generation_rmin"),
                "radius_range_max": to_float(window_row, "generation_rmax"),
                "estimated_kr": to_float(kr_row, "kr_hat"),
                "estimated_kr_lower": to_float(kr_boot_row, "kr_ci_low") if kr_boot_row else "not_available",
                "estimated_kr_upper": to_float(kr_boot_row, "kr_ci_high") if kr_boot_row else "not_available",
                "estimated_P32": to_float(p32_row, "P32_hat") if p32_row else "not_available",
                "estimated_P32_lower": to_float(p32_row, "P32_ci_low") if p32_row else "not_available",
                "estimated_P32_upper": to_float(p32_row, "P32_ci_high") if p32_row else "not_available",
                "estimator_name": "effective_rmin_window_mc_plus_unit_p32_forward_mc",
                "estimator_version": "benchmark1_parsimonious_baseline",
                "script_name": "scripts/run_parsimonious_baseline_report.py",
                "notes": str(kr_row.get("notes", "")),
            }
        )

    common_csv = os.path.join(OUTPUT_DIR, "common_estimator_results.csv")
    write_csv(common_rows, common_csv)

    # Validation-only truth stage starts after the GT-free common estimator report is written.
    truth_kr_rows, truth_p32_rows = truth_stage_rows()
    gt_rows: List[dict] = []
    mismatch_rows: List[dict] = []

    for key in keys:
        site, set_id = key
        kr_row = truth_kr_rows.get(key, {})
        p32_row = truth_p32_rows.get(key, {})
        trace_row = trace_stats_rows.get(key, {})
        censor_row = censor_rows.get(key, {})
        clipping_row = clipping_rows.get(key, {})
        orientation_row = orientation_rows.get(key, {})
        km_row = km_rows.get(key, {})
        max_length = trace_max_lengths.get(key, float("nan"))

        estimated_kr = to_float(kr_row, "kr_hat")
        gt_kr = to_float(kr_row, "kr_true")
        estimated_p32 = to_float(p32_row, "P32_hat") if p32_row else float("nan")
        gt_p32 = to_float(p32_row, "P32_reference") if p32_row else float("nan")
        radius_min = to_float(kr_row, "set_effective_generation_rmin")
        radius_max = to_float(window_rows.get(key, {}), "generation_rmax")

        gt_rows.append(
            {
                "site": site,
                "set_id": set_id,
                "estimated_kr": estimated_kr,
                "gt_kr": gt_kr,
                "kr_error": estimated_kr - gt_kr,
                "kr_relative_error": (estimated_kr - gt_kr) / gt_kr if gt_kr else float("nan"),
                "estimated_P32": estimated_p32,
                "gt_P32": gt_p32,
                "P32_error": estimated_p32 - gt_p32,
                "P32_relative_error": (estimated_p32 - gt_p32) / gt_p32 if gt_p32 else float("nan"),
                "radius_range_min": radius_min,
                "radius_range_max": radius_max,
                "validation_only_truth_used": "true",
            }
        )

        n_observed = int(float(trace_row.get("n_total", 0))) if trace_row else 0
        censor_ratio = to_float(censor_row, "censoring_ratio_total") if censor_row else float("nan")
        orientation_mean = to_float(orientation_row, "orientation_factor_mean") if orientation_row else float("nan")
        finite_window = None if not clipping_row else str(clipping_row.get("window_aware_priority", "")) == "high"
        low_sample = None if not trace_row else n_observed < 30
        orientation_flag = None if not orientation_row else orientation_mean <= 0.5
        identifiability = None if not trace_row else (
            n_observed < 30
            or (censor_ratio == censor_ratio and censor_ratio >= 0.5)
            or str(km_row.get("mc_km_consistency_status", "")) == "mc_km_tail_mismatch"
        )
        main_source = build_main_possible_source(
            low_sample,
            finite_window,
            orientation_flag,
            identifiability,
            str(km_row.get("mc_km_consistency_status", "")),
            clipping_row if clipping_row else None,
        )
        diagnostic_parts = [
            f"window_priority={clipping_row.get('window_aware_priority', 'not_evaluated')}" if clipping_row else "window_priority=not_evaluated",
            f"dominant_failure_mode={clipping_row.get('dominant_failure_mode', 'not_evaluated')}" if clipping_row else "dominant_failure_mode=not_evaluated",
            f"km_status={km_row.get('mc_km_consistency_status', 'not_evaluated')}" if km_row else "km_status=not_evaluated",
        ]
        mismatch_rows.append(
            {
                "site": site,
                "set_id": set_id,
                "n_observed_traces": n_observed,
                "trace_length_mean": to_float(trace_row, "total_trace_length") / n_observed if n_observed else float("nan"),
                "trace_length_median": to_float(trace_row, "median_length_uncensored"),
                "trace_length_max": max_length,
                "fraction_clipped_or_censored": censor_ratio,
                "finite_window_effect_flag": bool_text(finite_window),
                "size_bias_expected_flag": "true",
                "low_sample_size_flag": bool_text(low_sample),
                "orientation_effect_flag": bool_text(orientation_flag),
                "identifiability_issue_flag": bool_text(identifiability),
                "main_possible_mismatch_source": main_source,
                "diagnostic_comment": "; ".join(diagnostic_parts),
            }
        )

    gt_csv = os.path.join(OUTPUT_DIR, "gt_comparison.csv")
    mismatch_csv = os.path.join(OUTPUT_DIR, "mismatch_diagnostics.csv")
    methodology_md = os.path.join(OUTPUT_DIR, "methodology_summary.md")

    write_csv(gt_rows, gt_csv)
    write_csv(mismatch_rows, mismatch_csv)
    with open(methodology_md, "w", encoding="utf-8") as handle:
        handle.write(METHODOLOGY_TEXT)

    print(f"[*] Common estimator results written to: {common_csv}")
    print(f"[*] GT comparison written to: {gt_csv}")
    print(f"[*] Mismatch diagnostics written to: {mismatch_csv}")
    print(f"[*] Methodology summary written to: {methodology_md}")
    print("[*] GT was not used to fit or rescale the estimator in this script; validation is written only after the common result table is produced.")


if __name__ == "__main__":
    main()
