import argparse
import os
import sys
from typing import Dict, List, Tuple

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dfn_analysis.estimate_p32_mc_calibrated import read_csv, to_float, write_csv


DEFAULT_KR_SUMMARY_CSV = "storage/output/final_kr_recovery_summary_effective_rmin.csv"
DEFAULT_P32_FULL_UNIT_CSV = "storage/output/p32_mc_calibrated_effective_rmin/full_unit_p32/p32_full_unit_summary.csv"
DEFAULT_P32_FINAL_PILOT_CSV = "storage/output/p32_mc_calibrated_effective_rmin/p32_final_pilot/p32_combined_bootstrap_summary.csv"
DEFAULT_FORSMARK5_DENSE_CSV = "storage/output/p32_mc_calibrated_effective_rmin/forsmark_set5_dense_ckr/forsmark_set5_dense_ckr_bootstrap_summary.csv"


def load_rows(paths: List[str]) -> List[dict]:
    out: List[dict] = []
    for path in paths:
        out.extend(read_csv(path))
    return out


def build_p32_status_map() -> Dict[Tuple[str, int], str]:
    status_map: Dict[Tuple[str, int], str] = {}
    for row in read_csv(DEFAULT_P32_FULL_UNIT_CSV):
        status_map[(str(row["site"]), int(row["set_id"]))] = str(row.get("p32_status", ""))
    for row in read_csv(DEFAULT_P32_FINAL_PILOT_CSV):
        status_map[(str(row["site"]), int(row["set_id"]))] = str(row.get("p32_final_pilot_status", ""))
    for row in read_csv(DEFAULT_FORSMARK5_DENSE_CSV):
        status_map[(str(row["site"]), int(row["set_id"]))] = str(row.get("p32_final_pilot_status", row.get("status_update", "")))
    return status_map


def main() -> None:
    parser = argparse.ArgumentParser(description="Build final KM vs MC comparison summary.")
    parser.add_argument("--km-mc-csv", nargs="+", required=True)
    parser.add_argument("--kr-summary-csv", default=DEFAULT_KR_SUMMARY_CSV)
    parser.add_argument("--outcsv", required=True)
    parser.add_argument("--outmd", required=True)
    args = parser.parse_args()

    km_mc_rows = load_rows(args.km_mc_csv)
    kr_rows = {(str(row["site"]), int(row["set_id"])): row for row in read_csv(args.kr_summary_csv)}
    p32_status_map = build_p32_status_map()

    grouped: Dict[Tuple[str, int], List[dict]] = {}
    for row in km_mc_rows:
        grouped.setdefault((str(row["site"]), int(row["set_id"])), []).append(row)

    out_rows: List[dict] = []
    for (site, set_id), rows in sorted(grouped.items()):
        kr_row = kr_rows.get((site, set_id), {})
        best_lmin = to_float(kr_row, "best_lmin_fit")
        selected = None
        for row in rows:
            if abs(float(row["lmin_fit"]) - best_lmin) <= 1e-12:
                selected = row
                break
        if selected is None:
            selected = rows[0]
        notes = []
        if kr_row.get("notes"):
            notes.append(str(kr_row["notes"]))
        notes.append("comparison_uses_mc_km_emulated_survival")
        out_rows.append(
            {
                "site": site,
                "set_id": set_id,
                "best_lmin_fit": float(selected["lmin_fit"]),
                "km_p90": float(selected["km_p90"]),
                "mc_p90": float(selected["mc_p90"]),
                "p90_ratio_mc_vs_km": float(selected["p90_ratio_mc_vs_km"]),
                "km_p95": float(selected["km_p95"]),
                "mc_p95": float(selected["mc_p95"]),
                "p95_ratio_mc_vs_km": float(selected["p95_ratio_mc_vs_km"]),
                "survival_l1_distance": float(selected["survival_l1_distance"]),
                "survival_l2_distance": float(selected["survival_l2_distance"]),
                "mc_km_consistency_status": str(selected["mc_km_consistency_status"]),
                "kr_adoption_status": str(kr_row.get("adoption_status", "")),
                "p32_final_status": str(p32_status_map.get((site, set_id), "")),
                "notes": "; ".join(notes),
            }
        )

    os.makedirs(os.path.dirname(args.outcsv) or ".", exist_ok=True)
    write_csv(out_rows, args.outcsv)

    consistent = [row for row in out_rows if str(row["mc_km_consistency_status"]) == "mc_consistent_with_km"]
    mismatch = [row for row in out_rows if str(row["mc_km_consistency_status"]) == "mc_km_tail_mismatch"]
    consistent_text = ", ".join([f"{row['site']} set {row['set_id']}" for row in consistent]) if consistent else "none"
    mismatch_text = ", ".join([f"{row['site']} set {row['set_id']}" for row in mismatch]) if mismatch else "none"
    lines = [
        "# KM vs MC Final Comparison Report",
        "",
        "KM is a non-parametric censoring diagnostic, while MC is the fitted window-aware likelihood prediction.",
        "The direct comparison between observed KM survival and MC visible-length survival is diagnostic only, because KM estimates a censoring-adjusted survival curve whereas MC visible-length survival represents clipped observed lengths.",
        "The primary consistency check here uses MC KM-emulated survival, where the same Kaplan-Meier procedure is applied to simulated observed lengths and simulated censoring classes.",
        "This comparison does not replace final kr or P32 estimates.",
        "",
        "## Summary",
        "",
        "| site | set_id | best_lmin | km_p90 | mc_p90 | p90_ratio | km_p95 | mc_p95 | p95_ratio | l1 | l2 | status |",
        "|------|--------|-----------|--------|--------|-----------|--------|--------|-----------|----|----|--------|",
    ]
    for row in out_rows:
        lines.append(
            f"| {row['site']} | {row['set_id']} | {row['best_lmin_fit']:.2f} | {row['km_p90']:.3f} | {row['mc_p90']:.3f} | "
            f"{row['p90_ratio_mc_vs_km']:.3f} | {row['km_p95']:.3f} | {row['mc_p95']:.3f} | "
            f"{row['p95_ratio_mc_vs_km']:.3f} | {row['survival_l1_distance']:.3f} | {row['survival_l2_distance']:.3f} | "
            f"{row['mc_km_consistency_status']} |"
        )
    lines += [
        "",
        "## Sets",
        f"- mc_consistent_with_km: {consistent_text}",
        f"- mc_km_tail_mismatch: {mismatch_text}",
        "",
        "## Interpretation",
        "- Mismatch can reflect polygon clipping mismatch, censoring-class treatment differences, lmin-fit interaction, trace filtering differences, or small-sample / face-level variability.",
        "- Do not update accepted/provisional/rejected statuses from KM vs MC comparison alone.",
        "",
    ]
    with open(args.outmd, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[*] KM/MC final summary written to: {args.outcsv}")
    print(f"[*] KM/MC final report written to: {args.outmd}")


if __name__ == "__main__":
    main()
