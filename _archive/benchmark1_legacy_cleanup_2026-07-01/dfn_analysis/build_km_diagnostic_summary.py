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
    parser = argparse.ArgumentParser(description="Build final Kaplan-Meier diagnostic summary across sites.")
    parser.add_argument("--km-summary-csv", nargs="+", required=True)
    parser.add_argument("--km-lmin-csv", nargs="+", required=True)
    parser.add_argument("--outcsv", required=True)
    parser.add_argument("--outmd", required=True)
    parser.add_argument("--kr-summary-csv", default=DEFAULT_KR_SUMMARY_CSV)
    args = parser.parse_args()

    km_summary_rows = load_rows(args.km_summary_csv)
    km_lmin_rows = load_rows(args.km_lmin_csv)
    kr_rows = {
        (str(row["site"]), int(row["set_id"])): row
        for row in read_csv(args.kr_summary_csv)
    }
    p32_status_map = build_p32_status_map()
    lmin_map = {
        (str(row["site"]), int(row["set_id"])): row
        for row in km_lmin_rows
    }

    out_rows: List[dict] = []
    for row in km_summary_rows:
        site = str(row["site"])
        set_id = int(row["set_id"])
        kr_row = kr_rows.get((site, set_id), {})
        best_lmin = to_float(kr_row, "best_lmin_fit")
        if not (abs(float(row["lmin_fit"]) - best_lmin) <= 1e-12):
            continue
        lmin_row = lmin_map.get((site, set_id), {})
        notes = []
        if kr_row.get("notes"):
            notes.append(str(kr_row["notes"]))
        out_rows.append(
            {
                "site": site,
                "set_id": set_id,
                "best_lmin_fit": float(row["lmin_fit"]),
                "n_traces": int(row["n_traces"]),
                "n_exact": int(row["n_exact"]),
                "n_censored": int(row["n_censored"]),
                "censoring_fraction": float(row["censoring_fraction"]),
                "raw_p50": float(row["raw_p50"]),
                "raw_p90": float(row["raw_p90"]),
                "raw_p95": float(row["raw_p95"]),
                "km_p50": float(row["km_p50"]),
                "km_p90": float(row["km_p90"]),
                "km_p95": float(row["km_p95"]),
                "p90_ratio_km_vs_raw": float(row["p90_ratio_km_vs_raw"]),
                "p95_ratio_km_vs_raw": float(row["p95_ratio_km_vs_raw"]),
                "km_status": str(row["km_status"]),
                "km_tail_stability_status": str(lmin_row.get("km_tail_stability_status", "")),
                "kr_adoption_status": str(kr_row.get("adoption_status", "")),
                "p32_final_status": str(p32_status_map.get((site, set_id), "")),
                "notes": "; ".join(notes),
            }
        )

    os.makedirs(os.path.dirname(args.outcsv) or ".", exist_ok=True)
    write_csv(out_rows, args.outcsv)

    strong = [row for row in out_rows if str(row["km_status"]) == "strong_censoring_effect"]
    sensitive = [row for row in out_rows if str(row["km_tail_stability_status"]) == "km_tail_sensitive_to_lmin"]
    strong_text = ", ".join([f"{row['site']} set {row['set_id']}" for row in strong]) if strong else "none"
    sensitive_text = ", ".join([f"{row['site']} set {row['set_id']}" for row in sensitive]) if sensitive else "none"
    lines = [
        "# KM Final Diagnostic Report",
        "",
        "Kaplan-Meier is used only as an auxiliary diagnostic for trace-length censoring.",
        "It does not replace the final kr inversion or the final unit-P32 pilot estimator.",
        "",
        "## Summary",
        "",
        "| site | set_id | best_lmin | censoring_fraction | p90_ratio_km_vs_raw | p95_ratio_km_vs_raw | km_status | km_tail_stability | kr_status | p32_status |",
        "|------|--------|-----------|--------------------|---------------------|---------------------|-----------|-------------------|-----------|------------|",
    ]
    for row in out_rows:
        lines.append(
            f"| {row['site']} | {row['set_id']} | {row['best_lmin_fit']:.2f} | {row['censoring_fraction']:.3f} | "
            f"{row['p90_ratio_km_vs_raw']:.3f} | {row['p95_ratio_km_vs_raw']:.3f} | {row['km_status']} | "
            f"{row['km_tail_stability_status']} | {row['kr_adoption_status']} | {row['p32_final_status']} |"
        )
    lines += [
        "",
        "## Flags",
        f"- strong_censoring_effect: {strong_text}",
        f"- km_tail_sensitive_to_lmin: {sensitive_text}",
        "",
        "## Use",
        "- Use this summary to explain censoring and tail sensitivity.",
        "- Do not update kr_hat, P32_hat, or adoption status from KM alone.",
        "",
    ]
    with open(args.outmd, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[*] KM final summary written to: {args.outcsv}")
    print(f"[*] KM final report written to: {args.outmd}")


if __name__ == "__main__":
    main()
