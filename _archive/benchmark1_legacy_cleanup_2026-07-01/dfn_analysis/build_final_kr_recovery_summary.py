import argparse
import csv
import math
import os
from typing import Dict, List, Sequence, Tuple


ADOPTION_PRIORITY = {
    "accepted": 0,
    "provisional_accepted": 1,
    "rejected": 2,
    "diagnostic_only_rmin_support_mismatch": 3,
}


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


def best_row(rows: Sequence[dict]) -> dict:
    def sort_key(row: dict) -> Tuple[float, float, float, float]:
        adoption = row.get("adoption_status", "rejected")
        return (
            ADOPTION_PRIORITY.get(adoption, 9),
            abs(to_float(row, "kr_window_mc_hat") - to_float(row, "kr_true")),
            to_float(row, "class_fraction_l1_error"),
            abs(to_float(row, "q90_ratio_model_observed") - 1.0),
        )

    return sorted(rows, key=sort_key)[0]


def build_notes(site: str, set_id: int, fit_row: dict, mixture_row: dict) -> str:
    notes: List[str] = []
    if fit_row.get("rmin_support_status") == "matched":
        notes.append("effective_rmin_applied")
    else:
        notes.append("support_mismatch_diagnostic_only")
    ratio = to_float(mixture_row, "observed_vs_mc_radius_p90_ratio")
    if math.isfinite(ratio):
        if ratio > 1.5:
            notes.append("observed_radius_mixture_still_larger_than_mc")
        elif ratio > 1.1:
            notes.append("minor_observed_radius_mixture_excess")
        else:
            notes.append("radius_mixture_near_matched")
    if site == "forsmark" and set_id == 2:
        notes.append("forsmark_set2_visibility_opportunity_residual_bias")
    return "; ".join(notes)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build final kr recovery summary under effective-rmin mode.")
    parser.add_argument("--fit-csv", nargs="+", required=True, help="One or more window_mc_fit_by_set.csv files.")
    parser.add_argument("--mixture-csv", nargs="+", required=True, help="One or more observed_radius_mixture_summary.csv files.")
    parser.add_argument("--outcsv", required=True, help="Output CSV path.")
    args = parser.parse_args()

    fit_rows_all: List[dict] = []
    for path in args.fit_csv:
        fit_rows_all.extend(read_csv(path))

    mixture_map: Dict[Tuple[str, int], dict] = {}
    for path in args.mixture_csv:
        for row in read_csv(path):
            mixture_map[(str(row["site"]), int(row["set_id"]))] = row

    grouped: Dict[Tuple[str, int], List[dict]] = {}
    for row in fit_rows_all:
        key = (str(row["dfn_model"]), int(row["set_id"]))
        grouped.setdefault(key, []).append(row)

    summary_rows: List[dict] = []
    for (site, set_id), rows in sorted(grouped.items()):
        chosen = best_row(rows)
        mixture_row = mixture_map.get((site, set_id), {})
        kr_true = to_float(chosen, "kr_true")
        kr_hat = to_float(chosen, "kr_window_mc_hat")
        kr_abs_error = abs(kr_hat - kr_true) if math.isfinite(kr_true) and math.isfinite(kr_hat) else float("nan")
        rel_err = 100.0 * kr_abs_error / abs(kr_true) if math.isfinite(kr_abs_error) and kr_true != 0.0 else float("nan")
        summary_rows.append(
            {
                "site": site,
                "set_id": set_id,
                "kr_true": kr_true,
                "set_effective_generation_rmin": to_float(chosen, "set_effective_generation_rmin"),
                "set_likelihood_rmin": to_float(chosen, "set_likelihood_rmin"),
                "best_lmin_fit": to_float(chosen, "lmin_fit"),
                "kr_hat": kr_hat,
                "kr_abs_error": kr_abs_error,
                "kr_relative_error_percent": rel_err,
                "fit_status": chosen.get("fit_status", ""),
                "recovery_status": chosen.get("recovery_status", ""),
                "adoption_status": chosen.get("adoption_status", ""),
                "q90_ratio": to_float(chosen, "q90_ratio_model_observed"),
                "q95_ratio": to_float(chosen, "q95_ratio_model_observed"),
                "class_l1": to_float(chosen, "class_fraction_l1_error"),
                "observed_mc_radius_p90_ratio": to_float(mixture_row, "observed_vs_mc_radius_p90_ratio"),
                "notes": build_notes(site, set_id, chosen, mixture_row),
            }
        )

    os.makedirs(os.path.dirname(args.outcsv), exist_ok=True)
    write_csv(summary_rows, args.outcsv)
    print(f"[*] Final kr recovery summary written to: {args.outcsv}")


if __name__ == "__main__":
    main()
