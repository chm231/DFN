import argparse
import csv
import os
from typing import Dict, List, Tuple


def read_csv(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(rows: List[dict], path: str) -> None:
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Build bootstrap CI summary for accepted/provisional sets.")
    parser.add_argument("--bootstrap-fit-csv", nargs="+", required=True, help="Bootstrap window_mc_fit_by_set.csv files.")
    parser.add_argument("--reference-summary-csv", required=True, help="Reference final effective-rmin recovery summary CSV.")
    parser.add_argument("--outcsv", required=True, help="Output bootstrap summary CSV.")
    args = parser.parse_args()

    reference_rows = read_csv(args.reference_summary_csv)
    reference_map: Dict[Tuple[str, int], dict] = {
        (str(row["site"]), int(row["set_id"])): row for row in reference_rows
    }

    bootstrap_rows_all: List[dict] = []
    for path in args.bootstrap_fit_csv:
        bootstrap_rows_all.extend(read_csv(path))

    bootstrap_map: Dict[Tuple[str, int, float], dict] = {}
    for row in bootstrap_rows_all:
        key = (str(row["dfn_model"]), int(row["set_id"]), float(row["lmin_fit"]))
        bootstrap_map[key] = row

    summary_rows: List[dict] = []
    for (site, set_id), ref_row in sorted(reference_map.items()):
        best_lmin_fit = float(ref_row["best_lmin_fit"])
        bootstrap_row = bootstrap_map.get((site, set_id, best_lmin_fit))
        if bootstrap_row is None:
            continue
        kr_true = to_float(bootstrap_row, "kr_true")
        kr_ci_low = to_float(bootstrap_row, "kr_ci_low")
        kr_ci_high = to_float(bootstrap_row, "kr_ci_high")
        true_inside = bool(kr_ci_low <= kr_true <= kr_ci_high) if kr_true == kr_true and kr_ci_low == kr_ci_low and kr_ci_high == kr_ci_high else False
        summary_rows.append(
            {
                "site": site,
                "set_id": set_id,
                "kr_true": kr_true,
                "kr_hat": to_float(bootstrap_row, "kr_window_mc_hat"),
                "kr_boot_mean": to_float(bootstrap_row, "kr_boot_mean"),
                "kr_boot_std": to_float(bootstrap_row, "kr_boot_std"),
                "kr_ci_low": kr_ci_low,
                "kr_ci_high": kr_ci_high,
                "true_kr_inside_ci": true_inside,
                "bootstrap_boundary_fraction": to_float(bootstrap_row, "bootstrap_boundary_fraction"),
                "recovery_ci_status": bootstrap_row.get("recovery_ci_status", ""),
                "adoption_status": bootstrap_row.get("adoption_status", ""),
            }
        )

    os.makedirs(os.path.dirname(args.outcsv), exist_ok=True)
    write_csv(summary_rows, args.outcsv)
    print(f"[*] Final bootstrap summary written to: {args.outcsv}")


if __name__ == "__main__":
    main()
