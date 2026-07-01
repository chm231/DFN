import argparse
import csv
import os
from typing import Dict, List, Sequence, Tuple


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


def merge_notes(summary_row: dict, diag_row: dict) -> str:
    notes = [str(summary_row.get("notes", ""))]
    site = str(summary_row.get("site", ""))
    set_id = int(summary_row.get("set_id", 0))
    status = str(diag_row.get("p32_calibration_status", ""))
    if site == "laxemar" and set_id == 2 and status == "calibration_marginal":
        notes.append("laxemar_set2_p32_calibration_marginal")
    return "; ".join(note for note in notes if note)


def resolve_p32_status(summary_row: dict, diag_row: dict) -> str:
    current = str(summary_row.get("p32_status", ""))
    diag_status = str(diag_row.get("p32_calibration_status", ""))
    site = str(summary_row.get("site", ""))
    set_id = int(summary_row.get("set_id", 0))
    if site == "forsmark" and set_id == 2:
        return "p32_mc_provisional_systematic_bias"
    if diag_status == "calibration_marginal" and current == "p32_mc_pilot_candidate":
        return "p32_mc_marginal_candidate"
    if diag_status == "calibration_failed" and current != "p32_mc_provisional_systematic_bias":
        return "p32_mc_hold_for_calibration"
    return current


def main() -> None:
    parser = argparse.ArgumentParser(description="Build merged full-scale unit-P32 summary from site summaries and calibration diagnostics.")
    parser.add_argument("--summary-csv", nargs="+", required=True, help="One or more unit_p32_forward_mc summary CSVs.")
    parser.add_argument("--diagnostic-csv", nargs="+", required=True, help="One or more p32 calibration diagnostic CSVs.")
    parser.add_argument("--outcsv", required=True, help="Merged output CSV path.")
    args = parser.parse_args()

    summary_rows: List[dict] = []
    for path in args.summary_csv:
        summary_rows.extend(read_csv(path))

    diag_map: Dict[Tuple[str, int], dict] = {}
    for path in args.diagnostic_csv:
        for row in read_csv(path):
            diag_map[(str(row["site"]), int(row["set_id"]))] = row

    out_rows: List[dict] = []
    for row in sorted(summary_rows, key=lambda r: (str(r["site"]), int(r["set_id"]))):
        key = (str(row["site"]), int(row["set_id"]))
        diag_row = diag_map.get(key, {})
        out_rows.append(
            {
                "site": row["site"],
                "set_id": row["set_id"],
                "p32_label": row["p32_label"],
                "kr_used": row["kr_used"],
                "kr_ci_low": row["kr_ci_low"],
                "kr_ci_high": row["kr_ci_high"],
                "observed_P21": row["observed_P21"],
                "calibration_factor_C": row["calibration_factor_C"],
                "calibration_factor_std": row.get("calibration_factor_std", ""),
                "calibration_factor_ci_low": row.get("calibration_factor_ci_low", ""),
                "calibration_factor_ci_high": row.get("calibration_factor_ci_high", ""),
                "P32_hat": row["P32_hat"],
                "P32_ci_low": row["P32_ci_low"],
                "P32_ci_high": row["P32_ci_high"],
                "P32_reference": row["P32_reference"],
                "P32_relative_error_percent": row["P32_relative_error_percent"],
                "C_ratio": diag_row.get("C_ratio", ""),
                "p32_calibration_status": diag_row.get("p32_calibration_status", ""),
                "kr_adoption_status": row["kr_adoption_status"],
                "p32_status": resolve_p32_status(row, diag_row),
                "notes": merge_notes(row, diag_row),
            }
        )

    os.makedirs(os.path.dirname(args.outcsv) or ".", exist_ok=True)
    write_csv(out_rows, args.outcsv)
    print(f"[*] Full unit-P32 summary written to: {args.outcsv}")


if __name__ == "__main__":
    main()
