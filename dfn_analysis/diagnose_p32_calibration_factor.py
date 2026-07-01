import argparse
import csv
import os
import sys
from typing import Dict, List, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dfn_analysis.build_p32_pilot_summary import SITE_SET_CONFIG, support_scaled_p32


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


def calibration_status(c_ratio: float, mode: str) -> str:
    if mode != "unit_p32_forward_mc":
        return "proxy_mode_not_final"
    if c_ratio == c_ratio:
        if 0.8 <= c_ratio <= 1.25:
            return "calibration_reasonable"
        if 0.67 <= c_ratio < 0.8 or 1.25 < c_ratio <= 1.5:
            return "calibration_marginal"
        return "calibration_failed"
    return "calibration_unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose whether P32 calibration factor behaves like a unit-P32 observed-P21 calibration.")
    parser.add_argument(
        "--p32-summary-csv",
        required=True,
        help="MC-calibrated P32 summary CSV from estimate_p32_mc_calibrated.py",
    )
    parser.add_argument(
        "--reference-p32-source",
        choices=["site_table"],
        default="site_table",
        help="Reference P32 source for synthetic benchmark comparison.",
    )
    parser.add_argument("--site", choices=["forsmark", "laxemar"], required=True)
    parser.add_argument("--target-set", nargs="+", type=int, required=True)
    parser.add_argument(
        "--outcsv",
        default="storage/output/p32_mc_calibrated_effective_rmin/p32_calibration_diagnostic.csv",
        help="Output diagnostic CSV path.",
    )
    args = parser.parse_args()

    rows = [
        row
        for row in read_csv(args.p32_summary_csv)
        if str(row["site"]) == args.site and int(row["set_id"]) in set(args.target_set)
    ]
    if not rows:
        raise ValueError("No matching rows found in the requested P32 summary CSV.")

    out_rows: List[dict] = []
    for row in rows:
        set_id = int(row["set_id"])
        p32_label = str(row.get("p32_label", ""))
        p32_reference = to_float(row, "P32_reference")
        if not (p32_reference == p32_reference):
            kr_used = to_float(row, "kr_used")
            p32_reference = support_scaled_p32(
                args.site,
                set_id,
                kr_used,
                to_float(row, "set_effective_generation_rmin"),
                250.0,
            )
        observed_p21 = to_float(row, "observed_P21")
        calibration_factor_c = to_float(row, "calibration_factor_C")
        c_empirical = observed_p21 / p32_reference if p32_reference == p32_reference and p32_reference > 0.0 else float("nan")
        c_ratio = calibration_factor_c / c_empirical if c_empirical == c_empirical and c_empirical > 0.0 else float("nan")
        mode = str(row.get("calibration_factor_mode", ""))
        notes = str(row.get("notes", ""))
        out_rows.append(
            {
                "site": args.site,
                "set_id": set_id,
                "calibration_factor_mode": mode,
                "p32_label": p32_label,
                "P32_reference": p32_reference,
                "observed_P21": observed_p21,
                "calibration_factor_C": calibration_factor_c,
                "C_empirical": c_empirical,
                "C_ratio": c_ratio,
                "P32_hat": to_float(row, "P32_hat"),
                "P32_abs_error": to_float(row, "P32_abs_error"),
                "P32_relative_error_percent": to_float(row, "P32_relative_error_percent"),
                "p32_calibration_status": calibration_status(c_ratio, mode),
                "notes": notes,
            }
        )

    os.makedirs(os.path.dirname(args.outcsv) or ".", exist_ok=True)
    write_csv(out_rows, args.outcsv)
    print(f"[*] P32 calibration diagnostic written to: {args.outcsv}")


if __name__ == "__main__":
    main()
