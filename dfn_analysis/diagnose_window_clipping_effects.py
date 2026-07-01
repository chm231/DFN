import argparse
import csv
import json
import os
from typing import Dict, List, Optional, Sequence

import h5py
import numpy as np


def _decode(value) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def load_trace_rows_from_csv(csv_path: str) -> List[dict]:
    rows: List[dict] = []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {
            "set_id",
            "face_id",
            "observed_length_m",
            "censoring_class",
            "p0_x",
            "p0_y",
            "p0_z",
            "p1_x",
            "p1_y",
            "p1_z",
        }
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing required CSV fields: {sorted(missing)}")
        for row in reader:
            rows.append(
                {
                    "set_id": int(row["set_id"]),
                    "face_id": int(row["face_id"]),
                    "face_x_m": float(row["face_x_m"]) if row.get("face_x_m") not in (None, "") else float("nan"),
                    "observed_length_m": float(row["observed_length_m"]),
                    "censoring_class": int(row["censoring_class"]),
                    "p0_x": float(row["p0_x"]),
                    "p0_y": float(row["p0_y"]),
                    "p0_z": float(row["p0_z"]),
                    "p1_x": float(row["p1_x"]),
                    "p1_y": float(row["p1_y"]),
                    "p1_z": float(row["p1_z"]),
                    "endpoint0_type": row.get("endpoint0_type") or row.get("p0_endpoint_type") or "",
                    "endpoint1_type": row.get("endpoint1_type") or row.get("p1_endpoint_type") or "",
                    "trace_normal_valid": int(row["trace_normal_valid"]) if row.get("trace_normal_valid") not in (None, "") else -1,
                }
            )
    return rows


def load_trace_rows_from_h5(h5_path: str) -> List[dict]:
    rows: List[dict] = []
    with h5py.File(h5_path, "r") as f:
        if "traces" not in f:
            raise ValueError(f"Could not find /traces in: {h5_path}")
        grp = f["traces"]
        n_rows = len(grp["set_id"])
        p0 = grp["p0_xyz"][:].astype(np.float64)
        p1 = grp["p1_xyz"][:].astype(np.float64)
        p0_type = grp["p0_endpoint_type"][:] if "p0_endpoint_type" in grp else [b""] * n_rows
        p1_type = grp["p1_endpoint_type"][:] if "p1_endpoint_type" in grp else [b""] * n_rows
        trace_normal_valid = grp["trace_normal_valid"][:] if "trace_normal_valid" in grp else np.full(n_rows, -1)
        for idx in range(n_rows):
            rows.append(
                {
                    "set_id": int(grp["set_id"][idx]),
                    "face_id": int(grp["face_id"][idx]),
                    "face_x_m": float(grp["face_x_m"][idx]) if "face_x_m" in grp else float("nan"),
                    "observed_length_m": float(grp["observed_length_m"][idx]),
                    "censoring_class": int(grp["censoring_class"][idx]),
                    "p0_x": float(p0[idx, 0]),
                    "p0_y": float(p0[idx, 1]),
                    "p0_z": float(p0[idx, 2]),
                    "p1_x": float(p1[idx, 0]),
                    "p1_y": float(p1[idx, 1]),
                    "p1_z": float(p1[idx, 2]),
                    "endpoint0_type": _decode(p0_type[idx]),
                    "endpoint1_type": _decode(p1_type[idx]),
                    "trace_normal_valid": int(trace_normal_valid[idx]),
                }
            )
    return rows


def _safe_mean(values: Sequence[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def _safe_median(values: Sequence[float]) -> float:
    return float(np.median(values)) if values else float("nan")


def _safe_p90(values: Sequence[float]) -> float:
    return float(np.percentile(values, 90)) if values else float("nan")


def _ratio(numer: float, denom: float) -> float:
    return float(numer / denom) if np.isfinite(numer) and np.isfinite(denom) and denom != 0.0 else float("nan")


def load_v3_best_rows(paths: Sequence[str]) -> Dict[int, dict]:
    rows_by_set: Dict[int, List[dict]] = {}
    for path in paths:
        with open(path, "r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                set_id = int(row["set_id"])
                rows_by_set.setdefault(set_id, []).append(row)

    status_priority = {
        "ok": 0,
        "provisional_ok": 1,
        "high_censoring": 2,
        "weak_identifiability": 3,
        "posterior_predictive_failed": 4,
        "low_tail_sample": 5,
        "boundary_solution": 6,
        "high_two_end_censoring": 7,
        "window_aware_required": 8,
    }
    best_rows: Dict[int, dict] = {}
    for set_id, rows in rows_by_set.items():
        def sort_key(row: dict) -> tuple:
            status = row.get("final_status") or row.get("radius_model_status") or ""
            rejected = str(row.get("model_rejected", "")).lower() == "true"
            n_tail = int(float(row.get("n_used_tail", 0) or 0))
            return (1 if rejected else 0, status_priority.get(status, 99), -n_tail)

        best = sorted(rows, key=sort_key)[0]
        best_rows[set_id] = {
            "best_lmin_fit": float(best.get("lmin_fit", "nan")),
            "kr_radius_candidate": float(best.get("kr_radius_hat", "nan")),
            "final_status": best.get("final_status") or best.get("radius_model_status", ""),
            "q90_ratio_model_observed": float(best.get("q90_ratio_model_observed", "nan")),
            "q95_ratio_model_observed": float(best.get("q95_ratio_model_observed", "nan")),
            "rejection_reason": best.get("rejection_reason", ""),
        }
    return best_rows


def classify_failure(
    set_id: int,
    censoring_ratio_total: float,
    two_end_ratio: float,
    censored_to_uncensored_mean_ratio: float,
    n_total: int,
    v3_status: str,
) -> str:
    if censoring_ratio_total >= 0.8 or two_end_ratio >= 0.2:
        return "window_clipping_dominated"
    if np.isfinite(censored_to_uncensored_mean_ratio) and censored_to_uncensored_mean_ratio >= 1.5:
        return "long_traces_censored"
    if v3_status == "posterior_predictive_failed" and censoring_ratio_total >= 0.5:
        return "observation_model_mismatch"
    if n_total < 50:
        return "low_sample"
    return "mild_clipping"


def priority_for_set(set_id: int) -> str:
    if set_id == 4:
        return "required"
    if set_id in (1, 2):
        return "high"
    if set_id == 5:
        return "low_sample_first"
    if set_id == 3:
        return "low / reference"
    return "review"


def v3_failure_and_next_model(set_id: int, summary: dict, v3_row: Optional[dict]) -> tuple[str, str]:
    if set_id == 4:
        return "high_two_end_censoring", "window_aware_likelihood_required"
    if set_id in (1, 2) and v3_row and v3_row.get("final_status") == "posterior_predictive_failed":
        if summary["censoring_ratio_total"] >= 0.5:
            return "posterior_predictive_failed_with_high_censoring", "window_aware_likelihood"
        return "posterior_predictive_failed", "window_or_orientation_diagnostic"
    if set_id == 5:
        return "low_sample_or_boundary_solution", "bootstrap_or_pooling"
    if set_id == 3:
        return "reference_set_accepted", "use_as_reference_before_window_aware_model"
    if not v3_row:
        return "v3_result_missing", "run_radius_powerlaw_v3"
    return v3_row.get("final_status", ""), "review"


def build_set_summary(rows: Sequence[dict], v3_best: Dict[int, dict]) -> List[dict]:
    summaries: List[dict] = []
    for set_id in sorted({int(row["set_id"]) for row in rows}):
        set_rows = [row for row in rows if int(row["set_id"]) == set_id]
        uncensored = [float(row["observed_length_m"]) for row in set_rows if int(row["censoring_class"]) == 0]
        censored = [float(row["observed_length_m"]) for row in set_rows if int(row["censoring_class"]) > 0]
        n_total = len(set_rows)
        n_uncensored = len(uncensored)
        n_one_end = int(sum(int(row["censoring_class"]) == 1 for row in set_rows))
        n_two_end = int(sum(int(row["censoring_class"]) == 2 for row in set_rows))
        n_censored = n_one_end + n_two_end
        censoring_ratio = n_censored / n_total if n_total else float("nan")
        two_end_ratio = n_two_end / n_total if n_total else float("nan")
        mean_unc = _safe_mean(uncensored)
        mean_cen = _safe_mean(censored)
        p90_unc = _safe_p90(uncensored)
        p90_cen = _safe_p90(censored)
        v3_row = v3_best.get(set_id)
        v3_status = v3_row["final_status"] if v3_row else ""
        dominant = classify_failure(set_id, censoring_ratio, two_end_ratio, _ratio(mean_cen, mean_unc), n_total, v3_status)
        v3_failed_because, recommended_next_model = v3_failure_and_next_model(
            set_id,
            {"censoring_ratio_total": censoring_ratio},
            v3_row,
        )
        summaries.append(
            {
                "set_id": set_id,
                "n_total": n_total,
                "n_uncensored": n_uncensored,
                "n_one_end_censored": n_one_end,
                "n_two_end_censored": n_two_end,
                "censoring_ratio_total": censoring_ratio,
                "two_end_censoring_ratio": two_end_ratio,
                "mean_length_uncensored": mean_unc,
                "mean_length_censored": mean_cen,
                "median_length_uncensored": _safe_median(uncensored),
                "median_length_censored": _safe_median(censored),
                "p90_length_uncensored": p90_unc,
                "p90_length_censored": p90_cen,
                "censored_to_uncensored_mean_ratio": _ratio(mean_cen, mean_unc),
                "censored_to_uncensored_p90_ratio": _ratio(p90_cen, p90_unc),
                "n_faces_observed": len({int(row["face_id"]) for row in set_rows}),
                "dominant_failure_mode": dominant,
                "window_aware_priority": priority_for_set(set_id),
                "notes": "trace clipping diagnostic only; no new kr/P32 estimate",
                "best_lmin_fit": v3_row["best_lmin_fit"] if v3_row else float("nan"),
                "kr_radius_candidate": v3_row["kr_radius_candidate"] if v3_row else float("nan"),
                "final_status": v3_status,
                "q90_ratio_model_observed": v3_row["q90_ratio_model_observed"] if v3_row else float("nan"),
                "q95_ratio_model_observed": v3_row["q95_ratio_model_observed"] if v3_row else float("nan"),
                "rejection_reason": v3_row["rejection_reason"] if v3_row else "",
                "v3_failed_because": v3_failed_because,
                "recommended_next_model": recommended_next_model,
            }
        )
    return summaries


def build_face_summary(rows: Sequence[dict]) -> List[dict]:
    face_rows: List[dict] = []
    keys = sorted({(int(row["set_id"]), int(row["face_id"])) for row in rows})
    for set_id, face_id in keys:
        subset = [row for row in rows if int(row["set_id"]) == set_id and int(row["face_id"]) == face_id]
        lengths = [float(row["observed_length_m"]) for row in subset]
        n_traces = len(subset)
        n_censored = int(sum(int(row["censoring_class"]) > 0 for row in subset))
        face_rows.append(
            {
                "set_id": set_id,
                "face_id": face_id,
                "n_traces": n_traces,
                "n_censored": n_censored,
                "censoring_ratio": n_censored / n_traces if n_traces else float("nan"),
                "total_trace_length_m": float(sum(lengths)),
                "mean_trace_length_m": _safe_mean(lengths),
                "p90_trace_length_m": _safe_p90(lengths),
            }
        )
    return face_rows


def write_csv(rows: Sequence[dict], path: str) -> None:
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose finite window/clipping effects in trace observations.")
    parser.add_argument("--trace-csv", help="Input trace CSV")
    parser.add_argument("--trace-h5", help="Input trace HDF5")
    parser.add_argument("--v3-fit-csv", action="append", default=[], help="One or more v3 fit CSV files")
    parser.add_argument("--outdir", default="storage/output/window_clipping_diagnostics")
    args = parser.parse_args()

    if bool(args.trace_csv) == bool(args.trace_h5):
        raise ValueError("Provide exactly one of --trace-csv or --trace-h5.")

    rows = load_trace_rows_from_csv(args.trace_csv) if args.trace_csv else load_trace_rows_from_h5(args.trace_h5)
    v3_best = load_v3_best_rows(args.v3_fit_csv) if args.v3_fit_csv else {}
    set_summary = build_set_summary(rows, v3_best)
    face_summary = build_face_summary(rows)

    os.makedirs(args.outdir, exist_ok=True)
    set_csv = os.path.join(args.outdir, "window_clipping_summary_by_set.csv")
    face_csv = os.path.join(args.outdir, "window_clipping_summary_by_face.csv")
    joined_csv = os.path.join(args.outdir, "window_clipping_v3_joined_diagnostics.csv")
    json_path = os.path.join(args.outdir, "window_clipping_diagnostics.json")
    write_csv(set_summary, set_csv)
    write_csv(face_summary, face_csv)
    write_csv(set_summary, joined_csv)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"set_summary": set_summary, "face_summary": face_summary}, f, indent=2)

    print("[*] Window clipping diagnostic summary")
    for row in set_summary:
        print(
            f"    - Set {row['set_id']}: failure_mode={row['dominant_failure_mode']}, "
            f"priority={row['window_aware_priority']}, v3={row['final_status'] or 'missing'}, "
            f"next={row['recommended_next_model']}"
        )
    print(f"[*] Set CSV written to: {set_csv}")
    print(f"[*] Face CSV written to: {face_csv}")
    print(f"[*] Joined CSV written to: {joined_csv}")
    print(f"[*] JSON written to: {json_path}")


if __name__ == "__main__":
    main()
