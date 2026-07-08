import argparse
import csv
import math
import os
import sys
from typing import Dict, List, Sequence, Tuple

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dfn_analysis.export_setwise_3d_traces import load_hdf5_dfn
from dfn_analysis.summarize_setwise_trace_statistics import (
    build_summary_rows,
    compute_total_observation_area,
    load_rough_face_collection_from_h5,
    load_trace_rows_from_h5,
)


SITE_SET_CONFIG = {
    "forsmark": {
        1: {"p32_base": 0.602, "dist_type": "powerlaw", "r0": 0.28},
        2: {"p32_base": 2.069, "dist_type": "powerlaw", "r0": 0.25},
        3: {"p32_base": 0.448, "dist_type": "powerlaw", "r0": 0.14},
        4: {"p32_base": 0.226, "dist_type": "powerlaw", "r0": 0.15},
        5: {"p32_base": 0.605, "dist_type": "powerlaw", "r0": 0.25},
    },
    "laxemar": {
        1: {"p32_base": 1.310, "dist_type": "powerlaw", "r0": 0.328},
        2: {"p32_base": 1.026, "dist_type": "powerlaw", "r0": 0.977},
        3: {"p32_base": 0.975, "dist_type": "powerlaw", "r0": 0.858},
        4: {"p32_base": 2.320, "dist_type": "exponential", "r0": 4.0},
        5: {"p32_base": 1.400, "dist_type": "powerlaw", "r0": 0.400},
    },
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


def support_scaled_p32(
    site: str,
    set_id: int,
    kr: float,
    support_rmin: float,
    rmax: float,
) -> float:
    cfg = SITE_SET_CONFIG.get(site, {}).get(set_id)
    if cfg is None or not math.isfinite(kr) or not math.isfinite(support_rmin) or not math.isfinite(rmax):
        return float("nan")
    p32_base = float(cfg["p32_base"])
    dist_type = str(cfg["dist_type"])
    r0 = float(cfg["r0"])
    # 크기 참조가 없는 config(자기완결 추정: p32_base/r0 미제공)면 검증 reference는 NaN
    if not math.isfinite(p32_base) or not math.isfinite(r0) or r0 <= 0.0:
        return float("nan")

    if dist_type == "powerlaw":
        pow_val = 2.0 - kr
        if abs(pow_val) < 1e-12:
            int_r0 = math.log(rmax) - math.log(r0)
            int_rmin = math.log(rmax) - math.log(support_rmin)
        else:
            int_r0 = (rmax**pow_val - r0**pow_val) / pow_val
            int_rmin = (rmax**pow_val - support_rmin**pow_val) / pow_val
        if abs(int_r0) < 1e-12:
            return float("nan")
        return p32_base * (int_rmin / int_r0)

    if dist_type == "exponential":
        lam = 1.0 / r0

        def integral_fn(radius: float) -> float:
            return -math.exp(-lam * radius) * (radius**2 + 2.0 * radius / lam + 2.0 / (lam**2))

        int_r0 = integral_fn(rmax) - integral_fn(0.0)
        int_rmin = integral_fn(rmax) - integral_fn(support_rmin)
        if abs(int_r0) < 1e-12:
            return float("nan")
        return p32_base * (int_rmin / int_r0)

    return float("nan")


def load_p21_summary(trace_h5: str, rough_mesh_h5: str) -> Dict[int, dict]:
    rows = load_trace_rows_from_h5(trace_h5)
    rough_faces = load_rough_face_collection_from_h5(rough_mesh_h5)
    observation_area_m2 = compute_total_observation_area(rough_faces)
    summary_rows = build_summary_rows(rows, observation_area_m2)
    return {int(row["set_id"]): row for row in summary_rows}


def load_orientation_factors(dfn_h5: str) -> Dict[int, float]:
    data = load_hdf5_dfn(dfn_h5)
    factors: Dict[int, float] = {}
    for set_id in sorted({int(v) for v in np.asarray(data["set_ids"]).tolist()}):
        set_mask = np.asarray(data["set_ids"]) == set_id
        set_normals = np.asarray(data["normals"])[set_mask]
        if set_normals.size == 0:
            factors[set_id] = float("nan")
            continue
        g_factors = np.sqrt(np.clip(1.0 - set_normals[:, 0] ** 2, 0.0, 1.0))
        factors[set_id] = float(np.mean(g_factors))
    return factors


def classify_p32_status(site: str, set_id: int, adoption_status: str, recovery_ci_status: str) -> str:
    if adoption_status == "rejected":
        return "p32_hold"
    if site == "forsmark" and set_id == 2:
        return "p32_provisional_systematic_bias"
    if site == "forsmark" and set_id in {1, 5}:
        return "p32_pilot_candidate_with_bootstrap_uncertainty"
    if adoption_status == "provisional_accepted":
        return "p32_provisional"
    if recovery_ci_status == "systematic_bias":
        return "p32_provisional_systematic_bias"
    return "p32_pilot_candidate"


def build_notes(recovery_notes: str, orientation_proxy: float, p32_hat: float) -> str:
    notes = [recovery_notes] if recovery_notes else []
    if math.isfinite(orientation_proxy):
        notes.append(f"orientation_corrected_p21_proxy={orientation_proxy:.6f}")
    if math.isfinite(p32_hat):
        notes.append("p32_method=support_scaled_generator_proxy")
    return "; ".join(note for note in notes if note)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build P32 pilot summary for accepted/provisional effective-rmin sets.")
    parser.add_argument(
        "--final-summary-csv",
        default="storage/output/final_kr_recovery_summary_effective_rmin.csv",
        help="Final effective-rmin kr recovery summary CSV.",
    )
    parser.add_argument(
        "--bootstrap-summary-csv",
        default="storage/output/final_kr_bootstrap_effective_rmin/final_kr_bootstrap_summary_effective_rmin.csv",
        help="Bootstrap CI summary CSV.",
    )
    parser.add_argument(
        "--p32-label",
        default="P32_r_ge_0p5m",
        help="P32 label to report in the output summary.",
    )
    parser.add_argument(
        "--rough-mesh-h5",
        default="storage/output/rough_face_mesh_collection/synthetic_rough_face_collection.h5",
        help="Rough face mesh collection used to compute observed P21.",
    )
    parser.add_argument(
        "--forsmark-dfn-h5",
        default="storage/output/dfn_forsmark_rmin0p5/dfn_export_for_python.h5",
        help="Forsmark DFN export HDF5.",
    )
    parser.add_argument(
        "--forsmark-trace-h5",
        default="storage/output/forsmark_rmin0p5_trace_dataset_collection/trace_dataset_3d.h5",
        help="Forsmark trace dataset HDF5.",
    )
    parser.add_argument(
        "--laxemar-dfn-h5",
        default="storage/output/dfn_laxemar_rmin0p5/dfn_export_for_python.h5",
        help="Laxemar DFN export HDF5.",
    )
    parser.add_argument(
        "--laxemar-trace-h5",
        default="storage/output/laxemar_rmin0p5_trace_dataset_collection/trace_dataset_3d.h5",
        help="Laxemar trace dataset HDF5.",
    )
    parser.add_argument(
        "--accepted-only",
        action="store_true",
        default=False,
        help="If set, exclude provisional_accepted rows and keep only accepted sets.",
    )
    parser.add_argument(
        "--outcsv",
        default="storage/output/p32_pilot_effective_rmin/p32_pilot_summary_effective_rmin.csv",
        help="Output CSV path.",
    )
    args = parser.parse_args()

    final_rows = read_csv(args.final_summary_csv)
    bootstrap_rows = read_csv(args.bootstrap_summary_csv)
    bootstrap_map: Dict[Tuple[str, int], dict] = {
        (str(row["site"]), int(row["set_id"])): row for row in bootstrap_rows
    }

    site_context = {
        "forsmark": {
            "p21_map": load_p21_summary(args.forsmark_trace_h5, args.rough_mesh_h5),
            "orientation_map": load_orientation_factors(args.forsmark_dfn_h5),
        },
        "laxemar": {
            "p21_map": load_p21_summary(args.laxemar_trace_h5, args.rough_mesh_h5),
            "orientation_map": load_orientation_factors(args.laxemar_dfn_h5),
        },
    }

    out_rows: List[dict] = []
    for row in final_rows:
        site = str(row["site"])
        set_id = int(row["set_id"])
        adoption_status = str(row.get("adoption_status", ""))
        if adoption_status == "rejected":
            continue
        if adoption_status == "provisional_accepted" and args.accepted_only:
            continue

        boot_row = bootstrap_map.get((site, set_id), {})
        p21_row = site_context[site]["p21_map"].get(set_id, {})
        orientation_factor = site_context[site]["orientation_map"].get(set_id, float("nan"))
        observed_p21 = to_float(p21_row, "P21_observed")
        orientation_proxy = (
            observed_p21 / orientation_factor
            if math.isfinite(observed_p21) and math.isfinite(orientation_factor) and orientation_factor > 0.0
            else float("nan")
        )

        set_likelihood_rmin = to_float(row, "set_likelihood_rmin")
        kr_used = to_float(row, "kr_hat")
        kr_ci_low = to_float(boot_row, "kr_ci_low")
        kr_ci_high = to_float(boot_row, "kr_ci_high")
        p32_hat = support_scaled_p32(
            site=site,
            set_id=set_id,
            kr=kr_used,
            support_rmin=set_likelihood_rmin,
            rmax=250.0,
        )
        p32_at_kr_ci_low = support_scaled_p32(
            site=site,
            set_id=set_id,
            kr=kr_ci_low,
            support_rmin=set_likelihood_rmin,
            rmax=250.0,
        )
        p32_at_kr_ci_high = support_scaled_p32(
            site=site,
            set_id=set_id,
            kr=kr_ci_high,
            support_rmin=set_likelihood_rmin,
            rmax=250.0,
        )
        p32_ci_candidates = [value for value in (p32_at_kr_ci_low, p32_at_kr_ci_high) if math.isfinite(value)]
        if p32_ci_candidates:
            p32_ci_low = min(p32_ci_candidates)
            p32_ci_high = max(p32_ci_candidates)
        else:
            p32_ci_low = float("nan")
            p32_ci_high = float("nan")

        recovery_ci_status = str(boot_row.get("recovery_ci_status", ""))
        p32_status = classify_p32_status(site, set_id, adoption_status, recovery_ci_status)

        out_rows.append(
            {
                "site": site,
                "set_id": set_id,
                "p32_label": args.p32_label,
                "set_effective_generation_rmin": to_float(row, "set_effective_generation_rmin"),
                "set_likelihood_rmin": set_likelihood_rmin,
                "kr_used": kr_used,
                "kr_ci_low": kr_ci_low,
                "kr_ci_high": kr_ci_high,
                "kr_adoption_status": adoption_status,
                "observed_P21": observed_p21,
                "orientation_factor": orientation_factor,
                "P32_hat": p32_hat,
                "P32_ci_low": p32_ci_low,
                "P32_ci_high": p32_ci_high,
                "p32_pilot_status": p32_status,
                "notes": build_notes(str(row.get("notes", "")), orientation_proxy, p32_hat),
            }
        )

    os.makedirs(os.path.dirname(args.outcsv), exist_ok=True)
    write_csv(out_rows, args.outcsv)
    print(f"[*] P32 pilot summary written to: {args.outcsv}")


if __name__ == "__main__":
    main()
