"""Step 4: compare the fresh trial-pipeline estimates against Laxemar ground truth.

Reads the pipeline outputs (orientation/kappa recomputed from the trace HDF5, kr from
kr_summary_by_set.csv, P32 from p32_summary.csv) and builds a per-set comparison table
vs the Laxemar generation parameters. GT is used for comparison only.

Orientation is compared convention-safely: the GT pole vector is rebuilt with the SAME
ENU formula generate_dfn.py used to create the fractures, and the error is the axial
angular separation between the estimated mean pole and the GT pole in model xyz.
"""
import argparse
import csv
import os
import sys

import numpy as np
import h5py

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dfn_analysis.estimate_fisher_kappa import estimate_fisher_k_axial

# Laxemar generation parameters (from dfn generator v1/python/generate_dfn.py).
# Set 4 is exponential -> excluded from power-law kr / P32 comparison.
LAXEMAR_GT = {
    1: {"trend": 338.1, "plunge": 4.5, "kappa": 13.06, "kr": 2.85},
    2: {"trend": 100.4, "plunge": 0.2, "kappa": 19.62, "kr": 3.04},
    3: {"trend": 212.9, "plunge": 0.9, "kappa": 10.46, "kr": 3.01},
    5: {"trend": 243.0, "plunge": 24.4, "kappa": 23.52, "kr": 3.60},
}
SETS = [1, 2, 3, 5]


def pole_vec_enu(trend_deg: float, plunge_deg: float) -> np.ndarray:
    """GT pole in model xyz, identical to generate_dfn.mean_pole_vector_from_trend_plunge
    (x=East, y=North, z=Up; plunge positive downward)."""
    tr = np.radians(trend_deg)
    pl = np.radians(plunge_deg)
    n = np.array([np.cos(pl) * np.sin(tr), np.cos(pl) * np.cos(tr), -np.sin(pl)])
    return n / np.linalg.norm(n)


def pole_to_trend_plunge_enu(n_xyz: np.ndarray) -> tuple:
    """Inverse of pole_vec_enu: model xyz pole -> lower-hemisphere trend/plunge (ENU)."""
    n = np.asarray(n_xyz, dtype=np.float64)
    n = n / np.linalg.norm(n)
    if n[2] > 0.0:  # lower hemisphere: downward pole has nz <= 0 (z = Up)
        n = -n
    trend = float(np.degrees(np.arctan2(n[0], n[1])) % 360.0)
    plunge = float(np.degrees(np.arcsin(np.clip(-n[2], -1.0, 1.0))))
    return trend, plunge


def axial_angle_deg(a_xyz: np.ndarray, b_xyz: np.ndarray) -> float:
    a = np.asarray(a_xyz, np.float64)
    b = np.asarray(b_xyz, np.float64)
    d = abs(float(np.dot(a / np.linalg.norm(a), b / np.linalg.norm(b))))
    return float(np.degrees(np.arccos(np.clip(d, 0.0, 1.0))))


def read_csv_by_set(path: str) -> dict:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return {int(r["set_id"]): r for r in csv.DictReader(f)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare trial pipeline estimates vs Laxemar GT.")
    parser.add_argument("--base", default="storage/output/pipeline_test_laxemar",
                        help="Pipeline output base directory.")
    args = parser.parse_args()

    trace_h5 = os.path.join(args.base, "trace_dataset", "trace_dataset_3d.h5")
    kr_csv = os.path.join(args.base, "kr", "kr_summary_by_set.csv")
    p32_csv = os.path.join(args.base, "p32", "p32_summary.csv")
    out_csv = os.path.join(args.base, "comparison_vs_laxemar.csv")

    with h5py.File(trace_h5, "r") as f:
        g = f["traces"]
        set_id = g["set_id"][:].ravel().astype(int)
        trace_normal_xyz = g["trace_normal_xyz"][:].astype(np.float64)
        trace_normal_valid = g["trace_normal_valid"][:].ravel().astype(int)

    kr_rows = read_csv_by_set(kr_csv)
    p32_rows = read_csv_by_set(p32_csv)

    rows = []
    for s in SETS:
        gt = LAXEMAR_GT[s]
        mask = (set_id == s) & (trace_normal_valid == 1)
        normals = trace_normal_xyz[mask]
        n_valid = int(len(normals))

        fs = estimate_fisher_k_axial(normals) if n_valid >= 2 else {"mean_normal": None, "kappa": float("nan")}
        mean_normal = fs.get("mean_normal")
        est_kappa = float(fs.get("kappa", float("nan")))

        gt_vec = pole_vec_enu(gt["trend"], gt["plunge"])
        if mean_normal is not None:
            # Poles are axial; align the estimated pole to the same end as GT so the
            # trend/plunge readout is directly comparable (avoids 180 deg antipodal flips).
            mn = np.asarray(mean_normal, dtype=np.float64)
            if float(np.dot(mn, gt_vec)) < 0.0:
                mn = -mn
            mn = mn / np.linalg.norm(mn)
            est_trend = float(np.degrees(np.arctan2(mn[0], mn[1])) % 360.0)
            est_plunge = float(np.degrees(np.arcsin(np.clip(-mn[2], -1.0, 1.0))))
            orient_err = axial_angle_deg(mn, gt_vec)
        else:
            est_trend = est_plunge = orient_err = float("nan")

        kr_hat = float(kr_rows[s]["kr_hat"])
        kr_true = gt["kr"]
        p32_hat = float(p32_rows[s]["P32_hat"])
        p32_ref = float(p32_rows[s]["P32_reference"])

        rows.append({
            "set_id": s,
            "n_traces": int(np.sum(set_id == s)),
            "n_valid_normals": n_valid,
            "est_trend": round(est_trend, 1),
            "gt_trend": gt["trend"],
            "est_plunge": round(est_plunge, 1),
            "gt_plunge": gt["plunge"],
            "orient_ang_err_deg": round(orient_err, 1),
            "est_kappa": round(est_kappa, 2),
            "gt_kappa": gt["kappa"],
            "kr_hat": round(kr_hat, 3),
            "kr_true": kr_true,
            "kr_rel_err_pct": round(100.0 * (kr_hat - kr_true) / kr_true, 1),
            "p32_hat": round(p32_hat, 3),
            "p32_ref": round(p32_ref, 3),
            "p32_rel_err_pct": round(100.0 * (p32_hat - p32_ref) / p32_ref, 1),
        })

    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    hdr = (f"{'set':>3} {'nTr':>4} {'nNrm':>5} | {'estTr':>6} {'gtTr':>6} {'estPl':>6} {'gtPl':>5} "
           f"{'angErr':>6} | {'estK':>6} {'gtK':>6} | {'kr^':>5} {'krGT':>5} {'kr%':>6} | "
           f"{'P32^':>6} {'P32ref':>6} {'P32%':>6}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['set_id']:>3} {r['n_traces']:>4} {r['n_valid_normals']:>5} | "
              f"{r['est_trend']:>6} {r['gt_trend']:>6} {r['est_plunge']:>6} {r['gt_plunge']:>5} "
              f"{r['orient_ang_err_deg']:>6} | {r['est_kappa']:>6} {r['gt_kappa']:>6} | "
              f"{r['kr_hat']:>5} {r['kr_true']:>5} {r['kr_rel_err_pct']:>6} | "
              f"{r['p32_hat']:>6} {r['p32_ref']:>6} {r['p32_rel_err_pct']:>6}")
    print(f"\n[*] Comparison table written to: {out_csv}")


if __name__ == "__main__":
    main()
