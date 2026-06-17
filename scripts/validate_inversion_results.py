"""
Quantitative validation: compare pipeline inversion results against REAL ground-truth DFN parameters.

Real ground truth (from user-provided table):
  Set  | Trend  | Plunge | Kappa  | k_r  | r0    | P32
  S1   | 338.1  | 4.5    | 13.06  | 2.85 | 0.328 | 1.310
  S2   | 100.4  | 0.2    | 19.62  | 3.04 | 0.977 | 1.026
  S3   | 212.9  | 0.9    | 10.46  | 3.01 | 0.858 | 0.975
  S4   |   3.3  | 62.1   | 10.13  | exp(mean=4) | — | 2.320
  S5   | 243.0  | 24.4   | 23.52  | 3.602| 0.400 | 1.400

  Note: S4 k_r uses exponential distribution with lambda=1/4 (mean=4).
        Power-law k_r = alpha - 1; GT k_r values here are the CCDF exponent directly.

Pipeline inversion results (from verify_dfnrec_pipeline.py):
  S1: Trend=179.35, Plunge=6.75,  kappa=24.01, k_r=0.500, P32=4.9268
  S2: Trend=1.12,   Plunge=5.65,  kappa=3.58,  k_r=0.500, P32=3.6805
  S3: Trend=179.92, Plunge=0.36,  kappa=15.22, k_r=0.500, P32=14.0984
  S4: Trend=179.39, Plunge=63.84, kappa=20.03, k_r=0.500, P32=45.1852
  S5: Trend=1.13,   Plunge=47.29, kappa=6.68,  k_r=0.641, P32=2.3498

Raw-normal Fisher MLE results (debug_orientation.py):
  S1: Trend=160.58, Plunge=6.28,  kappa=18.51
  S2: Trend=279.14, Plunge=0.91,  kappa=16.68
  S3: Trend=216.25, Plunge=0.96,  kappa=10.02
  S4: Trend=185.76, Plunge=64.31, kappa=11.88
  S5: Trend=66.41,  Plunge=29.12, kappa=13.40
"""

from __future__ import annotations

import os
import sys
import math
import numpy as np
import pandas as pd

_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_here)
sys.path.insert(0, _root)

# ---------------------------------------------------------------------------
# REAL Ground Truth (user-provided measurement table)
# k_r: power-law CCDF exponent (= alpha - 1)
# S4 uses exponential distribution; kr_is_exp=True, kr_val=mean radius
# ---------------------------------------------------------------------------
GT = {
    "S1": dict(trend=338.1, plunge=4.5,  kappa=13.06, kr=2.85,  r0=0.328, P32=1.310, kr_is_exp=False),
    "S2": dict(trend=100.4, plunge=0.2,  kappa=19.62, kr=3.04,  r0=0.977, P32=1.026, kr_is_exp=False),
    "S3": dict(trend=212.9, plunge=0.9,  kappa=10.46, kr=3.01,  r0=0.858, P32=0.975, kr_is_exp=False),
    "S4": dict(trend=3.3,   plunge=62.1, kappa=10.13, kr=4.0,   r0=None,  P32=2.320, kr_is_exp=True),
    "S5": dict(trend=243.0, plunge=24.4, kappa=23.52, kr=3.602, r0=0.400, P32=1.400, kr_is_exp=False),
}

# ---------------------------------------------------------------------------
# Method A: Full Pipeline results
# ---------------------------------------------------------------------------
METHOD_A = {
    "S1": dict(trend=179.35, plunge=6.75,  kappa=24.01, kr=0.500, P32=4.9268),
    "S2": dict(trend=1.12,   plunge=5.65,  kappa=3.58,  kr=0.500, P32=3.6805),
    "S3": dict(trend=179.92, plunge=0.36,  kappa=15.22, kr=0.500, P32=14.0984),
    "S4": dict(trend=179.39, plunge=63.84, kappa=20.03, kr=0.500, P32=45.1852),
    "S5": dict(trend=1.13,   plunge=47.29, kappa=6.68,  kr=0.641, P32=2.3498),
}

# ---------------------------------------------------------------------------
# Method B: Raw CSV-normal Fisher MLE
# ---------------------------------------------------------------------------
METHOD_B = {
    "S1": dict(trend=160.58, plunge=6.28,  kappa=18.51, kr=None, P32=None),
    "S2": dict(trend=279.14, plunge=0.91,  kappa=16.68, kr=None, P32=None),
    "S3": dict(trend=216.25, plunge=0.96,  kappa=10.02, kr=None, P32=None),
    "S4": dict(trend=185.76, plunge=64.31, kappa=11.88, kr=None, P32=None),
    "S5": dict(trend=66.41,  plunge=29.12, kappa=13.40, kr=None, P32=None),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def axial_angle_deg(t1, p1, t2, p2):
    """Axial (undirected) great-circle angle between two pole directions [0, 90] deg."""
    def to_unit(t, p):
        tr, pr = math.radians(t), math.radians(p)
        return np.array([
            math.cos(pr) * math.sin(tr),
            math.cos(pr) * math.cos(tr),
            math.sin(pr),
        ])
    v1 = to_unit(t1, p1)
    v2 = to_unit(t2, p2)
    dot = float(np.clip(np.dot(v1, v2), -1.0, 1.0))
    return math.degrees(math.acos(abs(dot)))


def pct_err(estimated, truth):
    if truth is None or estimated is None:
        return None
    if abs(truth) < 1e-9:
        return None
    return (estimated - truth) / abs(truth) * 100.0


def fmt_f(val, decimals=2):
    if val is None:
        return "N/A"
    return f"{val:.{decimals}f}"


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def compare_method(label, method_dict):
    rows = []
    print(f"\n{'='*72}")
    print(f"  Method: {label}")
    print(f"{'='*72}")
    header = f"  {'Set':<4}  {'Param':<18}  {'GT':>10}  {'Inverted':>10}  {'Err':>10}"
    print(header)
    print(f"  {'-'*68}")

    for sid in ["S1", "S2", "S3", "S4", "S5"]:
        gt  = GT[sid]
        est = method_dict[sid]

        ang_err   = axial_angle_deg(gt["trend"], gt["plunge"], est["trend"], est["plunge"])
        kappa_err = pct_err(est.get("kappa"), gt["kappa"])
        kr_err    = pct_err(est.get("kr"), gt["kr"]) if not gt.get("kr_is_exp") else None
        p32_err   = pct_err(est.get("P32"), gt["P32"])

        s4_note = " [EXP]" if gt.get("kr_is_exp") else ""
        print(f"\n  {sid}")
        print(f"  {'':4}  {'Trend (deg)':<18}  {fmt_f(gt['trend']):>10}  {fmt_f(est['trend']):>10}  {'---':>10}")
        print(f"  {'':4}  {'Plunge (deg)':<18}  {fmt_f(gt['plunge']):>10}  {fmt_f(est['plunge']):>10}  {'---':>10}")
        print(f"  {'':4}  {'Pole axial err':<18}  {'':>10}  {'':>10}  {fmt_f(ang_err)+'deg':>10}")
        print(f"  {'':4}  {'kappa':<18}  {fmt_f(gt['kappa']):>10}  {fmt_f(est.get('kappa')):>10}  "
              f"  {(fmt_f(kappa_err,1)+'%') if kappa_err is not None else 'N/A':>9}")
        kr_gt_str = (fmt_f(gt['kr']) + s4_note) if not gt.get('kr_is_exp') else (f"exp(mean={gt['kr']:.0f})")
        kr_est_str = fmt_f(est.get("kr"), 3) if est.get("kr") is not None else "N/A"
        print(f"  {'':4}  {'k_r'+s4_note:<18}  {kr_gt_str:>10}  {kr_est_str:>10}  "
              f"  {(fmt_f(kr_err,1)+'%') if kr_err is not None else 'N/A':>9}")
        r0_str = fmt_f(gt.get("r0")) if gt.get("r0") else "N/A"
        print(f"  {'':4}  {'r0 (r_min)':<18}  {r0_str:>10}  {'N/A':>10}  {'---':>10}")
        print(f"  {'':4}  {'P32 (m2/m3)':<18}  {fmt_f(gt['P32'],4):>10}  {fmt_f(est.get('P32'),4):>10}  "
              f"  {(fmt_f(p32_err,1)+'%') if p32_err is not None else 'N/A':>9}")

        rows.append({
            "method": label,
            "set_id": sid,
            "gt_trend": gt["trend"],
            "gt_plunge": gt["plunge"],
            "gt_kappa": gt["kappa"],
            "gt_kr": gt["kr"],
            "gt_r0": gt.get("r0"),
            "gt_P32": gt["P32"],
            "gt_kr_is_exp": gt.get("kr_is_exp", False),
            "inv_trend": est["trend"],
            "inv_plunge": est["plunge"],
            "inv_kappa": est.get("kappa"),
            "inv_kr": est.get("kr"),
            "inv_P32": est.get("P32"),
            "angular_err_deg": round(ang_err, 3),
            "kappa_pct_err": round(kappa_err, 2) if kappa_err is not None else None,
            "kr_pct_err": round(kr_err, 2) if kr_err is not None else None,
            "P32_pct_err": round(p32_err, 2) if p32_err is not None else None,
        })

    return rows


def print_summary(all_rows):
    print(f"\n{'='*72}")
    print("  SUMMARY vs REAL GROUND TRUTH")
    print(f"{'='*72}")
    df = pd.DataFrame(all_rows)
    for method, grp in df.groupby("method"):
        mean_ang = grp["angular_err_deg"].mean()
        max_ang  = grp["angular_err_deg"].max()
        mean_kappa = grp["kappa_pct_err"].dropna().abs().mean()
        mean_kr    = grp["kr_pct_err"].dropna().abs().mean()
        mean_p32   = grp["P32_pct_err"].dropna().abs().mean()
        print(f"\n  [{method}]")
        print(f"    Mean pole angular error : {mean_ang:.2f} deg  (max {max_ang:.2f} deg)")
        if not pd.isna(mean_kappa):
            print(f"    Mean |kappa| error       : {mean_kappa:.1f}%")
        if not pd.isna(mean_kr):
            print(f"    Mean |k_r| error         : {mean_kr:.1f}%")
        if not pd.isna(mean_p32):
            print(f"    Mean |P32| error         : {mean_p32:.1f}%")
    print()
    return df


def main():
    print("=" * 72)
    print("  DFN Inversion Validation vs REAL Ground Truth")
    print("=" * 72)
    print()
    print("  Real GT (user-provided table):")
    for sid, gt in GT.items():
        kr_str = f"exp(mean={gt['kr']:.0f})" if gt.get("kr_is_exp") else f"k_r={gt['kr']:.3f}"
        r0_str = f"r0={gt['r0']:.3f}" if gt.get("r0") else "r0=N/A"
        print(f"    {sid}: Trend={gt['trend']:6.1f} Plunge={gt['plunge']:5.1f}  "
              f"kappa={gt['kappa']:6.2f}  {kr_str}  {r0_str}  P32={gt['P32']:.4f}")

    all_rows = []
    all_rows += compare_method("A: Full Pipeline (run_pipeline)", METHOD_A)
    all_rows += compare_method("B: Raw-Normal Fisher MLE only", METHOD_B)

    df = print_summary(all_rows)

    out_csv = os.path.join(_root, "storage", "output", "validation_comparison_real_gt.csv")
    df.to_csv(out_csv, index=False)
    print(f"  [*] Results saved: {out_csv}")
    print("  [Done]")


if __name__ == "__main__":
    main()
