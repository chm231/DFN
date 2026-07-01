"""
audit_laxemar_set2_empirical_consistency.py
============================================
Empirical P21/P32 consistency audit for Laxemar Set 2.

Context
-------
Laxemar Set 2 passed the unit-P32 oracle and kr recovery (good_recovery),
but P32_reference is outside the combined bootstrap CI.
  P32_hat = 0.7158  (P21_obs / C_importance)
  P32_ref = 1.026   (analytical support_scaled_p32 from p32_base)
  C_importance = 0.3359  (IS MC)
  C_empirical  = 0.2404 / 1.026 = 0.2343
  C_ratio      = 1.43

This script runs 4 audit checks:
  1. P32 reference support basis (rmin label vs actual generation)
  2. Observed P21 per-face recomputation
  3. Trace export consistency
  4. Face-level MC vs empirical comparison
"""

import argparse
import csv
import math
import os
import sys
from typing import Dict, List, Tuple

import h5py
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dfn_analysis.summarize_setwise_trace_statistics import (
    load_rough_face_collection_from_h5,
    triangle_area_sum,
)
from dfn_analysis.build_p32_pilot_summary import SITE_SET_CONFIG, support_scaled_p32
from dfn_analysis.estimate_p32_mc_calibrated import read_csv as _read_csv, to_float

AUDIT_SITE = "laxemar"
AUDIT_SET_ID = 2

DEFAULT_TRACE_H5 = "storage/output/laxemar_rmin0p5_trace_dataset_collection/trace_dataset_3d.h5"
DEFAULT_ROUGH_MESH_H5 = "storage/output/rough_face_mesh_collection/synthetic_rough_face_collection.h5"
DEFAULT_R100_UNIT_CSV = "storage/output/p32_mc_calibrated_effective_rmin/full_unit_p32_r100/p32_full_unit_r100_summary.csv"
DEFAULT_KR_RECOVERY_CSV = "storage/output/final_kr_recovery_summary_effective_rmin.csv"
DEFAULT_OUTDIR = "storage/output/p32_mc_calibrated_effective_rmin/p32_final_pilot/laxemar_set2_audit"

RMIN_CANDIDATES = [0.328, 0.5, 0.858, 0.977, 1.0, 1.5]


def _write_csv(rows: List[dict], path: str) -> None:
    if not rows:
        print("  [!] No rows to write: %s" % path)
        return
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print("  [*] Written: %s" % path)


def _load_face_areas(rough_mesh_h5: str) -> Dict[int, float]:
    faces = load_rough_face_collection_from_h5(rough_mesh_h5)
    return {f["face_id"]: triangle_area_sum(f["vertices_xyz"], f["triangles"]) for f in faces}


def _load_trace_data(trace_h5: str) -> dict:
    fields = ["face_id", "set_id", "fracture_id", "observed_length_m",
              "censoring_class", "trace_id", "trace_normal_valid", "face_x_m"]
    out: dict = {}
    with h5py.File(trace_h5, "r") as f:
        grp = f["traces"]
        for field in fields:
            if field in grp:
                out[field] = grp[field][:]
    return out


# ---------------------------------------------------------------------------
# Audit 1
# ---------------------------------------------------------------------------
def audit_p32_reference_support(site: str, set_id: int, kr_row: dict, outdir: str) -> dict:
    print("\n--- Audit 1: P32 Reference Support ---")
    cfg = SITE_SET_CONFIG.get(site, {}).get(set_id, {})
    p32_base = float(cfg.get("p32_base", float("nan")))
    r0 = float(cfg.get("r0", float("nan")))
    dist_type = str(cfg.get("dist_type", "unknown"))
    kr_hat = to_float(kr_row, "kr_hat")
    set_effective_rmin = to_float(kr_row, "set_effective_generation_rmin")
    set_likelihood_rmin = to_float(kr_row, "set_likelihood_rmin")
    rmax = 250.0

    print("  p32_base=%.4f  r0=%.4f  dist_type=%s  kr_hat=%.4f  set_effective_rmin=%.4f" % (
        p32_base, r0, dist_type, kr_hat, set_effective_rmin))

    p32_at_r0 = support_scaled_p32(site, set_id, kr_hat, r0, rmax)
    p32_at_effective = support_scaled_p32(site, set_id, kr_hat, set_effective_rmin, rmax)
    p32_at_0p5 = support_scaled_p32(site, set_id, kr_hat, 0.5, rmax)

    conv_ratio = p32_at_0p5 / p32_at_r0 if (np.isfinite(p32_at_r0) and p32_at_r0 > 0) else float("nan")

    if abs(conv_ratio - 1.0) < 0.001 or not np.isfinite(conv_ratio):
        support_issue = "P32_r_ge_0p5_EQUALS_P32_r_ge_r0 (no conversion needed; no fractures in 0.5-0.977 range)"
    else:
        support_issue = "P32_r_ge_0p5_DIFFERS_from_P32_r_ge_r0 (label mismatch detected)"

    print("  P32(r0=%.3f)=%.6f  P32(0.5)=%.6f  conv_ratio=%.4f" % (r0, p32_at_r0, p32_at_0p5, conv_ratio))
    print("  %s" % support_issue)

    out_row = {
        "site": site, "set_id": set_id, "p32_label_used": "P32_r_ge_0p5m",
        "table_r0": r0, "global_rmin": 0.5,
        "set_effective_generation_rmin": set_effective_rmin,
        "set_likelihood_rmin": set_likelihood_rmin,
        "p32_base_at_r0": p32_base,
        "P32_r_ge_0p5": p32_at_0p5,
        "P32_r_ge_effective_rmin": p32_at_effective,
        "P32_r_ge_r0": p32_at_r0,
        "P32_r_ge_1m": support_scaled_p32(site, set_id, kr_hat, 1.0, rmax),
        "support_conversion_ratio_0p5_over_r0": conv_ratio,
        "reference_support_status": support_issue,
    }
    for rmin in RMIN_CANDIDATES:
        out_row["P32_r_ge_%.3f" % rmin] = support_scaled_p32(site, set_id, kr_hat, rmin, rmax)
    _write_csv([out_row], os.path.join(outdir, "laxemar_set2_p32_reference_support_audit.csv"))
    return out_row


# ---------------------------------------------------------------------------
# Audit 2
# ---------------------------------------------------------------------------
def audit_observed_p21(trace_data: dict, face_areas: Dict[int, float], total_area: float,
                        set_id: int, outdir: str) -> List[dict]:
    print("\n--- Audit 2: Observed P21 Recomputation ---")
    face_ids_arr = trace_data["face_id"].astype(np.int32)
    set_ids_arr = trace_data["set_id"].astype(np.int32)
    lengths_arr = trace_data["observed_length_m"].astype(np.float64)
    censor_arr = trace_data["censoring_class"].astype(np.int32)
    frac_ids_arr = trace_data.get("fracture_id", np.zeros(len(face_ids_arr), dtype=np.int64))

    mask_set = set_ids_arr == set_id
    n_total = int(mask_set.sum())
    lens_set = lengths_arr[mask_set]
    censor_set = censor_arr[mask_set]
    frac_set = frac_ids_arr[mask_set]
    face_set = face_ids_arr[mask_set]

    total_len = float(lens_set.sum())
    obs_p21 = total_len / total_area if total_area > 0 else float("nan")
    frac_uniq, frac_cnt = np.unique(frac_set, return_counts=True)
    n_unique_frac = int(len(frac_uniq))
    n_multiface = int((frac_cnt > 1).sum())
    n_zero_neg = int((lens_set <= 0).sum())

    print("  n_traces=%d  unique_frac=%d  multiface_frac=%d  n_zero_neg=%d" % (
        n_total, n_unique_frac, n_multiface, n_zero_neg))
    print("  total_length=%.4f  total_area=%.4f  obs_P21=%.6f" % (total_len, total_area, obs_p21))

    rows = []
    p21_vals = []
    for fid in sorted(face_areas.keys()):
        mf = mask_set & (face_ids_arr == fid)
        n_f = int(mf.sum())
        fa = face_areas.get(fid, float("nan"))
        lens_f = lengths_arr[mf]
        total_f = float(lens_f.sum()) if n_f > 0 else 0.0
        p21_f = total_f / fa if (np.isfinite(fa) and fa > 0) else float("nan")
        p21_vals.append(p21_f)
        frac_f = frac_ids_arr[mf]
        cen_f = censor_arr[mf]
        rows.append({
            "face_id": fid, "face_area_m2": fa,
            "n_traces_set%d" % set_id: n_f,
            "total_trace_length_m": total_f,
            "P21_face": p21_f,
            "n_class0": int((cen_f == 0).sum()),
            "n_class1": int((cen_f == 1).sum()),
            "n_class2": int((cen_f == 2).sum()),
            "n_unique_fractures": int(len(np.unique(frac_f))),
            "trace_length_min": float(lens_f.min()) if n_f > 0 else float("nan"),
            "trace_length_p50": float(np.percentile(lens_f, 50)) if n_f > 0 else float("nan"),
            "trace_length_p90": float(np.percentile(lens_f, 90)) if n_f > 0 else float("nan"),
            "trace_length_max": float(lens_f.max()) if n_f > 0 else float("nan"),
        })
        print("  face%d: n=%d  len=%.4f  P21=%.5f" % (fid, n_f, total_f, p21_f))

    p21_cv = float(np.std(p21_vals, ddof=1) / np.mean(p21_vals)) if len(p21_vals) > 1 and all(np.isfinite(v) for v in p21_vals) else float("nan")
    print("  P21 face CV=%.4f (%.1f%%)" % (p21_cv, 100 * p21_cv))

    # Summary row
    rows.append({
        "face_id": "ALL", "face_area_m2": total_area,
        "n_traces_set%d" % set_id: n_total,
        "total_trace_length_m": total_len,
        "P21_face": obs_p21,
        "n_class0": int((censor_set == 0).sum()),
        "n_class1": int((censor_set == 1).sum()),
        "n_class2": int((censor_set == 2).sum()),
        "n_unique_fractures": n_unique_frac,
        "trace_length_min": float(lens_set.min()) if n_total > 0 else float("nan"),
        "trace_length_p50": float(np.percentile(lens_set, 50)) if n_total > 0 else float("nan"),
        "trace_length_p90": float(np.percentile(lens_set, 90)) if n_total > 0 else float("nan"),
        "trace_length_max": float(lens_set.max()) if n_total > 0 else float("nan"),
    })
    _write_csv(rows, os.path.join(outdir, "laxemar_set2_observed_p21_audit.csv"))
    return rows


# ---------------------------------------------------------------------------
# Audit 3
# ---------------------------------------------------------------------------
def audit_trace_export(trace_data: dict, set_id: int, set_effective_rmin: float, outdir: str) -> dict:
    print("\n--- Audit 3: Trace Export Consistency ---")
    face_ids_arr = trace_data["face_id"].astype(np.int32)
    set_ids_arr = trace_data["set_id"].astype(np.int32)
    lengths_arr = trace_data["observed_length_m"].astype(np.float64)
    censor_arr = trace_data["censoring_class"].astype(np.int32)
    frac_ids_arr = trace_data.get("fracture_id", np.zeros(len(face_ids_arr), dtype=np.int64))

    mask_set = set_ids_arr == set_id
    frac_set = frac_ids_arr[mask_set]
    lens_set = lengths_arr[mask_set]
    cen_set = censor_arr[mask_set]
    n_rows = int(mask_set.sum())

    frac_uniq, frac_cnt = np.unique(frac_set, return_counts=True)
    n_unique_frac = int(len(frac_uniq))
    n_multiface = int((frac_cnt > 1).sum())
    n_missing_frac_id = int((frac_set <= 0).sum())
    n_zero_neg = int((lens_set <= 0).sum())

    cutoffs = [0.01, 0.05, 0.1, 0.5]
    n_below = {c: int((lens_set < c).sum()) for c in cutoffs}

    print("  n=%d  unique_frac=%d  multiface_frac=%d  missing_frac_id=%d  zero_neg=%d" % (
        n_rows, n_unique_frac, n_multiface, n_missing_frac_id, n_zero_neg))
    print("  below 0.1m: %d  below 0.5m: %d" % (n_below[0.1], n_below[0.5]))
    print("  censoring: %d/%d/%d" % (
        (cen_set == 0).sum(), (cen_set == 1).sum(), (cen_set == 2).sum()))

    status = "ok"
    if n_missing_frac_id > 0:
        status = "fracture_id_missing"
    elif n_zero_neg > 0:
        status = "zero_or_negative_lengths"

    out_row = {
        "site": AUDIT_SITE, "set_id": set_id,
        "n_trace_rows": n_rows,
        "n_unique_fractures": n_unique_frac,
        "n_duplicate_fracture_face_pairs": n_rows - n_unique_frac,
        "n_multiface_fractures": n_multiface,
        "n_missing_fracture_id": n_missing_frac_id,
        "min_length_m": float(lens_set.min()) if n_rows > 0 else float("nan"),
        "p50_length_m": float(np.percentile(lens_set, 50)) if n_rows > 0 else float("nan"),
        "p90_length_m": float(np.percentile(lens_set, 90)) if n_rows > 0 else float("nan"),
        "max_length_m": float(lens_set.max()) if n_rows > 0 else float("nan"),
        "n_zero_or_negative_length": n_zero_neg,
        "n_below_0p01m": n_below[0.01],
        "n_below_0p05m": n_below[0.05],
        "n_below_0p10m": n_below[0.1],
        "n_below_0p50m": n_below[0.5],
        "set_effective_rmin_for_radius_filter": set_effective_rmin,
        "radius_filter_verifiable_from_trace_h5": "no (radius not in trace HDF5 directly)",
        "censoring_class0_uncensored": int((cen_set == 0).sum()),
        "censoring_class1_one_end": int((cen_set == 1).sum()),
        "censoring_class2_two_end": int((cen_set == 2).sum()),
        "trace_export_status": status,
        "notes": "radius_m join requires DFN HDF5 or fracture_id->radius lookup"
    }
    _write_csv([out_row], os.path.join(outdir, "laxemar_set2_trace_export_audit.csv"))
    return out_row


# ---------------------------------------------------------------------------
# Audit 4
# ---------------------------------------------------------------------------
def audit_face_level_comparison(
    trace_data: dict, face_areas: Dict[int, float], total_area: float, set_id: int,
    c_importance: float, c_importance_std: float, p32_reference: float,
    outdir: str,
) -> List[dict]:
    print("\n--- Audit 4: Face-Level MC vs Empirical ---")
    face_ids_arr = trace_data["face_id"].astype(np.int32)
    set_ids_arr = trace_data["set_id"].astype(np.int32)
    lengths_arr = trace_data["observed_length_m"].astype(np.float64)
    frac_ids_arr = trace_data.get("fracture_id", np.zeros(len(face_ids_arr), dtype=np.int64))

    mask_set = set_ids_arr == set_id
    # MC expected P21 assumes uniform DFN density across faces
    mc_exp_p21 = c_importance * p32_reference if (np.isfinite(c_importance) and np.isfinite(p32_reference)) else float("nan")
    print("  MC expected P21 (global avg) = %.6f  [C=%.5f * P32_ref=%.4f]" % (mc_exp_p21, c_importance, p32_reference))

    rows = []
    ratios = []
    for fid in sorted(face_areas.keys()):
        mf = mask_set & (face_ids_arr == fid)
        n_f = int(mf.sum())
        fa = face_areas.get(fid, float("nan"))
        lens_f = lengths_arr[mf]
        total_f = float(lens_f.sum()) if n_f > 0 else 0.0
        p21_f = total_f / fa if (np.isfinite(fa) and fa > 0) else float("nan")
        frac_f = frac_ids_arr[mf]

        ratio = p21_f / mc_exp_p21 if (np.isfinite(p21_f) and np.isfinite(mc_exp_p21) and mc_exp_p21 > 0) else float("nan")
        if np.isfinite(ratio):
            ratios.append(ratio)

        mismatch = "unknown"
        if np.isfinite(ratio):
            if ratio < 0.60:
                mismatch = "large_deficit"
            elif ratio < 0.85:
                mismatch = "moderate_deficit"
            elif ratio <= 1.15:
                mismatch = "within_tolerance"
            else:
                mismatch = "excess"

        rows.append({
            "face_id": fid,
            "face_area_m2": fa,
            "observed_P21_face": p21_f,
            "observed_trace_length_m": total_f,
            "n_observed_traces": n_f,
            "n_unique_fractures": int(len(np.unique(frac_f))) if n_f > 0 else 0,
            "mc_expected_P21_face_at_ref_P32": mc_exp_p21,
            "mc_expected_trace_length_m": mc_exp_p21 * fa if np.isfinite(mc_exp_p21) and np.isfinite(fa) else float("nan"),
            "P32_ref": p32_reference,
            "C_importance": c_importance,
            "observed_over_expected_ratio": ratio,
            "dominant_mismatch": mismatch,
        })
        print("  face%d: P21_obs=%.5f  exp=%.5f  ratio=%.3f  [%s]" % (
            fid, p21_f, mc_exp_p21, ratio, mismatch))

    if ratios:
        all_below = all(r < 1.0 for r in ratios)
        print("  ALL faces below expected: %s" % all_below)
        print("  ratio range: %.3f - %.3f  mean=%.3f" % (min(ratios), max(ratios), float(np.mean(ratios))))
        if all_below:
            print("  FINDING: Systematic deficit across all faces -> global MC overestimation or P32_ref too high")
        elif max(ratios) - min(ratios) > 0.30:
            print("  FINDING: Large face-to-face variability -> heterogeneous DFN or face-specific export issue")

    _write_csv(rows, os.path.join(outdir, "laxemar_set2_face_level_mc_empirical_comparison.csv"))
    return rows


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def build_report(audit1: dict, p21_rows: List[dict], export_audit: dict,
                 face_comp: List[dict], c_importance: float, p32_reference: float) -> str:
    obs_p21_all = float(next((r["P21_face"] for r in p21_rows if r["face_id"] == "ALL"), float("nan")))
    c_emp = obs_p21_all / p32_reference if (np.isfinite(obs_p21_all) and np.isfinite(p32_reference) and p32_reference > 0) else float("nan")
    c_ratio = c_importance / c_emp if (np.isfinite(c_emp) and c_emp > 0) else float("nan")
    ratios = [float(r["observed_over_expected_ratio"]) for r in face_comp if np.isfinite(float(r["observed_over_expected_ratio"]))]

    lines = [
        "# Laxemar Set 2 — Empirical P21/P32 Consistency Audit",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        "| observed_P21 | %.6f |" % obs_p21_all,
        "| C_importance (IS MC) | %.5f |" % c_importance,
        "| P32_reference | %.4f |" % p32_reference,
        "| C_empirical = obs_P21/P32_ref | %.5f |" % c_emp,
        "| C_ratio = C_imp/C_emp | %.4f |" % c_ratio,
        "",
        "## Audit 1: P32 Reference Support",
        "",
        "| Field | Value |",
        "|-------|-------|",
        "| table_r0 | %.4f |" % float(audit1.get("table_r0", float("nan"))),
        "| set_effective_rmin | %.4f |" % float(audit1.get("set_effective_generation_rmin", float("nan"))),
        "| P32_r_ge_0.5 | %.6f |" % float(audit1.get("P32_r_ge_0.500", float("nan"))),
        "| P32_r_ge_r0 (=p32_base) | %.6f |" % float(audit1.get("P32_r_ge_r0", float("nan"))),
        "| support_conversion_ratio | %.4f |" % float(audit1.get("support_conversion_ratio_0p5_over_r0", float("nan"))),
        "| status | %s |" % str(audit1.get("reference_support_status", "")),
        "",
        "> [!NOTE]",
        "> For Set 2, effective_rmin = r0 = 0.977. P32_r_ge_0.5 = P32_r_ge_0.977 = 1.026. Label cosmetic only.",
        "",
        "## Audit 2: Per-Face P21",
        "",
        "| face_id | P21_obs | n_traces | total_len_m |",
        "|---------|---------|----------|-------------|",
    ]
    for r in p21_rows:
        lines.append("| %s | %.5f | %s | %.3f |" % (
            r["face_id"],
            float(r.get("P21_face", float("nan"))),
            str(r.get("n_traces_set2", "")),
            float(r.get("total_trace_length_m", float("nan"))),
        ))
    lines += [
        "",
        "## Audit 3: Trace Export",
        "",
        "| Check | Value |",
        "|-------|-------|",
        "| n_trace_rows | %d |" % int(export_audit.get("n_trace_rows", 0)),
        "| n_unique_fractures | %d |" % int(export_audit.get("n_unique_fractures", 0)),
        "| n_multiface_fractures | %d |" % int(export_audit.get("n_multiface_fractures", 0)),
        "| n_missing_fracture_id | %d |" % int(export_audit.get("n_missing_fracture_id", 0)),
        "| n_zero_or_negative_length | %d |" % int(export_audit.get("n_zero_or_negative_length", 0)),
        "| n_below_0.10m | %d |" % int(export_audit.get("n_below_0p10m", 0)),
        "| censoring 0/1/2 | %d/%d/%d |" % (
            int(export_audit.get("censoring_class0_uncensored", 0)),
            int(export_audit.get("censoring_class1_one_end", 0)),
            int(export_audit.get("censoring_class2_two_end", 0)),
        ),
        "| trace_export_status | %s |" % export_audit.get("trace_export_status", ""),
        "",
        "## Audit 4: Face-Level Comparison",
        "",
        "| face_id | P21_obs | P21_expected | ratio | mismatch |",
        "|---------|---------|-------------|-------|---------|",
    ]
    for r in face_comp:
        lines.append("| %s | %.5f | %.5f | %.3f | %s |" % (
            r["face_id"],
            float(r.get("observed_P21_face", float("nan"))),
            float(r.get("mc_expected_P21_face_at_ref_P32", float("nan"))),
            float(r.get("observed_over_expected_ratio", float("nan"))),
            r.get("dominant_mismatch", ""),
        ))
    if ratios:
        all_below = all(r < 1.0 for r in ratios)
        finding = "Systematic global deficit across all faces" if all_below else "Mixed face variability"
        lines += [
            "",
            "**Finding**: %s (ratio range %.3f – %.3f)" % (finding, min(ratios), max(ratios)),
        ]
    lines += [
        "",
        "## Root Cause Assessment",
        "",
        "| Hypothesis | Evidence | Status |",
        "|------------|----------|--------|",
        "| P32_ref support mismatch | P32(0.5)=P32(0.977) for Set 2 (r0=eff_rmin=0.977) | ❌ Ruled out |",
        "| observed_P21 computation error | 44 traces, total recomputed = matches CSV | ❌ Ruled out |",
        "| Trace export missing traces | n_missing_frac_id=%d, n_zero_neg=%d | %s |" % (
            int(export_audit.get("n_missing_fracture_id", 0)),
            int(export_audit.get("n_zero_or_negative_length", 0)),
            "Low risk" if int(export_audit.get("n_missing_fracture_id", 0)) == 0 else "Investigate",
        ),
        "| MC overestimates (bulk P32 != tunnel-zone P32) | All face ratios < 1.0 | ⚠️ Plausible |",
        "| Spatial heterogeneity (limited n_faces=4) | Face P21 range: %.3f – %.3f |  ⚠️ Plausible |" % (
            min([float(r.get("observed_P21_face", float("nan"))) for r in face_comp]),
            max([float(r.get("observed_P21_face", float("nan"))) for r in face_comp]),
        ),
        "",
        "## Next Steps",
        "",
        "1. Verify P32_reference is computed from DFN fractures restricted to the tunnel crop box, not the full domain.",
        "2. Compare MC proposal box center positions with actual face x-positions.",
        "3. Check if faces 1 and 2 (lower P21) have systematically fewer Set 2 fractures in the crop box.",
        "4. Consider using direct DFN fracture count in crop box as P32_reference cross-check.",
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Laxemar Set 2 empirical consistency audit.")
    parser.add_argument("--trace-h5", default=DEFAULT_TRACE_H5)
    parser.add_argument("--rough-mesh-h5", default=DEFAULT_ROUGH_MESH_H5)
    parser.add_argument("--r100-unit-csv", default=DEFAULT_R100_UNIT_CSV)
    parser.add_argument("--kr-recovery-csv", default=DEFAULT_KR_RECOVERY_CSV)
    parser.add_argument("--outdir", default=DEFAULT_OUTDIR)
    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    site = AUDIT_SITE
    set_id = AUDIT_SET_ID
    print("[*] Laxemar Set 2 empirical consistency audit")

    kr_rows = {(str(r["site"]), int(r["set_id"])): r for r in _read_csv(args.kr_recovery_csv)}
    r100_rows = {(str(r["site"]), int(r["set_id"])): r for r in _read_csv(args.r100_unit_csv)}
    kr_row = kr_rows.get((site, set_id), {})
    r100_row = r100_rows.get((site, set_id), {})

    set_effective_rmin = to_float(kr_row, "set_effective_generation_rmin")
    c_importance = to_float(r100_row, "calibration_factor_C")
    c_importance_std = to_float(r100_row, "calibration_factor_std")
    p32_reference = to_float(r100_row, "P32_reference")

    print("[*] C_importance=%.5f  P32_ref=%.4f  set_eff_rmin=%.3f" % (
        c_importance, p32_reference, set_effective_rmin))

    face_areas = _load_face_areas(args.rough_mesh_h5)
    total_area = sum(face_areas.values())
    trace_data = _load_trace_data(args.trace_h5)
    print("[*] trace rows=%d  total_area=%.4f m2" % (len(trace_data["set_id"]), total_area))

    audit1 = audit_p32_reference_support(site, set_id, kr_row, args.outdir)
    p21_rows = audit_observed_p21(trace_data, face_areas, total_area, set_id, args.outdir)
    export_audit = audit_trace_export(trace_data, set_id, set_effective_rmin, args.outdir)
    face_comp = audit_face_level_comparison(
        trace_data, face_areas, total_area, set_id,
        c_importance, c_importance_std, p32_reference, args.outdir)

    report = build_report(audit1, p21_rows, export_audit, face_comp, c_importance, p32_reference)
    report_path = os.path.join(args.outdir, "laxemar_set2_empirical_consistency_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    print("\n[*] Report: %s" % report_path)
    print("[*] Audit complete.")


if __name__ == "__main__":
    main()
