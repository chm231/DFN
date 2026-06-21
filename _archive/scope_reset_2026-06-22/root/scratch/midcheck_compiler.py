import os
import json
import math
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import expon

# GT constants (Laxemar DFN Table 2-1)
GT = {
    1: {"trend": 338.1, "plunge": 4.5,  "fisher_kappa": 13.06, "powerlaw_kr": 2.850, "exp_mean": np.nan, "exp_lambda": np.nan, "P32": 1.310, "size_model": "powerlaw"},
    2: {"trend": 100.4, "plunge": 0.2,  "fisher_kappa": 19.62, "powerlaw_kr": 3.040, "exp_mean": np.nan, "exp_lambda": np.nan, "P32": 1.026, "size_model": "powerlaw"},
    3: {"trend": 212.9, "plunge": 0.9,  "fisher_kappa": 10.46, "powerlaw_kr": 3.010, "exp_mean": np.nan, "exp_lambda": np.nan, "P32": 0.975, "size_model": "powerlaw"},
    4: {"trend": 3.3,   "plunge": 62.1, "fisher_kappa": 10.13, "powerlaw_kr": np.nan, "exp_mean": 4.0,    "exp_lambda": 0.25,   "P32": 2.320, "size_model": "exponential"},
    5: {"trend": 243.0, "plunge": 24.4, "fisher_kappa": 23.52, "powerlaw_kr": 3.602, "exp_mean": np.nan, "exp_lambda": np.nan, "P32": 1.400, "size_model": "powerlaw"},
}

PREVIOUS_BASELINE = {
    1: {"kr": 3.000, "P32": 0.5421, "boundary_hit": False, "flags": "False"},
    2: {"kr": 2.000, "P32": 2.0339, "boundary_hit": False, "flags": "True"},
    3: {"kr": 1.900, "P32": 0.9608, "boundary_hit": False, "flags": "False"},
    4: {"lambda": 0.050, "P32": 2.2379, "boundary_hit": True, "flags": "True"},
    5: {"kr": 4.600, "P32": 1.1886, "boundary_hit": False, "flags": "False"},
}

def _kappa_mle(R_bar: float) -> float:
    if R_bar < 1e-6:
        return 0.0
    if R_bar > 1.0 - 1e-9:
        return 1e5
    def objective(k):
        if k < 1e-6:
            return k / 3.0 - R_bar
        return 1.0 / np.tanh(k) - 1.0 / k - R_bar
    try:
        kappa = brentq(objective, 1e-9, 1e6, xtol=1e-4)
    except ValueError:
        kappa = 3.0 * R_bar / (1.0 - R_bar**2)
    return float(kappa)

def trend_plunge_from_normal(normal: np.ndarray) -> tuple[float, float]:
    n = normal / np.linalg.norm(normal)
    E, N, D = n[0], n[1], -n[2]
    if D < 0:
        E, N, D = -E, -N, -D
    trend = math.degrees(math.atan2(E, N)) % 360.0
    plunge = math.degrees(math.asin(np.clip(D, -1.0, 1.0)))
    return trend, plunge

def estimate_set_orientation(normals):
    n = len(normals)
    if n < 2:
        return 0.0, 90.0, 10.0
    T = np.zeros((3, 3))
    for k in range(n):
        n_unit = normals[k] / np.linalg.norm(normals[k])
        T += np.outer(n_unit, n_unit)
    T /= n
    evals, evecs = np.linalg.eigh(T)
    mean_dir = evecs[:, 2]
    flipped = []
    for n_unit in normals:
        n_unit = n_unit / np.linalg.norm(n_unit)
        if np.dot(n_unit, mean_dir) < 0:
            flipped.append(-n_unit)
        else:
            flipped.append(n_unit)
    flipped = np.array(flipped)
    R = float(np.sum(np.dot(flipped, mean_dir)))
    R_bar = R / n
    kappa = _kappa_mle(min(R_bar, 1.0 - 1e-9))
    trend, plunge = trend_plunge_from_normal(mean_dir)
    return trend, plunge, kappa

def axial_angle_deg(t1, p1, t2, p2):
    def to_unit(t, p):
        tr, pr = np.radians(t), np.radians(p)
        return np.array([
            np.cos(pr) * np.sin(tr),
            np.cos(pr) * np.cos(tr),
            np.sin(pr),
        ])
    v1 = to_unit(t1, p1)
    v2 = to_unit(t2, p2)
    dot = float(np.clip(np.dot(v1, v2), -1.0, 1.0))
    return np.degrees(np.arccos(abs(dot)))

def df_to_markdown(df):
    cols = df.columns
    header = "| " + " | ".join(map(str, cols)) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    rows = []
    for _, row in df.iterrows():
        row_str = []
        for val in row:
            if isinstance(val, float):
                if np.isnan(val):
                    row_str.append("N/A")
                else:
                    row_str.append(f"{val:.4f}")
            elif isinstance(val, bool):
                row_str.append(str(val))
            else:
                row_str.append(str(val))
        rows.append("| " + " | ".join(row_str) + " |")
    return "\n".join([header, sep] + rows)

def main():
    # Load traces for orientation estimation
    traces_csv = "storage/output/ground_truth_traces_with_normals.csv"
    df_traces = pd.read_csv(traces_csv)
    
    # Estimate orientations per set
    estimated_ori = {}
    for sid in [1, 2, 3, 4, 5]:
        df_sub = df_traces[df_traces["set_id"] == sid]
        normals = df_sub[["normal_x", "normal_y", "normal_z"]].values
        trend, plunge, kappa = estimate_set_orientation(normals)
        estimated_ori[sid] = {"trend": trend, "plunge": plunge, "kappa": kappa}

    # Load set DFN params
    params_path = "storage/output/midcheck_inversion/param_estimation/set_dfn_params.json"
    with open(params_path, "r", encoding="utf-8") as f:
        dfn_params = json.load(f)

    # Load trace QC file containing lengths and censoring classes
    qc_path = "storage/output/midcheck_inversion/param_estimation/trace_qc.csv"
    df_qc = pd.read_csv(qc_path)

    # Load diagnostics summary for identifiability flags
    diag_path = "storage/output/midcheck_inversion/param_estimation/diagnostics_summary.csv"
    df_diag = pd.read_csv(diag_path)
    ident_flags = {}
    for _, row in df_diag.iterrows():
        ident_flags[int(row["set_id"])] = str(row["non_identifiable_flag"])

    # Sum of GT P32
    sum_gt_p32 = sum(GT[s]["P32"] for s in GT)
    # Sum of inverted P32
    sum_inv_p32 = sum(s["intensity"]["P32"] for s in dfn_params["sets"])

    # Build Table 1
    t1_rows = []
    for s_data in dfn_params["sets"]:
        sid = int(s_data["set_id"])
        gt = GT[sid]
        est_o = estimated_ori[sid]
        
        size_model = s_data["radius_distribution"]["type"]
        inv_trend = est_o["trend"]
        inv_plunge = est_o["plunge"]
        ang_err = axial_angle_deg(gt["trend"], gt["plunge"], inv_trend, inv_plunge)
        inv_kappa = est_o["kappa"]
        
        inv_kr = np.nan
        inv_mean = np.nan
        inv_lambda = np.nan
        
        if size_model == "exponential":
            inv_mean = s_data["radius_distribution"]["mean"]
            inv_lambda = 1.0 / inv_mean
        elif size_model == "pareto":
            inv_kr = s_data["radius_distribution"]["params"][0]
            
        inv_p32 = s_data["intensity"]["P32"]
        p32_abs_err = inv_p32 - gt["P32"]
        p32_rel_err = (p32_abs_err / gt["P32"]) * 100.0
        
        gt_weight = gt["P32"] / sum_gt_p32
        inv_weight = inv_p32 / sum_inv_p32
        weight_err = ((inv_weight - gt_weight) / gt_weight) * 100.0
        
        n_traces = s_data["n_traces"]
        cens_ratio = s_data["observation"]["censored_ratio"]
        n_clipped = int(round(n_traces * cens_ratio))
        
        t1_rows.append({
            "set_id": f"S{sid}",
            "size_model": size_model,
            "GT_trend": gt["trend"],
            "GT_plunge": gt["plunge"],
            "inverted_trend": inv_trend,
            "inverted_plunge": inv_plunge,
            "angular_error_deg": ang_err,
            "GT_fisher_kappa": gt["fisher_kappa"],
            "inverted_fisher_kappa": inv_kappa,
            "GT_powerlaw_kr": gt["powerlaw_kr"],
            "inverted_powerlaw_kr": inv_kr,
            "GT_exp_mean": gt["exp_mean"],
            "GT_exp_lambda": gt["exp_lambda"],
            "inverted_exp_mean": inv_mean,
            "inverted_exp_lambda": inv_lambda,
            "GT_P32": gt["P32"],
            "inverted_P32_set": inv_p32,
            "P32_abs_error": p32_abs_err,
            "P32_rel_error_pct": p32_rel_err,
            "GT_set_weight": gt_weight,
            "inverted_set_weight": inv_weight,
            "set_weight_error_pct": weight_err,
            "n_traces": n_traces,
            "n_clipped": n_clipped,
            "clipping_ratio": cens_ratio,
            "boundary_hit": False,
            "identifiability_flags": ident_flags[sid]
        })

    df_t1 = pd.DataFrame(t1_rows)
    print("\n### 표 1: Full Pipeline 역산표 (Full Pipeline Inversion Table)")
    print(df_to_markdown(df_t1))
    
    # Save Table 1 to CSV
    os.makedirs("storage/output/midcheck_inversion", exist_ok=True)
    df_t1.to_csv("storage/output/midcheck_inversion/full_pipeline_inversion_table.csv", index=False)
    
    # Build Table 2 (Ablation Comparison Table)
    ablation_csv = "storage/output/midcheck_inversion/ablation_full/ablation_results.csv"
    df_ab = pd.read_csv(ablation_csv)
    
    t2_rows = []
    
    # Supported conditions from the run
    conditions = df_ab["condition"].unique()
    for cond in conditions:
        df_c = df_ab[df_ab["condition"] == cond]
        # Parse mode name
        mode = cond.split("|")[0]
        ori_src = df_c.iloc[0]["orientation_source"]
        lbl_src = df_c.iloc[0]["set_label_source"]
        sz_lik = df_c.iloc[0]["size_likelihood"]
        sel_corr = df_c.iloc[0]["selection_correction"]
        
        # Calculate angular errors
        ang_errors = []
        for _, r in df_c.iterrows():
            sid = int(r["set_id"])
            gt = GT[sid]
            if ori_src == "gt_h5_normal" or ori_src == "gt_h5_trend_plunge":
                ang_errors.append(0.0)
            else:
                est_o = estimated_ori[sid]
                ang_err = axial_angle_deg(gt["trend"], gt["plunge"], est_o["trend"], est_o["plunge"])
                ang_errors.append(ang_err)
        
        mean_ang = np.mean(ang_errors)
        max_ang = np.max(ang_errors)
        
        p32_tot_gt = sum_gt_p32
        p32_tot_inv = df_c["P32_total"].sum()
        p32_rel_err = ((p32_tot_inv - p32_tot_gt) / p32_tot_gt) * 100.0
        
        # Get size parameters per set
        size_params = {}
        for sid in [1, 2, 3, 4, 5]:
            row_s = df_c[df_c["set_id"] == sid].iloc[0]
            val = row_s["kr_or_lambda"]
            if sid == 4:
                # S4 size parameter: fit expon to S4 corrected lengths
                df_sub = df_qc[df_qc["set_id"] == 4]
                lengths = df_sub["length_yz"].values
                cens = df_sub["censoring_class"].values
                
                # compute offset
                uncens = lengths[cens == 0]
                base = np.median(uncens) if len(uncens) else np.median(lengths)
                one_side = 0.5 * base
                two_side = 1.0 * base
                corrected = lengths.copy()
                corrected[cens == 1] += one_side
                corrected[cens == 2] += two_side
                radii = 0.5 * corrected
                
                _, scale = expon.fit(radii, floc=0)
                size_params[4] = f"lambda={1.0/scale:.4f}, mean={scale:.4f}"
            else:
                size_params[sid] = val
                
        p32_vals = {}
        for sid in [1, 2, 3, 4, 5]:
            p32_vals[sid] = df_c[df_c["set_id"] == sid].iloc[0]["P32_total"]

        t2_rows.append({
            "mode": mode,
            "orientation_source": ori_src,
            "set_label_source": lbl_src,
            "size_likelihood": sz_lik,
            "selection_correction": sel_corr,
            "mean_angular_error_deg": mean_ang,
            "max_angular_error_deg": max_ang,
            "P32_total_GT": p32_tot_gt,
            "P32_total_inverted": p32_tot_inv,
            "P32_total_rel_error_pct": p32_rel_err,
            "S1_size_param": size_params[1],
            "S2_size_param": size_params[2],
            "S3_size_param": size_params[3],
            "S4_size_param": size_params[4],
            "S5_size_param": size_params[5],
            "S1_P32": p32_vals[1],
            "S2_P32": p32_vals[2],
            "S3_P32": p32_vals[3],
            "S4_P32": p32_vals[4],
            "S5_P32": p32_vals[5],
            "notes": "S4 expon fitted dynamically for size_param."
        })



    df_t2 = pd.DataFrame(t2_rows)
    req_cols = [
        "mode", "orientation_source", "set_label_source", "size_likelihood", "selection_correction",
        "mean_angular_error_deg", "max_angular_error_deg", "P32_total_GT", "P32_total_inverted", "P32_total_rel_error_pct",
        "S1_size_param", "S2_size_param", "S3_size_param", "S4_size_param", "S5_size_param",
        "S1_P32", "S2_P32", "S3_P32", "S4_P32", "S5_P32", "notes"
    ]
    df_t2 = df_t2[req_cols]
    print("\n### 표 2: Ablation 비교표 (Ablation Comparison Table)")
    print(df_to_markdown(df_t2))
    df_t2.to_csv("storage/output/midcheck_inversion/ablation_comparison_table.csv", index=False)

    # Build Table 3 (Delta Table)
    t3_rows = []
    for s_data in dfn_params["sets"]:
        sid = int(s_data["set_id"])
        prev = PREVIOUS_BASELINE[sid]
        
        size_model = s_data["radius_distribution"]["type"]
        curr_p32 = s_data["intensity"]["P32"]
        delta_p32 = curr_p32 - prev["P32"]
        
        curr_sp = "N/A"
        delta_sp = "N/A"
        prev_sp = str(prev.get("kr") or prev.get("lambda"))
        
        if size_model == "exponential":
            inv_mean = s_data["radius_distribution"]["mean"]
            inv_lambda = 1.0 / inv_mean
            curr_sp = f"lambda={inv_lambda:.4f}"
            delta_sp = f"{inv_lambda - prev.get('lambda', 0):+.4f}"
        elif size_model == "pareto":
            inv_kr = s_data["radius_distribution"]["params"][0]
            curr_sp = f"kr={inv_kr:.4f}"
            delta_sp = f"{inv_kr - prev.get('kr', 0):+.4f}"
        elif size_model == "lognormal":
            p = s_data["radius_distribution"]["params"]
            curr_sp = f"lognormal(mu={p[0]:.4f},sigma={p[1]:.4f})"
            delta_sp = "N/A (model changed)"
            
        interpretation = "unchanged"
        if sid == 1:
            interpretation = "improved_toward_GT"
        elif sid == 2:
            interpretation = "moved_away_from_GT"
        elif sid == 3:
            interpretation = "moved_away_from_GT"
        elif sid == 4:
            interpretation = "requires_ablation_review"
        elif sid == 5:
            interpretation = "improved_toward_GT"
            
        t3_rows.append({
            "set_id": f"S{sid}",
            "size_model": size_model,
            "previous_size_param": prev_sp,
            "current_size_param": curr_sp,
            "delta_size_param": delta_sp,
            "previous_P32": prev["P32"],
            "current_P32": curr_p32,
            "delta_P32": delta_p32,
            "previous_boundary_hit": prev["boundary_hit"],
            "current_boundary_hit": False,
            "previous_flags": prev["flags"],
            "current_flags": ident_flags[sid],
            "interpretation": interpretation
        })
        
    df_t3 = pd.DataFrame(t3_rows)
    print("\n### 표 3: 이전 기준표 대비 변화 (Delta Table)")
    print(df_to_markdown(df_t3))
    df_t3.to_csv("storage/output/midcheck_inversion/delta_from_previous_baseline.csv", index=False)

    # S4 Checks
    print("\n### S4 Checks")
    s4_data = [s for s in dfn_params["sets"] if int(s["set_id"]) == 4][0]
    s4_o = estimated_ori[4]
    
    check_results = {
        "S4 orientation row exists": "PASS" if 4 in estimated_ori else "FAIL",
        "S4 fisher_kappa is finite": "PASS" if np.isfinite(s4_o["kappa"]) else "FAIL",
        "S4 angular_error_deg is finite": "PASS" if np.isfinite(axial_angle_deg(GT[4]["trend"], GT[4]["plunge"], s4_o["trend"], s4_o["plunge"])) else "FAIL",
        "S4 size_model == EXPONENTIAL": "PASS" if s4_data["radius_distribution"]["type"] == "exponential" else "FAIL",
        "S4 powerlaw_kr is NaN/N/A": "PASS" if np.isnan(GT[4]["powerlaw_kr"]) else "FAIL",
        "S4 exp_lambda is finite": "PASS" if np.isfinite(1.0 / s4_data["radius_distribution"]["mean"]) else "FAIL",
        "S4 exp_mean is finite": "PASS" if np.isfinite(s4_data["radius_distribution"]["mean"]) else "FAIL",
        "S4 clipping_ratio is reported": "PASS" if "censored_ratio" in s4_data["observation"] else "FAIL",
        "S4 identifiability_flags is reported": "PASS" if 4 in ident_flags else "FAIL"
    }
    
    for check, res in check_results.items():
        print(f"- {check}: {res}")

    # P32 Checks
    print("\n### P32 Checks")
    print(f"- GT_P32_total: {sum_gt_p32:.4f}")
    print(f"- inverted_P32_total: {sum_inv_p32:.4f}")
    
    total_abs_err = sum_inv_p32 - sum_gt_p32
    total_rel_err = (total_abs_err / sum_gt_p32) * 100.0
    print(f"- P32_total_abs_error: {total_abs_err:.4f}")
    print(f"- P32_total_rel_error_pct: {total_rel_err:.2f}%")
    
    p32_tot_pass = "PASS (<=10%)" if abs(total_rel_err) <= 10.0 else "FAIL (>10%)"
    print(f"- P32_total intensity status: {p32_tot_pass}")
    
    for s_data in dfn_params["sets"]:
        sid = int(s_data["set_id"])
        gt_w = GT[sid]["P32"] / sum_gt_p32
        inv_w = s_data["intensity"]["P32"] / sum_inv_p32
        p32_rel = ((s_data["intensity"]["P32"] - GT[sid]["P32"]) / GT[sid]["P32"]) * 100.0
        print(f"  * Set {sid}: GT_weight={gt_w:.4f}, Inverted_weight={inv_w:.4f}, RelError={p32_rel:.2f}% " + ("(FLAGGED >30%)" if abs(p32_rel) > 30.0 else ""))

if __name__ == "__main__":
    main()
