import os
import sys
import pandas as pd
import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
_parent = os.path.dirname(_here)
sys.path.insert(0, _parent)

from dfnrec.models import Face, Trace, CensorType
from dfnrec.size_intensity.chord_likelihood import censored_chord_log_likelihood

df_traces = pd.read_csv("storage/output/ground_truth_traces_with_normals.csv")
df_tunnel = pd.read_csv("storage/output/tunnel_polygon.csv")
poly_uv = df_tunnel[["y", "z"]].values
unique_face_ids = sorted(df_traces["face_id"].unique())
face_x_coords = {f_id: float(df_traces[df_traces["face_id"] == f_id]["x_face"].mean()) for f_id in unique_face_ids}

# Function to classify censoring
def classify_censoring(pt_uv, poly_uv, tol=0.05):
    min_dist = float("inf")
    for i in range(len(poly_uv)):
        p_start = poly_uv[i]
        p_end = poly_uv[(i + 1) % len(poly_uv)]
        edge = p_end - p_start
        edge_len = np.linalg.norm(edge)
        if edge_len < 1e-12:
            dist = np.linalg.norm(pt_uv - p_start)
        else:
            t = np.clip(np.dot(pt_uv - p_start, edge) / (edge_len**2), 0.0, 1.0)
            proj = p_start + t * edge
            dist = np.linalg.norm(pt_uv - proj)
        if dist < min_dist:
            min_dist = dist
    return CensorType.CLIPPED if min_dist < tol else CensorType.NATURAL

traces = []
for _, row in df_traces.iterrows():
    p0_xyz = [float(row["p0_x"]), float(row["p0_y"]), float(row["p0_z"])]
    p1_xyz = [float(row["p1_x"]), float(row["p1_y"]), float(row["p1_z"])]
    
    p0_uv = np.array([row["p0_y"], row["p0_z"]])
    p1_uv = np.array([row["p1_y"], row["p1_z"]])
    c0 = classify_censoring(p0_uv, poly_uv, tol=0.05)
    c1 = classify_censoring(p1_uv, poly_uv, tol=0.05)
    
    set_id = f"S{int(row['set_id'])}"
    traces.append(Trace(
        trace_id=f"T{int(row['trace_id'])}",
        face_id=str(int(row["face_id"])),
        set_id=set_id,
        p0_xyz=p0_xyz,
        p1_xyz=p1_xyz,
        censor_p0=c0,
        censor_p1=c1,
        measurement_sigma=0.02,
    ))

print("Joint MLE size model selection:")
for sid in ["S1", "S2", "S3", "S4", "S5"]:
    set_traces = [t for t in traces if t.set_id == sid]
    chord_lengths = np.array([t.observed_length for t in set_traces])
    is_contained = np.array([t.is_contained for t in set_traces])
    
    # Fit POWER_LAW
    best_pl_ll = -1e10
    best_pl_alpha = 0
    best_pl_rmin = 0
    
    for r_min_candidate in np.linspace(0.1, 1.5, 29):
        for alpha_candidate in np.linspace(1.5, 6.0, 46):
            ll = censored_chord_log_likelihood(
                chord_lengths, is_contained, alpha_candidate, r_min_candidate, 30.0, 0.1, size_model="POWER_LAW"
            )
            if ll > best_pl_ll:
                best_pl_ll = ll
                best_pl_alpha = alpha_candidate
                best_pl_rmin = r_min_candidate
                
    # Fit EXPONENTIAL
    best_exp_ll = -1e10
    best_exp_lambda = 0
    best_exp_rmin = 0
    
    for r_min_candidate in np.linspace(0.1, 1.5, 29):
        for lambda_candidate in np.linspace(0.05, 1.0, 39):
            ll = censored_chord_log_likelihood(
                chord_lengths, is_contained, lambda_candidate, r_min_candidate, 30.0, 0.1, size_model="EXPONENTIAL"
            )
            if ll > best_exp_ll:
                best_exp_ll = ll
                best_exp_lambda = lambda_candidate
                best_exp_rmin = r_min_candidate
                
    print(f"\n--- Set {sid} ---")
    print(f"POWER_LAW: best r_min={best_pl_rmin:.3f}, alpha={best_pl_alpha:.3f} (k_r={best_pl_alpha-1:.3f}), LL={best_pl_ll:.4f}")
    print(f"EXPONENTIAL: best r_min={best_exp_rmin:.3f}, lambda={best_exp_lambda:.3f} (scale={1/best_exp_lambda:.2f}), LL={best_exp_ll:.4f}")
    if best_pl_ll > best_exp_ll:
        print("Selected: POWER_LAW")
    else:
        print("Selected: EXPONENTIAL")
