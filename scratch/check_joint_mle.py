import os
import sys
import pandas as pd
import numpy as np
from scipy.optimize import minimize

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

faces = []
for idx, f_id in enumerate(unique_face_ids):
    faces.append(Face(
        face_id=str(f_id),
        order_index=idx,
        origin_xyz=[face_x_coords[f_id], 0.0, 0.0],
        normal_xyz=[1.0, 0.0, 0.0],
        axis_u_xyz=[0.0, 1.0, 0.0],
        axis_v_xyz=[0.0, 0.0, 1.0],
        observation_window_polygon_uv=poly_uv.tolist(),
        L_min=0.1,
    ))

faces_dict = {f.face_id: f for f in faces}

traces = []
for _, row in df_traces.iterrows():
    p0_xyz = [float(row["p0_x"]), float(row["p0_y"]), float(row["p0_z"])]
    p1_xyz = [float(row["p1_x"]), float(row["p1_y"]), float(row["p1_z"])]
    
    # Determine censoring
    p0_uv = np.array([row["p0_y"], row["p0_z"]])
    p1_uv = np.array([row["p1_y"], row["p1_z"]])
    c0 = CensorType.NATURAL
    c1 = CensorType.NATURAL
    
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

print("Grid search for MLE of (alpha, r_min):")
for sid in ["S1", "S2", "S3", "S5"]:
    set_traces = [t for t in traces if t.set_id == sid]
    chord_lengths = np.array([t.observed_length for t in set_traces])
    is_contained = np.array([t.is_contained for t in set_traces])
    
    best_ll = -1e10
    best_alpha = 0
    best_rmin = 0
    
    # Grid search
    for r_min_candidate in np.linspace(0.1, 1.5, 15):
        for alpha_candidate in np.linspace(1.5, 6.0, 46):
            ll = censored_chord_log_likelihood(
                chord_lengths, is_contained, alpha_candidate, r_min_candidate, 30.0, 0.1
            )
            if ll > best_ll:
                best_ll = ll
                best_alpha = alpha_candidate
                best_rmin = r_min_candidate
                
    print(f"Set {sid}: best r_min={best_rmin:.3f}, best alpha={best_alpha:.3f} (k_r={best_alpha-1:.3f}), log-lik={best_ll:.4f}")
