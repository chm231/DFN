import os
import sys
import pandas as pd
import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
_parent = os.path.dirname(_here)
sys.path.insert(0, _parent)

from dfnrec.models import Face, Trace, CensorType
from dfnrec.reconstruction import build_candidate_graph, select_non_overlapping_tracks, estimate_disc_map

# Setup faces and traces from csv
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
    
    # Orientation from normal if present
    trend_val = None
    plunge_val = None
    if "normal_x" in row:
        n_vec = np.array([float(row["normal_x"]), float(row["normal_y"]), float(row["normal_z"])])
        from dfnrec.geometry.vector import trend_plunge_from_normal
        trend_val, plunge_val = trend_plunge_from_normal(n_vec)

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
        trend_deg=trend_val,
        plunge_deg=plunge_val,
    ))

edges = build_candidate_graph(traces, faces_dict, log_bf_threshold=-15.0)
tracks = select_non_overlapping_tracks(traces, edges, min_faces=1)

discs = []
for i, track in enumerate(tracks):
    disc = estimate_disc_map(track, faces_dict, disc_id_prefix=f"D_{i:04d}")
    if disc is not None:
        discs.append(disc)

print("Check reconstructed discs:")
for sid in ["S1", "S2", "S3", "S4", "S5"]:
    set_discs = [d for d in discs if d.set_id == sid]
    multi = [d for d in set_discs if d.n_faces_observed >= 2]
    single_with_x = [d for d in set_discs if d.n_faces_observed < 2 and abs(d.normal_xyz[0]) >= 1e-6]
    single_without_x = [d for d in set_discs if d.n_faces_observed < 2 and abs(d.normal_xyz[0]) < 1e-6]
    print(f"Set {sid}: total={len(set_discs)}, multi-face={len(multi)}, single_with_x={len(single_with_x)}, single_without_x={len(single_without_x)}")
