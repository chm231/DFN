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
df_traces = pd.read_csv("storage/output/ground_truth_traces.csv")
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
    set_id = f"S{int(row['set_id'])}"
    traces.append(Trace(
        trace_id=f"T{int(row['trace_id'])}",
        face_id=str(int(row["face_id"])),
        set_id=set_id,
        p0_xyz=p0_xyz,
        p1_xyz=p1_xyz,
        censor_p0=CensorType.NATURAL,
        censor_p1=CensorType.NATURAL,
        measurement_sigma=0.02,
    ))

edges = build_candidate_graph(traces, faces_dict, log_bf_threshold=-15.0)
tracks = select_non_overlapping_tracks(traces, edges, min_faces=1)

discs = []
for i, track in enumerate(tracks):
    disc = estimate_disc_map(track, faces_dict, disc_id_prefix=f"D_{i:04d}")
    if disc is not None:
        discs.append(disc)

print(f"Total reconstructed discs: {len(discs)}")
for sid in ["S1", "S2", "S3", "S4", "S5"]:
    set_discs = [d for d in discs if d.set_id == sid]
    multi = [d for d in set_discs if d.n_faces_observed >= 2]
    print(f"Set {sid}: total={len(set_discs)}, multi-face={len(multi)}")
    for d in multi:
        print(f"  - {d.disc_id}: trend={d.trend_deg:.2f}, plunge={d.plunge_deg:.2f}")
