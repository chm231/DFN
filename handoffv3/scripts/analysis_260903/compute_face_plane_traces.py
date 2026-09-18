# -*- coding: utf-8 -*-
"""도메인 DFN JSON을 x=face_x 평면으로 절단해 관측창(터널 폴리곤) 내
가시 절리선 통계를 npz로 저장 (MC 면 대조용).

사용: (handoffv2에서, PYTHONPATH=.)
  python compute_face_plane_traces.py <json> <face_x> <out_npz>
저장: lengths(가시길이), angles_deg(yz 평면 방향 0~180), observed(0/1)
"""
import json
import sys

import numpy as np

try:
    from dfn_analysis.radius_powerlaw_likelihood import (
        clip_segments_to_convex_polygon_vectorized)
except ImportError:
    from radius_powerlaw_likelihood import clip_segments_to_convex_polygon_vectorized

json_path, face_x, out_npz = sys.argv[1], float(sys.argv[2]), sys.argv[3]

with open(json_path, encoding="utf-8") as f:
    data = json.load(f)
fr = data["fractures"]
C = np.array([f_["center_xyz_m"] for f_ in fr], dtype=np.float64)
N = np.array([f_["normal_xyz"] for f_ in fr], dtype=np.float64)
N /= np.linalg.norm(N, axis=1, keepdims=True)
R = np.array([f_["radius_m"] for f_ in fr], dtype=np.float64)
obs = np.array([1 if f_.get("observed") else 0 for f_ in fr], dtype=np.int8)
poly_yz = np.asarray(data["meta"]["domain"]["tunnel_polygon_yz_m"], dtype=float)

sin_phi = np.sqrt(np.clip(1.0 - N[:, 0] ** 2, 1e-12, None))
ok_phi = sin_phi > 1e-6
t_in = (C[:, 0] - face_x) / sin_phi
cand = ok_phi & (np.abs(t_in) < R - 1e-9)
idx = np.nonzero(cand)[0]

half = np.sqrt(np.clip(R[idx] ** 2 - t_in[idx] ** 2, 0.0, None))
dir_yz = np.column_stack([N[idx, 2], -N[idx, 1]]) / sin_phi[idx, None]
e_x = np.array([1.0, 0.0, 0.0])
mid = C[idx] - t_in[idx, None] * (e_x[None, :] - N[idx, 0:1] * N[idx]) / sin_phi[idx, None]
mid_yz = mid[:, 1:3]

vis_len, _ = clip_segments_to_convex_polygon_vectorized(
    mid_yz, dir_yz, 2.0 * half, poly_yz)
m = vis_len > 1e-6
ang = np.degrees(np.arctan2(dir_yz[m, 1], dir_yz[m, 0])) % 180.0
np.savez_compressed(out_npz, lengths=vis_len[m], angles_deg=ang,
                    observed=obs[idx][m])
n05 = (vis_len[m] >= 0.5).sum()
print(f"result 절단 {m.sum()}개, 가시>=0.5m {n05}개, 총길이(>=0.5) "
      f"{vis_len[m][vis_len[m]>=0.5].sum():.1f} m")
