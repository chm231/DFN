# -*- coding: utf-8 -*-
"""슬라이딩 윈도우 3세트 집계: 확률장 vti + 장면 vtp + 대상면 실측/예측 대조.

사용: (handoffv2에서) python aggregate_slide.py
"""
import glob
import json

import numpy as np
import pandas as pd
import pyvista as pv
from scipy.ndimage import gaussian_filter

SP = "C:/Users/user/AppData/Local/Temp/claude/c--Users-user-OneDrive-2026-1-3D-DFN-modeling/c501f425-c178-491d-b844-fdbebd260c97/scratchpad"
H2 = "c:/Users/user/OneDrive/2026-1/3D DFN modeling/handoffv2"
LMIN = 0.5

WINDOWS = [
    ("w14", "면 1-4 → 5 예측", 5, 8.4979, -0.2036),
    ("w25", "면 2-5 → 6 예측", 6, 11.3064, 1.6338),
    ("w36", "면 3-6 → 7 예측", 7, 12.63, 4.0731),
]

tr = pd.read_csv(f"{H2}/demo_output/dfm_demo/trace_dataset/trace_dataset_3d.csv")

for tag, label, tface, tx, x_back0 in WINDOWS:
    out = f"{H2}/example_io/f_slide/{tag}"
    dom = json.load(open(f"{out}/dfn_canonical.json", encoding="utf-8"))["meta"]["domain"]
    poly = np.asarray(dom["tunnel_polygon_yz_m"], float)
    x0, x1 = [float(v) for v in dom["x_range_m"]]
    yz = dom["yz_bounds_m"]
    area = 0.5 * abs(np.sum(poly[:, 0] * np.roll(poly[:, 1], -1)
                            - np.roll(poly[:, 0], -1) * poly[:, 1]))

    # ── 확률장 ──
    files = sorted(glob.glob(f"{SP}/slide/{tag}/blk_seed*_labels.npz"))
    count, vols = None, []
    for f in files:
        z = np.load(f)
        hit = z["labels"] > 0
        count = hit.astype(np.uint8) if count is None else count + hit
        origin, vs = z["origin"], float(z["voxel_size"])
        vols.append(hit.sum() * vs ** 3)
    p = count.astype(np.float32) / len(files)
    ps = gaussian_filter(p, sigma=3.0)
    grid = pv.ImageData(dimensions=p.shape, spacing=(vs, vs, vs),
                        origin=tuple(float(v) + vs / 2 for v in origin))
    grid.point_data["p_block"] = p.ravel(order="F")
    grid.point_data["p_smooth"] = ps.ravel(order="F")
    grid.save(f"{out}/block_prob_mc{len(files)}.vti")

    # ── 장면 기하: tunnel_behind / face_cap / domain_box ──
    n = len(poly)
    ring0 = np.column_stack([np.full(n, x_back0), poly])
    ring1 = np.column_stack([np.full(n, x0), poly])
    quads = [[4, i, (i+1) % n, n+(i+1) % n, n+i] for i in range(n)]
    pv.PolyData(np.vstack([ring0, ring1]),
                faces=np.array(quads).ravel()).save(f"{out}/tunnel_behind.vtp")
    cap = pv.PolyData(ring1, faces=np.concatenate([[n], np.arange(n)])).triangulate()
    cap.save(f"{out}/face_cap.vtp")
    box = pv.Box(bounds=(x0, x1, yz["y_min"], yz["y_max"], yz["z_min"], yz["z_max"]))
    box.extract_all_edges().save(f"{out}/domain_box.vtp")

    # ── 대상면 실측 절리선 (>=0.5 m, 3D 라인) ──
    f8 = tr[(tr.face_id == tface) & (tr.observed_length_m >= LMIN)]
    pts = np.vstack([f8[["p0_x", "p0_y", "p0_z"]].to_numpy(float),
                     f8[["p1_x", "p1_y", "p1_z"]].to_numpy(float)])
    nseg = len(f8)
    lines = np.column_stack([np.full(nseg, 2), np.arange(nseg),
                             np.arange(nseg) + nseg]).ravel()
    pv.PolyData(pts, lines=lines).save(f"{out}/target_face_traces.vtp")

    # ── 대상면 통계 대조 ──
    n_a = len(f8)
    p21_a = f8.observed_length_m.sum() / area
    counts, p21s = [], []
    for f in sorted(glob.glob(f"{SP}/slide/{tag}/tr_seed*.npz")):
        z = np.load(f)
        L = z["lengths"]
        m = L >= LMIN
        counts.append(int(m.sum()))
        p21s.append(L[m].sum() / area)
    counts = np.array(counts); p21s = np.array(p21s)
    print(f"[{tag}] {label} | 도메인 x [{x0:.2f},{x1:.2f}] | 실현 {len(files)}개, "
          f"블록부피 {min(vols):.1f}~{max(vols):.1f} m³")
    print(f"       대상면 {tface:02d}(x={tx}) 절리선: 예측 {counts.mean():.0f}±{counts.std():.0f} "
          f"vs 실측 {n_a} (백분위 {(counts<n_a).mean()*100:.0f}%) | "
          f"P21 {p21s.mean():.2f}±{p21s.std():.2f} vs {p21_a:.2f} "
          f"(백분위 {(p21s<p21_a).mean()*100:.0f}%)")
    print(f"       확률장: P>=0.5 {(p>=0.5-1e-9).sum()*vs**3:.2f} m³, 최대 P={p.max():.2f}, "
          f"p_smooth max {ps.max():.2f}")
