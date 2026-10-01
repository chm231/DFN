# -*- coding: utf-8 -*-
"""12개 막장면 점군을 기준 단면과 대조해 면별 여굴(윤곽 초과)을 정량화한다.

기준 단면: win_f01-06_l05 의 /tunnel/poly_YZ (면적 중앙값 면의 convex hull).
초과량 = 단면 중심에서의 반경 − 같은 각도에서의 단면 경계 반경.
출력: docs/data/overbreak_by_face.csv
"""
import json, os, sys
import h5py, numpy as np
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
R = "c:/Users/user/OneDrive/2026-1/3D DFN modeling"
DFM = f"{R}/handoffv2/demodata/DFM_Export/DFM_Export"

d = json.load(open(f"{R}/handoffv2/demo_output/dfm_demo/conversion_diagnostics.json", encoding="utf-8"))
Rot = np.array(d["coordinate_system"]["world_to_pipeline_rotation"], float)
AREA = d["face_area_m2"]
with h5py.File(f"{R}/handoffv2/demo_output/win_f01-06_l05/dfn_export_for_python.h5", "r") as f:
    poly = f["/tunnel/poly_YZ"][...]
cy, cz = poly[:, 0].mean(), poly[:, 1].mean()
pa = np.arctan2(poly[:, 1] - cz, poly[:, 0] - cy)
pr = np.hypot(poly[:, 0] - cy, poly[:, 1] - cz)
o = np.argsort(pa); pa, pr = pa[o], pr[o]

rows = []
for n in sorted(x for x in os.listdir(DFM) if x.isdigit()):
    p = f"{DFM}/{n}/Export 3D DS Data/3D_tunnel_face_point_cloud.ply"
    with open(p) as f:
        nv = ln = 0
        for line in f:
            ln += 1
            if line.startswith("element vertex"): nv = int(line.split()[-1])
            if line.strip() == "end_header": break
    P = np.loadtxt(p, skiprows=ln, usecols=(0, 1, 2), max_rows=nv) @ Rot.T
    y, z = P[:, 1], P[:, 2]
    ang = np.arctan2(z - cz, y - cy)
    exc = np.hypot(y - cy, z - cz) - np.interp(ang, pa, pr, period=2 * np.pi)
    r = dict(face=n, face_x_m=d["face_x_m"][n], face_area_m2=AREA[n], n_pts=len(P),
             exc_max_m=round(float(exc.max()), 3), exc_p99_m=round(float(np.percentile(exc, 99)), 3),
             frac_gt0=round(float((exc > 0).mean()), 4))
    for t in (0.2, 0.3, 0.5, 0.8):
        m = exc > t
        r[f"frac_gt{t}"] = round(float(m.mean()), 5)
        if m.sum() >= 50:
            arc = (ang[m].max() - ang[m].min()) * float(np.median(np.hypot(y[m]-cy, z[m]-cz)))
            r[f"area_gt{t}_m2"] = round(arc * float(exc[m].mean()), 3)
            r[f"ymed_gt{t}"] = round(float(np.median(y[m])), 2)
            r[f"zmed_gt{t}"] = round(float(np.median(z[m])), 2)
            r[f"upper_left_frac_gt{t}"] = round(float(((y[m] > cy) & (z[m] > cz)).mean()), 3)
        else:
            for k in (f"area_gt{t}_m2", f"ymed_gt{t}", f"zmed_gt{t}", f"upper_left_frac_gt{t}"):
                r[k] = ""
    rows.append(r)
    print(f"면 {n}: x={r['face_x_m']:>7.3f} | >0 {r['frac_gt0']*100:5.1f}% | >0.3 {r['frac_gt0.3']*100:5.2f}% "
          f"| 최대 {r['exc_max_m']:.2f} m | 단면적>0.3 {r['area_gt0.3_m2'] or '-'} m2", flush=True)

import csv
out = f"{R}/docs/data/overbreak_by_face.csv"
with open(out, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print("\n저장:", out)
