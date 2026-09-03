# -*- coding: utf-8 -*-
"""확인사항 2 회신용 그림: 면별 절리선 수·P21 — 원자료 vs 검출하한 0.4 m 통일."""
import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "Malgun Gothic"
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

P = "../handoffv1/demo_output/dfm_demo/trace_dataset/trace_dataset_3d.h5"
with h5py.File(P) as f:
    fx = np.round(f["traces/face_x_m"][:].ravel(), 4)
    L = f["traces/observed_length_m"][:].ravel()
    p0 = f["traces/p0_xyz"][:]
    p1 = f["traces/p1_xyz"][:]

xs = np.unique(fx)
n_raw, n_cut, p21_raw, p21_cut, lmin_face = [], [], [], [], []
for v in xs:
    m = fx == v
    pts = np.vstack([p0[m][:, 1:], p1[m][:, 1:]])
    A = ConvexHull(pts).volume
    mc = m & (L >= 0.4)
    n_raw.append(m.sum()); n_cut.append(mc.sum())
    p21_raw.append(L[m].sum() / A); p21_cut.append(L[mc].sum() / A)
    lmin_face.append(L[m].min())

idx = np.arange(len(xs))
labels = [f"{v:.1f}" for v in xs]
anom = xs == 12.63

fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

ax = axes[0]
ax.bar(idx - 0.2, n_raw, 0.4, label="원자료 전체", color="#4878a8")
ax.bar(idx + 0.2, n_cut, 0.4, label="길이 ≥ 0.4 m만", color="#e08214")
ax.set_title("면별 절리선 수")
ax.set_ylabel("절리선 수 [개]")

ax = axes[1]
ax.bar(idx - 0.2, p21_raw, 0.4, label="원자료 전체", color="#4878a8")
ax.bar(idx + 0.2, p21_cut, 0.4, label="길이 ≥ 0.4 m만", color="#e08214")
ax.set_title("면별 P21 (총길이/면적)")
ax.set_ylabel("P21 [m/m²]")

ax = axes[2]
ax.bar(idx, lmin_face, 0.6, color="#4878a8")
ax.axhline(0.15, color="gray", ls="--", lw=0.8)
ax.set_title("면별 최단 절리선 길이 (추출 하한 지표)")
ax.set_ylabel("min 길이 [m]")

for ax in axes:
    ax.set_xticks(idx)
    ax.set_xticklabels(labels, rotation=60, fontsize=8)
    ax.set_xlabel("막장면 x [m]")
    for i, a in enumerate(anom):
        if a:
            ax.axvspan(i - 0.45, i + 0.45, color="red", alpha=0.10)
axes[0].legend(fontsize=9)
axes[1].legend(fontsize=9)
fig.suptitle("면별 절리선 수·강도 — 붉은 음영 = x=12.63 (검출하한 통일 후에도 이상)", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.94])
out = "c:/Users/user/OneDrive/2026-1/3D DFN modeling/docs/figures/facewise_trace_counts_0902.png"
fig.savefig(out, dpi=150)
print("saved", out)
