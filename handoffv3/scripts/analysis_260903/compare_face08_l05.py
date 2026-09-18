# -*- coding: utf-8 -*-
"""면 08 블라인드: 검출하한 통일(lmin=0.5) 재보정 전/후 MC50 vs 실측.

출력: docs/figures/blind_face08_lmin05.png + 콘솔 요약
"""
import glob
import json

import numpy as np
import pandas as pd

H2 = "c:/Users/user/OneDrive/2026-1/3D DFN modeling/handoffv2"
SP = "C:/Users/user/AppData/Local/Temp/claude/c--Users-user-OneDrive-2026-1-3D-DFN-modeling/c501f425-c178-491d-b844-fdbebd260c97/scratchpad"
LMIN = 0.5

poly = np.asarray(json.load(open(
    f"{H2}/example_io/f06/dfn_domain_f06_forward.json", encoding="utf-8"))
    ["meta"]["domain"]["tunnel_polygon_yz_m"], float)
area = 0.5 * abs(np.sum(poly[:, 0] * np.roll(poly[:, 1], -1)
                        - np.roll(poly[:, 0], -1) * poly[:, 1]))

df = pd.read_csv(f"{H2}/demo_output/dfm_demo/trace_dataset/trace_dataset_3d.csv")
f8 = df[(df.face_id == 8) & (df.observed_length_m >= LMIN)]
len_a = f8.observed_length_m.to_numpy()
n_a, p21_a = len(f8), len_a.sum() / area


def load(tag):
    counts, p21s, lens = [], [], []
    for f in sorted(glob.glob(f"{SP}/{tag}/tr_seed*.npz")):
        z = np.load(f)
        L = z["lengths"]; m = L >= LMIN
        counts.append(int(m.sum())); p21s.append(L[m].sum() / area)
        lens.append(L[m])
    return np.array(counts), np.array(p21s), np.concatenate(lens)


c0, p0, l0 = load("face08mc")
c1, p1, l1 = load("face08mc_l05")
print(f"[실측] {n_a}개, P21 {p21_a:.2f}")
for name, c, p in (("lmin=0 (기존)", c0, p0), ("lmin=0.5 (통일)", c1, p1)):
    print(f"[{name}] 개수 {c.mean():.0f}±{c.std():.0f} [{c.min()},{c.max()}] "
          f"(실측 백분위 {(c<n_a).mean()*100:.0f}%) | "
          f"P21 {p.mean():.2f}±{p.std():.2f} (백분위 {(p<p21_a).mean()*100:.0f}%)")

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "Malgun Gothic"
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.8))

ax = axes[0]
bins = np.arange(90, 245, 8)
ax.hist(c0, bins=bins, alpha=0.55, color="#c98b3d", label="MC lmin=0 (기존)")
ax.hist(c1, bins=bins, alpha=0.55, color="#3a6ea8", label="MC lmin=0.5 (통일)")
ax.axvline(n_a, color="crimson", lw=2.5, label=f"실측 {n_a}개")
ax.set_title("절리선 개수 (≥0.5 m)")
ax.set_xlabel("개수 / 실현"); ax.set_ylabel("실현 수"); ax.legend(fontsize=9)

ax = axes[1]
bins = np.linspace(1.4, 4.3, 30)
ax.hist(p0, bins=bins, alpha=0.55, color="#c98b3d", label="MC lmin=0")
ax.hist(p1, bins=bins, alpha=0.55, color="#3a6ea8", label="MC lmin=0.5")
ax.axvline(p21_a, color="crimson", lw=2.5, label=f"실측 {p21_a:.2f}")
ax.set_title("P21 (≥0.5 m) [m/m²]")
ax.set_xlabel("P21"); ax.legend(fontsize=9)

ax = axes[2]
for arr, col, lab in ((len_a, "crimson", "실측"),
                      (l0, "#c98b3d", "MC lmin=0"),
                      (l1, "#3a6ea8", "MC lmin=0.5")):
    xs = np.sort(arr)
    ax.step(xs, 1 - np.arange(len(xs)) / len(xs), color=col, lw=2, label=lab)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_title("길이 CCDF (≥0.5 m)")
ax.set_xlabel("가시 길이 [m]"); ax.set_ylabel("P(L≥l)"); ax.legend(fontsize=9)

fig.suptitle("면 08 블라인드 재검증 — P32 보정 검출하한 통일(lmin 0→0.5) 전후, MC 50 실현",
             fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.93])
out = "c:/Users/user/OneDrive/2026-1/3D DFN modeling/docs/figures/blind_face08_lmin05.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("saved", out)
