# -*- coding: utf-8 -*-
"""면 08 블라인드 통계 대조: 1~6면 MC50 예측 절리선 vs 실측 절리선.

비교 기준: 검출하한 0.5 m (예측 조건화와 동일). 관측창 = 터널 단면 폴리곤.
출력: docs/figures/blind_face08_trace_stats.png + 콘솔 요약
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
print(f"관측창 면적 {area:.1f} m² (터널 단면)")

# ── 실측 면 08 ───────────────────────────────────────────────
df = pd.read_csv(f"{H2}/demo_output/dfm_demo/trace_dataset/trace_dataset_3d.csv")
f8 = df[df.face_id == 8].copy()
f8 = f8[f8.observed_length_m >= LMIN]
dy = f8.p1_y - f8.p0_y
dz = f8.p1_z - f8.p0_z
ang_a = np.degrees(np.arctan2(dz, dy)) % 180.0
len_a = f8.observed_length_m.to_numpy()
n_a, L_a = len(f8), len_a.sum()
print(f"[실측] 면08 절리선(>=0.5m) {n_a}개, 총길이 {L_a:.1f} m, "
      f"P21 {L_a/area:.3f} m/m², 길이 중앙값 {np.median(len_a):.2f}")

# ── 예측 MC50 ────────────────────────────────────────────────
files = sorted(glob.glob(f"{SP}/face08mc/tr_seed*.npz"))
print(f"[예측] 실현 {len(files)}개")
counts, p21s, med_lens = [], [], []
all_len, all_ang = [], []
for f in files:
    z = np.load(f)
    L = z["lengths"]; A = z["angles_deg"]
    m = L >= LMIN
    counts.append(int(m.sum()))
    p21s.append(L[m].sum() / area)
    med_lens.append(np.median(L[m]))
    all_len.append(L[m]); all_ang.append(A[m])
counts = np.array(counts); p21s = np.array(p21s)
all_len = np.concatenate(all_len); all_ang = np.concatenate(all_ang)


def pct_rank(v, arr):
    return (arr < v).mean() * 100


print(f"  개수: MC {counts.mean():.0f}±{counts.std():.0f} "
      f"[{counts.min()},{counts.max()}]  | 실측 {n_a} "
      f"(MC 백분위 {pct_rank(n_a, counts):.0f}%)")
print(f"  P21 : MC {p21s.mean():.3f}±{p21s.std():.3f} "
      f"[{p21s.min():.3f},{p21s.max():.3f}] | 실측 {L_a/area:.3f} "
      f"(MC 백분위 {pct_rank(L_a/area, p21s):.0f}%)")
print(f"  길이 중앙값: MC {np.mean(med_lens):.2f} | 실측 {np.median(len_a):.2f}")
from scipy.stats import ks_2samp
ks = ks_2samp(len_a, all_len)
print(f"  길이분포 KS(실측 vs MC풀) D={ks.statistic:.3f} p={ks.pvalue:.3f}")

# ── 그림 ─────────────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "Malgun Gothic"
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(13, 9))

ax = axes[0, 0]
ax.hist(counts, bins=15, color="#88a8c8", edgecolor="k", alpha=0.8)
ax.axvline(n_a, color="crimson", lw=2.5, label=f"실측 {n_a}개")
ax.set_title("절리선 개수 (≥0.5 m)")
ax.set_xlabel("개수 / 실현"); ax.set_ylabel("실현 수"); ax.legend()

ax = axes[0, 1]
ax.hist(p21s, bins=15, color="#88a8c8", edgecolor="k", alpha=0.8)
ax.axvline(L_a / area, color="crimson", lw=2.5, label=f"실측 {L_a/area:.2f}")
ax.set_title("P21 (≥0.5 m) [m/m²]")
ax.set_xlabel("P21"); ax.set_ylabel("실현 수"); ax.legend()

ax = axes[1, 0]
xs = np.sort(len_a); ax.step(xs, 1 - np.arange(len(xs)) / len(xs),
                             color="crimson", lw=2, label="실측")
xs = np.sort(all_len); ax.step(xs, 1 - np.arange(len(xs)) / len(xs),
                               color="#3a6ea8", lw=2, alpha=0.8, label="MC 풀(50)")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_title("길이 CCDF (≥0.5 m)")
ax.set_xlabel("가시 길이 [m]"); ax.set_ylabel("P(L≥l)"); ax.legend()

ax = axes[1, 1]
bins = np.linspace(0, 180, 19)
ax.hist(ang_a, bins=bins, density=True, color="crimson", alpha=0.55, label="실측")
ax.hist(all_ang, bins=bins, density=True, histtype="step", lw=2,
        color="#3a6ea8", label="MC 풀(50)")
ax.set_title("절리선 방향 분포 (yz 평면 각도)")
ax.set_xlabel("각도 [deg, 0=+y]"); ax.set_ylabel("밀도"); ax.legend()

fig.suptitle("블라인드 검증(면 08, x=15.43): 1~6면 예측 MC 50 실현 vs 실측 절리선 — "
             "검출하한 0.5 m, 관측창=터널 단면", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.95])
out = "c:/Users/user/OneDrive/2026-1/3D DFN modeling/docs/figures/blind_face08_trace_stats.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("saved", out)
