# -*- coding: utf-8 -*-
"""면 08 블라인드 검사 통계 그림(흑백, 캡션 없음): 1~6면 MC50 예측 vs 면 08 실측.
   ① 절리선 개수  ② P21  ③ 길이 CCDF.  재보정 전(lmin 0, 연회색 빗금) / 후(lmin 0.5, 진회색) / 실측(검정 선)."""
import glob, json
import numpy as np, pandas as pd, matplotlib, matplotlib.ticker
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Malgun Gothic"; plt.rcParams["axes.unicode_minus"] = False
K, DG, MG, LG = "#111111", "#444444", "#9A9A9A", "#D9D9D9"
H2 = r"C:\Users\user\OneDrive\2026-1\3D DFN modeling\handoffv2"
SP = r"C:\Users\user\AppData\Local\Temp\claude\c--Users-user-OneDrive-2026-1-3D-DFN-modeling\c501f425-c178-491d-b844-fdbebd260c97\scratchpad"
LMIN = 0.5
poly = np.asarray(json.load(open(H2 + r"\example_io\f06\dfn_domain_f06_forward.json", encoding="utf-8"))["meta"]["domain"]["tunnel_polygon_yz_m"], float)
area = 0.5 * abs(np.sum(poly[:, 0] * np.roll(poly[:, 1], -1) - np.roll(poly[:, 0], -1) * poly[:, 1]))
df = pd.read_csv(H2 + r"\demo_output\dfm_demo\trace_dataset\trace_dataset_3d.csv")
f8 = df[(df.face_id == 8) & (df.observed_length_m >= LMIN)]
len_a = f8.observed_length_m.to_numpy(); n_a, p21_a = len(f8), len_a.sum() / area
def load(tag):
    counts, p21s, lens = [], [], []
    for f in sorted(glob.glob(SP + "\\" + tag + r"\tr_seed*.npz")):
        z = np.load(f); L = z["lengths"]; m = L >= LMIN
        counts.append(int(m.sum())); p21s.append(L[m].sum() / area); lens.append(L[m])
    return np.array(counts), np.array(p21s), lens
c0, p0, l0 = load("face08mc"); c1, p1, l1 = load("face08mc_l05")
fig, axes = plt.subplots(1, 3, figsize=(14, 4.3))
def hist2(ax, v0, v1, obs, xl, bins):
    ax.hist(v0, bins=bins, facecolor="white", edgecolor=MG, hatch="////", lw=0.8, label="재보정 전 (lmin 0)")
    ax.hist(v1, bins=bins, facecolor=LG, edgecolor=DG, lw=0.8, alpha=0.95, label="재보정 후 (lmin 0.5)")
    ax.axvline(obs, color=K, lw=2.6, label="실측 (면 08)")
    ax.set_xlabel(xl); ax.set_ylabel("실현 수 (N = 50)")
    for s in ("top", "right"): ax.spines[s].set_visible(False)
hist2(axes[0], c0, c1, n_a, "절리선 개수 (≥ 0.5 m)", np.arange(85, 245, 8))
hist2(axes[1], p0, p1, p21_a, "P21 [m/m²] (≥ 0.5 m)", np.arange(1.4, 4.1, 0.12))
axes[0].legend(fontsize=9, frameon=False, loc="upper right")
# ③ 길이 CCDF: 실측 vs MC(lmin 0.5) 실현별 곡선(연회색) + 중앙 곡선(진회색)
ax = axes[2]
xg = np.logspace(np.log10(0.5), np.log10(8.0), 60)
def ccdf(L): return np.array([(L >= x).mean() for x in xg])
curves = np.array([ccdf(L) for L in l1])
for cv in curves: ax.plot(xg, cv, color=LG, lw=0.7)
ax.plot(xg, np.median(curves, axis=0), color=DG, lw=2.0, label="MC 중앙값 (lmin 0.5)")
ax.plot(xg, ccdf(len_a), color=K, lw=2.6, label="실측 (면 08)")
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("절리선 길이 ℓ [m]"); ax.set_ylabel("P(길이 ≥ ℓ)")
ax.set_xticks([0.5, 1, 2, 4, 8]); ax.set_xticklabels(["0.5", "1", "2", "4", "8"])
ax.set_yticks([1, 0.1, 0.01]); ax.set_yticklabels(["1", "0.1", "0.01"]); ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax.legend(fontsize=9, frameon=False, loc="lower left")
for s in ("top", "right"): ax.spines[s].set_visible(False)
fig.tight_layout(w_pad=2.0); fig.savefig("fig_blind_stats.png", dpi=200, facecolor="white"); plt.close(fig)
print(f"실측 {n_a}개 P21 {p21_a:.2f} | lmin0: {c0.mean():.0f}±{c0.std():.0f}, P21 {p0.mean():.2f}±{p0.std():.2f} (실측 백분위 {(c0<n_a).mean()*100:.0f}%/{(p0<p21_a).mean()*100:.0f}%) | lmin0.5: {c1.mean():.0f}±{c1.std():.0f}, P21 {p1.mean():.2f}±{p1.std():.2f} (백분위 {(c1<n_a).mean()*100:.0f}%/{(p1<p21_a).mean()*100:.0f}%) | 길이 중앙값 실측 {np.median(len_a):.2f} MC {np.median(np.concatenate(l1)):.2f}")
