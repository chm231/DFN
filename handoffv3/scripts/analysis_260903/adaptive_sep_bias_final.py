# -*- coding: utf-8 -*-
"""방향 편향 그림 최종판: 이전(고정 게이트 3.5 m) vs 최종(적응 게이트) vs 모집단. 계수 표기 없음."""
import sys, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np
sys.path.insert(0, "c:/Users/user/OneDrive/2026-1/3D DFN modeling/handoffv2")
import dfn_analysis.reconstruct_discs_from_traces as R
plt.rcParams["font.family"] = "Malgun Gothic"; plt.rcParams["axes.unicode_minus"] = False
H5 = "c:/Users/user/OneDrive/2026-1/3D DFN modeling/handoffv2/demo_output/win_f01-06_l05/trace_dataset/trace_dataset_3d.h5"
OUTS = ["c:/Users/user/OneDrive/2026-1/3D DFN modeling/docs/figures/adaptive_sep_direction_bias_final.png",
        "c:/Users/user/OneDrive/2026-2/현대건설/그림_적응게이트_방향편향.png"]
traces = [t for t in R.load_traces(H5) if t["set_id"] in {1, 2, 3}]
pop_nx = np.array([abs(float(t["normal"][0])) for t in traces])
def multi_nx(labels):
    cl = {}
    for t, lb in zip(traces, labels): cl.setdefault(lb, []).append(t)
    out = []
    for m in cl.values():
        if len({round(float(x["face_x"]), 3) for x in m}) < 2: continue
        n, _, _ = R._fit_plane_svd(np.vstack([t["verts"] for t in m])); out.append(abs(float(n[0])))
    return np.array(out)
nx_prev = multi_nx(R.associate_agglomerative(traces, 15.0, 0.15, 3.5))
nx_final = multi_nx(R.associate_agglomerative(traces, 15.0, 0.15, 4.5, adaptive_sep=True))
fig, ax = plt.subplots(1, 2, figsize=(13, 5)); bins = np.linspace(0, 1, 21)
a = ax[0]
a.hist(pop_nx, bins=bins, density=True, alpha=0.35, color="gray", label=f"전체 절리선 모집단 (n={len(pop_nx)})")
a.hist(nx_prev, bins=bins, density=True, histtype="step", lw=2.2, color="#2c6fb3", label=f"이전: 고정 게이트 3.5 m (다면 원판 n={len(nx_prev)})")
a.hist(nx_final, bins=bins, density=True, histtype="step", lw=2.2, color="#c0392b", label=f"최종: 적응 게이트 (다면 원판 n={len(nx_final)})")
a.axvline(0.8, color="green", ls=":", lw=1); a.text(0.8, a.get_ylim()[1]*0.9, " 막장면 평행", color="green", fontsize=9)
a.set_xlabel("|nx| (막장면 평행도: 0 = 수직, 1 = 평행)"); a.set_ylabel("정규화 밀도")
a.set_title("(a) 다면 원판의 방향 분포 vs 모집단\n(모집단에 가까울수록 방향 편향이 적음)"); a.legend(fontsize=9); a.grid(alpha=0.3)
a = ax[1]; thr = np.linspace(0.5, 0.95, 30)
for arr, c, lb in [(pop_nx, "gray", "모집단"), (nx_prev, "#2c6fb3", "이전: 고정 게이트 3.5 m"), (nx_final, "#c0392b", "최종: 적응 게이트")]:
    a.plot(thr, [(arr >= t).mean() for t in thr], color=c, lw=2, label=lb)
a.set_xlabel("|nx| 임계"); a.set_ylabel("임계 이상 원판 비율"); a.set_title("(b) 막장면 평행 원판 비율\n(최종이 모집단에 근접 = 편향 감소)"); a.legend(fontsize=9); a.grid(alpha=0.3)
fig.suptitle("적응 게이트의 방향 편향 감소 (1~6면, 절리군 1~3)", y=1.02, fontsize=13); fig.tight_layout()
for o in OUTS: fig.savefig(o, dpi=140, bbox_inches="tight")
print(f"모집단 평균|nx| {pop_nx.mean():.3f} | 이전(3.5) 다면 {len(nx_prev)}개 평균|nx| {nx_prev.mean():.3f} | 최종(적응) 다면 {len(nx_final)}개 평균|nx| {nx_final.mean():.3f}")
print(f"막장면평행(|nx|>=0.8) 비율: 모집단 {(pop_nx>=0.8).mean()*100:.1f}% / 이전 3.5: {(nx_prev>=0.8).mean()*100:.1f}% ({(nx_prev>=0.8).sum()}개) / 최종 적응: {(nx_final>=0.8).mean()*100:.1f}% ({(nx_final>=0.8).sum()}개)")
