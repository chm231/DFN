# -*- coding: utf-8 -*-
"""몬테카를로 실현(시드 12개) 블록 통계 집계: 평균±표준편차, CCDF 밴드 그림."""
import csv
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "Malgun Gothic"
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt

SC = Path(__file__).parent
BL = SC / "blocks"
OUT = SC / "block_figs"

# 기존 3개 + MC 9개 = 12개 실현
RUNS = [("seed2026", 2026), ("seed7", 7), ("seed123", 123)] + \
       [(f"mc_seed{s}", s) for s in [101, 202, 303, 404, 505, 606, 707, 808, 909]]

rows_all = {}
stats = []
for tag, seed in RUNS:
    path = BL / f"{tag}_full_blocks.csv"
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    vols = np.array([float(r["volume_m3"]) for r in rows])
    rows_all[seed] = vols
    stats.append(dict(seed=seed, n=len(vols), total=float(vols.sum()),
                      vmax=float(vols.max()), med=float(np.median(vols)),
                      p90=float(np.percentile(vols, 90))))

n_arr = np.array([s["n"] for s in stats], dtype=float)
t_arr = np.array([s["total"] for s in stats])
m_arr = np.array([s["vmax"] for s in stats])

print(f"{'seed':>6} {'블록수':>6} {'총부피':>8} {'최대':>8} {'p90':>8}")
for s in stats:
    print(f"{s['seed']:>6} {s['n']:>6d} {s['total']:>8.2f} {s['vmax']:>8.3f} {s['p90']:>8.4f}")
print()
print(f"N = {len(stats)}개 실현")
print(f"블록 수   : {n_arr.mean():.1f} ± {n_arr.std(ddof=1):.1f}  (CV {n_arr.std(ddof=1)/n_arr.mean()*100:.0f}%)  범위 [{n_arr.min():.0f}, {n_arr.max():.0f}]")
print(f"총부피 m³ : {t_arr.mean():.2f} ± {t_arr.std(ddof=1):.2f}  (CV {t_arr.std(ddof=1)/t_arr.mean()*100:.0f}%)  범위 [{t_arr.min():.2f}, {t_arr.max():.2f}]")
print(f"최대블록  : {m_arr.mean():.2f} ± {m_arr.std(ddof=1):.2f}  (CV {m_arr.std(ddof=1)/m_arr.mean()*100:.0f}%)  범위 [{m_arr.min():.2f}, {m_arr.max():.2f}]")

# CCDF 밴드 그림
fig, ax = plt.subplots(figsize=(7.5, 4.6))
grid_v = np.logspace(np.log10(0.008), np.log10(3.0), 200)
ccdfs = []
for seed, vols in rows_all.items():
    ccdf = np.array([(vols >= v).sum() for v in grid_v], dtype=float)
    ccdfs.append(ccdf)
    ax.step(grid_v, ccdf, color="#2b6777", alpha=0.25, lw=1)
C = np.array(ccdfs)
ax.plot(grid_v, C.mean(axis=0), color="#b5762a", lw=2.2, label=f"평균 (N={len(stats)})")
ax.fill_between(grid_v, np.percentile(C, 5, axis=0), np.percentile(C, 95, axis=0),
                color="#2b6777", alpha=0.15, label="5–95 백분위")
ax.set_xscale("log")
from matplotlib.ticker import FuncFormatter, LogLocator
ax.xaxis.set_major_locator(LogLocator(base=10))
ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
ax.set_xlabel("블록 부피 V [m³]")
ax.set_ylabel("V 이상 블록 수")
ax.set_title(f"블록 부피 CCDF — 몬테카를로 {len(stats)}개 실현 (voxel 0.1 m, 6-conn)")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "block_ccdf_mc_band.png", dpi=150)
print(f"[fig] {OUT/'block_ccdf_mc_band.png'}")

with open(OUT / "mc_stats.json", "w", encoding="utf-8") as f:
    json.dump(dict(runs=stats,
                   n_mean=n_arr.mean(), n_std=n_arr.std(ddof=1),
                   total_mean=t_arr.mean(), total_std=t_arr.std(ddof=1),
                   vmax_mean=m_arr.mean(), vmax_std=m_arr.std(ddof=1)), f, indent=2)
print(f"[out] {OUT/'mc_stats.json'}")
