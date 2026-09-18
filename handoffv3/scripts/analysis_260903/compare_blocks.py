# -*- coding: utf-8 -*-
"""시드별 블록 판정 결과 비교: 통계 + 교차 시드 매칭 + 그림 생성."""
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
OUT.mkdir(exist_ok=True)

TAGS = ["seed2026", "seed7", "seed123", "seed2026_rmax10"]
MODES = ["full", "top30"]


def load(tag, mode):
    with open(BL / f"{tag}_{mode}_summary.json", encoding="utf-8") as f:
        summ = json.load(f)
    rows = list(csv.DictReader(open(BL / f"{tag}_{mode}_blocks.csv", encoding="utf-8")))
    vols = np.array([float(r["volume_m3"]) for r in rows])
    cents = np.array([[float(r["centroid_x"]), float(r["centroid_y"]), float(r["centroid_z"])]
                      for r in rows]) if rows else np.zeros((0, 3))
    return summ, vols, cents


def block_voxel_sets(tag, mode):
    z = np.load(BL / f"{tag}_{mode}_labels.npz")
    lab = z["labels"]
    return lab


print("=" * 90)
print(f"{'run':22s} {'모드':6s} {'블록수':>6s} {'총부피':>10s} {'최대':>8s} {'중앙값':>8s} {'p90':>8s}")
data = {}
for tag in TAGS:
    for mode in MODES:
        summ, vols, cents = load(tag, mode)
        data[(tag, mode)] = (summ, vols, cents)
        p90 = np.percentile(vols, 90) if len(vols) else 0
        med = np.median(vols) if len(vols) else 0
        print(f"{tag:22s} {mode:6s} {len(vols):6d} {vols.sum():10.2f} "
              f"{(vols.max() if len(vols) else 0):8.3f} {med:8.4f} {p90:8.4f}")

print()
print("=== 교차 시드 블록 매칭 (중심거리 0.5 m 이내, 부피비 0.5~2배) ===")
SEEDS = ["seed2026", "seed7", "seed123"]
for mode in MODES:
    for i in range(len(SEEDS)):
        for j in range(i + 1, len(SEEDS)):
            a, b = SEEDS[i], SEEDS[j]
            _, va, ca = data[(a, mode)]
            _, vb, cb = data[(b, mode)]
            if len(va) == 0 or len(vb) == 0:
                print(f"[{mode}] {a} vs {b}: 블록 없음")
                continue
            d = np.linalg.norm(ca[:, None, :] - cb[None, :, :], axis=2)
            jmin = d.argmin(axis=1)
            dmin = d.min(axis=1)
            ratio = va / vb[jmin]
            matched = (dmin < 0.5) & (ratio > 0.5) & (ratio < 2.0)
            print(f"[{mode}] {a}({len(va)}) vs {b}({len(vb)}): 매칭 {int(matched.sum())}개 "
                  f"({matched.mean()*100:.0f}%)")

print()
print("=== 복셀 수준 Jaccard (전체 블록 마스크; full 모드) ===")
masks = {t: block_voxel_sets(t, "full") > 0 for t in SEEDS}
for i in range(len(SEEDS)):
    for j in range(i + 1, len(SEEDS)):
        a, b = SEEDS[i], SEEDS[j]
        inter = np.logical_and(masks[a], masks[b]).sum()
        union = np.logical_or(masks[a], masks[b]).sum()
        print(f"{a} vs {b}: Jaccard = {inter/union:.3f}  (교집합 {inter:,} / 합집합 {union:,} 복셀)")

# ---- 그림 1: 블록 부피 ECDF (full, 시드 3종) ----
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
for mode, ax in zip(MODES, axes):
    for tag, colr in zip(SEEDS, ["#1f77b4", "#d62728", "#2ca02c"]):
        _, vols, _ = data[(tag, mode)]
        if len(vols) == 0:
            continue
        v = np.sort(vols)
        ax.step(v, np.arange(1, len(v) + 1)[::-1], where="post", label=tag, color=colr)
    ax.set_xscale("log")
    ax.set_xlabel("block volume [m³]")
    ax.set_ylabel("N blocks ≥ V")
    ax.set_title(f"{mode} DFN")
    ax.legend()
    ax.grid(alpha=0.3)
fig.suptitle("Block volume complementary CDF by seed (voxel 0.1 m, 6-connectivity)")
fig.tight_layout()
fig.savefig(OUT / "block_volume_ccdf_by_seed.png", dpi=150)
print(f"\n[fig] {OUT/'block_volume_ccdf_by_seed.png'}")

# ---- 그림 2: 요약 막대 (블록 수 / 총부피) ----
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
labels4 = TAGS
x = np.arange(len(labels4))
w = 0.35
for k, (metric, title) in enumerate([("n", "블록 수"), ("v", "총 블록 부피 [m³]")]):
    ax = axes[k]
    for m_i, mode in enumerate(MODES):
        ys = []
        for tag in labels4:
            _, vols, _ = data[(tag, mode)]
            ys.append(len(vols) if metric == "n" else vols.sum())
        ax.bar(x + (m_i - 0.5) * w, ys, w, label=mode)
    ax.set_xticks(x)
    ax.set_xticklabels(labels4, rotation=15)
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "block_counts_by_seed.png", dpi=150)
print(f"[fig] {OUT/'block_counts_by_seed.png'}")

# ---- 저장: 비교 요약 JSON ----
comp = {}
for (tag, mode), (summ, vols, cents) in data.items():
    comp[f"{tag}_{mode}"] = dict(n_blocks=len(vols), total_v=float(vols.sum()),
                                 max_v=float(vols.max()) if len(vols) else 0.0,
                                 top10=[float(v) for v in vols[:10]])
with open(OUT / "comparison_summary.json", "w", encoding="utf-8") as f:
    json.dump(comp, f, ensure_ascii=False, indent=2)
print(f"[out] {OUT/'comparison_summary.json'}")
