# -*- coding: utf-8 -*-
"""association 파라미터 격자 탐색 — 합성 Laxemar 정답 기준.

지표: pair precision(1-과병합) / recall(1-과분할) / F1,
      그리고 반지름 하한 분포의 정답 대비 오차(다면 관통 신호의 편향).
"""
import sys
from collections import defaultdict
from itertools import product
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from dfn_analysis import reconstruct_discs_from_traces as R

H5 = (ROOT.parent / "_archive/보고서최종_그림코드_2026-08/storage_output_증거물"
      / "pipeline_v2_laxemar/trace_dataset/trace_dataset_3d.h5")
TH = [1.0, 1.5, 2.0]


def pair_prf(true_lab, pred_lab):
    tab, tsz, psz = defaultdict(int), defaultdict(int), defaultdict(int)
    for t, p in zip(true_lab, pred_lab):
        tab[(t, p)] += 1; tsz[t] += 1; psz[p] += 1
    c2 = lambda n: n * (n - 1) // 2
    tp = sum(c2(v) for v in tab.values())
    pp, tpp = sum(c2(v) for v in psz.values()), sum(c2(v) for v in tsz.values())
    pr = tp / pp if pp else 1.0
    rc = tp / tpp if tpp else 1.0
    return pr, rc, (2 * pr * rc / (pr + rc) if pr + rc else 0.0)


def rbound(groups, traces):
    out = []
    for idx in groups.values():
        fx = sorted({round(float(traces[i]["face_x"]), 3) for i in idx})
        if len(fx) == 1:
            out.append(0.0); continue
        n = np.mean([traces[i]["normal"] for i in idx], axis=0); n /= np.linalg.norm(n)
        out.append((fx[-1] - fx[0]) / (2 * np.sqrt(max(1 - n[0] ** 2, 1e-9))))
    return np.array(out)


traces = R.load_traces(str(H5))
with h5py.File(H5) as f:
    fid = f["traces/fracture_id"][:].ravel()
assert len(fid) == len(traces), f"트레이스 수 불일치 {len(fid)} vs {len(traces)}"
fid = fid.tolist()   # load_traces 가 하나도 버리지 않아 인덱스가 그대로 대응한다
tg = defaultdict(list)
for i, v in enumerate(fid):
    tg[int(v)].append(i)
rb0 = rbound(tg, traces)
base = np.array([(rb0 >= t).mean() * 100 for t in TH])

rows = []
for ang, cop, cap, ad in product([8.0, 10.0, 12.0, 15.0], [0.05, 0.08, 0.10, 0.15],
                                 [3.0, 4.5, 6.0, 8.0], [True, False]):
    lab = R.associate_agglomerative(traces, ang, cop, cap, adaptive_sep=ad,
                                    sep_safety=1.4, sep_cap=cap)
    g = defaultdict(list)
    for i, l in enumerate(lab):
        g[l].append(i)
    pr, rc, f1 = pair_prf(fid, list(lab))
    rb = rbound(g, traces)
    v = np.array([(rb >= t).mean() * 100 for t in TH])
    rows.append((f1, pr, rc, ang, cop, cap, ad, len(g), int((rb > 0).sum()),
                 v, float(np.abs(v - base).mean())))

rows.sort(key=lambda r: -r[0])
print(f"정답: 원판 {len(tg)}개, 다면 {int((rb0>0).sum())}개, "
      f"r>=1.0/1.5/2.0 = {base[0]:.1f}/{base[1]:.1f}/{base[2]:.1f}%\n")
print(f"  {'F1':>6s} {'prec':>6s} {'rec':>6s} | {'각도':>4s} {'공면':>5s} {'cap':>4s} {'적응':>4s} | "
      f"{'군집':>5s} {'다면':>4s} | {'r>=1.0':>7s} {'r>=1.5':>7s} {'r>=2.0':>7s} {'하한오차':>7s}")
print("  " + "-" * 104)
for r in rows[:10]:
    f1, pr, rc, ang, cop, cap, ad, ng, nm, v, err = r
    print(f"  {f1:6.3f} {pr:6.3f} {rc:6.3f} | {ang:4.0f} {cop:5.2f} {cap:4.1f} {'O' if ad else 'X':>4s} | "
          f"{ng:5d} {nm:4d} | {v[0]:6.1f}% {v[1]:6.1f}% {v[2]:6.1f}% {err:6.2f}p")
print("  ...")
cur = [r for r in rows if abs(r[3]-15) < .1 and abs(r[4]-0.15) < .001 and abs(r[5]-8.0) < .1 and r[6]]
for r in cur:
    f1, pr, rc, ang, cop, cap, ad, ng, nm, v, err = r
    print(f"  {f1:6.3f} {pr:6.3f} {rc:6.3f} | {ang:4.0f} {cop:5.2f} {cap:4.1f} {'O':>4s} | "
          f"{ng:5d} {nm:4d} | {v[0]:6.1f}% {v[1]:6.1f}% {v[2]:6.1f}% {err:6.2f}p   <== 현행 기본 (순위 {rows.index(r)+1}/{len(rows)})")
best_err = min(rows, key=lambda r: r[10])
f1, pr, rc, ang, cop, cap, ad, ng, nm, v, err = best_err
print(f"\n  하한오차 최소: 각도 {ang:.0f} 공면 {cop:.2f} cap {cap:.1f} 적응 {'O' if ad else 'X'} "
      f"→ F1 {f1:.3f}, 오차 {err:.2f}p")
