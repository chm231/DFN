# -*- coding: utf-8 -*-
"""association 정확도 검증 — 합성 Laxemar(정답 라벨 보유) 기준.

[왜] 2026-09-21 진단에서 '다면 관통을 반지름 정보로 쓰자'는 아이디어가
  association 설정에 따라 신호가 7.1%~0.0% 로 요동해 결론이 불가했다.
  실측 DFM 에는 원판 정답이 없으므로 합성 자료로 설정별 정확도를 잰다.

[정답] fracture_id = 생성된 원판 ID (여러 면의 트레이스가 같은 값을 공유),
       radius_m = 참 반지름.

[지표]
  pair precision = 같은 군집으로 묶은 쌍 중 실제로 같은 원판인 비율 (1-과병합)
  pair recall    = 실제 같은 원판인 쌍 중 같은 군집으로 묶은 비율 (1-과분할)
  다면 관통 원판 수: 정답 vs 추정
  반지름 하한 분포: 정답 vs 추정  (하한 = span/(2 sinφ))

사용: (handoffv3) python scripts/multiface/assoc_validation.py
"""
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from dfn_analysis import reconstruct_discs_from_traces as R

H5 = (ROOT.parent / "_archive/보고서최종_그림코드_2026-08/storage_output_증거물"
      / "pipeline_v2_laxemar/trace_dataset/trace_dataset_3d.h5")

SETTINGS = [
    ("기본 (adaptive, 15deg, 0.15m, cap8)", dict(a=15.0, c=0.15, ad=True, cap=8.0, ms=4.5)),
    ("각도 엄격 10deg",                      dict(a=10.0, c=0.15, ad=True, cap=8.0, ms=4.5)),
    ("공면 엄격 0.08m",                      dict(a=15.0, c=0.08, ad=True, cap=8.0, ms=4.5)),
    ("근접 상한 cap 4.5",                    dict(a=15.0, c=0.15, ad=True, cap=4.5, ms=4.5)),
    ("비적응 max_sep 2.0",                   dict(a=15.0, c=0.15, ad=False, cap=8.0, ms=2.0)),
    ("모두 엄격 (10deg, 0.08m, cap4.5)",     dict(a=10.0, c=0.08, ad=True, cap=4.5, ms=4.5)),
]


def pair_scores(true_lab, pred_lab):
    """군집 교차표로 pair precision/recall 계산 (O(n^2) 루프 없이)."""
    tab = defaultdict(int)
    tsz, psz = defaultdict(int), defaultdict(int)
    for t, p in zip(true_lab, pred_lab):
        tab[(t, p)] += 1
        tsz[t] += 1
        psz[p] += 1
    c2 = lambda n: n * (n - 1) // 2
    tp = sum(c2(v) for v in tab.values())
    pred_pairs = sum(c2(v) for v in psz.values())
    true_pairs = sum(c2(v) for v in tsz.values())
    prec = tp / pred_pairs if pred_pairs else float("nan")
    rec = tp / true_pairs if true_pairs else float("nan")
    return prec, rec, tp, pred_pairs, true_pairs


def rbound(groups, traces):
    """군집별 반지름 하한 = 면 span/(2 sinφ). 단일면은 0."""
    out = []
    for idx in groups.values():
        fx = sorted({round(float(traces[i]["face_x"]), 3) for i in idx})
        if len(fx) == 1:
            out.append(0.0)
            continue
        n = np.mean([traces[i]["normal"] for i in idx], axis=0)
        n = n / np.linalg.norm(n)
        sp = np.sqrt(max(1 - n[0] ** 2, 1e-9))
        out.append((fx[-1] - fx[0]) / (2 * sp))
    return np.array(out)


def main():
    traces = R.load_traces(str(H5))
    with h5py.File(H5) as f:
        fid_all = f["traces/fracture_id"][:].ravel()
        r_true_all = f["traces/radius_m"][:].ravel()
        fx_all = np.round(f["traces/face_x_m"][:].ravel(), 3)
    # load_traces 는 정점<2 인 트레이스를 버릴 수 있으므로 trace_id 로 정렬 대응
    keep = [int(t["trace_id"]) for t in traces] if "trace_id" in traces[0] else list(range(len(traces)))
    fid = fid_all[keep]
    r_true = r_true_all[keep]
    faces = sorted(set(fx_all.tolist()))
    print(f"합성 Laxemar  트레이스 {len(traces):,}  면 {len(faces)}개 x={faces}")
    print(f"참 원판 {len(np.unique(fid)):,}개,  참 반지름 중앙 {np.median(r_true):.2f} m\n")

    # --- 정답(oracle) 군집 ---
    true_groups = defaultdict(list)
    for i, v in enumerate(fid):
        true_groups[int(v)].append(i)
    rb_true = rbound(true_groups, traces)
    nmulti_true = int((rb_true > 0).sum())
    TH = [1.0, 1.5, 2.0, 3.0]
    print(f"  {'설정':34s} {'군집수':>6s} {'다면':>5s} {'pair_prec':>10s} {'pair_rec':>9s} " +
          "".join(f"{'r>='+str(t):>8s}" for t in TH))
    print(f"  {'정답(oracle)':34s} {len(true_groups):6,d} {nmulti_true:5d} {1.0:10.3f} {1.0:9.3f} " +
          "".join(f"{(rb_true>=t).mean()*100:7.1f}%" for t in TH))
    for name, kw in SETTINGS:
        lab = R.associate_agglomerative(traces, kw["a"], kw["c"], kw["ms"],
                                        adaptive_sep=kw["ad"], sep_safety=1.4, sep_cap=kw["cap"])
        g = defaultdict(list)
        for i, l in enumerate(lab):
            g[l].append(i)
        prec, rec, *_ = pair_scores(fid.tolist(), list(lab))
        rb = rbound(g, traces)
        print(f"  {name:34s} {len(g):6,d} {int((rb>0).sum()):5d} {prec:10.3f} {rec:9.3f} " +
              "".join(f"{(rb>=t).mean()*100:7.1f}%" for t in TH))

    print(f"\n  * pair_prec 낮음 = 과병합(다른 원판을 하나로), pair_rec 낮음 = 과분할")
    print(f"  * 정답 다면 관통 원판 {nmulti_true}개 / {len(true_groups)}개 "
          f"= {nmulti_true/len(true_groups)*100:.1f}%")


if __name__ == "__main__":
    main()
