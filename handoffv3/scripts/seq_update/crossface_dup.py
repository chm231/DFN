# -*- coding: utf-8 -*-
"""면 간 관측 상관(같은 원판이 여러 면에 남긴 절리선)이 kr 추정에 주는 편향 측정.

[문제] 우도는 loglik = Σ_trace log p(trace|kr) 로 모든 절리선을 독립 표본으로 본다.
  전방 모형(simulate_window_from_true_chords)은 '원판 1개 × 창 1개 → 가시길이 1개' 를
  모사한다. 그런데 관측은 7개 면을 풀링하므로, 큰 원판일수록 여러 면에 절리선을 남겨
  표본에 여러 번 들어간다. 반지름에 비례하는 추가 크기편향 → kr 과소추정(꼬리 과대) 의심.

[분리] 중복은 두 종류다.
  (a) 같은 면 조각  : 한 원판이 한 면에서 여러 토막으로 매핑된 것 (매핑 artifact)
  (b) 면 간 중복    : 한 원판이 여러 면을 관통한 것 (모형 가정 위반)

[방법] 복원 association(agglomerative) 으로 절리선→원판 군집을 얻고 세 데이터셋 비교:
  full          : 원본
  dedup_face    : (군집, 면) 당 1개만 남김        → (a) 제거
  dedup_cluster : 군집당 1개만 남김(무작위/최장) → (a)+(b) 제거

사용: (handoffv3 에서) python scripts/seq_update/crossface_dup.py
"""
import contextlib, csv, io, os, shutil, sys
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from dfn_analysis import estimate_kr
from dfn_analysis import reconstruct_discs_from_traces as R

SRC = ROOT / "demo_output/_seq2/cum_7/trace_dataset/trace_dataset_3d.h5"
WORK = ROOT / "demo_output/_dup"
SETS = ["1", "2", "3"]
BIN_UPPER = "12.0"
RNG = np.random.default_rng(2026)


def call(fn, argv):
    old = sys.argv; sys.argv = argv; buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            fn()
    except SystemExit as e:
        if e.code not in (0, None):
            raise RuntimeError(buf.getvalue()[-800:])
    finally:
        sys.argv = old


def subset_h5(src, dst, keep_idx):
    """per-trace 데이터셋을 keep_idx 로 자르고 폴리라인 인덱스를 재구성한다."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    keep = np.asarray(sorted(keep_idx))
    with h5py.File(src, "r") as f, h5py.File(dst, "w") as g:
        n = len(f["traces/set_id"])
        starts = f["traces/polyline_vertex_start"][...]
        counts = f["traces/polyline_vertex_count"][...]
        verts = f["traces/polyline_vertices_xyz"][...]
        gt = g.create_group("traces")
        for k, v in f["traces"].items():
            if k in ("polyline_vertices_xyz", "polyline_vertex_start", "polyline_vertex_count"):
                continue
            gt.create_dataset(k, data=v[...][keep] if v.shape[0] == n else v[...])
        nv, ns, nc = [], [], []
        off = 0
        for i in keep:
            c = int(counts[i]); s = int(starts[i])
            nv.append(verts[s:s + c]); ns.append(off); nc.append(c); off += c
        gt.create_dataset("polyline_vertices_xyz", data=np.vstack(nv))
        gt.create_dataset("polyline_vertex_start", data=np.array(ns, dtype=starts.dtype))
        gt.create_dataset("polyline_vertex_count", data=np.array(nc, dtype=counts.dtype))
        f.copy("meta", g)


def fit(tag, h5path):
    out = WORK / tag / "kr"
    if not (out / "kr_summary_by_set.csv").exists():
        os.chdir(ROOT)
        call(estimate_kr.main,
             ["kr", "--trace-h5", str(h5path), "--dfn-model", tag,
              "--generation-rmin", "0.5", "--target-set", *SETS,
              "--length-bin-upper", BIN_UPPER, "--outdir", str(out)])
    res = {}
    with open(out / "kr_summary_by_set.csv", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            res[int(r["set_id"])] = (float(r["kr_hat"]), int(r["n_used"]))
    return res


def ci(tag):
    """프로파일 우도 Δ<=2 구간 폭 (식별력 지표)."""
    prof = defaultdict(lambda: ([], []))
    with open(WORK / tag / "kr/kr_profile_likelihood.csv", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            s = int(r["set_id"])
            prof[s][0].append(float(r["kr_window_mc"])); prof[s][1].append(float(r["loglik"]))
    out = {}
    for s, (k, l) in prof.items():
        k, l = np.array(k), np.array(l)
        inside = k[(l.max() - l) <= 2.0]
        out[s] = float(inside.max() - inside.min()) if len(inside) else float("nan")
    return out


def main():
    shutil.rmtree(WORK, ignore_errors=True)
    traces = R.load_traces(str(SRC))
    labels = R.associate_agglomerative(traces, 15.0, 0.15, 4.5, adaptive_sep=True,
                                       sep_safety=1.4, sep_cap=8.0)
    by_cluster = defaultdict(list)
    for i, lab in enumerate(labels):
        by_cluster[lab].append(i)

    # --- 중복 구조 진단 ---
    nfaces = [len({round(float(traces[i]["face_x"]), 3) for i in idx}) for idx in by_cluster.values()]
    ntr = [len(idx) for idx in by_cluster.values()]
    nfaces, ntr = np.array(nfaces), np.array(ntr)
    print(f"절리선 {len(traces):,}개 → 복원 원판(군집) {len(by_cluster):,}개\n")
    print(f"  {'관통 면 수':>10s} {'원판 수':>8s} {'그 원판들의 절리선 수':>18s}")
    for k in range(1, nfaces.max() + 1):
        m = nfaces == k
        if m.sum():
            print(f"  {k:10d} {int(m.sum()):8,d} {int(ntr[m].sum()):18,d}")
    multi = nfaces >= 2
    print(f"\n  다면 관통 원판 {int(multi.sum()):,}개 ({multi.mean()*100:.1f}%) 가 "
          f"절리선 {int(ntr[multi].sum()):,}개 ({ntr[multi].sum()/len(traces)*100:.1f}%) 를 차지")

    # --- 세 데이터셋 구성 ---
    keep_face, keep_rand, keep_long = [], [], []
    for idx in by_cluster.values():
        per_face = defaultdict(list)
        for i in idx:
            per_face[round(float(traces[i]["face_x"]), 3)].append(i)
        for _, ii in per_face.items():                       # (군집, 면) 당 1개
            keep_face.append(max(ii, key=lambda j: traces[j]["chord"]))
        keep_rand.append(int(RNG.choice(idx)))               # 군집당 무작위 1개
        keep_long.append(max(idx, key=lambda j: traces[j]["chord"]))

    datasets = {"full": list(range(len(traces))), "dedup_face": keep_face,
                "dedup_cluster_rand": keep_rand, "dedup_cluster_long": keep_long}
    res, cis = {}, {}
    for tag, keep in datasets.items():
        h5p = WORK / tag / "traces.h5"
        subset_h5(SRC, h5p, keep) if tag != "full" else shutil.copy(SRC, _mk(h5p))
        res[tag] = fit(tag, h5p)
        cis[tag] = ci(tag)
        print(f"  적합 완료: {tag} (절리선 {len(keep):,})", flush=True)

    print(f"\n{'='*78}\n{'데이터셋':22s} {'절리선':>8s} " +
          " ".join(f"{'set'+str(s)+' kr':>10s}" for s in (1, 2, 3)) +
          " " + " ".join(f"{'CI폭'+str(s):>8s}" for s in (1, 2, 3)))
    for tag, keep in datasets.items():
        kr = " ".join(f"{res[tag].get(s,(float('nan'),0))[0]:10.2f}" for s in (1, 2, 3))
        cw = " ".join(f"{cis[tag].get(s,float('nan')):8.2f}" for s in (1, 2, 3))
        print(f"{tag:22s} {len(keep):8,d} {kr} {cw}")


def _mk(p):
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


if __name__ == "__main__":
    main()
