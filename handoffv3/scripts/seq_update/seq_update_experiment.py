# -*- coding: utf-8 -*-
"""순차 갱신(sequential Bayesian updating) 시험 — kr 대상.

[배경] 터널은 막장면마다 관측이 순차적으로 들어온다. 현행 파이프라인은 창을
옮길 때마다 처음부터 재역산하고 이전 창의 사후를 쓰지 않는다.

[묻는 것]
  Q1. 면별 우도를 누적한 순차 사후가 현행 배치 재역산과 같은 답을 주는가?
  Q2. 창을 넓힐 때 kr 추정치의 흔들림이 순차 쪽에서 더 작은가?

[방법]
  단일면 f   : 면 f 만으로 estimate_kr → 프로파일 우도 logL_f(kr)
  누적창 1..n: 면 1..n 을 한 번에 estimate_kr → logL_batch,n(kr)   (= 현행)
  순차 사후  : logpost_n(kr) = Σ_{f<=n} logL_f(kr)                (균등 사전)

[주의] 다항우도 loglik = Σ counts·log(prob) 는 counts 에 가법적이지만,
  bin 경계(make_length_edges)와 방향 풀이 관측에서 만들어지므로 면별 실행과
  창 실행의 prob 이 달라진다 → 정확한 가법성은 성립하지 않는다. 그 괴리를 잰다.
  또한 큰 원판은 여러 면에 절리선을 남기므로 면 간 관측은 독립이 아니다.
  순차·배치 **둘 다** 이 독립 가정을 쓰므로 이 시험이 그 문제를 다루지는 않는다.

사용: (handoffv3 에서) python scripts/seq_update/seq_update_experiment.py [면수]
"""
import contextlib
import csv
import importlib.util
import io
import os
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dfn_analysis import estimate_kr

_spec = importlib.util.spec_from_file_location(
    "_conv", ROOT / "scripts/convert_dfm_export_to_trace_dataset.py")
convert = importlib.util.module_from_spec(_spec)
sys.modules["_conv"] = convert
_spec.loader.exec_module(convert)

NFACE = int(sys.argv[1]) if len(sys.argv) > 1 else 7
FACES = [f"{i:02d}" for i in range(1, NFACE + 1)]
DFM = Path(os.environ.get("DFN_DFM_DIR",
                          ROOT.parent / "handoffv2/demodata/DFM_Export/DFM_Export"))
WORK = ROOT / "demo_output" / "_seq"
SETS = ["1", "2", "3"]


def call(fn, argv):
    """모듈 main() 을 같은 프로세스에서 호출(sys.argv 교체 + SystemExit 흡수)."""
    old = sys.argv
    sys.argv = argv
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            fn()
    except SystemExit as e:
        if e.code not in (0, None):
            raise RuntimeError(buf.getvalue()[-800:])
    finally:
        sys.argv = old


def build_and_fit(tag, faces):
    """면 목록으로 trace dataset 을 만들고 kr 프로파일 우도까지 생성한다."""
    out = WORK / tag
    if (out / "kr/kr_profile_likelihood.csv").exists():
        return out
    indir = WORK / "_in" / tag
    shutil.rmtree(indir, ignore_errors=True)
    indir.mkdir(parents=True)
    for f in faces:
        shutil.copytree(DFM / f, indir / f, dirs_exist_ok=True)
    os.chdir(ROOT)
    call(convert.main, ["conv", "--dfm-dir", str(indir), "--outdir", str(out)])
    call(estimate_kr.main,
         ["kr", "--trace-h5", str(out / "trace_dataset/trace_dataset_3d.h5"),
          "--dfn-model", tag, "--generation-rmin", "0.5",
          "--target-set", *SETS, "--outdir", str(out / "kr")])
    shutil.rmtree(indir, ignore_errors=True)   # 입력 사본은 크므로 즉시 정리
    return out


def load_profile(out):
    """set -> (kr 격자, loglik). 적합에 실패한 set 은 빠진다."""
    prof = defaultdict(lambda: ([], []))
    with open(out / "kr/kr_profile_likelihood.csv", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            s = int(r["set_id"])
            prof[s][0].append(float(r["kr_window_mc"]))
            prof[s][1].append(float(r["loglik"]))
    return {s: (np.array(k), np.array(l)) for s, (k, l) in prof.items()}


def ci95(grid, logpost):
    """정규화 후 2.5/97.5 백분위 (격자 기준)."""
    w = logpost - logpost.max()
    p = np.exp(w)
    p /= p.sum()
    c = np.cumsum(p)
    lo = float(grid[np.searchsorted(c, 0.025)])
    hi = float(grid[min(int(np.searchsorted(c, 0.975)), len(grid) - 1)])
    return lo, hi


def main():
    t0 = time.perf_counter()
    print(f"면 {FACES[0]}~{FACES[-1]} / set {SETS}\n작업폴더 {WORK}\n")
    single, batch = {}, {}
    for i, f in enumerate(FACES, 1):
        single[i] = load_profile(build_and_fit(f"single_{f}", [f]))
        batch[i] = load_profile(build_and_fit(f"cum_{i}", FACES[:i]))
        print(f"  적합: 단일면 {f} (set {sorted(single[i])}) / "
              f"누적창 1~{i} (set {sorted(batch[i])})", flush=True)
    print(f"\n전처리+적합 {time.perf_counter() - t0:.1f}s")

    all_sets = sorted({s for p in batch.values() for s in p})
    summary = {}
    for s in all_sets:
        print(f"\n{'=' * 78}\nset {s}")
        print(f"{'n':>2s} {'배치 kr(현행)':>13s} {'순차 kr':>9s} {'차이':>7s} "
              f"{'배치 창간변동':>13s} {'순차 창간변동':>13s} {'순차 95%CI':>18s}")
        acc = grid = prev_b = prev_q = None
        db_list, dq_list = [], []
        for i in range(1, NFACE + 1):
            if s not in batch[i]:
                print(f"{i:2d}   (배치 미적합)")
                continue
            kb, lb = batch[i][s]
            bkr = float(kb[np.argmax(lb)])
            if s in single[i]:
                ks, ls = single[i][s]
                if grid is None:
                    grid, acc = ks, np.zeros_like(ls)
                if len(ks) == len(grid) and np.allclose(ks, grid):
                    acc = acc + ls
            if acc is None:
                print(f"{i:2d} {bkr:13.2f}      (순차 불가: 단일면 적합 없음)")
                prev_b = bkr
                continue
            qkr = float(grid[np.argmax(acc)])
            lo, hi = ci95(grid, acc)
            db = dq = "-"
            if prev_b is not None:
                db_list.append(abs(bkr - prev_b)); db = f"{abs(bkr - prev_b):.2f}"
            if prev_q is not None:
                dq_list.append(abs(qkr - prev_q)); dq = f"{abs(qkr - prev_q):.2f}"
            print(f"{i:2d} {bkr:13.2f} {qkr:9.2f} {qkr - bkr:7.2f} "
                  f"{db:>13s} {dq:>13s} {f'[{lo:.2f}, {hi:.2f}]':>18s}")
            prev_b, prev_q = bkr, qkr
        if db_list and dq_list:
            summary[s] = (float(np.mean(db_list)), float(np.mean(dq_list)))

    if summary:
        print(f"\n{'=' * 78}\n창을 하나 넓힐 때 kr 변동 평균 (작을수록 안정)")
        print(f"{'set':>4s} {'배치(현행)':>12s} {'순차':>10s} {'감소율':>9s}")
        for s, (b, q) in summary.items():
            print(f"{s:4d} {b:12.3f} {q:10.3f} {(1 - q / b) * 100 if b else 0:8.0f}%")


if __name__ == "__main__":
    main()
