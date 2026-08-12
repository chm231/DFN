# -*- coding: utf-8 -*-
"""DFN 역산 파이프라인 데모 러너.

하나의 명령으로 다음을 순서대로 실행한다 (보고서 제6장 그림 6-1의 흐름과 동일):

  [1] 합성 DFN 생성 (시드 지정, 참값 파라미터)     dfn generator v1/python/generate_dfn.py
  [2] 평면 막장면 교차 -> 2차원 절리선 데이터셋     dfn_analysis.export_flat_face_traces
  [3] 반지름 지수 k_r 역산 (하이브리드 우도)        dfn_analysis.estimate_kr
  [4] 절리군별 방향/kappa 정리 -> dataset config    dfn_analysis.build_dataset_config_from_traces
  [5] P32 역산 (해석식 C = E[sin phi])              dfn_analysis.estimate_p32_mc_calibrated
  [6] 관측 균열 복원 (원판)                         dfn_analysis.reconstruct_discs_from_traces
  [7] 조건부 암반균열망 생성 (관측 모순 균열 제거)  dfn_analysis.generate_conditional_hidden_dfn
  [8] 안정성 해석 입력 JSON 추출                    dfn_analysis.export_domain_dfn_json
  [9] 3차원 시각화 (pyvista, 선택)                  dfn_analysis.plot_domain_dfn_3d

사용 예:
    python run_demo.py --seed 2026
    python run_demo.py --seed 2026 --n-bootstrap 0 --skip-3d   (빠른 실행)

참값(벤치마크 파라미터)은 마지막 비교표에서만 사용하며, 역산 과정에는 입력되지 않는다.
좌표계: x = East(터널 진행축), y = North, z = Up. 막장면 = x = 상수 평면.
"""
import argparse
import csv
import os
import subprocess
import sys
import time

# cp949 콘솔에서도 한글/특수문자 출력이 죽지 않도록 stdout/stderr 를 utf-8 로 강제
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
SETS = ["1", "2", "3", "5"]  # 멱법칙 반지름 분포 절리군 (Set 4 는 지수분포라 k_r/P32 제외)


def banner(idx, total, title):
    print()
    print("=" * 72)
    print(f" [{idx}/{total}] {title}")
    print("=" * 72, flush=True)


def run(cmd, env, allow_fail=False):
    print("  $ " + " ".join(f'"{c}"' if " " in str(c) else str(c) for c in cmd), flush=True)
    r = subprocess.run([str(c) for c in cmd], cwd=HERE, env=env)
    if r.returncode != 0 and not allow_fail:
        sys.exit(f"\n[중단] 위 단계가 실패했습니다 (exit {r.returncode}).")
    return r.returncode


def read_csv_rows(path):
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def fill_reference_p32(p32_csv, site):
    """P32_reference 가 비어 있으면 벤치마크 참값 k_r 기반 유효 P32 로 채운다.

    --config 로 절리군 프리셋이 역산값으로 대체되면 estimate_p32 단계가 참조 P32 를
    계산하지 못한다(의도된 동작 — 역산에는 참값을 쓰지 않는다). 비교표 표시용으로만
    여기서 사후 계산한다.
    """
    sys.path.insert(0, HERE)
    try:
        from dfn_analysis.estimate_kr import SITE_TO_KR_TRUE
        from dfn_analysis.build_p32_pilot_summary import SITE_SET_CONFIG, support_scaled_p32
    except ImportError:
        return
    rows = read_csv_rows(p32_csv)
    changed = False
    for r in rows:
        try:
            ref = float(r.get("P32_reference", "nan"))
        except (TypeError, ValueError):
            ref = float("nan")
        if ref == ref:  # 이미 유효
            continue
        sid = int(r["set_id"])
        kr_true = SITE_TO_KR_TRUE.get(site, {}).get(sid)
        cfg_set = SITE_SET_CONFIG.get(site, {}).get(sid)
        if kr_true is None or cfg_set is None:
            continue
        r0 = float(cfg_set.get("r0", float("nan")))
        ref = support_scaled_p32(site, sid, kr_true, max(0.5, r0), 250.0)
        if ref != ref:
            continue
        hat = float(r["P32_hat"])
        r["P32_reference"] = f"{ref:.6f}"
        r["P32_abs_error"] = f"{abs(hat - ref):.6f}"
        r["P32_relative_error_percent"] = f"{100.0 * (hat - ref) / ref:.3f}"
        changed = True
    if changed:
        with open(p32_csv, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print("  [i] 참조 P32(벤치마크 참값 기반)를 비교표용으로 채웠습니다.")


def main():
    ap = argparse.ArgumentParser(description="DFN 역산 파이프라인 데모 러너",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--seed", type=int, default=2026, help="DFN 생성 난수 시드")
    ap.add_argument("--site", default="laxemar", choices=["laxemar", "forsmark"],
                    help="벤치마크 사이트 (참값 파라미터 세트)")
    ap.add_argument("--face-x", nargs="+", type=float, default=[0.0, 1.0, 2.0, 3.0],
                    help="관측 막장면 x 위치 [m]")
    ap.add_argument("--outdir", default=None,
                    help="출력 폴더 (기본: demo_output/seed<seed>)")
    ap.add_argument("--n-bootstrap", type=int, default=100,
                    help="k_r bootstrap 반복 수 (0 = 신뢰구간 생략, 실행 단축)")
    ap.add_argument("--rmax-local", type=float, default=25.0,
                    help="조건부 생성의 국소 반지름 상한 [m]")
    ap.add_argument("--lmin-det", type=float, default=0.5,
                    help="검출 하한 [m] (조건부 생성의 관측 모순 판정에 사용)")
    ap.add_argument("--skip-3d", action="store_true", help="3차원 시각화 생략")
    ap.add_argument("--p32-fisher", choices=["preset", "estimated"], default="preset",
                    help="P32 환산계수 C=E[sin phi] 의 방향분포 소스. "
                         "preset = 사이트 방향 모델 표(보고서 표 6-5 수치 재현), "
                         "estimated = 이번 실행에서 역산한 방향분포(자기완결)")
    args = ap.parse_args()

    base = os.path.abspath(args.outdir or os.path.join(HERE, "demo_output", f"seed{args.seed}"))
    os.makedirs(base, exist_ok=True)

    env = dict(os.environ, PYTHONPATH=HERE, PYTHONIOENCODING="utf-8")

    dfn_h5 = os.path.join(base, "dfn_export_for_python.h5")
    tr_dir = os.path.join(base, "trace_dataset")
    tr_h5 = os.path.join(tr_dir, "trace_dataset_3d.h5")
    kr_dir = os.path.join(base, "kr")
    kr_sum = os.path.join(kr_dir, "kr_summary_by_set.csv")
    cfg = os.path.join(base, "dataset_config.json")
    p32_dir = os.path.join(base, "p32")
    p32_csv = os.path.join(p32_dir, "p32_summary.csv")
    rec_csv = os.path.join(base, "reconstruct", "reconstructed_discs.csv")

    total = 9
    t0 = time.time()
    times = {}

    banner(1, total, f"합성 DFN 생성  (site={args.site}, seed={args.seed})")
    t = time.time()
    run([PY, os.path.join(HERE, "dfn generator v1", "python", "generate_dfn.py"),
         "--site", args.site, "--seed", args.seed, "--output-dir", base], env)
    times["생성"] = time.time() - t

    banner(2, total, f"평면 막장면 교차 -> 절리선 데이터셋  (막장면 x = {args.face_x})")
    t = time.time()
    run([PY, "-m", "dfn_analysis.export_flat_face_traces",
         "--dfn-h5", dfn_h5, "--face-x", *[f"{x:g}" for x in args.face_x],
         "--outdir", tr_dir], env)
    times["절리선"] = time.time() - t

    banner(3, total, "반지름 지수 k_r 역산  (하이브리드 우도, lmin_fit = 0.5 m 고정)")
    t = time.time()
    cmd = [PY, "-m", "dfn_analysis.estimate_kr",
           "--trace-h5", tr_h5, "--site", args.site, "--target-set", *SETS,
           "--outdir", kr_dir]
    if args.n_bootstrap > 0:
        cmd += ["--run-bootstrap", "--n-bootstrap", args.n_bootstrap]
    run(cmd, env)
    times["k_r"] = time.time() - t

    banner(4, total, "절리군별 방향/kappa 정리 -> dataset config")
    t = time.time()
    run([PY, "-m", "dfn_analysis.build_dataset_config_from_traces",
         "--trace-h5", tr_h5, "--kr-summary-csv", kr_sum,
         "--dataset-name", args.site, "--target-set", *SETS, "--out", cfg], env)
    times["config"] = time.time() - t

    banner(5, total, f"P32 역산  (해석식 환산계수 C = E[sin phi], P21 전량 집계, "
                     f"방향분포={args.p32_fisher})")
    t = time.time()
    os.makedirs(p32_dir, exist_ok=True)
    p32_cmd = [PY, "-m", "dfn_analysis.estimate_p32_mc_calibrated",
               "--trace-h5", tr_h5, "--site", args.site,
               "--target-set", *SETS, "--kr-summary-csv", kr_sum, "--bootstrap-csv", kr_sum,
               "--dfn-h5", dfn_h5, "--outcsv", p32_csv]
    if args.p32_fisher == "estimated":
        # config 를 넘기면 사이트 프리셋이 역산된 방향분포로 대체된다 (자기완결 역산)
        p32_cmd += ["--config", cfg]
    run(p32_cmd, env)
    fill_reference_p32(p32_csv, args.site)
    times["P32"] = time.time() - t

    banner(6, total, "관측 균열 복원  (다중 막장면 연결 + 원판 복원)")
    t = time.time()
    os.makedirs(os.path.dirname(rec_csv), exist_ok=True)
    run([PY, "-m", "dfn_analysis.reconstruct_discs_from_traces",
         "--trace-h5", tr_h5, "--kr-summary-csv", kr_sum, "--out-csv", rec_csv], env)
    times["복원"] = time.time() - t

    banner(7, total, f"조건부 암반균열망 생성  (rmax_local={args.rmax_local:g} m, "
                     f"lmin_det={args.lmin_det:g} m)")
    t = time.time()
    run([PY, "-m", "dfn_analysis.generate_conditional_hidden_dfn",
         "--pipeline-dir", base, "--rmax-local", args.rmax_local,
         "--lmin-det", args.lmin_det, "--seed", args.seed], env)
    times["조건부"] = time.time() - t

    banner(8, total, "안정성 해석 입력 JSON 추출")
    t = time.time()
    run([PY, "-m", "dfn_analysis.export_domain_dfn_json",
         "--pipeline-dir", base, "--rmax-local", args.rmax_local,
         "--lmin-det", args.lmin_det, "--seed", args.seed], env)
    times["JSON"] = time.time() - t

    export_dir = os.path.join(base, "export")
    json_files = sorted(
        (os.path.join(export_dir, f) for f in os.listdir(export_dir) if f.endswith(".json")),
        key=os.path.getmtime) if os.path.isdir(export_dir) else []
    out_json = json_files[-1] if json_files else None

    banner(9, total, "3차원 시각화  (pyvista)")
    if args.skip_3d:
        print("  --skip-3d 지정으로 생략합니다.")
    elif out_json is None:
        print("  [경고] export JSON 을 찾지 못해 생략합니다.")
    else:
        t = time.time()
        rc = run([PY, "-m", "dfn_analysis.plot_domain_dfn_3d",
                  "--json", out_json, "--out-dir", export_dir], env, allow_fail=True)
        if rc != 0:
            print("  [안내] 3차원 시각화가 실패했습니다 (pyvista/그래픽 환경 문제일 수 "
                  "있습니다). 데모의 나머지 결과에는 영향이 없습니다.")
        times["3D"] = time.time() - t

    # ------------------------------------------------------------------
    # 최종 요약: 참값 vs 역산 (참값은 이 비교표에서만 사용)
    # ------------------------------------------------------------------
    banner("끝", total, "요약 — 벤치마크 참값 대비 역산 결과")
    if args.site == "laxemar":
        rc = run([PY, os.path.join(HERE, "scripts", "compare_pipeline_to_laxemar.py"),
                  "--base", base], env, allow_fail=True)
        if rc != 0:
            print("  [경고] 비교표 생성 실패 — 개별 CSV 를 직접 확인하세요.")

    try:
        rec_rows = read_csv_rows(rec_csv)
        tiers = {}
        for r in rec_rows:
            tiers[r.get("radius_status", "?")] = tiers.get(r.get("radius_status", "?"), 0) + 1
        print(f"\n복원 균열 {len(rec_rows)}개: " +
              ", ".join(f"{k}={v}" for k, v in sorted(tiers.items())))
    except OSError:
        pass
    cond_csv = os.path.join(base, "conditional_hidden", "conditional_dfn.csv")
    try:
        cond_rows = read_csv_rows(cond_csv)
        kinds = {}
        for r in cond_rows:
            kinds[r.get("source_type", "?")] = kinds.get(r.get("source_type", "?"), 0) + 1
        print(f"조건부 암반균열망 {len(cond_rows)}개: " +
              ", ".join(f"{k}={v}" for k, v in sorted(kinds.items())))
    except OSError:
        pass

    print(f"\n출력 폴더  : {base}")
    if out_json:
        print(f"해석 입력 JSON: {out_json}")
    print("단계별 소요: " + ", ".join(f"{k} {v:.1f}s" for k, v in times.items()))
    print(f"총 소요    : {(time.time() - t0) / 60:.1f}분")


if __name__ == "__main__":
    main()
