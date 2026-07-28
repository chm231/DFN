"""
막장면 굴착 간격(face interval) 민감도 분석 — 기본 Set 5, kappa / kr / P32.

개념
----
관측 막장면 개수 N 을 고정하고(기본 4개), 막장면 사이 x 간격 d 만 바꾼다.
 - N 고정 → 총 관측면적(= N x 막장면 폴리곤 면적) 은 간격과 무관하게 일정.
 - 따라서 간격 효과(공간 상관/독립성)만 분리된다:
     간격이 작으면 같은 fracture 가 인접 여러 막장면에 걸쳐 상관된 trace 를 만들고,
     간격이 크면 서로 독립적인 fracture 를 관측하여 유효 표본이 늘어난다.
회전 없음(터널은 전역 x축 정렬). 각 (간격 d, realization seed) 마다 파이프라인:
    generate_dfn(seed) -> traces(간격 d 의 mesh) -> kr -> config(kappa/kr_hat) -> P32
수집: set별 kappa / kr_hat / P32_hat. 간격 x realization boxplot 을 GT(빨간 점선)와 함께 출력.

개인/로컬 실행용. 대용량 임시 h5 는 --work-dir(기본 시스템 temp)에만 쓰고 실행 후 삭제.
라이브 결과는 로컬에 기록 후 주기적으로 out-dir(동기화 폴더)로 복사(publish)한다.
좌표계: x=East(터널 진행축), y=North, z=Up. 막장면 = x=상수 평면.
"""
import argparse
import csv
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GENERATOR = os.path.join(PROJ, "dfn generator v1", "python", "generate_dfn.py")
MESH_GEN = "dfn_analysis/generate_synthetic_rough_face_mesh.py"
DEFAULT_DAT = os.path.join(PROJ, "storage", "data", "단면_폴리곤.dat")
DEFAULT_OUT = os.path.join(PROJ, "storage", "output", "sensitivity_face_interval")

PARAMS = [
    ("kappa",   "Fisher 집중도 κ",      "kappa"),
    ("kr_hat",  "반지름 멱법칙 지수 kr", "kr_hat"),
    ("P32_hat", "체적 절리밀도 P32",     "P32 (m²/m³)"),
]


# ----------------------------------------------------------------------------
# Ground truth (kappa: 생성기 파싱 / kr: SITE_TO_KR_TRUE / P32: support_scaled_p32)
# ----------------------------------------------------------------------------
def parse_generator_kappa(site: str) -> dict:
    try:
        text = open(GENERATOR, "r", encoding="utf-8").read()
    except OSError:
        return {}
    m = re.search(rf'site\s*==\s*"{re.escape(site)}"', text)
    if not m:
        return {}
    sub = text[m.end():]
    b0 = sub.find("sets = [")
    if b0 < 0:
        return {}
    b1 = sub.find("\n        ]", b0)
    block = sub[b0:b1 if b1 > 0 else len(sub)]
    out = {}
    for sid, kap in re.findall(r"'name':\s*'Set_(\d+)'.*?'kappa':\s*([-\d.]+)", block):
        out[int(sid)] = float(kap)
    return out


def build_ground_truth(site, target_sets, estimation_rmin, rmax):
    from dfn_analysis.estimate_kr import SITE_TO_KR_TRUE
    from dfn_analysis.build_p32_pilot_summary import SITE_SET_CONFIG, support_scaled_p32
    kap_map = parse_generator_kappa(site)
    kr_map = SITE_TO_KR_TRUE.get(site, {})
    cfg_map = SITE_SET_CONFIG.get(site, {})
    gt = {"kappa": {}, "kr_hat": {}, "P32_hat": {}}
    for sid in target_sets:
        si = int(sid)
        if si in kap_map:
            gt["kappa"][str(sid)] = kap_map[si]
        if si in kr_map:
            gt["kr_hat"][str(sid)] = kr_map[si]
        cfg, kr_true = cfg_map.get(si), kr_map.get(si)
        if cfg is not None and kr_true is not None:
            r0 = float(cfg.get("r0", float("nan")))
            p32 = support_scaled_p32(site, si, kr_true, max(estimation_rmin, r0), rmax)
            if math.isfinite(p32):
                gt["P32_hat"][str(sid)] = p32
    return gt


# ----------------------------------------------------------------------------
def _run(cmd, env, label):
    r = subprocess.run(cmd, cwd=PROJ, env=env, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    if r.returncode != 0:
        raise RuntimeError(f"[{label}] 실패 (exit {r.returncode})\n"
                           f"CMD: {' '.join(cmd)}\nSTDERR:\n{r.stderr[-1500:]}")
    return r


def load_done(csv_path):
    done = set()
    if os.path.exists(csv_path):
        for r in csv.DictReader(open(csv_path)):
            done.add((float(r["interval_m"]), int(r["seed"])))
    return done


def fmt_d(d):
    return f"{d:g}"


def generate_meshes(py, env, args, intervals, mesh_dir):
    """간격별 mesh 를 1회 생성(모든 realization 이 공유). {d: mesh_h5} 반환."""
    os.makedirs(mesh_dir, exist_ok=True)
    mesh_map = {}
    for d in intervals:
        out = os.path.join(mesh_dir, f"mesh_d{fmt_d(d)}")
        mesh_h5 = os.path.join(out, "mesh.h5")
        if not os.path.exists(mesh_h5):
            _run([py, MESH_GEN, "--tunnel-dat", args.tunnel_dat,
                  "--num-faces", str(args.num_faces), "--face-step", str(d),
                  "--base-x", str(args.base_x), "--seed-base", str(args.mesh_seed_base),
                  "--outdir", out, "--out-h5", "mesh.h5"], env, f"mesh(d={d})")
        mesh_map[d] = mesh_h5
        print(f"[mesh] d={fmt_d(d)}m  faces={args.num_faces}  -> {mesh_h5}", flush=True)
    return mesh_map


def process_one(py, env, sets, label, args, d, sd, wd, mesh_h5):
    """DFN(seed) + 간격 d mesh -> set별 결과 rows. wd 정리는 호출자 담당."""
    os.makedirs(wd, exist_ok=True)
    base = os.path.join(wd, "dfn_export_for_python.h5")
    tr_dir = os.path.join(wd, "traces")
    kr_dir = os.path.join(wd, "kr")
    cfg = os.path.join(wd, "config.json")
    p32_csv = os.path.join(wd, "p32.csv")
    tr_h5 = os.path.join(tr_dir, "trace_dataset_3d.h5")
    kr_sum = os.path.join(kr_dir, "kr_summary_by_set.csv")

    _run([py, GENERATOR, "--site", args.site, "--seed", str(sd),
          "--output-dir", wd], env, "generate_dfn")
    _run([py, "dfn_analysis/export_setwise_3d_traces.py",
          "--input", base, "--rough-mesh-h5", mesh_h5, "--outdir", tr_dir], env, "traces")
    _run([py, "dfn_analysis/estimate_kr.py", "--trace-h5", tr_h5,
          "--dfn-model", label, "--target-set", *sets,
          "--mc-samples-per-grid", str(args.kr_mc_samples),
          "--profile-grid-size", str(args.kr_grid_size),
          "--outdir", kr_dir], env, "kr")
    _run([py, "-m", "dfn_analysis.build_dataset_config_from_traces",
          "--trace-h5", tr_h5, "--kr-summary-csv", kr_sum,
          "--dataset-name", label, "--target-set", *sets, "--out", cfg], env, "config")
    _run([py, "-m", "dfn_analysis.estimate_p32_mc_calibrated",
          "--trace-h5", tr_h5, "--config", cfg, "--site", label,
          "--target-set", *sets, "--kr-summary-csv", kr_sum, "--bootstrap-csv", kr_sum,
          "--dfn-h5", base, "--rough-mesh-h5", mesh_h5,
          "--calibration-factor-mode", "unit_p32_forward_mc",
          "--mc-samples", str(args.p32_mc_samples),
          "--unit-p32-mc-replicates", str(args.p32_replicates),
          "--outcsv", p32_csv], env, "p32")

    cfg_sets = json.load(open(cfg))["sets"]
    p32_map = {r["set_id"]: r for r in csv.DictReader(open(p32_csv))}
    rows = []
    for sid in sets:
        cc = cfg_sets.get(sid, {})
        pp = p32_map.get(sid, {})
        rows.append({"interval_m": fmt_d(d), "seed": sd, "set_id": sid,
                     "kappa": cc.get("kappa", ""),
                     "kr_hat": cc.get("kr_hat_estimated", ""),
                     "P32_hat": pp.get("P32_hat", "")})
    return rows


def run_pipeline(args):
    py = args.python
    env = dict(os.environ, PYTHONPATH=PROJ, PYTHONIOENCODING="utf-8")
    sets = [str(s) for s in args.target_set]
    label = args.label
    intervals = list(args.intervals)
    seeds = [args.seed_base + i for i in range(args.n_realizations)]

    work = args.work_dir or os.path.join(tempfile.gettempdir(), "dfn_interval_work")
    os.makedirs(work, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)
    out_csv = os.path.join(args.out_dir, "interval_results.csv")
    live_csv = os.path.join(work, "interval_results_live.csv")
    fieldnames = ["interval_m", "seed", "set_id", "kappa", "kr_hat", "P32_hat"]

    done = set()
    if args.resume and os.path.exists(out_csv):
        with open(out_csv, newline="") as f:
            hdr = f.readline().strip().split(",")
        if hdr == fieldnames:
            shutil.copyfile(out_csv, live_csv)
            done = load_done(live_csv)
        else:
            print(f"[경고] 기존 CSV 헤더가 달라 --resume 무시, 새로 시작.\n  기존:{hdr}\n  현재:{fieldnames}")
    if not done and os.path.exists(live_csv):
        os.remove(live_csv)
    new_file = not os.path.exists(live_csv)
    fcsv = open(live_csv, "a", newline="")
    writer = csv.DictWriter(fcsv, fieldnames=fieldnames)
    if new_file:
        writer.writeheader(); fcsv.flush()

    def publish():
        try:
            fcsv.flush(); shutil.copyfile(live_csv, out_csv)
        except OSError:
            pass

    combos = [(d, sd) for sd in seeds for d in intervals
              if (float(fmt_d(d)), sd) not in done]
    total = len(combos)
    skipped = len(intervals) * len(seeds) - total

    print("=" * 70)
    print(" 막장면 굴착 간격 민감도 분석 (kappa / kr / P32)")
    print("=" * 70)
    print(f" site={args.site}  sets={sets}  (막장면 {args.num_faces}개 고정, 회전 없음)")
    print(f" 간격 {[fmt_d(d) for d in intervals]} m  ({len(intervals)}개)")
    print(f" realization {seeds}  ({len(seeds)}개)")
    print(f" 총 {len(intervals)*len(seeds)} run  (완료 {skipped} 건너뜀 → 실행 {total})")
    print(f" kr: mc={args.kr_mc_samples} grid={args.kr_grid_size} | "
          f"P32: mc={args.p32_mc_samples} rep={args.p32_replicates} | jobs={args.jobs}")
    print(f" 결과 CSV: {out_csv}")
    print(f" 작업 임시폴더: {work}")
    print("=" * 70, flush=True)
    if total == 0:
        print("[i] 실행할 run 이 없습니다 (모두 완료).")
        fcsv.close()
        return out_csv

    # 간격별 mesh 사전 생성(공유)
    mesh_map = generate_meshes(py, env, args, intervals, os.path.join(work, "meshes"))

    import time
    t0 = time.time()
    n_done = 0

    if args.jobs <= 1:
        for d, sd in combos:
            wd = os.path.join(work, f"d{fmt_d(d)}_s{sd}")
            tr = time.time()
            try:
                rows = process_one(py, env, sets, label, args, d, sd, wd, mesh_map[d])
                for r in rows:
                    writer.writerow(r)
                fcsv.flush()
            finally:
                if not args.keep_intermediate:
                    shutil.rmtree(wd, ignore_errors=True)
            n_done += 1
            el = time.time() - t0
            print(f"[{n_done}/{total}] d={fmt_d(d):>3}m seed={sd} ✓  "
                  f"{time.time()-tr:.1f}s  경과 {el/60:.1f}분  "
                  f"ETA {el/n_done*(total-n_done)/60:.1f}분", flush=True)
            if n_done % 10 == 0:
                publish()
    else:
        import threading
        from concurrent.futures import ThreadPoolExecutor, as_completed
        env_par = dict(env, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                       OPENBLAS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
        lock = threading.Lock()

        def task(d, sd):
            wd = os.path.join(work, f"d{fmt_d(d)}_s{sd}")
            try:
                rows = process_one(py, env_par, sets, label, args, d, sd, wd, mesh_map[d])
                return (d, sd, rows, None)
            except Exception as e:
                return (d, sd, None, str(e))
            finally:
                if not args.keep_intermediate:
                    shutil.rmtree(wd, ignore_errors=True)

        ex = ThreadPoolExecutor(max_workers=args.jobs)
        futs = [ex.submit(task, d, sd) for (d, sd) in combos]
        try:
            for fut in as_completed(futs):
                d, sd, rows, err = fut.result()
                with lock:
                    n_done += 1
                    el = time.time() - t0
                    eta = el / n_done * (total - n_done)
                    if err is None and rows is not None:
                        for r in rows:
                            writer.writerow(r)
                        fcsv.flush()
                        print(f"[{n_done}/{total}] d={fmt_d(d):>3}m seed={sd} ✓  "
                              f"경과 {el/60:.1f}분  ETA {eta/60:.1f}분", flush=True)
                    else:
                        print(f"[{n_done}/{total}] d={fmt_d(d):>3}m seed={sd} ✗ 실패: "
                              f"{(err or 'unknown').splitlines()[0]}", flush=True)
                    if n_done % 10 == 0:
                        publish()
            ex.shutdown(wait=True)
        except KeyboardInterrupt:
            print("\n[중단] Ctrl+C — 진행분 저장 후 종료. --resume 로 이어서 실행.", flush=True)
            for f in futs:
                f.cancel()
            ex.shutdown(wait=False, cancel_futures=True)
            with lock:
                publish()
            fcsv.close()
            return out_csv

    fcsv.close()
    shutil.copyfile(live_csv, out_csv)
    if not args.keep_intermediate:
        shutil.rmtree(os.path.join(work, "meshes"), ignore_errors=True)
    print("-" * 70)
    print(f"[완료] {n_done} run  ->  {out_csv}  (총 {(time.time()-t0)/60:.1f}분)", flush=True)
    return out_csv


# ----------------------------------------------------------------------------
def make_plots(out_csv, out_dir, site, gt, target_sets):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from collections import defaultdict

    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False

    sets = [str(s) for s in target_sets]
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    ivset = set()
    for r in csv.DictReader(open(out_csv)):
        iv = float(r["interval_m"]); sid = r["set_id"]
        ivset.add(iv)
        for col, _, _ in PARAMS:
            try:
                v = float(r.get(col, ""))
            except (ValueError, TypeError):
                continue
            if math.isfinite(v):
                data[col][sid][iv].append(v)
    intervals = sorted(ivset)
    xpos = np.arange(len(intervals))

    nrow, ncol = len(sets), len(PARAMS)
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.2 * ncol, 4.2 * nrow),
                             squeeze=False)
    n_r = 0
    for ri, sid in enumerate(sets):
        for ci, (col, disp, ylab) in enumerate(PARAMS):
            ax = axes[ri][ci]
            box_data = [data[col][sid].get(iv, []) for iv in intervals]
            bp = ax.boxplot(box_data, positions=xpos, widths=0.6,
                            patch_artist=True, showfliers=True,
                            medianprops=dict(color="black"))
            for patch in bp["boxes"]:
                patch.set_facecolor("#4C9BE0"); patch.set_alpha(0.7)
            for i, vals in enumerate(box_data):
                if vals:
                    jit = (np.random.RandomState(i).rand(len(vals)) - 0.5) * 0.25
                    ax.scatter(np.full(len(vals), xpos[i]) + jit, vals,
                               s=10, color="#08306B", alpha=0.5, zorder=3)
            gtv = gt.get(col, {}).get(sid)
            if gtv is not None:
                ax.axhline(gtv, color="red", ls="--", lw=1.6, zorder=4,
                           label=f"ground truth = {gtv:g}")
                ax.legend(loc="upper right", fontsize=8, framealpha=0.85)
            n_r = max((len(v) for v in box_data), default=0)
            ax.set_title(f"Set {sid} — {disp}  (n={n_r})")
            ax.set_ylabel(ylab)
            ax.grid(True, axis="y", alpha=0.3)
            ax.set_xticks(xpos)
            ax.set_xticklabels([fmt_d(iv) for iv in intervals])
            ax.set_xlabel("막장면 굴착 간격 d (m)")
    fig.suptitle(f"막장면 굴착 간격 민감도 ({site}, 막장면 고정, "
                 f"realization {n_r}, {len(intervals)}개 간격)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    out = os.path.join(out_dir, "sensitivity_face_interval.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[plot] {out}")
    return out


# ----------------------------------------------------------------------------
def build_argparser():
    p = argparse.ArgumentParser(
        description="막장면 굴착 간격 민감도 분석 (kappa/kr/P32)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--site", default="laxemar", choices=["forsmark", "laxemar"])
    p.add_argument("--target-set", nargs="+", type=int, default=[5])
    p.add_argument("--intervals", nargs="+", type=float, default=[0.5, 1.0, 2.0, 3.0, 6.0],
                   help="막장면 간격 d 목록 (m)")
    p.add_argument("--num-faces", type=int, default=4, help="고정 막장면 개수")
    p.add_argument("--base-x", type=float, default=0.0, help="기준 막장면 x 위치 (m)")
    p.add_argument("--n-realizations", type=int, default=20)
    p.add_argument("--seed-base", type=int, default=1)
    p.add_argument("--mesh-seed-base", type=int, default=42, help="mesh roughness 시드(간격별 고정)")
    p.add_argument("--tunnel-dat", default=DEFAULT_DAT)
    p.add_argument("--kr-mc-samples", type=int, default=20000)
    p.add_argument("--kr-grid-size", type=int, default=61)
    p.add_argument("--p32-mc-samples", type=int, default=10000)
    p.add_argument("--p32-replicates", type=int, default=16)
    p.add_argument("--estimation-rmin", type=float, default=0.5)
    p.add_argument("--rmax", type=float, default=250.0)
    p.add_argument("--out-dir", default=DEFAULT_OUT)
    p.add_argument("--work-dir", default=None)
    p.add_argument("--label", default="sens")
    p.add_argument("--jobs", type=int, default=1, help="동시 실행 run 개수(병렬). 권장 8~12.")
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--keep-intermediate", action="store_true")
    p.add_argument("--no-plot", action="store_true")
    p.add_argument("--plot-only", action="store_true")
    return p


def main():
    args = build_argparser().parse_args()
    if not args.plot_only and not os.path.exists(args.tunnel_dat):
        sys.exit(f"[오류] 터널 단면 DAT 없음: {args.tunnel_dat}")

    gt = build_ground_truth(args.site, args.target_set, args.estimation_rmin, args.rmax)
    print("[GT] ground truth (빨간 점선):")
    for col in ("kappa", "kr_hat", "P32_hat"):
        print(f"     {col:8s}: {{{', '.join(f'{s}:{round(v,4)}' for s,v in gt.get(col,{}).items())}}}")

    out_csv = os.path.join(args.out_dir, "interval_results.csv")
    if args.plot_only:
        if not os.path.exists(out_csv):
            sys.exit(f"[오류] CSV 없음: {out_csv}")
    else:
        out_csv = run_pipeline(args)

    if not args.no_plot:
        make_plots(out_csv, args.out_dir, args.site, gt, args.target_set)


if __name__ == "__main__":
    main()
