"""
터널 방위각(azimuth) 민감도 분석 — kappa / kr / P32 만.

개념
----
코드의 관측 막장면은 항상 전역 x=상수 평면으로 고정되어 있어 '터널 방위각' 파라미터가 없다.
따라서 '터널을 방위각 theta 로 굴착' = 'DFN 전체를 연직 z축 기준 -theta 회전' 으로 구현한다
(관측창=터널 단면 polygon·거친 막장면 mesh 는 고정, fracture centers/normals 만 회전).

각 (방위각 theta, realization seed) 마다 파이프라인:
    generate_dfn(seed) -> rotate(-theta) -> traces -> kr(truth-free)
    -> trace 기반 config(trend/plunge/kappa/kr_hat) -> P32(unit_p32_forward_mc)
수집: set별 kappa / kr_hat / P32_hat. 각도 x realization boxplot 을 GT(빨간 점선)와 함께 출력.

사용 변수(모두 CLI 로 조절)
    --angle-start/--angle-stop/--angle-step  또는  --angles 0 30 60 ...
    --n-realizations   각 각도당 realization(무작위 seed) 개수
    --seed-base        realization seed 시작값
    --site --target-set --kr-mc-samples --kr-grid-size --p32-mc-samples --p32-replicates ...

개인/로컬 실행용. 대용량 임시 h5 는 --work-dir(기본 시스템 temp)에만 쓰고 실행 후 삭제한다.
중단되어도 --resume 로 이어서 실행 가능(결과 CSV 를 매 run 마다 append).

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
import time

import h5py
import numpy as np

# 프로젝트 루트(= 이 스크립트의 상위 폴더)
PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GENERATOR = os.path.join(PROJ, "dfn generator v1", "python", "generate_dfn.py")
DEFAULT_MESH = os.path.join(PROJ, "storage", "output",
                            "rough_face_mesh_collection",
                            "synthetic_rough_face_collection.h5")
DEFAULT_OUT = os.path.join(PROJ, "storage", "output", "sensitivity_tunnel_angle")

# boxplot 파라미터: (CSV 컬럼, 표시이름, y라벨)
PARAMS = [
    ("kappa",   "Fisher 집중도 κ",      "kappa"),
    ("kr_hat",  "반지름 멱법칙 지수 kr", "kr_hat"),
    ("P32_hat", "체적 절리밀도 P32",     "P32 (m²/m³)"),
]


# ----------------------------------------------------------------------------
# DFN 회전 (터널 방위각 theta = DFN 을 z축 -theta 회전)
# ----------------------------------------------------------------------------
def rotate_dfn_about_z(in_h5: str, out_h5: str, rot_deg: float) -> None:
    a = math.radians(rot_deg)
    c, s = math.cos(a), math.sin(a)
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)

    def copy_group(src, dst):
        for k, v in src.attrs.items():
            dst.attrs[k] = v
        for name, item in src.items():
            if isinstance(item, h5py.Group):
                copy_group(item, dst.require_group(name))
            else:
                dst.create_dataset(name, data=item[...])

    with h5py.File(in_h5, "r") as fin, h5py.File(out_h5, "w") as fout:
        for k, v in fin.attrs.items():
            fout.attrs[k] = v
        for top in fin.keys():
            if top == "fractures":
                g = fout.require_group("fractures")
                fr = fin["fractures"]
                for name, item in fr.items():
                    if name in ("centers", "normals"):
                        rot = (item[...].astype(np.float64) @ R.T).astype(np.float32)
                        g.create_dataset(name, data=rot)
                    else:
                        g.create_dataset(name, data=item[...])
                for k, v in fr.attrs.items():
                    g.attrs[k] = v
            else:
                copy_group(fin[top], fout.require_group(top))
        fout.require_group("meta").attrs["sensitivity_rot_deg_about_z"] = float(rot_deg)


# ----------------------------------------------------------------------------
# Ground truth
# ----------------------------------------------------------------------------
def parse_generator_kappa(site: str) -> dict:
    """generate_dfn.py 의 site별 set 정의에서 생성 입력 kappa 를 파싱한다.
    (SITE_FISHER_PARAMS 는 P32 forward MC 용 별도 테이블이라 생성 입력값과 다르므로 사용 불가.)"""
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


def build_ground_truth(site: str, target_sets, estimation_rmin: float, rmax: float) -> dict:
    """set별 GT: kappa(생성기 파싱), kr(SITE_TO_KR_TRUE), P32(support_scaled_p32)."""
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
        cfg = cfg_map.get(si)
        kr_true = kr_map.get(si)
        if cfg is not None and kr_true is not None:
            r0 = float(cfg.get("r0", float("nan")))
            support_rmin = max(estimation_rmin, r0)  # = set_likelihood_rmin
            p32 = support_scaled_p32(site, si, kr_true, support_rmin, rmax)
            if math.isfinite(p32):
                gt["P32_hat"][str(sid)] = p32
    return gt


# ----------------------------------------------------------------------------
# 파이프라인 실행
# ----------------------------------------------------------------------------
def _run(cmd, env, label):
    # 하위 스크립트가 한글(UTF-8)을 출력하므로 명시적으로 UTF-8 디코딩
    # (Windows 기본 cp949 로 캡처하면 리더 스레드에서 UnicodeDecodeError 발생)
    r = subprocess.run(cmd, cwd=PROJ, env=env, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    if r.returncode != 0:
        raise RuntimeError(f"[{label}] 실패 (exit {r.returncode})\n"
                           f"CMD: {' '.join(cmd)}\nSTDERR:\n{r.stderr[-1500:]}")
    return r


def load_done(csv_path):
    """--resume 용: 이미 완료된 (angle, seed) 집합."""
    done = set()
    if os.path.exists(csv_path):
        for r in csv.DictReader(open(csv_path)):
            done.add((int(r["angle_deg"]), int(r["seed"])))
    return done


def _process_one(py, env, sets, label, args, a, sd, wd, base_h5, report=None):
    """base_h5(회전 전 DFN) + 방위각 a -> set별 결과 rows.
    wd 생성/정리는 호출자 책임. report(str) 지정 시 substep 진척을 흘려보낸다."""
    os.makedirs(wd, exist_ok=True)
    rot_h5 = os.path.join(wd, "dfn_rot.h5")
    tr_dir = os.path.join(wd, "traces")
    kr_dir = os.path.join(wd, "kr")
    cfg = os.path.join(wd, "config.json")
    p32_csv = os.path.join(wd, "p32.csv")
    tr_h5 = os.path.join(tr_dir, "trace_dataset_3d.h5")
    kr_sum = os.path.join(kr_dir, "kr_summary_by_set.csv")

    def step(m):
        if report:
            report(m)

    rotate_dfn_about_z(base_h5, rot_h5, -a); step("rot✓ ")
    _run([py, "-m", "dfn_analysis.export_flat_face_traces",
          "--dfn-h5", rot_h5, "--face-x", *[str(x) for x in args.face_x],
          "--outdir", tr_dir], env, "traces"); step("trace✓ ")
    _run([py, "dfn_analysis/estimate_kr.py", "--trace-h5", tr_h5,
          "--dfn-model", label, "--target-set", *sets,
          "--mc-samples-per-grid", str(args.kr_mc_samples),
          "--profile-grid-size", str(args.kr_grid_size),
          "--outdir", kr_dir], env, "kr"); step("kr✓ ")
    _run([py, "-m", "dfn_analysis.build_dataset_config_from_traces",
          "--trace-h5", tr_h5, "--kr-summary-csv", kr_sum,
          "--dataset-name", label, "--target-set", *sets,
          "--out", cfg], env, "config"); step("cfg✓ ")
    _run([py, "-m", "dfn_analysis.estimate_p32_mc_calibrated",
          "--trace-h5", tr_h5, "--config", cfg, "--site", label,
          "--target-set", *sets, "--kr-summary-csv", kr_sum,
          "--bootstrap-csv", kr_sum, "--dfn-h5", rot_h5,
          "--rough-mesh-h5", args.mesh_h5,
          "--calibration-factor-mode", "forward_mc_lmin", "--lmin", str(args.lmin),
          "--mc-samples", str(args.p32_mc_samples),
          "--unit-p32-mc-replicates", str(args.p32_replicates),
          "--outcsv", p32_csv], env, "p32"); step("p32✓ ")

    cfg_sets = json.load(open(cfg))["sets"]
    p32_map = {r["set_id"]: r for r in csv.DictReader(open(p32_csv))}
    rows = []
    for sid in sets:
        cc = cfg_sets.get(sid, {})
        pp = p32_map.get(sid, {})
        rows.append({"angle_deg": a, "seed": sd, "set_id": sid,
                     "kappa": cc.get("kappa", ""),
                     "kr_hat": cc.get("kr_hat_estimated", ""),
                     "P32_hat": pp.get("P32_hat", "")})
    return rows


def run_pipeline(args):
    py = args.python
    env = dict(os.environ, PYTHONPATH=PROJ, PYTHONIOENCODING="utf-8")
    sets = [str(s) for s in args.target_set]
    label = args.label

    angles = args.angles if args.angles else list(
        range(args.angle_start, args.angle_stop + 1, args.angle_step))
    seeds = [args.seed_base + i for i in range(args.n_realizations)]

    work = args.work_dir or os.path.join(tempfile.gettempdir(),
                                         "dfn_sens_work")
    os.makedirs(work, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)
    out_csv = os.path.join(args.out_dir, "sensitivity_results.csv")
    # OneDrive 등 클라우드 동기화 폴더에 장시간 열린 append 핸들을 두면, 동기화가 파일을
    # 교체하면서 핸들이 분리되어 write 가 조용히 유실될 수 있다. → 라이브 기록은 로컬 임시
    # (live_csv)에 하고, 주기적으로 + 종료 시 out_csv(동기화 폴더)로 통째 복사(publish)한다.
    live_csv = os.path.join(work, "sensitivity_results_live.csv")
    fieldnames = ["angle_deg", "seed", "set_id", "kappa", "kr_hat", "P32_hat"]

    done = set()
    if args.resume and os.path.exists(out_csv):
        with open(out_csv, newline="") as f:
            hdr = f.readline().strip().split(",")
        if hdr == fieldnames:
            shutil.copyfile(out_csv, live_csv)
            done = load_done(live_csv)
        else:
            print(f"[경고] 기존 CSV 헤더가 현재 스키마와 달라 --resume 무시, 새로 시작합니다.\n"
                  f"       기존: {hdr}\n       현재: {fieldnames}")
    if not done and os.path.exists(live_csv):
        os.remove(live_csv)
    new_file = not os.path.exists(live_csv)
    fcsv = open(live_csv, "a", newline="")
    writer = csv.DictWriter(fcsv, fieldnames=fieldnames)
    if new_file:
        writer.writeheader()
        fcsv.flush()

    def publish():
        """로컬 라이브 CSV -> 동기화 out_csv 로 원자적 복사(핸들 분리 유실 회피)."""
        try:
            fcsv.flush()
            shutil.copyfile(live_csv, out_csv)
        except OSError:
            pass  # 다음 주기 / 종료 시 재시도

    combos = [(a, sd) for sd in seeds for a in angles if (a, sd) not in done]
    total = len(combos)
    skipped = len(angles) * len(seeds) - total

    print("=" * 70)
    print(" 터널 방위각 민감도 분석 (kappa / kr / P32)")
    print("=" * 70)
    print(f" site={args.site}  sets={sets}")
    print(f" 방위각 {angles}  ({len(angles)}개)")
    print(f" realization {seeds}  ({len(seeds)}개)")
    print(f" 총 {len(angles)*len(seeds)} run  (완료 {skipped} 건너뜀 → 실행 {total})")
    print(f" kr: mc={args.kr_mc_samples} grid={args.kr_grid_size} | "
          f"P32: mc={args.p32_mc_samples} rep={args.p32_replicates}")
    if args.jobs > 1:
        print(f" 병렬 실행: --jobs {args.jobs} (독립 태스크; 각 태스크가 DFN 자체 생성)")
    print(f" 결과 CSV: {out_csv}")
    print(f" 작업 임시폴더: {work}")
    print("=" * 70, flush=True)
    if total == 0:
        print("[i] 실행할 run 이 없습니다 (모두 완료).")
        fcsv.close()
        return out_csv, angles, seeds

    t0 = time.time()
    n_done = 0

    if args.jobs <= 1:
        # -------- 직렬: seed 별 base DFN 1회 생성 후 각도 재사용 + substep 스트리밍 --------
        seeds_pending = [sd for sd in seeds
                         if any((a, sd) not in done for a in angles)]
        for sd in seeds_pending:
            base_dir = os.path.join(work, f"base_seed{sd}")
            os.makedirs(base_dir, exist_ok=True)
            base_h5 = os.path.join(base_dir, "dfn_export_for_python.h5")
            sys.stdout.write(f"[seed {sd}] DFN 생성 중...")
            sys.stdout.flush()
            tg = time.time()
            _run([py, GENERATOR, "--site", args.site, "--seed", str(sd),
                  "--output-dir", base_dir], env, "generate_dfn")
            sys.stdout.write(f" 완료 ({time.time()-tg:.1f}s)\n")
            sys.stdout.flush()

            for a in angles:
                if (a, sd) in done:
                    continue
                wd = os.path.join(work, f"s{sd}_a{a}")
                tr = time.time()

                def report(msg):
                    sys.stdout.write(msg)
                    sys.stdout.flush()

                report(f"[{n_done+1}/{total}] seed={sd} θ={a:>3}°  ")
                try:
                    rows = _process_one(py, env, sets, label, args,
                                        a, sd, wd, base_h5, report)
                    for r in rows:
                        writer.writerow(r)
                    fcsv.flush()
                finally:
                    if not args.keep_intermediate:
                        shutil.rmtree(wd, ignore_errors=True)
                n_done += 1
                el = time.time() - t0
                eta = el / n_done * (total - n_done)
                report(f"| {time.time()-tr:.1f}s  경과 {el/60:.1f}분  "
                       f"ETA {eta/60:.1f}분\n")
                if n_done % 10 == 0:
                    publish()
            if not args.keep_intermediate:
                shutil.rmtree(base_dir, ignore_errors=True)
    else:
        # -------- 병렬: (seed,angle) 완전 독립 태스크(자체 DFN 생성) + 스레드풀 --------
        import threading
        from concurrent.futures import ThreadPoolExecutor, as_completed
        # 각 자식 프로세스는 1스레드로 고정(코어 과다구독 방지 → jobs 개가 jobs 코어 사용)
        env_par = dict(env, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                       OPENBLAS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
        lock = threading.Lock()

        def task(a, sd):
            wd = os.path.join(work, f"s{sd}_a{a}")
            try:
                os.makedirs(wd, exist_ok=True)
                _run([py, GENERATOR, "--site", args.site, "--seed", str(sd),
                      "--output-dir", wd], env_par, "generate_dfn")
                base = os.path.join(wd, "dfn_export_for_python.h5")
                rows = _process_one(py, env_par, sets, label, args,
                                    a, sd, wd, base, None)
                return (a, sd, rows, None)
            except Exception as e:  # 한 run 실패가 전체를 죽이지 않도록
                return (a, sd, None, str(e))
            finally:
                if not args.keep_intermediate:
                    shutil.rmtree(wd, ignore_errors=True)

        ex = ThreadPoolExecutor(max_workers=args.jobs)
        futs = [ex.submit(task, a, sd) for (a, sd) in combos]
        try:
            for fut in as_completed(futs):
                a, sd, rows, err = fut.result()
                with lock:
                    n_done += 1
                    el = time.time() - t0
                    eta = el / n_done * (total - n_done)
                    if err is None and rows is not None:
                        for r in rows:
                            writer.writerow(r)
                        fcsv.flush()
                        print(f"[{n_done}/{total}] seed={sd} θ={a:>3}° ✓  "
                              f"경과 {el/60:.1f}분  ETA {eta/60:.1f}분",
                              flush=True)
                    else:
                        print(f"[{n_done}/{total}] seed={sd} θ={a:>3}° ✗ 실패: "
                              f"{(err or 'unknown').splitlines()[0]}", flush=True)
                    if n_done % 10 == 0:
                        publish()
            ex.shutdown(wait=True)
        except KeyboardInterrupt:
            print("\n[중단] Ctrl+C 감지 — 진행분 저장 후 종료합니다. "
                  "남은 작업은 동일 명령에 --resume 를 붙여 이어서 실행하세요.", flush=True)
            for f in futs:
                f.cancel()
            ex.shutdown(wait=False, cancel_futures=True)
            with lock:
                publish()
            fcsv.close()
            return out_csv, angles, seeds

    fcsv.close()
    shutil.copyfile(live_csv, out_csv)  # 최종 발행(닫은 뒤라 반드시 성공)
    print("-" * 70)
    print(f"[완료] {n_done} run  ->  {out_csv}  (총 {(time.time()-t0)/60:.1f}분)",
          flush=True)
    return out_csv, angles, seeds


# ----------------------------------------------------------------------------
# Boxplot 출력 (kappa / kr / P32, GT 빨간 점선)
# ----------------------------------------------------------------------------
def make_plots(out_csv, out_dir, site, gt, target_sets):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from collections import defaultdict

    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False

    sets = [str(s) for s in target_sets]
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    angset = set()
    for r in csv.DictReader(open(out_csv)):
        ang = int(r["angle_deg"]); sid = r["set_id"]
        angset.add(ang)
        for col, _, _ in PARAMS:
            try:
                v = float(r.get(col, ""))
            except (ValueError, TypeError):
                continue
            if math.isfinite(v):
                data[col][sid][ang].append(v)
    angles = sorted(angset)
    xpos = np.arange(len(angles))

    n = len(sets)
    ncol = 2 if n > 1 else 1
    nrow = int(math.ceil(n / ncol))

    outs = []
    for col, disp, ylab in PARAMS:
        fig, axes = plt.subplots(nrow, ncol, figsize=(6.5 * ncol, 4.5 * nrow),
                                 squeeze=False, sharex=True)
        flat = axes.ravel()
        for k, sid in enumerate(sets):
            ax = flat[k]
            box_data = [data[col][sid].get(a, []) for a in angles]
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
            n_seeds = max((len(v) for v in box_data), default=0)
            ax.set_title(f"Set {sid}  (n={n_seeds} realizations/angle)")
            ax.set_ylabel(ylab)
            ax.grid(True, axis="y", alpha=0.3)
            ax.set_xticks(xpos)
            ax.set_xticklabels([str(a) for a in angles])
        for k in range(len(sets), len(flat)):
            flat[k].axis("off")
        for ax in axes[-1, :]:
            ax.set_xlabel("터널 방위각 θ (deg)  [DFN 을 -θ 회전]")
        fig.suptitle(f"터널 방위각 민감도 — {disp}  "
                     f"({site}, realization {n_seeds}, {len(angles)}개 각도)",
                     fontsize=14, fontweight="bold")
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
        out = os.path.join(out_dir, f"sensitivity_{col}.png")
        fig.savefig(out, dpi=150)
        plt.close(fig)
        outs.append(out)
        print(f"[plot] {out}")
    return outs


# ----------------------------------------------------------------------------
def build_argparser():
    p = argparse.ArgumentParser(
        description="터널 방위각 민감도 분석 (kappa/kr/P32)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--site", default="laxemar", choices=["forsmark", "laxemar"])
    p.add_argument("--target-set", nargs="+", type=int, default=[1, 2, 3, 5])
    # 각도 조밀도
    p.add_argument("--angle-start", type=int, default=0)
    p.add_argument("--angle-stop", type=int, default=180)
    p.add_argument("--angle-step", type=int, default=15)
    p.add_argument("--angles", nargs="+", type=int, default=None,
                   help="명시적 각도 리스트(지정 시 start/stop/step 무시)")
    # realization
    p.add_argument("--n-realizations", type=int, default=10,
                   help="각 각도당 무작위 seed 개수")
    p.add_argument("--seed-base", type=int, default=1, help="seed 시작값")
    # 추정 정밀도(속도 절충)
    p.add_argument("--kr-mc-samples", type=int, default=20000)
    p.add_argument("--kr-grid-size", type=int, default=61)
    p.add_argument("--p32-mc-samples", type=int, default=10000)
    p.add_argument("--p32-replicates", type=int, default=16)
    p.add_argument("--face-x", nargs="+", type=float, default=[0.0, 1.0, 2.0, 3.0],
                   help="평면 막장면 x 위치 목록 [m] (v2: 요철 mesh 미사용)")
    p.add_argument("--lmin", type=float, default=0.5,
                   help="최소 절리선 길이 [m] — 관측·가상 양쪽에 동일 적용")
    p.add_argument("--estimation-rmin", type=float, default=0.5,
                   help="P32 GT 지지구간 하한(모집단 정의)")
    p.add_argument("--rmax", type=float, default=250.0)
    # 경로
    p.add_argument("--mesh-h5", default=DEFAULT_MESH,
                   help="거친 막장면 mesh 컬렉션 h5")
    p.add_argument("--out-dir", default=DEFAULT_OUT)
    p.add_argument("--work-dir", default=None,
                   help="대용량 임시 h5 폴더(기본: 시스템 temp/dfn_sens_work)")
    p.add_argument("--label", default="sens", help="kr/config/P32 공통 site 라벨")
    p.add_argument("--jobs", type=int, default=1,
                   help="동시 실행 run 개수(병렬). >1 이면 (seed,angle) 독립 태스크로 "
                        "스레드풀 실행. 코어/메모리 여유 내에서 권장 8~12.")
    p.add_argument("--python", default=sys.executable)
    # 동작
    p.add_argument("--resume", action="store_true",
                   help="기존 결과 CSV 의 완료 (각도,seed) 는 건너뛰고 이어서 실행")
    p.add_argument("--keep-intermediate", action="store_true",
                   help="중간 h5/trace 를 지우지 않고 보존")
    p.add_argument("--no-plot", action="store_true", help="boxplot 생략")
    p.add_argument("--plot-only", action="store_true",
                   help="파이프라인 없이 기존 CSV 로 boxplot 만 재생성")
    return p


def main():
    args = build_argparser().parse_args()
    if not os.path.exists(args.mesh_h5):
        sys.exit(f"[오류] 거친 막장면 mesh 가 없습니다: {args.mesh_h5}\n"
                 f"       --mesh-h5 로 경로를 지정하거나 먼저 mesh 를 생성하세요.")

    gt = build_ground_truth(args.site, args.target_set,
                            args.estimation_rmin, args.rmax)
    print("[GT] ground truth (빨간 점선):")
    for col in ("kappa", "kr_hat", "P32_hat"):
        vals = {s: round(v, 4) for s, v in gt.get(col, {}).items()}
        print(f"     {col:8s}: {vals}")

    out_csv = os.path.join(args.out_dir, "sensitivity_results.csv")
    if args.plot_only:
        if not os.path.exists(out_csv):
            sys.exit(f"[오류] CSV 없음: {out_csv}")
    else:
        out_csv, _, _ = run_pipeline(args)

    if not args.no_plot:
        make_plots(out_csv, args.out_dir, args.site, gt, args.target_set)


if __name__ == "__main__":
    main()
