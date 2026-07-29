# =============================================================================
# 이 파일의 역할:
#   benchmark1 통합 kr(반경 멱법칙 지수) 추정 엔트리포인트 스크립트.
#   관측된 터널면 트레이스로부터 유효 rmin 기반 폴리곤 윈도우 MC 우도(likelihood)를
#   사용하여 세트별 반경 분포 멱지수 kr을 추정한다. (FINAL 버전 추정기)
#
# 주요 입력:
#   - 트레이스 데이터: --trace-h5(HDF5) 또는 --trace-csv(CSV) 중 하나
#   - 관측 윈도우 폴리곤: HDF5의 /meta/tunnel_poly_yz 또는 --tunnel-dat
#   - 추정 파라미터: rmin/rmax, set-rmin-mode, kr 격자 범위, lmin_fit 후보 등
#   - --likelihood-mode {window_mc(기본·최종 추정용)|hybrid(보조)}: 우도 계산 방식.
#     hybrid = 참 현길이 분포를 닫힌형(해석식)으로, 창·절단 변환은 kr불변
#     MC 커널 1회로 분해. 프로파일 평활·lmin 불변·속도가 강점(해석/민감도 스캔용).
#     20시드 검증에서 최종 추정 효율(RMSE)은 window_mc + per-set lmin 선택이
#     우위여서 기본값은 window_mc 를 유지한다(판정: docs/제6장_수정안_통합본.md 제3부 §4.6).
#     hybrid 는 lmin_fit 미지정 시 전 set 공통 0.5 m 고정을 기본으로 한다(§4.4).
#   - (검증용) site 프리셋 또는 --kr-true-map 으로 주어지는 kr 참값
#
# 주요 출력(--outdir 하위 CSV/JSON):
#   - kr_fit_by_lmin.csv        : lmin_fit 후보별 적합 결과
#   - kr_profile_likelihood.csv : kr 격자에 대한 프로파일 우도
#   - kr_posterior_predictive.csv: 사후 예측(길이 분포 등) 진단
#   - kr_summary_by_set.csv     : 세트별 최적 결과 요약
#   - kr_fit_by_lmin.json       : 입력 요약 + 적합/요약 행 통합 저장
#
# 핵심 처리 흐름:
#   1) CLI 인자 파싱 및 입력(트레이스/윈도우 폴리곤) 로드
#   2) 세트별로 그룹화하고 생성 rmin 메타데이터/일관성 확인
#   3) 각 세트 x 각 lmin_fit 후보에 대해 kr 적합(fit_set_lmin;
#      window-MC 우도 또는 hybrid 해석식+커널 우도)
#   4) 세트별 최적 행 선정(best_row) 후 요약 생성
#   5) CSV/JSON 출력 및 콘솔 요약 출력
# =============================================================================
import argparse
import json
import math
import os
import sys
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dfn_analysis.estimate_radius_powerlaw_window_mc import (
    build_set_rmin_lookup,
    fit_set_lmin,
    load_trace_data_from_csv,
    load_trace_data_from_h5,
    load_trace_rmin_metadata_from_h5,
    resolve_set_likelihood_rmin,
    write_csv,
)
from dfn_analysis.export_setwise_3d_traces import load_tunnel_polygon_from_dat


# 채택 상태(adoption_status)별 우선순위. 값이 작을수록 우선 채택.
# best_row 정렬 시 1차 정렬 키로 사용된다.
ADOPTION_PRIORITY = {
    "accepted": 0,
    "provisional_accepted": 1,
    "rejected": 2,
    "diagnostic_only_rmin_support_mismatch": 3,
}

# 사이트/세트별 kr 참값(ground-truth) 프리셋. 검증(validation) 목적으로만 사용.
SITE_TO_KR_TRUE: Dict[str, Dict[int, float]] = {
    "forsmark": {1: 2.88, 2: 3.02, 3: 2.81, 4: 2.95, 5: 2.92},
    "laxemar": {1: 2.85, 2: 3.04, 3: 3.01, 5: 3.60},
}


# 트레이스 행들을 set_id 기준으로 그룹화한다.
#   인자: rows(트레이스 행 목록), target_sets(대상 세트 집합, None이면 전체)
#   반환: set_id로 오름차순 정렬된 {set_id: [행,...]} 딕셔너리
def group_rows_by_set(rows: Sequence[dict], target_sets: Optional[Set[int]]) -> Dict[int, List[dict]]:
    # 대상 세트 필터링 후 set_id별로 행을 누적
    grouped: Dict[int, List[dict]] = {}
    for row in rows:
        set_id = int(row["set_id"])
        if target_sets is not None and set_id not in target_sets:
            continue
        grouped.setdefault(set_id, []).append(row)
    # set_id 오름차순으로 재구성하여 반환
    return {set_id: grouped[set_id] for set_id in sorted(grouped)}


# 검증용 kr 참값 매핑을 구성한다.
#   인자: site(프리셋 조회 키), tokens("SET_ID:KR_TRUE" 형식 문자열 목록)
#   반환: {set_id: kr_true} 딕셔너리. tokens 값이 프리셋을 덮어쓴다.
def parse_kr_true_map(site: str, tokens: Optional[Sequence[str]]) -> Dict[int, float]:
    # 사이트 프리셋으로 초기화한 뒤 사용자 지정 토큰으로 덮어쓴다
    kr_true_map = dict(SITE_TO_KR_TRUE.get(site, {})) if site else {}
    for token in tokens or []:
        sid_str, kr_str = token.split(":")
        kr_true_map[int(sid_str)] = float(kr_str)
    return kr_true_map


# 행 딕셔너리에서 특정 키 값을 float으로 안전하게 변환한다.
#   변환 불가(None/빈문자열/비수치)면 nan을 반환한다.
def to_float(row: dict, key: str) -> float:
    value = row.get(key, "")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


# 동일 세트의 여러 적합 행 중 가장 우수한 행 하나를 선정한다.
#   정렬 우선순위: (채택상태 우선순위 -> kr 오차 -> 클래스분율 L1오차 -> q90비 1과의 근접도)
#   반환: 정렬상 최상위 행
def best_row(rows: Sequence[dict]) -> dict:
    # 두 값의 절대 오차. 유한하지 않은 값이 있으면 inf(정렬상 최하위)로 처리
    def safe_abs_diff(value: float, ref: float) -> float:
        if not math.isfinite(value) or not math.isfinite(ref):
            return float("inf")
        return abs(value - ref)

    # 행의 정렬 키(튜플)를 구성. 앞쪽 요소일수록 우선순위가 높다
    def sort_key(row: dict) -> Tuple[float, float, float, float]:
        adoption = str(row.get("adoption_status", "rejected"))
        return (
            ADOPTION_PRIORITY.get(adoption, 9),
            safe_abs_diff(to_float(row, "kr_window_mc_hat"), to_float(row, "kr_true")),
            to_float(row, "class_fraction_l1_error"),
            safe_abs_diff(to_float(row, "q90_ratio_model_observed"), 1.0),
        )

    return sorted(rows, key=sort_key)[0]


# 세트별 최적 적합 행을 골라 요약 행 목록을 생성한다.
#   인자: fit_rows(모든 lmin_fit 적합 결과), site(출력에 기록할 사이트 라벨)
#   반환: 세트별 1개씩의 요약 딕셔너리 목록(kr_hat, 오차, CI, 상태 등 포함)
def build_summary_rows(fit_rows: Sequence[dict], site: str = "") -> List[dict]:
    # 먼저 set_id별로 적합 행들을 묶는다
    grouped: Dict[int, List[dict]] = {}
    for row in fit_rows:
        grouped.setdefault(int(row["set_id"]), []).append(row)

    # 각 세트에서 최적 행을 선정하고 요약 필드를 구성
    summary_rows: List[dict] = []
    for set_id, rows in sorted(grouped.items()):
        chosen = best_row(rows)
        summary_rows.append(
            {
                "site": site,
                "set_id": set_id,
                "best_lmin_fit": to_float(chosen, "lmin_fit"),
                "kr_hat": to_float(chosen, "kr_window_mc_hat"),
                "kr_true": to_float(chosen, "kr_true"),
                "kr_abs_error": safe_error(to_float(chosen, "kr_window_mc_hat"), to_float(chosen, "kr_true")),
                "set_effective_generation_rmin": to_float(chosen, "set_effective_generation_rmin"),
                "set_likelihood_rmin": to_float(chosen, "set_likelihood_rmin"),
                "generation_rmax": to_float(chosen, "generation_rmax"),
                "fit_status": str(chosen.get("fit_status", "")),
                "recovery_status": str(chosen.get("recovery_status", "")),
                "adoption_status": str(chosen.get("adoption_status", "")),
                "q90_ratio_model_observed": to_float(chosen, "q90_ratio_model_observed"),
                "q95_ratio_model_observed": to_float(chosen, "q95_ratio_model_observed"),
                "class_fraction_l1_error": to_float(chosen, "class_fraction_l1_error"),
                "kr_boot_mean": to_float(chosen, "kr_boot_mean"),
                "kr_boot_std": to_float(chosen, "kr_boot_std"),
                "kr_ci_low": to_float(chosen, "kr_ci_low"),
                "kr_ci_high": to_float(chosen, "kr_ci_high"),
                "bootstrap_boundary_fraction": to_float(chosen, "bootstrap_boundary_fraction"),
                "n_total": int(to_float(chosen, "n_total")) if math.isfinite(to_float(chosen, "n_total")) else "",
                "n_used": int(to_float(chosen, "n_used")) if math.isfinite(to_float(chosen, "n_used")) else "",
                "notes": str(chosen.get("warning", "")),
            }
        )
    return summary_rows


# 추정값과 참값의 절대 오차를 계산한다. 유한하지 않으면 nan 반환.
# (best_row의 safe_abs_diff와 달리 결측 시 nan을 반환하여 요약에 기록한다)
def safe_error(value: float, ref: float) -> float:
    if not math.isfinite(value) or not math.isfinite(ref):
        return float("nan")
    return abs(value - ref)


# 스크립트 엔트리포인트: CLI 인자 파싱 -> 입력 로드 -> 세트별 kr 적합 -> 결과 저장.
def main() -> None:
    # --- CLI 인자 정의 ---
    parser = argparse.ArgumentParser(
        description="Integrated benchmark1 kr estimator using the active effective-rmin polygon window-MC likelihood."
    )
    parser.add_argument("--trace-h5", help="Input trace HDF5")
    parser.add_argument("--trace-csv", help="Input trace CSV")
    parser.add_argument("--tunnel-dat", help="Optional tunnel polygon .dat file if trace input does not contain /meta/tunnel_poly_yz.")
    parser.add_argument("--target-set", nargs="+", type=int)
    parser.add_argument("--site", choices=["forsmark", "laxemar"], default="")
    parser.add_argument("--dfn-model", default="", help="Site/model label written to outputs.")
    parser.add_argument("--rmin", type=float, default=0.5)
    parser.add_argument("--rmax", type=float, default=250.0)
    parser.add_argument(
        "--set-rmin-mode",
        choices=["global", "effective_generation", "table_r0"],
        default="effective_generation",
    )
    parser.add_argument("--generation-rmin", type=float, default=0.5)
    parser.add_argument("--generation-rmax", type=float, default=250.0)
    parser.add_argument("--p32-label", default="P32_r_ge_0p5m")
    parser.add_argument("--kr-min", type=float, default=1.5)
    parser.add_argument("--kr-max", type=float, default=5.5)
    # lmin_fit 기본값은 우도 방식에 따라 다르다(--likelihood-mode 참조):
    #   window_mc → [0.1,0.2,0.3,0.5,0.75] 후보에서 set별 선택(기존 동작; per-set 선택이
    #               격자 재표집 잡음 회피 역할을 겸하므로 후보 탐색이 필요)
    #   hybrid    → [0.5] 고정(전 set 공통). hybrid 는 lmin 에 대한 kr_hat 요동이
    #               ≤0.05~0.30 수준으로 작고 잡음이 없어 후보 탐색의 실익이 없으며,
    #               0.5 m 는 mesh 해상도(0.2 m)·검출한계 위의 안정 구간이다
    #               (민감도 근거: docs/제6장_수정안_통합본.md 제3부 §4.4).
    parser.add_argument("--lmin-fit-values", nargs="+", type=float, default=None,
                        help="적합 길이 하한 후보 [m]. 미지정 시 window_mc=[0.1,0.2,0.3,0.5,0.75], hybrid=[0.5]")
    parser.add_argument("--allow-rmin-mismatch", action="store_true")
    parser.add_argument("--profile-grid-size", type=int, default=81)
    parser.add_argument("--mc-samples-per-grid", type=int, default=50000)
    parser.add_argument("--length-bin-count", type=int, default=40)
    parser.add_argument("--length-bin-mode", choices=["log", "linear"], default="log")
    parser.add_argument("--direction-mode", choices=["empirical_trace", "orientation_conditioned"], default="empirical_trace")
    parser.add_argument("--likelihood-mode", choices=["window_mc", "hybrid"], default="window_mc",
                        help="window_mc(기본)=kr별 전량 MC 시뮬레이션. hybrid=참 현길이 분포는 "
                             "닫힌형(해석식), 창·절단 변환만 kr불변 MC 커널로 1회 계산(분산↓·속도↑).")
    parser.add_argument("--center-weighting", choices=["unweighted", "proposal_area"], default="proposal_area")
    parser.add_argument("--likelihood-component", choices=["joint", "length_only", "class_only"], default="joint")
    parser.add_argument("--class-likelihood-weight", type=float, default=1.0)
    parser.add_argument("--oracle-radius-mode", choices=["none", "observed_trace_radii"], default="none")
    parser.add_argument("--run-bootstrap", action="store_true")
    parser.add_argument("--n-bootstrap", type=int, default=100)
    parser.add_argument("--kr-true-map", nargs="+", metavar="SET_ID:KR_TRUE")
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()

    # --- lmin_fit 기본값 결정(미지정 시): hybrid=0.5 고정, window_mc=후보 탐색 ---
    if args.lmin_fit_values is None:
        args.lmin_fit_values = [0.5] if args.likelihood_mode == "hybrid" else [0.1, 0.2, 0.3, 0.5, 0.75]

    # --- 입력 검증: 트레이스 소스는 H5/CSV 중 정확히 하나만 허용 ---
    if bool(args.trace_h5) == bool(args.trace_csv):
        raise ValueError("Provide exactly one of --trace-h5 or --trace-csv.")

    # --- 트레이스 행과 관측 윈도우 폴리곤(yz) 로드 ---
    rows, polygon_yz = load_trace_data_from_h5(args.trace_h5) if args.trace_h5 else load_trace_data_from_csv(args.trace_csv)
    # 트레이스에 폴리곤이 없으면 별도 .dat 파일에서 로드
    if polygon_yz is None and args.tunnel_dat:
        polygon_yz = load_tunnel_polygon_from_dat(args.tunnel_dat)
    # window-MC 우도에는 폴리곤이 필수
    if polygon_yz is None:
        raise ValueError("Window polygon is required for the active kr estimator. Provide --trace-h5 with /meta/tunnel_poly_yz or pass --tunnel-dat.")
    # --- 대상 세트 필터링 후 set_id별 그룹화 ---
    target_sets = set(args.target_set) if args.target_set else None
    grouped = group_rows_by_set(rows, target_sets)
    if not grouped:
        raise ValueError("No matching rows for target sets.")

    # --- 생성 rmin/rmax 확정: HDF5 메타데이터가 있으면 그 값으로 덮어씀 ---
    generation_rmin = float(args.generation_rmin)
    generation_rmax = float(args.generation_rmax)
    trace_rmin_metadata = {}
    if args.trace_h5:
        trace_rmin_metadata = load_trace_rmin_metadata_from_h5(args.trace_h5)
        if trace_rmin_metadata.get("generation_rmin") is not None:
            generation_rmin = float(trace_rmin_metadata["generation_rmin"])
        if trace_rmin_metadata.get("generation_rmax") is not None:
            generation_rmax = float(trace_rmin_metadata["generation_rmax"])

    # --- 생성 rmin과 추정 rmin 불일치 방어: 진단 모드가 아니면 오류 ---
    if abs(generation_rmin - float(args.rmin)) > 1e-6 and not args.allow_rmin_mismatch:
        raise ValueError(
            f"generation_rmin ({generation_rmin}) does not match estimator rmin ({args.rmin}). "
            "Use --allow-rmin-mismatch only for diagnostic runs."
        )

    # --- 사이트 라벨/검증 참값/세트별 rmin 조회표/kr 격자 준비 ---
    site = args.site or args.dfn_model
    kr_true_map = parse_kr_true_map(site, args.kr_true_map)
    set_rmin_lookup = build_set_rmin_lookup(site, generation_rmin, trace_rmin_metadata)
    kr_grid = np.linspace(args.kr_min, args.kr_max, args.profile_grid_size, dtype=np.float64)

    # 결과 누적용 컨테이너(적합/프로파일 우도/사후 예측)
    fit_rows: List[dict] = []
    profile_rows: List[dict] = []
    pp_rows: List[dict] = []

    # --- 각 세트 x 각 lmin_fit 후보에 대해 window-MC 우도로 kr 적합 ---
    for set_id, set_rows in grouped.items():
        for lmin_fit in args.lmin_fit_values:
            # 세트별 우도 rmin(및 유효 생성 rmin/table_r0/지지 상태) 결정
            set_likelihood_rmin, set_effective_generation_rmin, set_table_r0, set_support_status = resolve_set_likelihood_rmin(
                set_id,
                args.set_rmin_mode,
                float(args.rmin),
                set_rmin_lookup,
            )
            # 실제 kr 적합 수행: 폴리곤 윈도우 MC 우도 기반
            fit_row, profile, pp, _ = fit_set_lmin(
                set_id=set_id,
                set_rows=set_rows,
                polygon_yz=polygon_yz,
                kr_grid=kr_grid,
                rmin=set_likelihood_rmin,
                rmax=float(args.rmax),
                lmin_fit=float(lmin_fit),
                mc_samples_per_grid=int(args.mc_samples_per_grid),
                bin_count=int(args.length_bin_count),
                bin_mode=args.length_bin_mode,
                window_mode="polygon",
                direction_mode=args.direction_mode,
                site=site,
                kr_true=kr_true_map.get(set_id),
                center_weighting=args.center_weighting,
                likelihood_component=args.likelihood_component,
                class_likelihood_weight=float(args.class_likelihood_weight),
                oracle_radius_mode=args.oracle_radius_mode,
                run_bootstrap=bool(args.run_bootstrap),
                n_bootstrap=int(args.n_bootstrap),
                likelihood_mode=args.likelihood_mode,
            )
            # 적합/프로파일/사후예측 모든 행에 붙일 공통 메타데이터 구성
            metadata = {
                "dfn_model": site or args.dfn_model,
                "generation_rmin": generation_rmin,
                "generation_rmax": generation_rmax,
                "estimation_rmin": float(args.rmin),
                "estimation_rmax": float(args.rmax),
                "set_likelihood_rmin": float(set_likelihood_rmin),
                "set_effective_generation_rmin": float(set_effective_generation_rmin),
                "set_table_r0": float(set_table_r0) if math.isfinite(set_table_r0) else float("nan"),
                "set_rmin_mode": args.set_rmin_mode,
                "rmin_support_status": set_support_status,
                "p32_label": args.p32_label,
                "entrypoint_script": "dfn_analysis/estimate_kr.py",
            }
            # 각 결과 행에 메타데이터를 병합한 뒤 누적 컨테이너에 추가
            fit_row.update(metadata)
            for row in profile:
                row.update(metadata)
            for row in pp:
                row.update(metadata)
            fit_rows.append(fit_row)
            profile_rows.extend(profile)
            pp_rows.extend(pp)

    # --- 출력 디렉터리 및 파일 경로 준비 ---
    os.makedirs(args.outdir, exist_ok=True)
    fit_csv = os.path.join(args.outdir, "kr_fit_by_lmin.csv")
    profile_csv = os.path.join(args.outdir, "kr_profile_likelihood.csv")
    pp_csv = os.path.join(args.outdir, "kr_posterior_predictive.csv")
    summary_csv = os.path.join(args.outdir, "kr_summary_by_set.csv")
    fit_json = os.path.join(args.outdir, "kr_fit_by_lmin.json")

    # --- CSV 결과 저장(적합/프로파일/사후예측) 및 세트별 요약 생성/저장 ---
    write_csv(fit_rows, fit_csv)
    write_csv(profile_rows, profile_csv)
    write_csv(pp_rows, pp_csv)
    summary_rows = build_summary_rows(fit_rows, site)
    write_csv(summary_rows, summary_csv)
    # --- 입력 요약 + 적합/요약 행을 하나의 JSON으로 통합 저장 ---
    with open(fit_json, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "input_summary": {
                    "trace_h5": args.trace_h5,
                    "trace_csv": args.trace_csv,
                    "tunnel_dat": args.tunnel_dat,
                    "target_set": args.target_set,
                    "site": site,
                    "rmin": float(args.rmin),
                    "rmax": float(args.rmax),
                    "set_rmin_mode": args.set_rmin_mode,
                    "lmin_fit_values": [float(value) for value in args.lmin_fit_values],
                    "window_mode": "polygon",
                    "center_weighting": args.center_weighting,
                    "likelihood_component": args.likelihood_component,
                    "class_likelihood_weight": float(args.class_likelihood_weight),
                    "oracle_radius_mode": args.oracle_radius_mode,
                    "run_bootstrap": bool(args.run_bootstrap),
                    "n_bootstrap": int(args.n_bootstrap),
                    "truth_used_for_validation_only": bool(site or args.kr_true_map),
                },
                "fit_rows": fit_rows,
                "summary_rows": summary_rows,
            },
            handle,
            indent=2,
        )

    # --- 콘솔 요약 출력: 세트별 최적 lmin, kr 추정치, (있으면) 참값과 상태 ---
    print("[*] estimate_kr.py summary")
    for row in summary_rows:
        kr_true = to_float(row, "kr_true")
        truth_text = f", kr_true={kr_true:.3f}" if math.isfinite(kr_true) else ""
        print(
            f"    - Set {row['set_id']}: best_lmin={float(row['best_lmin_fit']):.3f}, "
            f"kr_hat={float(row['kr_hat']):.3f}{truth_text}, status={row['fit_status']}, adoption={row['adoption_status']}"
        )
    print(f"[*] Fit CSV written to: {fit_csv}")
    print(f"[*] Profile CSV written to: {profile_csv}")
    print(f"[*] Posterior predictive CSV written to: {pp_csv}")
    print(f"[*] Summary CSV written to: {summary_csv}")
    print(f"[*] Fit JSON written to: {fit_json}")


if __name__ == "__main__":
    main()
