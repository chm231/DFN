# =============================================================================
# 이 파일의 역할:
#   추정된 kr(반경 멱법칙 지수) 회수 결과로부터 P32(단위 부피당 균열 면적 밀도)의
#   파일럿 요약을 생성한다. 채택/잠정채택된 유효 rmin 세트에 대해,
#   생성기 프리셋(SITE_SET_CONFIG)과 지지 반경(support rmin)을 이용해
#   support-scaled P32를 계산하고 CI/상태 분류를 붙여 요약 CSV로 출력한다.
#
# 주요 입력(모두 CSV/HDF5 경로 인자):
#   - --final-summary-csv     : 최종 kr 회수 요약(세트별 kr_hat, rmin, 채택상태 등)
#   - --bootstrap-summary-csv : kr 부트스트랩 CI 요약
#   - --rough-mesh-h5         : 관측 P21 계산용 러프면 메쉬 컬렉션
#   - --{site}-dfn-h5         : 방향 보정계수 계산용 DFN export
#   - --{site}-trace-h5       : 관측 P21 계산용 트레이스 데이터셋
#
# 주요 출력:
#   - --outcsv : 세트별 P32_hat, P32 CI, 관측 P21, 방향계수, 상태 분류를 담은 요약 CSV
#
# 핵심 처리 흐름:
#   1) 최종/부트스트랩 요약 CSV 로드 및 사이트별 P21/방향계수 맵 구성
#   2) 각 최종 행에 대해 채택상태 필터링(rejected/accepted-only 처리)
#   3) support_scaled_p32로 kr_hat 및 CI 경계에서의 P32 계산
#   4) P32 상태 분류 및 진단 노트 구성 후 요약 행 누적
#   5) 요약 CSV 저장
# =============================================================================
import argparse
import csv
import math
import os
import sys
from typing import Dict, List, Sequence, Tuple

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dfn_analysis.export_setwise_3d_traces import load_hdf5_dfn
from dfn_analysis.summarize_setwise_trace_statistics import (
    build_summary_rows,
    compute_total_observation_area,
    load_rough_face_collection_from_h5,
    load_trace_rows_from_h5,
)


# 사이트/세트별 생성기 프리셋.
#   p32_base : 기준 rmin 지지에서의 기준 P32 값
#   dist_type: 반경 분포 형태("powerlaw" 또는 "exponential")
#   r0       : 분포 기준 반경(멱법칙의 최소반경 또는 지수분포의 특성반경)
# support_scaled_p32에서 support_rmin에 맞춰 P32를 재스케일할 때 사용된다.
SITE_SET_CONFIG = {
    "forsmark": {
        1: {"p32_base": 0.602, "dist_type": "powerlaw", "r0": 0.28},
        2: {"p32_base": 2.069, "dist_type": "powerlaw", "r0": 0.25},
        3: {"p32_base": 0.448, "dist_type": "powerlaw", "r0": 0.14},
        4: {"p32_base": 0.226, "dist_type": "powerlaw", "r0": 0.15},
        5: {"p32_base": 0.605, "dist_type": "powerlaw", "r0": 0.25},
    },
    "laxemar": {
        1: {"p32_base": 1.310, "dist_type": "powerlaw", "r0": 0.328},
        2: {"p32_base": 1.026, "dist_type": "powerlaw", "r0": 0.977},
        3: {"p32_base": 0.975, "dist_type": "powerlaw", "r0": 0.858},
        4: {"p32_base": 2.320, "dist_type": "exponential", "r0": 4.0},
        5: {"p32_base": 1.400, "dist_type": "powerlaw", "r0": 0.400},
    },
}


# CSV 파일을 읽어 dict 리스트로 반환한다. (헤더를 키로 사용)
def read_csv(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


# dict 리스트를 CSV로 저장한다. 첫 행의 키를 헤더(필드명)로 사용.
#   행이 비어 있으면 ValueError를 발생시킨다.
def write_csv(rows: Sequence[dict], path: str) -> None:
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


# 행 딕셔너리에서 특정 키 값을 float으로 안전하게 변환한다.
#   변환 불가면 nan 반환.
def to_float(row: dict, key: str) -> float:
    value = row.get(key, "")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


# 지지 반경(support_rmin) 기준으로 재스케일된 P32를 계산한다.
#   인자: site/set_id(프리셋 조회), kr(멱지수), support_rmin(우도 지지 최소반경), rmax(최대반경)
#   반환: support_rmin~rmax 지지에서의 P32 추정치. 프리셋 없거나 비수치 입력이면 nan.
#   원리: p32_base * (support_rmin 구간 반경^2 적분 / r0 구간 반경^2 적분)
def support_scaled_p32(
    site: str,
    set_id: int,
    kr: float,
    support_rmin: float,
    rmax: float,
) -> float:
    # 사이트/세트 프리셋 조회 및 입력 유효성 검사
    cfg = SITE_SET_CONFIG.get(site, {}).get(set_id)
    if cfg is None or not math.isfinite(kr) or not math.isfinite(support_rmin) or not math.isfinite(rmax):
        return float("nan")
    p32_base = float(cfg["p32_base"])
    dist_type = str(cfg["dist_type"])
    r0 = float(cfg["r0"])

    # 멱법칙 분포: 반경^2 가중 적분비로 지지 구간을 재스케일
    if dist_type == "powerlaw":
        # 반경^2 * r^(-kr) 적분의 지수. kr=2 근처면 로그 적분으로 분기
        pow_val = 2.0 - kr
        if abs(pow_val) < 1e-12:
            int_r0 = math.log(rmax) - math.log(r0)
            int_rmin = math.log(rmax) - math.log(support_rmin)
        else:
            # r0 기준 구간과 support_rmin 기준 구간의 적분값
            int_r0 = (rmax**pow_val - r0**pow_val) / pow_val
            int_rmin = (rmax**pow_val - support_rmin**pow_val) / pow_val
        # 분모가 0에 가까우면 정의 불가
        if abs(int_r0) < 1e-12:
            return float("nan")
        return p32_base * (int_rmin / int_r0)

    # 지수 분포: 반경^2 * exp(-lam r)의 해석적 적분(부분적분 결과)로 재스케일
    if dist_type == "exponential":
        lam = 1.0 / r0

        # 0~radius 구간 적분의 부정적분값(부호 포함)
        def integral_fn(radius: float) -> float:
            return -math.exp(-lam * radius) * (radius**2 + 2.0 * radius / lam + 2.0 / (lam**2))

        # r0(=0 하한) 기준 구간과 support_rmin 기준 구간의 적분값
        int_r0 = integral_fn(rmax) - integral_fn(0.0)
        int_rmin = integral_fn(rmax) - integral_fn(support_rmin)
        if abs(int_r0) < 1e-12:
            return float("nan")
        return p32_base * (int_rmin / int_r0)

    # 알 수 없는 분포 형태
    return float("nan")


# 관측 트레이스와 러프면 메쉬로부터 세트별 P21 관측 요약을 만든다.
#   인자: trace_h5(트레이스 HDF5), rough_mesh_h5(관측 면적 계산용 메쉬)
#   반환: {set_id: 요약행} 딕셔너리 (P21_observed 등 포함)
def load_p21_summary(trace_h5: str, rough_mesh_h5: str) -> Dict[int, dict]:
    # 트레이스 행 로드 -> 총 관측 면적 계산 -> 세트별 요약(P21 등) 생성
    rows = load_trace_rows_from_h5(trace_h5)
    rough_faces = load_rough_face_collection_from_h5(rough_mesh_h5)
    observation_area_m2 = compute_total_observation_area(rough_faces)
    summary_rows = build_summary_rows(rows, observation_area_m2)
    return {int(row["set_id"]): row for row in summary_rows}


# DFN export로부터 세트별 방향 보정계수(orientation factor)를 계산한다.
#   인자: dfn_h5(DFN export HDF5)
#   반환: {set_id: 평균 방향계수}. 계수 = sqrt(1 - n_x^2)의 세트 평균
#   (터널축 x에 대한 법선 성분을 제외한 성분 크기 = 면에 투영되는 정도)
def load_orientation_factors(dfn_h5: str) -> Dict[int, float]:
    data = load_hdf5_dfn(dfn_h5)
    factors: Dict[int, float] = {}
    # 세트별로 법선 벡터를 모아 방향계수의 평균을 구한다
    for set_id in sorted({int(v) for v in np.asarray(data["set_ids"]).tolist()}):
        set_mask = np.asarray(data["set_ids"]) == set_id
        set_normals = np.asarray(data["normals"])[set_mask]
        if set_normals.size == 0:
            factors[set_id] = float("nan")
            continue
        # n_x^2를 뺀 성분 크기(=면 투영 성분). [0,1] 클리핑 후 제곱근
        g_factors = np.sqrt(np.clip(1.0 - set_normals[:, 0] ** 2, 0.0, 1.0))
        factors[set_id] = float(np.mean(g_factors))
    return factors


# P32 파일럿 상태 라벨을 규칙 기반으로 분류한다.
#   인자: site/set_id, adoption_status(kr 채택 상태), recovery_ci_status(CI 회수 상태)
#   반환: p32_hold / p32_provisional* / p32_pilot_candidate* 등의 상태 문자열
def classify_p32_status(site: str, set_id: int, adoption_status: str, recovery_ci_status: str) -> str:
    # 거부된 세트는 보류(hold)
    if adoption_status == "rejected":
        return "p32_hold"
    # Forsmark 특정 세트에 대한 사이트 고유 예외 규칙
    if site == "forsmark" and set_id == 2:
        return "p32_provisional_systematic_bias"
    if site == "forsmark" and set_id in {1, 5}:
        return "p32_pilot_candidate_with_bootstrap_uncertainty"
    # 잠정 채택 / CI 계통편향 여부에 따른 일반 분류
    if adoption_status == "provisional_accepted":
        return "p32_provisional"
    if recovery_ci_status == "systematic_bias":
        return "p32_provisional_systematic_bias"
    return "p32_pilot_candidate"


# 출력 요약에 붙일 진단 노트 문자열을 조립한다.
#   인자: recovery_notes(원 회수 노트), orientation_proxy(방향보정 P21 프록시), p32_hat(P32 추정치)
#   반환: 유효한 항목만 "; "로 이어붙인 노트 문자열
def build_notes(recovery_notes: str, orientation_proxy: float, p32_hat: float) -> str:
    notes = [recovery_notes] if recovery_notes else []
    # 방향 보정된 P21 프록시가 유효하면 기록
    if math.isfinite(orientation_proxy):
        notes.append(f"orientation_corrected_p21_proxy={orientation_proxy:.6f}")
    # P32 추정 방법 태그 추가
    if math.isfinite(p32_hat):
        notes.append("p32_method=support_scaled_generator_proxy")
    return "; ".join(note for note in notes if note)


# 스크립트 엔트리포인트: 인자 파싱 -> 입력 로드 -> 세트별 P32 계산 -> 요약 CSV 저장.
def main() -> None:
    # --- CLI 인자 정의(입력 CSV/HDF5 경로 및 출력 경로) ---
    parser = argparse.ArgumentParser(description="Build P32 pilot summary for accepted/provisional effective-rmin sets.")
    parser.add_argument(
        "--final-summary-csv",
        default="storage/output/final_kr_recovery_summary_effective_rmin.csv",
        help="Final effective-rmin kr recovery summary CSV.",
    )
    parser.add_argument(
        "--bootstrap-summary-csv",
        default="storage/output/final_kr_bootstrap_effective_rmin/final_kr_bootstrap_summary_effective_rmin.csv",
        help="Bootstrap CI summary CSV.",
    )
    parser.add_argument(
        "--p32-label",
        default="P32_r_ge_0p5m",
        help="P32 label to report in the output summary.",
    )
    parser.add_argument(
        "--rough-mesh-h5",
        default="storage/output/rough_face_mesh_collection/synthetic_rough_face_collection.h5",
        help="Rough face mesh collection used to compute observed P21.",
    )
    parser.add_argument(
        "--forsmark-dfn-h5",
        default="storage/output/dfn_forsmark_rmin0p5/dfn_export_for_python.h5",
        help="Forsmark DFN export HDF5.",
    )
    parser.add_argument(
        "--forsmark-trace-h5",
        default="storage/output/forsmark_rmin0p5_trace_dataset_collection/trace_dataset_3d.h5",
        help="Forsmark trace dataset HDF5.",
    )
    parser.add_argument(
        "--laxemar-dfn-h5",
        default="storage/output/dfn_laxemar_rmin0p5/dfn_export_for_python.h5",
        help="Laxemar DFN export HDF5.",
    )
    parser.add_argument(
        "--laxemar-trace-h5",
        default="storage/output/laxemar_rmin0p5_trace_dataset_collection/trace_dataset_3d.h5",
        help="Laxemar trace dataset HDF5.",
    )
    parser.add_argument(
        "--accepted-only",
        action="store_true",
        default=False,
        help="If set, exclude provisional_accepted rows and keep only accepted sets.",
    )
    parser.add_argument(
        "--outcsv",
        default="storage/output/p32_pilot_effective_rmin/p32_pilot_summary_effective_rmin.csv",
        help="Output CSV path.",
    )
    args = parser.parse_args()

    # --- 최종 kr 요약 및 부트스트랩 CI 요약 로드, (site,set_id) 키 맵 구성 ---
    final_rows = read_csv(args.final_summary_csv)
    bootstrap_rows = read_csv(args.bootstrap_summary_csv)
    bootstrap_map: Dict[Tuple[str, int], dict] = {
        (str(row["site"]), int(row["set_id"])): row for row in bootstrap_rows
    }

    # --- 사이트별 관측 P21 맵과 방향계수 맵을 미리 계산해 둔다 ---
    site_context = {
        "forsmark": {
            "p21_map": load_p21_summary(args.forsmark_trace_h5, args.rough_mesh_h5),
            "orientation_map": load_orientation_factors(args.forsmark_dfn_h5),
        },
        "laxemar": {
            "p21_map": load_p21_summary(args.laxemar_trace_h5, args.rough_mesh_h5),
            "orientation_map": load_orientation_factors(args.laxemar_dfn_h5),
        },
    }

    # --- 각 최종 행을 순회하며 P32 요약 행을 생성 ---
    out_rows: List[dict] = []
    for row in final_rows:
        site = str(row["site"])
        set_id = int(row["set_id"])
        # 채택 상태 필터: 거부 세트 제외, accepted-only 옵션 시 잠정채택도 제외
        adoption_status = str(row.get("adoption_status", ""))
        if adoption_status == "rejected":
            continue
        if adoption_status == "provisional_accepted" and args.accepted_only:
            continue

        # 부트스트랩/관측 P21/방향계수 조회 및 방향 보정 P21 프록시 계산
        boot_row = bootstrap_map.get((site, set_id), {})
        p21_row = site_context[site]["p21_map"].get(set_id, {})
        orientation_factor = site_context[site]["orientation_map"].get(set_id, float("nan"))
        observed_p21 = to_float(p21_row, "P21_observed")
        orientation_proxy = (
            observed_p21 / orientation_factor
            if math.isfinite(observed_p21) and math.isfinite(orientation_factor) and orientation_factor > 0.0
            else float("nan")
        )

        # 지지 rmin, 사용 kr(point estimate), kr CI 경계값 추출
        set_likelihood_rmin = to_float(row, "set_likelihood_rmin")
        kr_used = to_float(row, "kr_hat")
        kr_ci_low = to_float(boot_row, "kr_ci_low")
        kr_ci_high = to_float(boot_row, "kr_ci_high")
        # 점추정 kr에서의 P32
        p32_hat = support_scaled_p32(
            site=site,
            set_id=set_id,
            kr=kr_used,
            support_rmin=set_likelihood_rmin,
            rmax=250.0,
        )
        # kr CI 하한/상한에서의 P32 (kr과 P32는 단조관계가 보장되지 않으므로 둘 다 계산)
        p32_at_kr_ci_low = support_scaled_p32(
            site=site,
            set_id=set_id,
            kr=kr_ci_low,
            support_rmin=set_likelihood_rmin,
            rmax=250.0,
        )
        p32_at_kr_ci_high = support_scaled_p32(
            site=site,
            set_id=set_id,
            kr=kr_ci_high,
            support_rmin=set_likelihood_rmin,
            rmax=250.0,
        )
        # 두 P32 값 중 min/max로 P32 CI 구간 구성(유한값이 없으면 nan)
        p32_ci_candidates = [value for value in (p32_at_kr_ci_low, p32_at_kr_ci_high) if math.isfinite(value)]
        if p32_ci_candidates:
            p32_ci_low = min(p32_ci_candidates)
            p32_ci_high = max(p32_ci_candidates)
        else:
            p32_ci_low = float("nan")
            p32_ci_high = float("nan")

        # P32 파일럿 상태 분류 후 요약 행 누적
        recovery_ci_status = str(boot_row.get("recovery_ci_status", ""))
        p32_status = classify_p32_status(site, set_id, adoption_status, recovery_ci_status)

        out_rows.append(
            {
                "site": site,
                "set_id": set_id,
                "p32_label": args.p32_label,
                "set_effective_generation_rmin": to_float(row, "set_effective_generation_rmin"),
                "set_likelihood_rmin": set_likelihood_rmin,
                "kr_used": kr_used,
                "kr_ci_low": kr_ci_low,
                "kr_ci_high": kr_ci_high,
                "kr_adoption_status": adoption_status,
                "observed_P21": observed_p21,
                "orientation_factor": orientation_factor,
                "P32_hat": p32_hat,
                "P32_ci_low": p32_ci_low,
                "P32_ci_high": p32_ci_high,
                "p32_pilot_status": p32_status,
                "notes": build_notes(str(row.get("notes", "")), orientation_proxy, p32_hat),
            }
        )

    # --- 출력 디렉터리 생성 후 요약 CSV 저장 ---
    os.makedirs(os.path.dirname(args.outcsv), exist_ok=True)
    write_csv(out_rows, args.outcsv)
    print(f"[*] P32 pilot summary written to: {args.outcsv}")


if __name__ == "__main__":
    main()
