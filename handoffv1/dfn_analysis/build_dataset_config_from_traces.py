# =============================================================================
# 파일 역할: 관측 trace로부터 추정한 방향(orientation)과 kr을 이용해, P32 추정용
#           dataset config(JSON)를 자동 생성한다.
#   - 기존에는 방향(trend/plunge/kappa)을 preset(참값)에서 가져왔으나, 여기서는
#     trace h5에 저장된 per-trace 법선(trace_normal_xyz)으로부터 추정한다.
#   - 이렇게 만든 config를 estimate_p32_mc_calibrated에 --config로 넘기면, P32가
#     "trace에서 추정한 크기·방향"만으로 계산되는 자기완결적 파이프라인이 된다.
#
# 주요 입력: trace h5(traces/set_id, trace_normal_xyz, trace_normal_valid),
#            kr 요약 CSV(kr_hat)
# 주요 출력: dataset config JSON (dataset_config.py 스키마)
# 핵심 처리 흐름:
#   1) trace h5에서 set별 유효 법선 로드
#   2) estimate_fisher_k_axial 로 평균 pole + kappa 추정
#   3) 평균 pole -> (trend, plunge)  [window_mc 규약: pole=[cosP cosT, cosP sinT, -sinP]]
#   4) kr_hat(추정값)을 함께 기록해 JSON으로 저장
# =============================================================================
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import h5py

from dfn_analysis.estimate_fisher_kappa import estimate_fisher_k_axial


# 모델 xyz 평균 pole 벡터를 window_mc 규약의 (trend, plunge)로 변환한다.
#   window_mc.mean_pole_from_trend_plunge 는 pole = [cosP cosT, cosP sinT, -sinP] 이므로
#   역변환은 trend = atan2(ny, nx), plunge = asin(-nz) (하반구: nz<=0) 가 되어야
#   같은 pole을 재현한다. (compare 스크립트의 atan2(nx,ny) 규약과 다름 — 주의)
def normal_to_trend_plunge_windowmc(normal_xyz: np.ndarray) -> tuple:
    n = np.asarray(normal_xyz, dtype=np.float64)
    n = n / np.linalg.norm(n)
    # 법선은 축(axial)이라 부호 무의미 → 하반구(z=Up 기준 nz<=0)로 정렬해 plunge>=0
    if n[2] > 0.0:
        n = -n
    trend = float(np.degrees(np.arctan2(n[1], n[0])) % 360.0)
    plunge = float(np.degrees(np.arcsin(np.clip(-n[2], -1.0, 1.0))))
    return trend, plunge


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trace-h5", required=True, help="Trace HDF5 with traces/trace_normal_xyz.")
    ap.add_argument("--kr-summary-csv", required=True, help="kr summary CSV (kr_hat per set).")
    ap.add_argument("--dataset-name", required=True,
                    help="dataset_name written into the config (must match the site label "
                         "used elsewhere, e.g. the kr summary 'site' column / P32 --site).")
    ap.add_argument("--target-set", nargs="+", type=int, required=True)
    ap.add_argument("--dist-type", default="powerlaw", choices=["powerlaw", "exponential"])
    ap.add_argument("--out", required=True, help="Output config JSON path.")
    args = ap.parse_args()

    # 1) trace h5에서 set_id / 법선 / 유효 플래그 로드
    with h5py.File(args.trace_h5, "r") as f:
        g = f["traces"]
        set_id = g["set_id"][:].ravel().astype(int)
        trace_normal_xyz = g["trace_normal_xyz"][:].astype(np.float64)
        trace_normal_valid = g["trace_normal_valid"][:].ravel().astype(int)

    # kr_hat 조회용 (set_id -> row)
    kr_by_set = {int(r["set_id"]): r for r in csv.DictReader(open(args.kr_summary_csv, newline=""))}

    sets: dict = {}
    print(f"{'set':>4} {'n_valid':>7} {'trend':>7} {'plunge':>7} {'kappa':>7} {'kr_hat':>7}")
    for s in args.target_set:
        # 2) set별 유효 법선만 추출
        mask = (set_id == s) & (trace_normal_valid == 1)
        normals = trace_normal_xyz[mask]
        if len(normals) < 2:
            print(f"{s:>4} {len(normals):>7}   <2 valid normals -> skipped")
            continue
        # 2) 평균 pole + Fisher kappa 추정
        fs = estimate_fisher_k_axial(normals)
        mean_normal = fs.get("mean_normal")
        kappa = float(fs.get("kappa", float("nan")))
        if mean_normal is None:
            print(f"{s:>4} {len(normals):>7}   no mean normal -> skipped")
            continue
        # 3) 평균 pole -> (trend, plunge) [window_mc 규약]
        trend, plunge = normal_to_trend_plunge_windowmc(mean_normal)
        # kr(추정값): P32는 kr을 kr_summary CSV에서 직접 읽지만, config에도 기록해 자기설명적으로 둔다
        kr_hat = float(kr_by_set[s]["kr_hat"]) if s in kr_by_set else float("nan")

        sets[str(s)] = {
            "dist_type": args.dist_type,
            "trend": round(trend, 2),
            "plunge": round(plunge, 2),
            "kappa": round(kappa, 2),
            "kr_hat_estimated": round(kr_hat, 3),
        }
        print(f"{s:>4} {len(normals):>7} {trend:>7.1f} {plunge:>7.1f} {kappa:>7.1f} {kr_hat:>7.2f}")

    cfg = {"dataset_name": args.dataset_name, "sets": sets}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    print(f"\nEstimated dataset config written to {out}")


if __name__ == "__main__":
    main()
