"""
Leave-One-Face-Out (LOFO) 외삽검증 — 참값 없이 복원의 정합성을 잰다.

원리: 막장면 하나(xf_test)를 빼고 나머지 면들로 DFN 을 복원한 뒤, 복원된 원판들이
      빼놓은 면에 남기는 chord(예측 trace)를 실제 관측 trace 와 비교한다.
      → "일부 면으로 복원한 DFN 이 보지 못한 면의 trace 를 예측하는가?"

지표(면·set 별):
  recall    = 예측된 held-out 관측 trace 수 / held-out 관측 trace 수   (놓친 절리 회수)
  precision = 실제 trace 와 맞는 예측 chord 수 / 예측 chord 수         (환상 예측 억제)
  length ratio = 예측 chord 총길이 / 관측 trace 총길이 (held-out 면)

관측 P21 대비 오차와 달리, LOFO 는 검열 복원(un-censoring)을 보상하지 않는다:
과대한 disc 는 보지 못한 면에 존재하지 않는 trace 를 예측 → precision 하락으로 벌점.
"""
# =============================================================================
# 파일 역할:
#   Leave-One-Face-Out(LOFO) 외삽검증. 참값(GT) 없이 복원 파이프라인의 정합성을
#   측정한다. 면 하나를 빼고 복원한 DFN 이 그 면의 관측 trace 를 예측하는지 본다.
#
# 주요 입력:
#   - --trace-h5 : trace HDF5. /traces(끝점·set_id·face_x_m) + /meta/tunnel_poly_yz +
#                  /meta/face_x_positions_m
#   - --kr-summary-csv (선택): set별 kr_hat (반지름 축소추정용; reconstruct 와 동일)
#   - --target-set / --rmax / --arc-min / --normal-angle-deg / --coplanar-dist /
#     --max-centroid-sep : reconstruct 로 전달되는 복원 파라미터
#
# 주요 출력:
#   - 콘솔 표(파일 저장 없음):
#     [면별]     hold-out 면의 관측 수 / recall / 예측 수 / precision
#     [set별 종합] recall(놓친 절리 회수), precision(환상 예측 억제), 길이비(예측/관측)
#
# 핵심 처리 흐름:
#   1) 관측 trace·터널 폴리곤·면 x위치 로드
#   2) 각 면 xf 를 held-out: 나머지 면만으로 reconstruct.reconstruct(exclude_faces=[xf])
#   3) 복원 disc 가 xf 면에 남기는 예측 chord(visible_trace_on_face) 계산
#   4) 예측 chord vs held-out 관측 trace 매칭(is_reproduced)으로 recall/precision/길이비 집계
#   5) 면별·set별 종합 표를 콘솔 출력
#
# 좌표계: x=East(터널축), y=North, z=Up. 막장면 = x=상수 평면.
# =============================================================================
import argparse
import csv

import h5py
import numpy as np

import dfn_analysis.reconstruct_discs_from_traces as R
from dfn_analysis.generate_conditional_hidden_dfn import visible_trace_on_face, _ccw_polygon
from dfn_analysis.visualize_reconstruction import is_reproduced


def load_meta(trace_h5):
    with h5py.File(trace_h5, "r") as f:
        g = f["traces"]
        p0 = g["p0_xyz"][...].astype(float)
        p1 = g["p1_xyz"][...].astype(float)
        sid = g["set_id"][...].astype(int)
        fx = g["face_x_m"][...].astype(float)
        poly = np.array(f["meta"]["tunnel_poly_yz"])
        faces = sorted(float(x) for x in np.array(f["meta"]["face_x_positions_m"]))
    return p0, p1, sid, fx, poly, faces


def lofo(trace_h5, kr_map, target_sets, angle, coplanar, sep, arc_min, rmax):
    p0, p1, sid, fx, poly, faces = load_meta(trace_h5)
    poly_ccw = _ccw_polygon(poly)
    sets = target_sets or sorted(set(sid.tolist()))

    # 결과 누적: per (set) 및 전체
    agg = {s: dict(n_test=0, recall_hit=0, n_pred=0, prec_hit=0,
                   len_obs=0.0, len_pred=0.0) for s in list(sets) + ["ALL"]}
    per_face = []

    for xf in faces:
        discs = R.reconstruct(trace_h5, "agglomerative", angle, coplanar, sep,
                              target_sets=target_sets, arc_min=arc_min,
                              kr_map=kr_map, rmax=rmax, exclude_faces=[xf])
        # 예측 chord (held-out 면), set별
        pred_by_set = {s: [] for s in sets}
        for d in discs:
            s = d["set_id"]
            if s not in pred_by_set:
                continue
            c = np.array([d["cx"], d["cy"], d["cz"]])
            n = np.array([d["nx"], d["ny"], d["nz"]])
            seg = visible_trace_on_face(c, n, d["radius"], xf, poly_ccw)
            if seg is not None:
                pred_by_set[s].append((seg[0][1:], seg[1][1:]))
        # 실제 held-out 관측 trace, set별
        test_by_set = {s: [] for s in sets}
        for i in range(len(sid)):
            if abs(fx[i] - xf) < 0.5 and sid[i] in test_by_set:
                test_by_set[sid[i]].append((p0[i, 1:], p1[i, 1:]))

        f_test = f_recall = f_pred = f_prec = 0
        for s in sets:
            tests = test_by_set[s]
            preds = pred_by_set[s]
            rec = sum(1 for o0, o1 in tests if is_reproduced(o0, o1, preds))
            prec = sum(1 for r0, r1 in preds if is_reproduced(r0, r1, tests))
            lo = sum(float(np.linalg.norm(o1 - o0)) for o0, o1 in tests)
            lp = sum(float(np.linalg.norm(r1 - r0)) for r0, r1 in preds)
            a = agg[s]
            a["n_test"] += len(tests); a["recall_hit"] += rec
            a["n_pred"] += len(preds); a["prec_hit"] += prec
            a["len_obs"] += lo; a["len_pred"] += lp
            f_test += len(tests); f_recall += rec; f_pred += len(preds); f_prec += prec
        per_face.append((xf, f_test, f_recall, f_pred, f_prec))

    # ALL 집계
    for s in sets:
        for k in ("n_test", "recall_hit", "n_pred", "prec_hit", "len_obs", "len_pred"):
            agg["ALL"][k] += agg[s][k]
    return agg, per_face, sets


def main():
    ap = argparse.ArgumentParser(description="LOFO 외삽검증 (복원 정합성)")
    ap.add_argument("--trace-h5", required=True)
    ap.add_argument("--kr-summary-csv", default=None)
    ap.add_argument("--target-set", nargs="+", type=int, default=None)
    ap.add_argument("--normal-angle-deg", type=float, default=15.0)
    ap.add_argument("--coplanar-dist", type=float, default=0.15)
    ap.add_argument("--max-centroid-sep", type=float, default=2.0)
    ap.add_argument("--arc-min", type=float, default=120.0)
    ap.add_argument("--rmax", type=float, default=250.0)
    args = ap.parse_args()

    kr_map = None
    if args.kr_summary_csv:
        kr_map = {int(r["set_id"]): float(r["kr_hat"])
                  for r in csv.DictReader(open(args.kr_summary_csv))
                  if r.get("kr_hat") not in (None, "", "nan")}

    agg, per_face, sets = lofo(args.trace_h5, kr_map, args.target_set,
                               args.normal_angle_deg, args.coplanar_dist,
                               args.max_centroid_sep, args.arc_min, args.rmax)

    def rate(a, b):
        return 100.0 * a / b if b else float("nan")

    print("=" * 68)
    print(" Leave-One-Face-Out 외삽검증 (참값 미사용)")
    print("=" * 68)
    print(" [면별] hold-out 면의 trace 를 나머지 면 복원으로 예측")
    print(f"   {'면 x':>5} {'관측':>6} {'recall':>8} {'예측':>6} {'precision':>10}")
    for xf, nt, rc, npd, pc in per_face:
        print(f"   {xf:>5g} {nt:>6} {rate(rc,nt):>7.0f}% {npd:>6} {rate(pc,npd):>9.0f}%")
    print("-" * 68)
    print(f" [set별 종합]  recall=놓친절리회수  precision=환상예측억제")
    print(f"   {'set':>4} {'관측':>6} {'recall':>8} {'예측':>6} {'precision':>10} {'길이비(예측/관측)':>16}")
    for s in list(sets) + ["ALL"]:
        a = agg[s]
        lr = a["len_pred"] / a["len_obs"] if a["len_obs"] else float("nan")
        print(f"   {str(s):>4} {a['n_test']:>6} {rate(a['recall_hit'],a['n_test']):>7.0f}% "
              f"{a['n_pred']:>6} {rate(a['prec_hit'],a['n_pred']):>9.0f}% {lr:>15.2f}")
    print("=" * 68)


if __name__ == "__main__":
    main()
