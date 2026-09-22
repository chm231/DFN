# -*- coding: utf-8 -*-
"""절리선 법선·공면 오차를 실측 자료에서 직접 측정 (정답 라벨 불필요).

[착안] 같은 막장면 위에서 '거의 같은 직선 위에 이어져 있는' 조각들은 한 균열의
  트레이스가 매핑 과정에서 쪼개진 것일 가능성이 매우 높다. 면 간 매칭과 달리
  기하 식별이 거의 모호하지 않다(공선 + 인접 + 같은 면).
  그 조각들은 '참 법선이 같다'고 볼 수 있으므로, 조각 간 법선 차이가 곧
  **법선 추정 오차**다. association 의 각도 허용치를 튜닝 대신 이 값으로 정할 수 있다.

[측정]
  1) 법선 축각 산포  -> 각도 허용치(현행 15deg)의 근거
  2) 이격거리 L 에 따른 면외 편차 -> 공면 허용치(현행 고정 0.15 m)의 근거
     면외 편차 ~ L * sin(법선오차) 이므로 고정값이 아니라 L 에 비례해야 한다.

사용: (handoffv3) python scripts/noise/measure_trace_noise.py [trace_h5]
"""
import sys
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
H5 = Path(sys.argv[1]) if len(sys.argv) > 1 else \
    ROOT / "demo_output/_seq2/cum_7/trace_dataset/trace_dataset_3d.h5"

# 조각쌍 식별 기준 (엄격) — 기본값
ANG_INPLANE = 3.0     # 면내 방향 차이 [deg]
PERP_MAX = 0.02       # 두 직선 사이 수직 이격 [m]
GAP_MAX = 0.30        # 직선 방향 틈 [m]


def load(h5):
    with h5py.File(h5) as f:
        g = f["traces"]
        return dict(
            p0=g["p0_xyz"][...].astype(float), p1=g["p1_xyz"][...].astype(float),
            n=g["trace_normal_xyz"][...].astype(float),
            valid=g["trace_normal_valid"][...].astype(bool),
            fx=np.round(g["face_x_m"][...].astype(float).ravel(), 3),
            sid=g["set_id"][...].astype(int).ravel(),
            L=g["observed_length_m"][...].astype(float).ravel(),
        )


def fragment_pairs(d, ang_inplane=ANG_INPLANE, perp_max=PERP_MAX, gap_max=GAP_MAX):
    """같은 면에서 공선·인접한 조각쌍 (i, j) 를 찾는다."""
    pairs = []
    mid = 0.5 * (d["p0"] + d["p1"])
    vec = d["p1"] - d["p0"]
    ln = np.linalg.norm(vec, axis=1)
    ok = ln > 1e-6
    u = np.zeros_like(vec)
    u[ok] = vec[ok] / ln[ok, None]
    cos_tol = np.cos(np.radians(ang_inplane))
    for fx in np.unique(d["fx"]):
        idx = np.nonzero((d["fx"] == fx) & ok & d["valid"])[0]
        if len(idx) < 2:
            continue
        # 면 내 좌표(y,z)로 판정
        m2, u2 = mid[idx][:, 1:3], u[idx][:, 1:3]
        nu = np.linalg.norm(u2, axis=1)
        keep = nu > 1e-9
        idx, m2, u2, nu = idx[keep], m2[keep], u2[keep], nu[keep]
        u2 = u2 / nu[:, None]
        for a in range(len(idx)):
            dm = m2 - m2[a]
            cosang = np.abs(u2 @ u2[a])
            along = dm @ u2[a]
            perp = np.abs(dm[:, 0] * u2[a, 1] - dm[:, 1] * u2[a, 0])
            half = 0.5 * (ln[idx] + ln[idx[a]])
            gap = np.maximum(np.abs(along) - half, 0.0)
            sel = (cosang >= cos_tol) & (perp <= perp_max) & (gap <= gap_max)
            sel[a] = False
            for b in np.nonzero(sel)[0]:
                if idx[a] < idx[b]:
                    pairs.append((int(idx[a]), int(idx[b])))
    return pairs


def main():
    d = load(H5)
    n = d["n"] / np.maximum(np.linalg.norm(d["n"], axis=1, keepdims=True), 1e-12)
    pairs = fragment_pairs(d)
    print(f"자료 {H5.name}   트레이스 {len(d['fx']):,}개, 면 {len(np.unique(d['fx']))}개")
    print(f"조각쌍 식별 기준: 면내각<={ANG_INPLANE}deg, 수직이격<={PERP_MAX}m, 틈<={GAP_MAX}m")
    print(f"  → 식별된 조각쌍 {len(pairs):,}개\n")
    if not pairs:
        return
    a = np.array([p[0] for p in pairs]); b = np.array([p[1] for p in pairs])
    # 1) 법선 축각 (같은 균열이라고 보므로 이 값이 곧 추정 오차)
    cos = np.abs(np.einsum("ij,ij->i", n[a], n[b])).clip(0, 1)
    ang = np.degrees(np.arccos(cos))
    # 2) 이격거리와 면외 편차
    mid = 0.5 * (d["p0"] + d["p1"])
    sep = np.linalg.norm(mid[a] - mid[b], axis=1)
    dvec = mid[b] - mid[a]
    outp = np.maximum(np.abs(np.einsum("ij,ij->i", dvec, n[a])),
                      np.abs(np.einsum("ij,ij->i", dvec, n[b])))
    q = [50, 75, 90, 95, 99]
    print("  [1] 법선 축각 산포 = 법선 추정 오차")
    print("      " + "  ".join(f"p{x}={np.percentile(ang,x):5.1f}deg" for x in q) +
          f"   평균 {ang.mean():.1f}deg")
    print(f"      → 현행 각도 허용치 15deg 는 p{int((ang<=15).mean()*100)} 수준을 덮는다")
    print()
    print("  [2] 트레이스 길이별 법선 오차 (짧을수록 부정확할 것)")
    Lp = 0.5 * (d["L"][a] + d["L"][b])
    for lo, hi in [(0, 0.2), (0.2, 0.5), (0.5, 1.0), (1.0, 99)]:
        m = (Lp >= lo) & (Lp < hi)
        if m.sum() > 20:
            print(f"      길이 {lo:.1f}~{hi if hi<99 else float('inf'):>4} m: n={int(m.sum()):5d}  "
                  f"중앙 {np.median(ang[m]):5.1f}deg  p90 {np.percentile(ang[m],90):5.1f}deg")
    print()
    print("  [3] 면외 편차 vs 이격거리  (공면 허용치는 고정이어야 하나?)")
    print(f"      {'이격 L[m]':>12s} {'n':>6s} {'면외편차 중앙':>13s} {'p90':>8s} {'p90/L':>8s}")
    for lo, hi in [(0, 0.2), (0.2, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 99)]:
        m = (sep >= lo) & (sep < hi)
        if m.sum() > 20:
            p90 = np.percentile(outp[m], 90)
            print(f"      {lo:5.1f}~{hi if hi<99 else 9.9:<5.1f} {int(m.sum()):6d} "
                  f"{np.median(outp[m]):12.4f} {p90:8.4f} {p90/np.median(sep[m]):8.3f}")
    print()
    print("  [4] 현행 고정 공면 허용치 0.15 m 를 각도로 환산하면")
    for L in (0.3, 1.0, 2.1, 4.2):
        print(f"      이격 {L:4.1f} m 에서 0.15 m  =  {np.degrees(np.arcsin(min(0.15/L,1))):5.1f}deg 상당")


if __name__ == "__main__":
    main()
