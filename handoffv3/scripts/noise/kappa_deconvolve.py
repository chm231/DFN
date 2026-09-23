# -*- coding: utf-8 -*-
"""법선 측정 노이즈를 제거한 Fisher kappa 추정 (과분산 보정).

[문제] kappa 는 원시 trace_normal_xyz 에서 직접 적합된다
  (build_dataset_config_from_traces.py:84,89 — 평균화 없음).
  2026-09-22 측정에서 이 법선의 쌍간 축각 산포가 중앙 ~18deg 로 나왔다.
  측정 노이즈가 섞이면 법선이 실제보다 퍼져 보이므로 **kappa 가 과소추정**된다.

[원리] 관측 법선 = 참 법선에 독립 회전 오차가 더해진 것이라 보면,
  합벡터 기댓값이 분해된다:   R_obs = R_true * R_noise
  두 독립 관측이 같은 참 법선을 가질 때 E[cos(쌍간각)] = R_noise^2 이므로
  조각쌍에서 R_noise 를 직접 잰다. 그러면 R_true = R_obs / R_noise.

[검증] 알려진 kappa_true + 알려진 노이즈로 합성해 회복되는지 확인한다.

사용: (handoffv3) python scripts/noise/kappa_deconvolve.py
"""
import sys
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from dfn_analysis.estimate_fisher_kappa import estimate_fisher_k_axial
from measure_trace_noise import load, fragment_pairs, H5


def kappa_from_R(r):
    r = float(np.clip(r, 0.0, 1 - 1e-12))
    return (3.0 * r - r ** 3) / max(1e-12, 1.0 - r ** 2)


def R_from_kappa(k, lo=0.0, hi=1 - 1e-12):
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if kappa_from_R(mid) < k:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def vmf(mean_dirs, kappa, rng):
    """각 행의 평균방향 주위로 vMF(kappa) 표본 1개씩 (벡터화)."""
    n = len(mean_dirs)
    u = rng.uniform(0, 1, n)
    w = 1.0 + np.log(u + (1 - u) * np.exp(-2.0 * kappa)) / kappa
    ph = rng.uniform(0, 2 * np.pi, n)
    s = np.sqrt(np.maximum(1 - w * w, 0.0))
    loc = np.stack([s * np.cos(ph), s * np.sin(ph), w], 1)
    # z축 -> mean_dirs 로 회전 (Rodrigues)
    z = np.array([0.0, 0.0, 1.0])
    v = np.cross(z, mean_dirs)
    c = mean_dirs[:, 2]
    k2 = np.linalg.norm(v, axis=1)
    out = np.empty_like(loc)
    flat = k2 < 1e-12
    out[flat] = loc[flat] * np.sign(np.where(c[flat] >= 0, 1.0, -1.0))[:, None]
    m = ~flat
    if m.any():
        vv = v[m] / k2[m][:, None]
        th = np.arctan2(k2[m], c[m])
        ct, st = np.cos(th)[:, None], np.sin(th)[:, None]
        L = loc[m]
        out[m] = (L * ct + np.cross(vv, L) * st +
                  vv * (np.einsum('ij,ij->i', vv, L))[:, None] * (1 - ct))
    return out / np.linalg.norm(out, axis=1, keepdims=True)


def R_obs_of(normals):
    fs = estimate_fisher_k_axial(np.asarray(normals, float))
    return float(fs.get("resultant_length", 0.0)) / max(1, len(normals)), float(fs.get("kappa", np.nan))


def main():
    rng = np.random.default_rng(11)

    # ---------- 1) 절차 검증 (합성) ----------
    print("[검증] 알려진 kappa_true + 알려진 노이즈에서 회복되는가")
    print(f"  {'kappa_true':>10s} {'kappa_noise':>11s} {'kappa_obs':>10s} {'회복 kappa':>10s} {'오차':>7s}")
    for kt in (15.0, 30.0, 60.0):
        for kn in (10.0, 20.0, 40.0):
            mean = np.tile(np.array([0.3, 0.2, 0.93]) / np.linalg.norm([0.3, 0.2, 0.93]), (80000, 1))
            true = vmf(mean, kt, rng)
            obs = vmf(true, kn, rng)
            r_obs, k_obs = R_obs_of(obs)
            r_noise = R_from_kappa(kn)
            k_rec = kappa_from_R(min(r_obs / r_noise, 1 - 1e-12))
            print(f"  {kt:10.1f} {kn:11.1f} {k_obs:10.1f} {k_rec:10.1f} {(k_rec/kt-1)*100:6.1f}%")

    # ---------- 2) 실측 노이즈 크기 ----------
    d = load(H5)
    n = d["n"] / np.maximum(np.linalg.norm(d["n"], axis=1, keepdims=True), 1e-12)
    print("\n[실측] 조각쌍에서 R_noise 추정   (E[cos(쌍간각)] = R_noise^2)")
    print(f"  {'식별 기준':28s} {'쌍':>5s} {'중앙각':>7s} {'E[cos]':>8s} {'R_noise':>8s} {'kappa_noise':>11s}")
    cands = []
    for lbl, (ai, pm, gm) in [("엄격 (0.5deg,2mm,0.4m)", (0.5, 0.002, 0.40)),
                              ("중간 (1.0deg,5mm,0.3m)", (1.0, 0.005, 0.30)),
                              ("기본 (3.0deg,20mm,0.3m)", (3.0, 0.020, 0.30))]:
        p = fragment_pairs(d, ai, pm, gm)
        if len(p) < 8:
            continue
        a = np.array([x[0] for x in p]); b = np.array([x[1] for x in p])
        c = np.abs(np.einsum("ij,ij->i", n[a], n[b])).clip(0, 1)
        med = np.degrees(np.arccos(np.median(c)))
        r_noise = float(np.sqrt(max(c.mean(), 1e-9)))
        cands.append((lbl, r_noise))
        print(f"  {lbl:28s} {len(p):5d} {med:6.1f}° {c.mean():8.3f} {r_noise:8.3f} {kappa_from_R(r_noise):11.1f}")

    # ---------- 3) set 별 kappa 보정 ----------
    with h5py.File(H5) as f:
        sid = f["traces/set_id"][...].astype(int).ravel()
        valid = f["traces/trace_normal_valid"][...].astype(bool)
    print("\n[보정] set 별 kappa  (현행 = 원시 법선 직접 적합)")
    print(f"  {'set':>4s} {'n':>6s} {'R_obs':>7s} {'현행 kappa':>10s} | " +
          " | ".join(f"{lbl.split()[0]} 보정" for lbl, _ in cands))
    for s in sorted(set(sid.tolist())):
        m = (sid == s) & valid
        if m.sum() < 50:
            continue
        r_obs, k_obs = R_obs_of(n[m])
        cells = []
        for lbl, rn in cands:
            ratio = r_obs / rn
            cells.append("  발산(>1)" if ratio >= 1 else f"{kappa_from_R(ratio):9.1f}")
        print(f"  {s:4d} {int(m.sum()):6d} {r_obs:7.3f} {k_obs:10.1f} | " + " | ".join(cells))
    print("\n  * 보정 kappa = kappa( R_obs / R_noise ). R_obs/R_noise>=1 이면 노이즈가 관측 산포를")
    print("    전부 설명한다는 뜻 — 참 분산이 0 에 가깝다는 비현실적 결과이므로 노이즈 과대추정 신호다.")


if __name__ == "__main__":
    main()
