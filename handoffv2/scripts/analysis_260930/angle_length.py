# -*- coding: utf-8 -*-
"""절리선 길이가 균열면-막장면 각도에 의존하는가? set·검열을 통제해 확인한다.

순수 현(chord) 모형에서는 교선 길이 분포가 이면각 φ와 무관하다.
  원판 중심의 면수직 거리 d ~ U[-r sinφ, r sinφ]  ->  반현 = r sqrt(1-u²), u~U[-1,1]
  E[반현]/r = π/4 ≈ 0.785  (φ에 무관)
sinφ는 '교차 개수'에만 들어가고 '길이 분포'에는 들어가지 않는다.
따라서 set 내부에서 길이~각도 상관이 있으면 관측 모형이 틀렸다는 신호다.
반대로 set 사이 상관은 set별 크기분포 차이로도 생기므로 증거가 되지 못한다.
"""
import csv, json, sys
import numpy as np
from scipy import stats
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = r"c:/Users/user/OneDrive/2026-1/3D DFN modeling/handoffv2/demo_output/dfm_demo"
D = json.load(open(ROOT + "/conversion_diagnostics.json", encoding="utf-8"))
Rot = np.array(D["coordinate_system"]["world_to_pipeline_rotation"], dtype=float)
FN = {int(k): Rot @ np.array(v, dtype=float) for k, v in D["face_normal_world"].items()}
for k in FN:
    FN[k] /= np.linalg.norm(FN[k])

sid, fid, L, N, cc = [], [], [], [], []
with open(ROOT + "/trace_dataset/trace_dataset_3d.csv", newline="", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        sid.append(int(r["set_id"])); fid.append(int(r["face_id"]))
        L.append(float(r["observed_length_m"])); cc.append(int(r["censoring_class"]))
        N.append([float(r["nx"]), float(r["ny"]), float(r["nz"])])
sid = np.array(sid); fid = np.array(fid); L = np.array(L); cc = np.array(cc)
N = np.array(N); N /= np.linalg.norm(N, axis=1, keepdims=True)

F = np.array([FN[i] for i in fid])
cosnn = np.abs(np.einsum("ij,ij->i", N, F))
alpha = np.degrees(np.arccos(np.clip(cosnn, 0, 1)))   # 0=균열이 막장면과 평행, 90=수직
sinphi = np.sqrt(np.maximum(1 - cosnn**2, 0))

print(f"전체 {len(L)}개 | 이면각 α 중앙 {np.median(alpha):.1f}° (5~95%: {np.percentile(alpha,5):.1f}~{np.percentile(alpha,95):.1f}°)")
u = cc == 0
print(f"비검열 {u.sum()}개만 사용 (한쪽 {int((cc==1).sum())}, 양쪽 {int((cc==2).sum())} 제외)\n")

def rep(tag, m):
    if m.sum() < 30:
        print(f"{tag:>22} n={m.sum():>5}  (표본 부족)"); return
    rho, p = stats.spearmanr(alpha[m], L[m])
    lo = alpha[m] < np.median(alpha[m]); hi = ~lo
    print(f"{tag:>22} n={m.sum():>5}  ρ={rho:+.3f} (p={p:.1e})  "
          f"α낮은절반 L중앙 {np.median(L[m][lo]):.3f} m / 높은절반 {np.median(L[m][hi]):.3f} m  "
          f"배율 {np.median(L[m][hi])/np.median(L[m][lo]):.2f}")

print("[통제 없음 — set 섞임]")
rep("전체", u)
print("\n[set 내부 — 크기분포 차이 통제]")
for s in np.unique(sid):
    rep(f"set {s}", u & (sid == s))
print("\n[set 내부 + 고해상 7면만 — 검출하한 차이 통제]")
hi7 = np.isin(fid, [1, 2, 3, 4, 5, 6, 8])
for s in np.unique(sid):
    rep(f"set {s}", u & (sid == s) & hi7)
print("\n[set 내부 + 고해상 7면 + L>=0.5 m — 검출하한 위]")
for s in np.unique(sid):
    rep(f"set {s}", u & (sid == s) & hi7 & (L >= 0.5))

print("\n[set별 각도·길이 — set 사이 상관의 정체]")
print(f"{'set':>5} {'n':>6} {'α중앙':>7} {'sinφ중앙':>9} {'L중앙':>7} {'L평균':>7}")
for s in np.unique(sid):
    m = u & (sid == s)
    print(f"{s:>5} {m.sum():>6} {np.median(alpha[m]):>6.1f}° {np.median(sinphi[m]):>9.3f} "
          f"{np.median(L[m]):>7.3f} {L[m].mean():>7.3f}")
