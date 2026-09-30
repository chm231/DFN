# -*- coding: utf-8 -*-
"""각도-길이 상관이 '패치 경계'인가 '짧은 절리선의 법선 추정 잡음'인가.

짧은 절리선은 패치 점이 적어 평면 적합이 불안정하다. 법선이 trace 축 주위로 거의
자유로우면 |n·f|가 사실상 무작위가 되고, 이때 α는 [0,90]에 고르게 퍼진다(중앙 45°).
-> set 평균축과의 편차가 짧은 절리선에서 커지는지, α 분포가 균등에 가까워지는지 본다.
"""
import csv, json, sys
import numpy as np
from scipy import stats
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = r"c:/Users/user/OneDrive/2026-1/3D DFN modeling/handoffv2/demo_output/dfm_demo"
D = json.load(open(ROOT + "/conversion_diagnostics.json", encoding="utf-8"))
Rot = np.array(D["coordinate_system"]["world_to_pipeline_rotation"], float)
FN = {int(k): Rot @ np.array(v, float) for k, v in D["face_normal_world"].items()}
for k in FN: FN[k] /= np.linalg.norm(FN[k])

sid, fid, L, N, cc, P0, P1 = [], [], [], [], [], [], []
with open(ROOT + "/trace_dataset/trace_dataset_3d.csv", newline="", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        sid.append(int(r["set_id"])); fid.append(int(r["face_id"]))
        L.append(float(r["observed_length_m"])); cc.append(int(r["censoring_class"]))
        N.append([float(r["nx"]), float(r["ny"]), float(r["nz"])])
        P0.append([float(r["p0_x"]), float(r["p0_y"]), float(r["p0_z"])])
        P1.append([float(r["p1_x"]), float(r["p1_y"]), float(r["p1_z"])])
sid=np.array(sid); fid=np.array(fid); L=np.array(L); cc=np.array(cc)
N=np.array(N); N/=np.linalg.norm(N,axis=1,keepdims=True)
T=np.array(P1)-np.array(P0); T/=np.linalg.norm(T,axis=1,keepdims=True)
F=np.array([FN[i] for i in fid])
alpha=np.degrees(np.arccos(np.clip(np.abs(np.einsum("ij,ij->i",N,F)),0,1)))
u=(cc==0)

def axmean(V):
    T_=(V.T@V)/len(V); w,Q=np.linalg.eigh(T_); return Q[:,-1]

BINS=[(0,0.2),(0.2,0.3),(0.3,0.5),(0.5,0.8),(0.8,1.5),(1.5,99)]
print("[set 1~4, 비검열] 길이대별 — set 평균축과의 축간각 편차 / α 분포")
print(f"{'길이구간':>12} {'n':>5} {'편차중앙':>8} {'편차90%':>8} {'α중앙':>7} {'α 1사분':>8} {'α 3사분':>8} {'KS(균등0-90) p':>14}")
for s in (1,2,3,4):
    m0=u&(sid==s)
    ax=axmean(N[m0])
    dev=np.degrees(np.arccos(np.clip(np.abs(N@ax),0,1)))
    print(f"  --- set {s} (n={m0.sum()}, 평균축 편차 중앙 {np.median(dev[m0]):.1f}°) ---")
    for lo,hi in BINS:
        m=m0&(L>=lo)&(L<hi)
        if m.sum()<20: continue
        ks=stats.kstest(alpha[m]/90.0,"uniform").pvalue
        print(f"{f'{lo}~{hi} m':>12} {m.sum():>5} {np.median(dev[m]):>7.1f}° {np.percentile(dev[m],90):>7.1f}° "
              f"{np.median(alpha[m]):>6.1f}° {np.percentile(alpha[m],25):>7.1f}° {np.percentile(alpha[m],75):>7.1f}° {ks:>14.2e}")

print("\n[법선이 trace 축에 수직인가 — 적합 자체의 일관성]")
d=np.abs(np.einsum("ij,ij->i",N,T))
print(f"  |n·t| 중앙 {np.median(d):.4f}, 90% {np.percentile(d,90):.4f}, 최대 {d.max():.4f}  (0이면 정확히 수직)")
for lo,hi in BINS:
    m=u&(L>=lo)&(L<hi)
    if m.sum()<20: continue
    print(f"  {lo}~{hi} m  n={m.sum():>5}  |n·t| 중앙 {np.median(d[m]):.4f}  90% {np.percentile(d[m],90):.4f}")

print("\n[요약 상관] 비검열 set1~4")
m=u&np.isin(sid,[1,2,3,4])
for s in (1,2,3,4):
    mm=u&(sid==s); ax=axmean(N[mm])
    dev=np.degrees(np.arccos(np.clip(np.abs(N@ax),0,1)))
    r1,p1_=stats.spearmanr(L[mm],dev[mm])
    print(f"  set {s}: 길이 vs 평균축편차 ρ={r1:+.3f} (p={p1_:.1e})")
