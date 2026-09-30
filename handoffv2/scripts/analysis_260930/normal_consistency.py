# -*- coding: utf-8 -*-
"""패치 법선이 자기 절리선 방향에 수직인가 — 원자료 3D 끝점(비투영)으로 확인.

절리선은 균열면 위의 선이므로 반드시 n · t = 0 이어야 한다.
2D_trace_data.csv는 '평면화' 끝점이라 투영 오차가 섞이므로,
3D_disk_data.csv의 TraceP1/P2(비투영 3D 끝점)로 검정한다.
또 원판 중심이 절리선 중점과 같은지(= 원판이 절리선에서 파생됐는지)도 본다.
"""
import csv, os, sys
import numpy as np
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = r"c:/Users/user/OneDrive/2026-1/3D DFN modeling/handoffv2/demodata/DFM_Export/DFM_Export"
faces = sorted(d for d in os.listdir(ROOT) if d.isdigit())

rows = []
for face in faces:
    with open(os.path.join(ROOT, face, "Export 3D DS Data", "3D_disk_data.csv"), newline="") as f:
        for r in csv.DictReader(f):
            rows.append((face,
                np.array([float(r[f"TraceP1{c}"]) for c in "XYZ"]),
                np.array([float(r[f"TraceP2{c}"]) for c in "XYZ"]),
                np.array([float(r[f"Normal{c}"]) for c in "XYZ"]),
                np.array([float(r[f"Center{c}"]) for c in "XYZ"]),
                float(r["Radius"])))
A = np.array([x[1] for x in rows]); B = np.array([x[2] for x in rows])
N = np.array([x[3] for x in rows]); C = np.array([x[4] for x in rows])
R = np.array([x[5] for x in rows])
N /= np.linalg.norm(N, axis=1, keepdims=True)
V = B - A; L = np.linalg.norm(V, axis=1); T = V / L[:, None]

dev = np.degrees(np.arcsin(np.clip(np.abs(np.einsum("ij,ij->i", N, T)), 0, 1)))
print(f"[n 과 t 의 수직성 위반]  n={len(L)}  (0°이면 정확히 수직)")
for q in (50, 75, 90, 95, 99, 100):
    print(f"   {q:>3}백분위  {np.percentile(dev, q):6.2f}°")
print(f"   5° 초과 {100*(dev>5).mean():.1f}%, 10° 초과 {100*(dev>10).mean():.1f}%, 20° 초과 {100*(dev>20).mean():.1f}%")

print("\n[길이대별]")
for lo, hi in ((0,0.2),(0.2,0.3),(0.3,0.5),(0.5,0.8),(0.8,1.5),(1.5,99)):
    m = (L >= lo) & (L < hi)
    if m.sum() < 20: continue
    print(f"   {lo}~{hi} m  n={m.sum():>5}  위반 중앙 {np.median(dev[m]):5.2f}°  90% {np.percentile(dev[m],90):5.2f}°")

mid = (A + B) / 2.0
print("\n[원판 중심이 절리선 중점인가]")
dc = np.linalg.norm(C - mid, axis=1)
print(f"   |중심 - 절리선중점| 중앙 {np.median(dc):.4f} m, 90% {np.percentile(dc,90):.4f} m, 최대 {dc.max():.4f} m")
print(f"   절반길이로 정규화: 중앙 {np.median(dc/(L/2)):.3f}, 90% {np.percentile(dc/(L/2),90):.3f}")
print(f"   반현/반지름 = (L/2)/R : 중앙 {np.median((L/2)/R):.3f}, "
      f"5~95% {np.percentile((L/2)/R,5):.3f}~{np.percentile((L/2)/R,95):.3f}")
print("   무작위 현 모형 기대값 0.785 (0~1 넓게 분포) / 지름이면 1.0 에 집중")
