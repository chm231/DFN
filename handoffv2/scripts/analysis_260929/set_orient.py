# -*- coding: utf-8 -*-
"""전역 절리군 평균 방향을 경사/경사방향으로 환산하고, 상대측 DS 대응을 대조한다."""
import csv, os, sys
from collections import defaultdict
import numpy as np
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = r"c:/Users/user/OneDrive/2026-1/3D DFN modeling/handoffv2/demodata/DFM_Export/DFM_Export"
MAP  = r"c:/Users/user/OneDrive/2026-1/3D DFN modeling/handoffv2/demo_output/dfm_demo/set_mapping.csv"

g = {}
with open(MAP, newline="", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        g[(r["face"], int(r["source_ds"]))] = int(r["global_set"])

acc, faces_of = defaultdict(list), defaultdict(set)
for face in sorted(d for d in os.listdir(ROOT) if d.isdigit()):
    with open(os.path.join(ROOT, face, "Export 3D DS Data", "3D_disk_data.csv"), newline="") as f:
        for r in csv.DictReader(f):
            k = (face, int(r["DS"]))
            if k not in g:
                continue
            acc[g[k]].append([float(r[f"Normal{c}"]) for c in "XYZ"])
            faces_of[g[k]].add(face)

def dipdir(n):
    n = n / np.linalg.norm(n)
    if n[2] < 0: n = -n                      # 위쪽 법선으로 통일
    dip = np.degrees(np.arccos(min(1.0, abs(n[2]))))
    az  = (np.degrees(np.arctan2(-n[0], -n[1]))) % 360.0   # X=동, Y=북 가정, 북기준 시계방향
    return dip, az

print(f"{'전역set':>7} {'절리선수':>8} {'경사':>6} {'경사방향':>8} {'R̄':>6} {'존재 막장면'}")
for s in sorted(acc):
    N = np.array(acc[s])
    N = N / np.linalg.norm(N, axis=1, keepdims=True)
    N[N[:, 2] < 0] *= -1.0                   # 축성 자료: 반구 통일
    m = N.mean(0); R = np.linalg.norm(m)
    dip, az = dipdir(m)
    print(f"{s:>7} {len(N):>8} {dip:>5.0f}° {az:>7.0f}° {R:>6.3f} {','.join(sorted(faces_of[s]))}")

print("\n[상대측 지적 대조] '88/305 급경사군 = 면1 DS2, 면2 DS3, 면3·4 DS1'")
for face, ds in (("01", 2), ("02", 3), ("03", 1), ("04", 1)):
    print(f"  면{face} DS{ds} -> 전역 set {g[(face, ds)]}")

print("\n[축성 자료 정식 처리: 방향텐서 주고유벡터]")
print(f"{'전역set':>7} {'절리선수':>8} {'경사':>6} {'경사방향':>8} {'집중도 e1':>9} {'존재 막장면'}")
for s in sorted(acc):
    N = np.array(acc[s]); N /= np.linalg.norm(N, axis=1, keepdims=True)
    T = (N.T @ N) / len(N)
    w, V = np.linalg.eigh(T)
    v = V[:, -1]
    dip, az = dipdir(v)
    print(f"{s:>7} {len(N):>8} {dip:>5.0f}° {az:>7.0f}° {w[-1]:>9.3f} {','.join(sorted(faces_of[s]))}")
