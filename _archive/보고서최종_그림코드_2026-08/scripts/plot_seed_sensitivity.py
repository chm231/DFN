"""
DFN seed(realization) 민감도 그림 — 형상 고정, seed 만 변화.

7.2/7.3 스윕 CSV 에서 '고정 형상' 부분집합(터널 방위각 θ=0°, 막장면 4개)만 뽑아
seed 30개에 대한 set별 κ / kr / P32 분포를 boxplot(+GT 빨간 점선) 으로 그린다.
좌표계·파이프라인은 sensitivity_tunnel_angle.py 와 동일. 결과 CSV 만 읽으므로 재실행 불필요.
"""
import csv
import math
import os
import sys

import numpy as np

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANGLE_CSV = os.path.join(PROJ, "storage", "output",
                         "sensitivity_tunnel_angle", "sensitivity_results.csv")
OUT_DIR = os.path.join(PROJ, "storage", "output", "sensitivity_seed")
FIX_ANGLE = 0  # 고정 형상 = 회전 없음(θ=0°)
SETS = ["1", "2", "3", "5"]
PARAMS = [("kappa", "Fisher 집중도 κ", "kappa"),
          ("kr_hat", "반지름 멱법칙 지수 kr", "kr_hat"),
          ("P32_hat", "체적 절리밀도 P32", "P32 (m²/m³)")]


def fnum(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except (ValueError, TypeError):
        return None


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False

    # 선택 인자: [angle_csv] [out_dir]  (미지정 시 기존 v0 경로)
    global ANGLE_CSV, OUT_DIR
    if len(sys.argv) > 1:
        ANGLE_CSV = sys.argv[1]
    if len(sys.argv) > 2:
        OUT_DIR = sys.argv[2]

    sys.argv = ["x"]  # build_ground_truth 가 argparse 안 건드리도록
    from scripts.sensitivity_tunnel_angle import build_ground_truth
    gt = build_ground_truth("laxemar", [int(s) for s in SETS], 0.5, 250.0)

    # θ=0° 고정 형상만 수집 → {param: {set: [seed별 값]}}
    data = {c: {s: [] for s in SETS} for c, _, _ in PARAMS}
    n_seed = 0
    seeds = set()
    for r in csv.DictReader(open(ANGLE_CSV)):
        if int(r["angle_deg"]) != FIX_ANGLE or r["set_id"] not in SETS:
            continue
        seeds.add(int(r["seed"]))
        for col, _, _ in PARAMS:
            v = fnum(r.get(col, ""))
            if v is not None:
                data[col][r["set_id"]].append(v)
    n_seed = len(seeds)

    fig, axes = plt.subplots(1, len(PARAMS), figsize=(6.0 * len(PARAMS), 5.0))
    xpos = np.arange(len(SETS))
    for ci, (col, disp, ylab) in enumerate(PARAMS):
        ax = axes[ci]
        box = [data[col][s] for s in SETS]
        bp = ax.boxplot(box, positions=xpos, widths=0.55, patch_artist=True,
                        showfliers=True, medianprops=dict(color="black"))
        for patch in bp["boxes"]:
            patch.set_facecolor("#4C9BE0"); patch.set_alpha(0.7)
        for i, vals in enumerate(box):
            if vals:
                jit = (np.random.RandomState(i).rand(len(vals)) - 0.5) * 0.25
                ax.scatter(np.full(len(vals), xpos[i]) + jit, vals,
                           s=12, color="#08306B", alpha=0.5, zorder=3)
        for i, s in enumerate(SETS):
            gtv = gt.get(col, {}).get(s)
            if gtv is not None:
                ax.hlines(gtv, xpos[i] - 0.3, xpos[i] + 0.3, color="red",
                          ls="--", lw=1.8, zorder=4)
        ax.plot([], [], color="red", ls="--", lw=1.8, label="ground truth")
        ax.legend(loc="upper right", fontsize=8, framealpha=0.85)
        ax.set_title(disp)
        ax.set_ylabel(ylab)
        ax.set_xticks(xpos)
        ax.set_xticklabels([f"Set {s}" for s in SETS])
        ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle(f"DFN seed(realization) 민감도 — 형상 고정(θ=0°, 막장면 4개), "
                 f"realization {n_seed}개", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, "sensitivity_seed.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[plot] {out}  (seeds={n_seed})")


if __name__ == "__main__":
    main()
