"""보고서 그림 6-14(벤치마크-역산 비교 막대) 재생성 — v1 파이프라인 수치.

comparison_vs_laxemar.csv 를 읽어 3패널 막대그래프를 그린다:
  (a) 반지름 지수 kr   : 실제(GT) vs 역산, 막대 위 % 오차 라벨
  (b) 강도 P32         : 참조(GT) vs 역산, 막대 위 % 오차 라벨
  (c) 방향 정확도       : 절리군별 축 각도차(°), 허용 상한 8° 빨간 점선

실행:
    python scripts/make_fig_validation_bars.py \
        [--base storage/output/pipeline_v1_laxemar] \
        [--out docs/figures/fig_validation_bars_v1.png]
"""
import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False
except Exception:
    pass

SETS = [1, 2, 3, 5]
C_GT = "#b8cce4"    # 실제/참조(연한 파랑)
C_EST = "#1f5fa8"   # 역산(진한 파랑)
C_WARN = "#e8a33d"  # 허용 상한 초과 강조(주황)
ALLOW_ANG = 8.0     # 방향 각오차 허용 상한 (deg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="storage/output/pipeline_v1_laxemar")
    ap.add_argument("--out", default="docs/figures/fig_validation_bars_v1.png")
    args = ap.parse_args()

    path = os.path.join(args.base, "comparison_vs_laxemar.csv")
    rows = {int(r["set_id"]): r for r in csv.DictReader(open(path, encoding="utf-8"))}

    kr_gt = [float(rows[s]["kr_true"]) for s in SETS]
    kr_es = [float(rows[s]["kr_hat"]) for s in SETS]
    kr_er = [float(rows[s]["kr_rel_err_pct"]) for s in SETS]
    p_gt = [float(rows[s]["p32_ref"]) for s in SETS]
    p_es = [float(rows[s]["p32_hat"]) for s in SETS]
    p_er = [float(rows[s]["p32_rel_err_pct"]) for s in SETS]
    ang = [float(rows[s]["orient_ang_err_deg"]) for s in SETS]

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    x = np.arange(len(SETS))
    w = 0.36

    # (a) kr
    axa = axes[0]
    axa.bar(x - w / 2, kr_gt, w, color=C_GT, label="실제(GT)")
    axa.bar(x + w / 2, kr_es, w, color=C_EST, label="역산")
    for i, (v, e) in enumerate(zip(kr_es, kr_er)):
        axa.text(x[i] + w / 2, v + 0.05, f"{e:+.1f}%", ha="center", va="bottom",
                 fontsize=8, color="#c00000" if abs(e) >= 10 else "#333")
    axa.set_title("(a) 반지름 지수 kr")
    axa.set_ylabel("kr")
    axa.set_ylim(0, max(kr_es + kr_gt) * 1.22)
    axa.legend(fontsize=8, loc="lower right")

    # (b) P32
    axb = axes[1]
    axb.bar(x - w / 2, p_gt, w, color="#cde6cf", label="참조(GT)")
    axb.bar(x + w / 2, p_es, w, color="#2e7d43", label="역산")
    for i, (v, e) in enumerate(zip(p_es, p_er)):
        axb.text(x[i] + w / 2, v + 0.015, f"{e:+.1f}%", ha="center", va="bottom",
                 fontsize=8, color="#c00000" if abs(e) >= 10 else "#333")
    axb.set_title("(b) 강도 P32")
    axb.set_ylabel("P32 (m²/m³)")
    axb.set_ylim(0, max(p_es + p_gt) * 1.25)
    axb.legend(fontsize=8, loc="lower right")

    # (c) 방향 정확도
    axc = axes[2]
    cols = [C_WARN if a > ALLOW_ANG else "#2e7d43" for a in ang]
    axc.bar(x, ang, 0.5, color=cols)
    for i, a in enumerate(ang):
        axc.text(x[i], a + 0.15, f"{a:.1f}°", ha="center", va="bottom", fontsize=9)
    axc.axhline(ALLOW_ANG, color="red", ls="--", lw=1.4)
    axc.text(len(SETS) - 0.55, ALLOW_ANG + 0.15, f"허용 상한 ~{ALLOW_ANG:.0f}°",
             color="red", fontsize=8, ha="right")
    axc.set_title("(c) 방향 정확도")
    axc.set_ylabel("평균 pole 축 각도차 (°)")
    axc.set_ylim(0, max(max(ang), ALLOW_ANG) * 1.3)

    for axx in axes:
        axx.set_xticks(x)
        axx.set_xticklabels([f"Set {s}" for s in SETS])
        axx.grid(True, axis="y", alpha=0.3)

    fig.suptitle("벤치마크 – 역산 비교 (v1 파이프라인: 외부 3D 방향 · hybrid kr · 해석식 P32)",
                 fontsize=12.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=180)
    plt.close(fig)
    print("[*] written:", args.out)


if __name__ == "__main__":
    main()
