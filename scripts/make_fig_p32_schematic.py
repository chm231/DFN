"""보고서 그림 6-11(P32 보정 모식도) 대체 — v1 해석식(analytic C=E[sinφ]) 흐름.

기존 그림과 같은 4상자 흐름도 형식을 유지하되, unit-P32 forward MC 대신
해석식 보정 경로를 나타낸다.

실행:
    python scripts/make_fig_p32_schematic.py
출력:
    docs/figures/fig_p32_analytic_schematic.png
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

try:
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False
except Exception:
    pass


def box(ax, x, y, w, h, text, fs=9.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.10",
                                fc="white", ec="#333", lw=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, linespacing=1.55)


def arrow(ax, x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                 mutation_scale=16, lw=1.5, color="#333",
                                 shrinkA=2, shrinkB=2))


def main():
    fig, ax = plt.subplots(figsize=(12.5, 3.2))
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 4.6)
    ax.axis("off")

    box(ax, 0.3, 1.2, 4.2, 2.2,
        "절리군 방향 분포\n(평균 pole · 집중도 κ,\ntrace 추정치)")
    box(ax, 5.6, 1.2, 4.6, 2.2,
        "환산계수 해석식 계산\nC = E[sinφ]\n(결정론적 구적 · 표집 없음)")
    box(ax, 11.3, 1.2, 3.9, 2.2,
        "관측 P21\n(가시 절리선 총길이\n/ 관측면적)")
    box(ax, 16.2, 1.2, 3.5, 2.2,
        "P32_hat\n= P21 / C")

    arrow(ax, 4.5, 2.3, 5.6, 2.3)
    arrow(ax, 10.2, 2.3, 11.3, 2.3)
    arrow(ax, 15.2, 2.3, 16.2, 2.3)

    ax.text(10.0, 0.45,
            "※ C는 방향 분포만의 함수(크기분포와 무관) → 크기지수 $k_r$ 추정과 분리되어 "
            "$k_r$ 쪽 설정 변화가 P32에 전파되지 않음",
            ha="center", va="center", fontsize=8.6, color="#555")
    ax.text(10.0, 4.25, "P32 해석식 보정 흐름 (v1 기본: analytic C = E[sinφ])",
            ha="center", va="center", fontsize=12, fontweight="bold")

    out = os.path.join("docs", "figures", "fig_p32_analytic_schematic.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("[*] written:", out)


if __name__ == "__main__":
    main()
