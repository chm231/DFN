"""보고서 그림 6-11(P32 보정 모식도) — v2: C_lmin = E[sinφ] · η_det 2성분 구조.

2026-08-07 결정 반영: 순수 해석식(C = E[sinφ])이 아니라, 방향 성분(해석식)과
관측조건 성분(최소길이·관측창 clipping)을 함께 담은 C_lmin 을 사용한다.
실제 계산은 두 성분을 분리하지 않고, 관측과 동일한 절차를 적용한 순방향
모사에서 C_lmin 을 직접 산정한다(B회 반복 → 평균·분산 → 신뢰구간).

실행:
    python scripts/make_fig_p32_schematic.py
출력:
    docs/figures/fig_p32_clmin_schematic.png
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

C_ANA = "#eaf2fb"   # 해석 성분
C_MC = "#fdf0e3"    # MC 성분
C_OBS = "#eef7ee"   # 관측
C_OUT = "#fff9e6"   # 산출


def box(ax, x, y, w, h, text, fc="white", ec="#333", fs=9.2, lw=1.2, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.12",
                                fc=fc, ec=ec, lw=lw, linestyle=ls))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, linespacing=1.5)


def arrow(ax, x0, y0, x1, y1, style="-|>", lw=1.5, color="#333"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style,
                                 mutation_scale=15, lw=lw, color=color,
                                 shrinkA=2, shrinkB=2))


def main():
    fig, ax = plt.subplots(figsize=(13.0, 5.6))
    ax.set_xlim(0, 22)
    ax.set_ylim(0, 9.4)
    ax.axis("off")

    ax.text(11, 8.9, "P32 보정계수 산정 — $C_{\\ell_{min}} = E[\\sin\\varphi]\\cdot\\eta_{det}$",
            ha="center", va="center", fontsize=13.5, fontweight="bold")
    ax.text(11, 8.05, "실제 계산은 두 성분을 분리하지 않고, 관측과 동일한 절차를 적용한 "
                      "순방향 모사에서 $C_{\\ell_{min}}$ 을 직접 산정한다",
            ha="center", va="center", fontsize=8.8, color="#555")

    # 상단: 두 성분의 의미
    box(ax, 0.4, 5.4, 6.4, 2.0,
        "방향 성분  $E[\\sin\\varphi]$\n절리군 평균 방향·집중도 κ에 따른\n기본 교차효율 (해석적 성분)",
        fc=C_ANA, ec="#1f4e79", fs=9.0)
    box(ax, 7.4, 5.4, 7.0, 2.0,
        "관측조건 성분  $\\eta_{det}$\n최소 길이 $\\ell_{min}$ · 터널 단면 clipping ·\n"
        "검출 누락 적용 후 유지되는 길이 비율",
        fc=C_MC, ec="#b06a1f", fs=9.0)
    ax.text(7.1, 6.4, "×", ha="center", va="center", fontsize=22, color="#333")
    box(ax, 15.9, 5.6, 5.7, 1.6, "$C_{\\ell_{min}}$\n(무차원 3D→2D 환산계수)",
        fc="#f2f2f2", ec="#333", fs=9.5)
    arrow(ax, 14.4, 6.4, 15.9, 6.4)

    # 하단: 실제 산정 절차 (순방향 모사에서 직접)
    y = 2.3
    box(ax, 0.4, y, 4.5, 2.2,
        "$P_{32}=1$ 기준 균열망\n$n_0 = 1/(\\pi E[R^2])$\n(추정 $k_r$·방향분포 사용)", fs=8.8)
    box(ax, 5.3, y, 4.7, 2.2,
        "막장면 교차 → 현 생성\n터널 단면 다각형 clipping\n동일한 $\\ell_{min}$ 적용", fs=8.8)
    box(ax, 10.4, y, 4.6, 2.2,
        "가상 $P_{21}^{ret}$ 집계\n$C^{(b)} = P_{21}^{sim,ret}/P_{32}^{sim}$\n"
        "B회 반복 → $\\bar C$, $s^2(C)$", fs=8.8)
    box(ax, 15.4, y, 6.2, 2.2,
        "$\\widehat{P_{32}} = P_{21,obs}^{ret} / \\bar{C}_{\\ell_{min}}$\n"
        "반복 분산 → 신뢰구간", fc=C_OUT, ec="#8a7a2e", fs=9.4)
    for x0, x1 in ((4.9, 5.3), (10.0, 10.4), (15.0, 15.4)):
        arrow(ax, x0, y + 1.1, x1, y + 1.1)
    arrow(ax, 18.7, 5.6, 18.7, y + 2.2, style="-|>", color="#777")

    # 관측 입력
    box(ax, 5.3, 0.25, 9.7, 1.5,
        "관측 절리선: $P_{21,obs}^{ret} = \\sum_i \\ell_i \\mathbf{1}(\\ell_i \\geq \\ell_{min}) / A_{obs}$"
        "     ($A_{obs}$ = 터널 단면 다각형 면적 × 막장면 수)",
        fc=C_OBS, ec="#3a7d44", fs=9.0)
    arrow(ax, 15.0, 1.0, 18.0, 1.0)
    arrow(ax, 18.0, 1.0, 18.0, y, color="#333")
    ax.text(9.0, 2.05, "↑ 관측자료와 가상자료에 같은 $\\ell_{min}$ 기준을 적용",
            ha="center", va="center", fontsize=8.2, color="#3a7d44")

    out = os.path.join("docs", "figures", "fig_p32_clmin_schematic.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("[*] written:", out)


if __name__ == "__main__":
    main()
