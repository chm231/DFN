"""보고서 P32 보정계수 모식도 — 해석식 C = E[sinφ] 기준.

2026-08-04 재확정: P32 보정계수의 기본·최종 모드는 해석식 C = E[sinφ] 이다.
방향분포만으로 결정되는 결정론적 구적이라 표집 변동이 없다. 유한 관측창과
최소 길이 기준의 효과(η_det)는 순방향 모사로 확인한 결과 0.95~0.99 로 작으며,
해석식은 η_det → 1 인 이상 조건 극한에 해당한다.

작도 원칙은 그림 6-1(`make_pipeline_overview_figure.py`)·6-5와 같다. 흑백 전용으로,
채움은 흰색·회색 계열만 쓰고 구분은 선 굵기와 실선/점선으로만 한다.

글자 크기: 보고서 단 폭(약 17 cm)에 넣으면 그림이 축소되므로, 도형 폭을 좁게 잡고
글자를 키워 본문 대비 상대 크기를 확보한다.

실행:
    python scripts/make_fig_p32_schematic.py
출력:
    docs/figures/fig_p32_calibration_schematic.png
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

C_EDGE = "#333333"      # 상자 테두리
C_ARROW = "#222222"
C_MID = "#f2f2f2"       # 보정계수 채움
C_OUT = "#e6e6e6"       # 입력·최종 산출 채움
DASH = (0, (4, 2.4))


def box(ax, x, y, w, h, text, fc="white", fs=10.0, lw=1.3, dashed=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.12",
                                fc=fc, ec=C_EDGE, lw=lw,
                                ls=DASH if dashed else "-"))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, color="#222", linespacing=1.6)


def arrow(ax, x0, y0, x1, y1, lw=1.6):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                 mutation_scale=15, lw=lw, color=C_ARROW,
                                 shrinkA=2, shrinkB=2))


def main():
    fig, ax = plt.subplots(figsize=(9.8, 5.9))
    ax.set_xlim(0, 22)
    ax.set_ylim(0, 9.9)
    ax.axis("off")

    ax.text(11, 9.4, "$P_{32}$ 보정계수 산정 — $C = E[\\sin\\varphi]$",
            ha="center", va="center", fontsize=15, fontweight="bold")
    ax.text(11, 8.6, "절리군 방향분포만으로 결정되는 해석식이며, 표집 변동이 없어 "
                     "같은 입력이면 같은 값을 준다",
            ha="center", va="center", fontsize=9.6, color="#555")

    # ── 위: 방향분포 → 보정계수 ────────────────────────────────
    y_up, h_up = 5.7, 2.1
    box(ax, 0.4, y_up, 6.2, h_up,
        "절리군 방향분포\n평균 pole ($T$, $P$) · 집중도 $\\kappa$")
    arrow(ax, 6.6, y_up + h_up / 2, 7.2, y_up + h_up / 2)
    box(ax, 7.2, y_up, 7.2, h_up,
        "$C = E[\\sin\\varphi] = E\\left[\\sqrt{1-n_x^2}\\,\\right]$\n"
        "결정론적 구적 (표집 없음)", fc=C_MID, lw=1.6)

    # ── 아래: 관측 절리선 ──────────────────────────────────────
    y_lo, h_lo = 2.7, 2.1
    box(ax, 0.4, y_lo, 14.0, h_lo,
        "관측 절리선 전량   $P_{21}^{obs} = \\sum_i \\ell_i \\,/\\, A_{obs}$\n"
        "$A_{obs}$ = 터널 단면 다각형 면적 × 막장면 수   "
        "(최소 길이 하한을 적용하지 않는다)",
        fc=C_OUT, fs=9.6, dashed=True)

    # ── 오른쪽: 최종 산출 ──────────────────────────────────────
    box(ax, 15.8, 4.2, 5.8, 2.1,
        "$\\widehat{P_{32}} = P_{21}^{obs} \\,/\\, C$\n(점추정으로 보고한다)",
        fc=C_OUT, dashed=True)
    for y_src, y_dst in ((y_up + h_up / 2, 6.3), (y_lo + h_lo / 2, 4.2)):
        ax.plot([14.4, 18.7], [y_src, y_src], "-", color=C_ARROW, lw=1.6)
        arrow(ax, 18.7, y_src, 18.7, y_dst)

    # ── 적용 범위 각주 ─────────────────────────────────────────
    box(ax, 1.6, 0.3, 18.8, 1.7,
        "해석식은 무한 관측면·하한 없음을 전제한 이상 조건의 결과이다. "
        "유한 관측창 절단과 최소 길이 기준의 효과는\n"
        "순방향 모사로 확인한 결과 $\\eta_{det} = 0.95 \\sim 0.99$ 로 작다. "
        "관측창이 좁거나 절리가 크면 이 가정을 다시 확인해야 한다.",
        fs=9.4, dashed=True)

    out = os.path.join("docs", "figures", "fig_p32_calibration_schematic.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("[*] written:", out)


if __name__ == "__main__":
    main()
