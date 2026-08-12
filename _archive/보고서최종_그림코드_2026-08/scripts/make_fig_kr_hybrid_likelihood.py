"""보고서 제3절 2항 삽입용 — 반지름 지수 k_r 하이브리드 우도의 구성 모식도.

우도를 k_r 의존부와 비의존부로 분리한다는 것이 이 그림의 요점이다.

    해석적 현길이 질량 w_j(k_r)   ×   창·절단 커널 K[j,i,c]
        (후보마다 해석식 계산)          (절리군마다 한 번만 모사)
                      ↓  모형 확률표 p_ic(k_r)
    관측 분할표 n_ic  →  다항 로그우도 → 최대우도 · ΔlogL ≤ 2

작도 원칙은 그림 6-1·6-5·6-6과 같다. 흑백 전용이며, 도형 폭을 좁게 잡고 글자를 키워
보고서 단 폭(약 17 cm)에서도 본문과 비슷한 크기로 읽히게 한다.

실행:
    python scripts/make_fig_kr_hybrid_likelihood.py
출력:
    docs/figures/fig_kr_hybrid_likelihood.png
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

C_EDGE = "#333333"
C_ARROW = "#222222"
C_OUT = "#e6e6e6"
DASH = (0, (4, 2.4))


def box(ax, x, y, w, h, name, desc, fc="white", name_fs=10.5, desc_fs=9.5,
        lw=1.3, dashed=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.12",
                                fc=fc, ec=C_EDGE, lw=lw,
                                ls=DASH if dashed else "-"))
    ax.text(x + w / 2, y + h - 0.55, name, ha="center", va="center",
            fontsize=name_fs, fontweight="bold", color="#111")
    ax.text(x + w / 2, y + (h - 0.9) / 2, desc, ha="center", va="center",
            fontsize=desc_fs, color="#333", linespacing=1.6)


def arrow(ax, x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                 mutation_scale=15, lw=1.6, color=C_ARROW,
                                 shrinkA=2, shrinkB=2))


def main():
    fig, ax = plt.subplots(figsize=(10.5, 6.5))
    ax.set_xlim(0, 22)
    ax.set_ylim(0, 13.4)
    ax.axis("off")

    ax.text(11, 12.9, "반지름 지수 $k_r$ 하이브리드 우도", ha="center", va="center",
            fontsize=15, fontweight="bold")
    ax.text(11, 12.15, "$k_r$ 에 의존하는 부분은 해석식으로, 의존하지 않는 부분은 "
                       "절리군마다 한 번의 수치 모사로 계산한다",
            ha="center", va="center", fontsize=9.5, color="#555")

    # ── 두 성분 ────────────────────────────────────────────────
    box(ax, 0.5, 8.8, 9.5, 2.9, "해석적 현길이 질량  $w_j(k_r)$",
        "크기 편향 반지름 분포 × 현길이 조건부 밀도\n"
        "→ 참 현길이 로그 구간 $j$ 의 확률질량\n"
        "후보 $k_r$ 마다 해석식으로 계산  ($k_r$ 의존)")
    ax.text(11, 10.25, "×", ha="center", va="center", fontsize=20, color="#333")
    box(ax, 12.0, 8.8, 9.5, 2.9, "창 · 절단 커널  $K[j,i,c]$",
        "면내 방향 재표집 · 제안영역 가중 · 다각형 클리핑\n"
        "절단등급 부여 · $\\ell_{min,fit}$ 적용\n"
        "절리군마다 한 번만 계산  ($k_r$ 무관)")

    for x in (5.25, 16.75):
        ax.plot([x, x], [8.8, 8.3], "-", color=C_ARROW, lw=1.5)
    ax.plot([5.25, 16.75], [8.3, 8.3], "-", color=C_ARROW, lw=1.5)
    arrow(ax, 11, 8.3, 11, 7.6)

    # ── 모형 확률표 ───────────────────────────────────────────
    box(ax, 5.0, 5.9, 12.0, 1.7, "모형 확률표  $p_{i,c}(k_r)$",
        "$p_{i,c}(k_r) \\;\\propto\\; \\sum_j w_j(k_r)\\, K[j,i,c]$    (총합으로 정규화)",
        desc_fs=10.0)

    ax.plot([11, 11], [5.9, 5.3], "-", color=C_ARROW, lw=1.5)
    ax.plot([11, 16.75], [5.3, 5.3], "-", color=C_ARROW, lw=1.5)
    arrow(ax, 16.75, 5.3, 16.75, 4.4)

    # ── 관측자료와 우도 ───────────────────────────────────────
    box(ax, 0.5, 1.6, 9.5, 2.8, "관측 분할표  $n_{i,c}$",
        "가시길이 40 로그구간 × 절단등급 3종\n"
        "$\\ell_{obs} \\geq \\ell_{min,fit}$ 인 관측 절리선만 집계")
    box(ax, 12.0, 1.6, 9.5, 2.8, "다항 로그우도",
        "$\\log L(k_r)=\\sum_{i,c} n_{i,c}\\,\\log p_{i,c}(k_r)$\n"
        "후보 격자 1.5 – 5.5, $\\Delta$ = 0.05\n"
        "최대우도 $\\hat{k_r}$ · $\\Delta\\log L \\leq 2$ 구간",
        fc=C_OUT, dashed=True)
    arrow(ax, 10.0, 3.0, 12.0, 3.0)

    ax.text(11, 0.65,
            "커널이 $k_r$ 에 무관하므로 후보마다 다시 모사하지 않는다. "
            "표집 잡음이 $k_r$ 방향으로 전파되지 않아 프로파일 우도가 매끄럽게 얻어진다.",
            ha="center", va="center", fontsize=9.5, color="#555")

    out = os.path.join("docs", "figures", "fig_kr_hybrid_likelihood.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("[*] written:", out)


if __name__ == "__main__":
    main()
