"""보고서 그림 6-5(절리선 전처리 흐름) — 제2절의 5단계 구성.

제6장의 전처리는 제5장에서 전달받은 2차원 절리선에서 시작하며 다음 5단계로 구성된다.

    1) 절리선 끝점 입력   2) 길이·중심점 계산   3) 접촉 유형 분류
    4) 최소 길이 기준 적용   5) 절리군 ID·3차원 방향 확인

작도 원칙은 그림 6-1(`make_pipeline_overview_figure.py`)과 같다. 흑백 전용으로,
채움은 흰색·회색 계열만 쓰고 구분은 선 굵기와 실선/점선으로만 한다.

글자 크기: 보고서 단 폭(약 17 cm)에 넣으면 그림이 축소되므로, 도형 폭을 좁게 잡고
글자를 키워 본문 대비 상대 크기를 확보한다. 상자 문구는 한 줄이 11자를 넘지 않게 쓴다.

실행:
    python scripts/make_fig_preprocess_flow.py
출력:
    docs/figures/fig_preprocess_flow.png
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
C_OUT = "#e6e6e6"       # 산출물 상자 채움
DASH = (0, (4, 2.4))


def box(ax, x, y, w, h, text, fc="white", fs=10.0, lw=1.3, dashed=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.12",
                                fc=fc, ec=C_EDGE, lw=lw,
                                ls=DASH if dashed else "-"))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, color="#222", linespacing=1.6)


def arrow(ax, x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                 mutation_scale=15, lw=1.6, color=C_ARROW,
                                 shrinkA=2, shrinkB=2))


def main():
    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    ax.set_xlim(0, 22)
    ax.set_ylim(0, 9.2)
    ax.axis("off")

    ax.text(11, 8.7, "절리선 전처리 흐름 (제2절)", ha="center", va="center",
            fontsize=16, fontweight="bold")

    # 외부 입력 (제5장) — 진입점이므로 테두리를 굵게
    ax.text(3.2, 7.9, "① 절리선 끝점 입력", ha="center", va="center",
            fontsize=10.5, color="#333")
    box(ax, 0.4, 5.2, 5.6, 2.3,
        "제5장 절리선 데이터셋\n끝점 $(y_1,z_1)$, $(y_2,z_2)$\n막장면 위치 $x$",
        lw=2.0)

    # 전처리 단계 ②~⑤
    steps = [
        (6.5, 3.5, "② 길이·중심점 계산",
         "$L=\\|P_2-P_1\\|$\n$C=(P_1+P_2)/2$", 10.0),
        (10.5, 3.5, "③ 접촉 유형 분류",
         "Type 0 · 1 · 2\n절단등급 분류", 10.0),
        (14.5, 3.5, "④ 최소 길이 기준",
         "$\\ell_{min,fit}$ : $k_r$ 적합\n현재 0.5 m\n($P_{21}$ 은 전량 사용)", 9.4),
        (18.5, 3.3, "⑤ 절리군·방향 확인",
         "절리군 ID와\n3차원 방향(법선)\n유효성 확인", 10.0),
    ]
    for x0, w, label, body, fs in steps:
        ax.text(x0 + w / 2, 7.9, label, ha="center", va="center",
                fontsize=10.5, color="#333")
        box(ax, x0, 5.2, w, 2.3, body, fs=fs)

    for x0, x1 in ((6.0, 6.5), (10.0, 10.5), (14.0, 14.5), (18.0, 18.5)):
        arrow(ax, x0, 6.35, x1, 6.35)

    # 산출물 → 다음 절 (⑤에서 꺾어 내려오는 경로)
    box(ax, 3.0, 1.7, 16.0, 1.9,
        "전처리 절리선 데이터셋  —  길이 $L$ · 중심 $C$ · 절단등급 · 절리군 ID · 3차원 방향\n"
        "→ 제3절 통계 역산 ($k_r$ · $\\kappa$ · $P_{32}$) 및 제4절 절리선 정합의 입력",
        fc=C_OUT, fs=10.5, dashed=True)
    ax.plot([20.15, 20.15], [5.2, 4.5], "-", color=C_ARROW, lw=1.6)
    ax.plot([20.15, 11.0], [4.5, 4.5], "-", color=C_ARROW, lw=1.6)
    arrow(ax, 11.0, 4.5, 11.0, 3.6)

    ax.text(11, 0.8,
            "제6장의 전처리는 제5장에서 전달받은 2차원 절리선에서 시작하며,\n"
            "막장면은 평면으로, 관측창은 터널 단면 다각형으로 이상화한다.",
            ha="center", va="center", fontsize=9.8, color="#555", linespacing=1.5)

    out = os.path.join("docs", "figures", "fig_preprocess_flow.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("[*] written:", out)


if __name__ == "__main__":
    main()
