"""보고서 제2절 1.1 삽입용 — 절리선 자료의 두 가지 역할(정방향 / 역방향) 모식도.

같은 절리선 자료가 앞에서는 검증자료 생성의 산출물이고 뒤에서는 역산의 입력이다.
이를 하나의 공유 상자로 그려, 정방향 흐름이 위에서 들어오고 역방향 흐름이 아래로
빠져나가도록 배치한다.

    정방향  합성 암반균열망 생성 → 평면 막장면과 균열 원판의 교차 계산 → 절리선 생성
                                   ↓  [ 절리선 데이터셋 ]  ↓
    역방향  절리군별 통계 파라미터 역산 → 관측 균열 복원 → 조건부 암반균열망 생성

작도 원칙은 그림 6-1·6-5와 같다. 흑백 전용, 도형 폭을 좁게 잡고 글자를 키워
보고서 단 폭(약 17 cm)에서도 본문과 비슷한 크기로 읽히게 한다.

실행:
    python scripts/make_fig_trace_dual_role.py
출력:
    docs/figures/fig_trace_dual_role.png
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
C_SHARED = "#e6e6e6"    # 공유 자료 상자
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


def edge_label(ax, x, y, text, ha="left"):
    ax.text(x, y, text, ha=ha, va="center", fontsize=9.6, color="#222",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none"))


def main():
    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    ax.set_xlim(0, 22)
    ax.set_ylim(0, 12.0)
    ax.axis("off")

    ax.text(11, 11.4, "절리선 자료의 두 가지 역할", ha="center", va="center",
            fontsize=15.5, fontweight="bold")

    # ── 정방향: 검증자료 생성 ──────────────────────────────────
    ax.text(21.0, 10.4, "정방향 — 검증자료 생성", ha="right", va="center",
            fontsize=10.5, color="#333")
    box(ax, 1.0, 8.0, 6.0, 1.9, "합성 암반균열망 생성\n(참값 파라미터)")
    box(ax, 7.8, 8.0, 6.4, 1.9, "평면 막장면과\n균열 원판의 교차 계산")
    box(ax, 15.0, 8.0, 6.0, 1.9, "절리선 생성")
    arrow(ax, 7.0, 8.95, 7.8, 8.95)
    arrow(ax, 14.2, 8.95, 15.0, 8.95)

    # ── 공유 자료 ─────────────────────────────────────────────
    box(ax, 3.0, 5.0, 16.0, 1.9,
        "절리선 데이터셋  —  막장면별 2차원 절리선 · 절리군 ID · 절리면 법선벡터\n"
        "같은 자료가 앞에서는 검증자료 생성의 산출물, 뒤에서는 역산의 입력이 된다",
        fc=C_SHARED, fs=9.8, dashed=True)
    arrow(ax, 18.0, 8.0, 18.0, 6.9)
    edge_label(ax, 17.5, 7.45, "산출", ha="right")
    arrow(ax, 4.0, 5.0, 4.0, 3.9)
    edge_label(ax, 4.5, 4.45, "입력", ha="left")

    # ── 역방향: 역산 · 복원 · 조건부 생성 ──────────────────────
    ax.text(21.0, 4.45, "역방향 — 역산 · 복원 · 조건부 생성", ha="right", va="center",
            fontsize=10.5, color="#333")
    box(ax, 1.0, 2.0, 6.0, 1.9, "절리군별 통계\n파라미터 역산")
    box(ax, 7.8, 2.0, 6.4, 1.9, "관측 균열 복원")
    box(ax, 15.0, 2.0, 6.0, 1.9, "조건부 암반균열망\n생성")
    arrow(ax, 7.0, 2.95, 7.8, 2.95)
    arrow(ax, 14.2, 2.95, 15.0, 2.95)

    ax.text(11, 0.75,
            "실제 현장 적용 시에는 정방향의 합성 절리선 생성이 제5장의 매핑 산출물로 대체되며,\n"
            "역방향 파이프라인은 동일한 구조로 사용한다.",
            ha="center", va="center", fontsize=9.6, color="#555", linespacing=1.5)

    out = os.path.join("docs", "figures", "fig_trace_dual_role.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("[*] written:", out)


if __name__ == "__main__":
    main()
