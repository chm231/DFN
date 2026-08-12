"""DFN 역산 파이프라인 개요도 생성 (보고서 그림 6-1).

비전공 독자(박사급이나 이 분야는 개요만 아는 독자)를 위해 각 모듈의 역할을
평문 한국어로 풀어 쓴 개요도. 하단에 설계 원칙과 용어 설명 포함.

작도 원칙
  - 흑백 전용: 채움은 흰색·회색 계열만 쓰고, 구분은 선 굵기와 실선/점선으로만 한다.
  - 겹침 방지: 단계 사이는 세로 화살표 하나로만 잇고, 단계 안에서는 이웃한 상자끼리만
    짧게 잇는다. 간선 설명에는 흰 배경을 깔아 선·상자와 겹치지 않게 한다.
  - 단계를 건너뛰는 자료 전달은 좌·우 여백의 점선 우회선으로 그린다. 우회선은 단계
    사이의 빈 띠와 상자 바깥 여백만 지나가므로 어떤 상자와도 겹치지 않는다.

실행:
    python scripts/make_pipeline_overview_figure.py
출력:
    docs/figures/fig_pipeline_overview.png
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

C_BAND = "#f2f2f2"        # 단계 묶음 배경
C_BAND_EDGE = "#b8b8b8"
C_EDGE = "#333333"        # 상자 테두리
C_ARROW = "#222222"
C_OUT = "#e6e6e6"         # 산출물 상자 채움
DASH = (0, (4, 2.4))      # 상자 테두리용 점선
BYPASS = (0, (3, 2.6))    # 우회선용 점선

BAND_X0, BAND_X1 = 14, 126
COL_X = [20, 54, 88]      # 3열 배치
COL_W = 32
CENTER = 70
CH_L, CH_R = 7, 131       # 좌·우 우회 통로 (상자 바깥)


def col_gap(i):
    """i열 오른쪽 끝과 i+1열 왼쪽 끝."""
    return COL_X[i] + COL_W, COL_X[i + 1]


def band(ax, y0, y1, title):
    ax.add_patch(FancyBboxPatch((BAND_X0, y0), BAND_X1 - BAND_X0, y1 - y0,
                                boxstyle="round,pad=0.4",
                                fc=C_BAND, ec=C_BAND_EDGE, lw=1.0))
    ax.text(BAND_X0 + 4, y1 - 2.6, title, ha="left", va="center", fontsize=11.5,
            fontweight="bold", color="#111")


def box(ax, x, y, w, h, name, desc, dashed=False, fc="white",
        name_fs=9.2, desc_fs=8.0):
    """모듈 상자: 위=모듈명(굵게), 아래=평문 설명."""
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.35",
                                fc=fc, ec=C_EDGE, lw=1.2,
                                ls=DASH if dashed else "-"))
    ax.text(x + w / 2, y + h - 2.4, name, ha="center", va="center",
            fontsize=name_fs, fontweight="bold", color="#111")
    ax.text(x + w / 2, y + (h - 3.4) / 2, desc, ha="center", va="center",
            fontsize=desc_fs, color="#333", linespacing=1.55)


def arrow(ax, x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                 mutation_scale=14, lw=1.5, color=C_ARROW,
                                 shrinkA=0, shrinkB=0))


def edge_label(ax, x, y, text, ha="left", fs=8.2, rot=0):
    """간선 설명 — 흰 배경을 깔아 선·상자와 겹치지 않게 한다."""
    ax.text(x, y, text, ha=ha, va="center", fontsize=fs, color="#222", rotation=rot,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none"))


def stage_arrow(ax, y_from, y_to, label, side="right"):
    """side: 라벨을 놓을 쪽. 우회선이 지나가는 쪽을 피해서 지정한다."""
    arrow(ax, CENTER, y_from, CENTER, y_to)
    dx, ha = (2.5, "left") if side == "right" else (-2.5, "right")
    edge_label(ax, CENTER + dx, (y_from + y_to) / 2, label, ha=ha)


def bypass(ax, pts):
    """단계를 건너뛰는 자료 전달 — 꺾은 점선, 마지막 구간에만 화살표."""
    for a, b in zip(pts[:-2], pts[1:-1]):
        ax.plot([a[0], b[0]], [a[1], b[1]], ls=BYPASS, color=C_ARROW, lw=1.3)
    ax.add_patch(FancyArrowPatch(pts[-2], pts[-1], arrowstyle="-|>",
                                 mutation_scale=13, lw=1.3, color=C_ARROW,
                                 linestyle=BYPASS, shrinkA=0, shrinkB=0))


def main():
    fig, ax = plt.subplots(figsize=(15.0, 19.2))
    ax.set_xlim(0, 136)
    ax.set_ylim(-16.5, 148)
    ax.axis("off")

    ax.text(CENTER, 145, "DFN 역산 파이프라인 개요", ha="center", va="center",
            fontsize=17, fontweight="bold")
    ax.text(CENTER, 141.2, "터널 굴착면(막장면)에서 관측한 2차원 절리선으로부터 3차원 절리망(DFN)의 "
                           "통계를 역산하고, 관측과 일치하는 조건부 절리망을 생성하는 과정",
            ha="center", va="center", fontsize=9.6, color="#444")

    # ---------------------- 1단계 · 입력 ----------------------
    band(ax, 114, 138, "1단계 · 입력 — 제5장에서 전달받은 2차원 절리선 데이터셋")
    box(ax, 20, 120, 46, 12, "제5장 절리선 데이터셋 (외부 입력)",
        "절리선 끝점 P1(y,z) · P2(y,z), 막장면 위치 x,\n"
        "절리군 ID, 3차원 단위 법선벡터 n\n"
        "(제6장은 이 값을 그대로 받아 쓰며 재추정하지 않는다)")
    ax.text(72, 126, "또는", ha="center", va="center", fontsize=9.5, color="#333",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=C_BAND_EDGE))
    box(ax, 78, 120, 46, 12, "export_flat_face_traces   (검증 전용)",
        "알고리즘 검증용 합성 절리선 생성기\n"
        "막장면을 x = 일정 평면으로 두고 균열 원판과의\n"
        "교선을 구한 뒤 터널 단면 다각형으로 정확히 절단\n"
        "→ 관측면적 = 다각형 면적 (모델 창 기준과 일치)",
        dashed=True, desc_fs=7.8)

    ax.text(BAND_X1 - 4, 135.4, "현장 적용은 왼쪽, 알고리즘 검증은 오른쪽",
            ha="right", va="center", fontsize=8.2, color="#555")

    for x in (43, 101):                     # 두 대안이 같은 산출물로 합류
        ax.plot([x, x], [120, 118], "-", color=C_ARROW, lw=1.4)
    ax.plot([43, 101], [118, 118], "-", color=C_ARROW, lw=1.4)

    stage_arrow(ax, 118, 110, "절리선 관측 데이터셋   (trace_dataset_3d .h5 / .csv)")

    # ---------------------- 2단계 · 통계 역산 ----------------------
    band(ax, 62, 110, "2단계 · 절리군별 통계 파라미터 역산")

    box(ax, COL_X[0], 94, COL_W, 10, "summarize_setwise_trace_statistics",
        "절리군별 절리선 개수 · 길이 기초 통계\n(자료 현황 요약)", name_fs=8.6)
    box(ax, COL_X[1], 94, COL_W, 10, "estimate_mean_orientation",
        "절리군의 평균 방향 추정\n(법선벡터의 축성 평균 → trend / plunge)")
    box(ax, COL_X[2], 94, COL_W, 10, "estimate_fisher_kappa",
        "방향의 흩어짐 정도 추정 — 절리 방향이\n평균 둘레에 모인 정도(Fisher 집중도 κ)")
    arrow(ax, col_gap(1)[0], 99, col_gap(1)[1], 99)

    box(ax, COL_X[0], 76, COL_W, 13, "estimate_radius_powerlaw_window_mc",
        "절리 크기 추정의 계산 엔진\n절리선이 관측창(막장면) 밖으로 잘려\n"
        "짧게 보이는 효과를 보정한 우도 계산\n하이브리드(해석식 + 창 · 절단 커널)",
        name_fs=8.4, desc_fs=7.8)
    box(ax, COL_X[1], 76, COL_W, 13, "estimate_kr",
        "절리 크기(반지름) 분포 추정 진입점\n반지름 멱법칙 지수 kr 를, 관측된 절리선\n"
        "길이 분포와 가장 잘 맞도록 탐색\n(최소길이 0.5 m 고정 · 참값 미사용 선택)",
        desc_fs=7.8)
    box(ax, COL_X[2], 76, COL_W, 13, "estimate_p32_mc_calibrated",
        "절리 밀도 P32 추정 (최종 산출)\n환산: P32 = P21(관측 전량) / C\n"
        "환산계수 C = E[sinφ] 를 방향분포에서\n해석식(결정론적 구적)으로 계산",
        desc_fs=7.8)
    arrow(ax, col_gap(0)[0], 82.5, col_gap(0)[1], 82.5)
    arrow(ax, col_gap(1)[0], 82.5, col_gap(1)[1], 82.5)
    edge_label(ax, (col_gap(1)[0] + col_gap(1)[1]) / 2, 86.0, "크기지수 kr",
               ha="center", fs=7.8)

    x_p32 = COL_X[2] + COL_W / 2
    arrow(ax, x_p32, 94, x_p32, 89)
    edge_label(ax, x_p32 - 2, 91.5, "평균 방향 · 집중도 κ", ha="right", fs=7.8)

    box(ax, COL_X[2], 64, COL_W, 9, "build_p32_pilot_summary / dataset_config",
        "P32 계산에 필요한 설정 관리 (절리군 구성 ·\n방향 · 크기분포) — 내장 preset 또는\n"
        "새 현장 자료용 JSON 설정", name_fs=8.2, desc_fs=7.6)
    arrow(ax, x_p32, 73, x_p32, 76)

    stage_arrow(ax, 62, 58, "크기지수 kr  (반지름 축소추정에 사용)", side="left")

    # ---------------------- 2.5단계 · 관측 절리 복원 ----------------------
    band(ax, 34, 58, "2.5단계 · 관측 절리 복원")
    box(ax, COL_X[0], 38, COL_W, 13, "reconstruct_discs_from_traces",
        "여러 막장면의 절리선 중 같은 절리에서\n나온 것끼리 연결(같은 평면 위인지 검증)하고,\n"
        "이를 지나는 3차원 원판의 위치 · 크기를\n추정해 복원", desc_fs=7.8)
    box(ax, COL_X[1], 38, COL_W, 13, "visualize_reconstruction",
        "복원 결과 검증용 그림 생성\n(3차원 배치 + 면별로 관측\n절리선과 복원 원판을 대조)")
    box(ax, COL_X[2], 38, COL_W, 13, "validate_reconstruction_lofo",
        "교차검증: 막장면 하나를 빼고\n복원한 뒤 그 면의 절리선을 얼마나\n맞히는지 평가 (정답 자료 불필요)",
        desc_fs=7.8)
    arrow(ax, col_gap(0)[0], 44.5, col_gap(0)[1], 44.5)
    arrow(ax, col_gap(1)[0], 44.5, col_gap(1)[1], 44.5)

    stage_arrow(ax, 34, 30, "복원 원판 목록  (reconstructed_discs.csv)")

    # ---------------------- 3단계 · 조건부 절리망 생성 ----------------------
    band(ax, 4, 30, "3단계 · 조건부 절리망 생성 · 시각화")
    box(ax, COL_X[0], 9, COL_W, 13, "generate_conditional_hidden_dfn",
        "조건부 절리망 생성 — 관측 · 복원된 절리는\n그대로 유지(visible)하고, 관측되지 않은\n"
        "영역은 역산 통계에 맞춰 확률적으로\n채움(hidden)", desc_fs=7.8)
    box(ax, COL_X[1], 9, COL_W, 13, "안정성 해석 입력 disc 데이터셋",
        "절리 원판 목록: 번호 · 절리군 ·\n출처(관측 / 생성) · 중심좌표 ·\n반지름 · 법선방향",
        dashed=True, fc=C_OUT)
    box(ax, COL_X[2], 9, COL_W, 13, "visualize_conditional_dfn_3d",
        "최종 3차원 절리망 시각화\n(원판 + 막장면 + 절리선,\nPyVista · 웹브라우저용\n대화형 HTML 저장 가능)",
        desc_fs=7.8)
    arrow(ax, col_gap(0)[0], 15.5, col_gap(0)[1], 15.5)
    arrow(ax, col_gap(1)[0], 15.5, col_gap(1)[1], 15.5)

    # ---------------------- 점선 우회선 ----------------------
    # ① 절리선 데이터셋은 2단계를 거치지 않고 복원 단계로도 직접 들어간다 (좌측 통로)
    bypass(ax, [(CENTER, 111.5), (CH_L, 111.5), (CH_L, 44.5), (COL_X[0], 44.5)])
    edge_label(ax, CH_L, 78, "절리선 관측 데이터셋", ha="center", fs=7.6, rot=90)

    # ② 역산 통계는 복원 단계를 거치지 않고 생성 단계로 직접 들어간다 (우측 통로)
    bypass(ax, [(CENTER, 60), (CH_R, 60), (CH_R, 6), (COL_X[0] + COL_W / 2, 6),
                (COL_X[0] + COL_W / 2, 9)])
    edge_label(ax, CH_R, 33, "역산 통계 (kr · P32 · κ)", ha="center", fs=7.6, rot=90)

    # ---------------------- 하단 주석 ----------------------
    ax.text(BAND_X0 + 2, 0.6, "점선 화살표 = 단계를 건너뛰어 전달되는 자료",
            ha="left", va="center", fontsize=8.0, color="#555")

    for y, t in [
        (-2.6, "① 제6장의 출발점은 제5장에서 전달받은 2차원 절리선과 절리면 법선벡터이며, "
               "막장면은 평면으로, 관측창은 터널 단면 다각형으로 이상화한다."),
        (-5.0, "② 크기지수 kr 는 하이브리드 우도(해석식 길이분포 × 창 · 절단 커널)로 추정하며, "
               "최소길이 0.5 m 를 전 절리군에 공통 적용하고 참값은 선택에 쓰지 않는다."),
        (-7.4, "③ 밀도 P32 는 해석식 환산계수 C = E[sinφ] (방향분포의 결정론적 구적)로 역산하며, "
               "유한 관측창 · 최소길이 효과는 순방향 모사로 별도 검증한다."),
    ]:
        ax.text(BAND_X0 + 2, y, t, ha="left", va="center", fontsize=8.6, color="#222")

    ax.text(BAND_X0 + 2, -10.8, "용어 |  절리선(trace): 절리가 굴착면과 만나 생기는 선 · "
                                "절리군(set): 방향이 비슷한 절리 묶음 · "
                                "kr: 절리 반지름 멱법칙 분포의 지수(클수록 작은 절리 비중 증가)",
            ha="left", va="center", fontsize=7.8, color="#555")
    ax.text(BAND_X0 + 2, -13.0, "        P21: 관측면 단위면적당 절리선 길이(m/m²) · "
                                "P32: 암반 단위부피당 절리 면적(m²/m³) · "
                                "Fisher κ: 방향 집중도(클수록 방향이 평균 주위에 모임)",
            ha="left", va="center", fontsize=7.8, color="#555")
    ax.text(BAND_X0 + 2, -15.6, "패키지 |  dfn_analysis",
            ha="left", va="center", fontsize=7.8, color="#777")

    out = os.path.join("docs", "figures", "fig_pipeline_overview.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("[*] written:", out)


if __name__ == "__main__":
    main()
