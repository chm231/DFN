"""DFN 역산 파이프라인 개요도 생성 (v2, 2026-08).

비전공 독자(박사급이나 이 분야는 개요만 아는 독자)를 위해 각 모듈의 역할을
평문 한국어로 풀어 쓴 개요도. 하단에 용어 설명 각주 포함.

v2(2026-08-07 결정) 반영:
  - 제6장 범위를 "제5장 전달 2D 절리선 데이터셋"부터로 축소
    (요철 mesh 생성·3D polyline·3점법 법선은 범위 밖)
  - 검증용 합성 절리선은 평면 막장면 × 터널 단면 다각형(export_flat_face_traces)
  - estimate_kr : hybrid 우도 · lmin 0.5 m 고정 · 참값 미사용 선택
  - estimate_p32: C(ℓmin) = E[sinφ]·η_det 를 순방향 모사에서 직접 산정

실행:
    python scripts/make_pipeline_overview_figure.py
출력:
    docs/figures/fig_pipeline_overview_v2.png
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

C_V1_FACE = "#eaf2fb"   # v2 변경 모듈 배경
C_V1_EDGE = "#1f4e79"   # v1 변경 모듈 테두리
C_BOX_EDGE = "#555555"
C_BAND = "#f6f6f6"
C_ARROW = "#444444"
C_DATA = "#fff9e6"      # 데이터(파일) 상자


def box(ax, x, y, w, h, name, desc, v1=False, name_fs=9.0, desc_fs=8.0):
    """모듈 상자: 위=파일명(bold), 아래=평문 설명. v1=True 면 파란 강조 + 배지."""
    fc = C_V1_FACE if v1 else "white"
    ec = C_V1_EDGE if v1 else C_BOX_EDGE
    lw = 1.7 if v1 else 1.1
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.35",
                                fc=fc, ec=ec, lw=lw))
    ax.text(x + w / 2, y + 2.6, name, ha="center", va="center",
            fontsize=name_fs, fontweight="bold", color="#111")
    ax.text(x + w / 2, y + 2.6 + (h - 2.6) / 2 + 0.3, desc, ha="center",
            va="center", fontsize=desc_fs, color="#333", linespacing=1.5)
    if v1:
        ax.text(x + w - 1.2, y + 0.15, "v2", ha="center", va="center",
                fontsize=7, color="white", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.28", fc=C_V1_EDGE, ec="none"))


def data_box(ax, x, y, w, h, title, desc, fs=8.0):
    """데이터(파일) 상자: 점선 테두리 + 연노랑."""
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.35",
                                fc=C_DATA, ec="#8a7a2e", lw=1.1, ls=(0, (4, 2))))
    ax.text(x + w / 2, y + 2.4, title, ha="center", va="center",
            fontsize=fs + 0.5, fontweight="bold", color="#4a3f10")
    ax.text(x + w / 2, y + 2.4 + (h - 2.4) / 2 + 0.2, desc, ha="center",
            va="center", fontsize=fs - 0.5, color="#4a3f10", linespacing=1.45)


def band(ax, y0, y1, title):
    ax.add_patch(FancyBboxPatch((1.2, y0), 97.6, y1 - y0, boxstyle="round,pad=0.3",
                                fc=C_BAND, ec="#bbbbbb", lw=0.8))
    ax.text(3.0, y0 + 2.4, title, fontsize=11.5, fontweight="bold",
            ha="left", va="center", color="#222")


def arrow(ax, p0, p1, label=None, rad=0.0, ls="-", lx=0.0, ly=0.0, fs=7.6):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=13,
                                 lw=1.3, color=C_ARROW, shrinkA=1.5, shrinkB=1.5,
                                 linestyle=ls,
                                 connectionstyle=f"arc3,rad={rad}"))
    if label:
        ax.text((p0[0] + p1[0]) / 2 + lx, (p0[1] + p1[1]) / 2 + ly, label,
                ha="center", va="center", fontsize=fs, style="italic",
                color="#222", bbox=dict(fc="white", ec="none", alpha=0.88, pad=0.6))


def main():
    fig, ax = plt.subplots(figsize=(12.4, 17.2))
    ax.set_xlim(0, 100)
    ax.set_ylim(170.5, 0)   # y 아래 방향 (위=0)
    ax.axis("off")

    ax.text(50, 2.3, "DFN 역산 파이프라인 개요 — v2 (2026-08)",
            ha="center", va="center", fontsize=15.5, fontweight="bold")
    ax.text(50, 5.5, "터널 굴착면(막장면)에서 관측한 2차원 절리선으로부터 3차원 절리망(DFN)의 통계를 역산하고, "
                     "관측과 일치하는 조건부 절리망을 생성하는 과정",
            ha="center", va="center", fontsize=9.2, color="#444")

    # ── 1단계 ──────────────────────────────────────────────────────────
    band(ax, 8, 29.5, "1단계 · 입력 — 제5장에서 전달받은 2차원 절리선 데이터셋")
    box(ax, 4, 13.5, 33, 13, "제5장 절리선 데이터셋 (외부 입력)",
        "절리선 끝점 P1(y,z)·P2(y,z), 막장면 위치 x,\n절리군 ID, 3차원 단위 법선벡터 n\n"
        "(요철 mesh·polyline·3점법은 제6장 범위 밖)", v1=True, name_fs=9.0, desc_fs=7.7)
    box(ax, 52, 13.5, 44, 13, "export_flat_face_traces  (검증 전용)",
        "알고리즘 검증용 합성 절리선 생성기\n막장면을 x=일정 평면으로 두고 균열 원판과의\n"
        "교선을 구한 뒤 터널 단면 다각형으로 정확히 절단\n"
        "→ 관측면적 = 다각형 면적 (모델 창 기준과 일치)", v1=True, name_fs=8.6, desc_fs=7.6)
    # 두 상자는 순차 흐름이 아니라 '실측 입력' 또는 '검증용 합성' 중 택일이다.
    ax.text(44.5, 20, "또는", ha="center", va="center", fontsize=10, color="#555",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#bbb"))
    ax.text(50, 28, "※ 실제 현장에서는 왼쪽(제5장 전달 자료)을 쓰고, 오른쪽 생성기는 알고리즘 검증에만 쓴다",
            ha="center", va="center", fontsize=7.6, color="#666")
    arrow(ax, (20, 26.8), (44, 31.5), rad=0.10)
    arrow(ax, (74, 26.8), (52, 31.5), rad=-0.10)
    arrow(ax, (48, 31.5), (48, 33.1), label="절리선 관측 데이터셋 (trace_dataset_3d .h5/.csv)",
          lx=0, ly=-2.4)

    # ── 2단계 ──────────────────────────────────────────────────────────
    band(ax, 33.5, 92, "2단계 · 절리군별 통계 파라미터 역산")
    # 1행: 방향 · 기초 통계
    box(ax, 4, 40, 27, 11, "estimate_mean_orientation",
        "절리군의 평균 방향 추정\n(법선벡터의 축성 평균 → trend/plunge)", name_fs=8.6, desc_fs=7.7)
    box(ax, 35, 40, 27, 11, "estimate_fisher_kappa",
        "방향의 흩어짐 정도 추정 — 절리 방향이\n평균 둘레에 모인 정도(Fisher 집중도 κ)", name_fs=8.6, desc_fs=7.7)
    box(ax, 66, 40, 30, 11, "summarize_setwise_trace_statistics",
        "절리군별 절리선 개수·길이 기초 통계\n(자료 현황 요약)", name_fs=8.0, desc_fs=7.7)
    arrow(ax, (31, 45.5), (35, 45.5))
    # 2행: 크기(kr) → 밀도(P32)
    box(ax, 4, 55.5, 29, 17, "estimate_radius_powerlaw_window_mc",
        "절리 크기 추정의 계산 엔진\n절리선이 관측창(막장면) 밖으로 잘려\n짧게 보이는 효과를 보정한 우도 계산\n"
        "hybrid(수식+커널, 기본) / 전량 MC(legacy)", v1=True, name_fs=7.8, desc_fs=7.7)
    box(ax, 36, 55.5, 28, 17, "estimate_kr",
        "절리 크기(반지름) 분포 추정 진입점\n반지름 멱법칙 지수 kr를, 관측된 절리선\n길이 분포와 가장 잘 맞도록 탐색\n"
        "(최소길이 0.5 m 고정 · 정답 미사용 선택)", v1=True, desc_fs=7.7)
    box(ax, 67, 55.5, 29, 17, "estimate_p32_mc_calibrated",
        "절리 밀도 P32 추정 (최종 산출)\n환산: P32 = P21(최소길이 통과분) / C\n"
        "환산계수 C(ℓmin) = E[sinφ]·η_det 를\n순방향 모사에서 직접 산정 (v2)", v1=True, name_fs=8.6, desc_fs=7.7)
    arrow(ax, (33, 64), (36, 64))
    arrow(ax, (64, 64), (67, 64), label="크기지수 kr", ly=-2.2)
    arrow(ax, (48, 51), (81, 55.5), label="평균 방향 · 집중도 κ", rad=0.18, lx=3, ly=-2.0)
    # 3행: P32 계산용 설정
    box(ax, 62, 76.5, 34, 13.5, "build_p32_pilot_summary\ndataset_config / build_dataset_config",
        "P32 계산에 필요한 설정 관리 (절리군 구성·\n방향·크기분포) — 내장 preset 또는 새 현장\n자료용 JSON 설정(절리선에서 자동 생성 가능)",
        name_fs=7.6, desc_fs=7.3)
    arrow(ax, (79, 76.5), (79, 72.9))

    # ── 2.5단계 ────────────────────────────────────────────────────────
    band(ax, 96, 122, "2.5단계 · 관측 절리 복원")
    box(ax, 4, 103, 36, 15.5, "reconstruct_discs_from_traces",
        "여러 막장면의 절리선 중 같은 절리에서 나온\n것끼리 연결(같은 평면 위인지 검증)하고, 이를\n지나는 3차원 원판의 위치·크기를 추정해 복원", desc_fs=7.7)
    box(ax, 44, 103, 25, 15.5, "visualize_reconstruction",
        "복원 결과 검증용 그림 생성\n(3차원 배치 + 면별로 관측\n절리선과 복원 원판을 대조)", desc_fs=7.7)
    box(ax, 73, 103, 25, 15.5, "validate_reconstruction_lofo",
        "교차검증: 막장면 하나를 빼고\n복원한 뒤 그 면의 절리선을 얼마나\n맞히는지 평가 (정답 자료 불필요)", name_fs=8.2, desc_fs=7.7)
    arrow(ax, (40, 110.75), (44, 110.75))
    arrow(ax, (69, 110.75), (73, 110.75))
    arrow(ax, (26, 92.4), (26, 102.6), label="관측 절리선", ls=(0, (4, 2)), lx=5.5)
    arrow(ax, (45, 72.9), (36, 102.6), label="크기지수 kr (반지름 추정에 활용)", lx=10.5)

    # ── 3단계 ──────────────────────────────────────────────────────────
    band(ax, 126, 152, "3단계 · 조건부 절리망 생성 · 시각화")
    box(ax, 4, 133, 36, 16, "generate_conditional_hidden_dfn",
        "조건부 절리망 생성 — 관측·복원된 절리는\n그대로 유지(visible)하고, 관측되지 않은\n영역은 역산 통계에 맞춰 확률적으로\n채움(hidden)", name_fs=8.6, desc_fs=7.7)
    data_box(ax, 44, 135, 26, 12, "안정성 해석 입력 disc 데이터셋",
             "절리 원판 목록: 번호 · 절리군 ·\n출처(관측/생성) · 중심좌표 ·\n반지름 · 법선방향", fs=7.9)
    box(ax, 74, 133, 22, 16, "visualize_conditional_dfn_3d",
        "최종 3차원 절리망 시각화\n(원판 + 막장면 + 절리선,\nPyVista) · 웹브라우저용\n대화형 HTML 저장 가능", name_fs=8.0, desc_fs=7.7)
    arrow(ax, (40, 141), (44, 141))
    arrow(ax, (70, 141), (74, 141))
    arrow(ax, (31, 118.9), (31, 132.6),
          label="복원 원판 목록 (reconstructed_discs.csv)", lx=-6, ly=-2.6)
    arrow(ax, (40, 122.4), (36.5, 132.6), ls=(0, (4, 2)),
          label="역산 통계 (크기 kr · 밀도 P32 · 방향 κ) → 미관측 절리 생성", lx=18)

    # ── 범례 · 용어 ────────────────────────────────────────────────────
    ax.add_patch(FancyBboxPatch((4, 154.5), 3.2, 2.2, boxstyle="round,pad=0.25",
                                fc=C_V1_FACE, ec=C_V1_EDGE, lw=1.5))
    ax.text(9, 155.6,
            "파란 상자 = v2(2026-08-07 결정)에서 변경된 모듈:  ① 제6장 범위를 '제5장 전달 절리선'부터로 축소 — 요철 mesh·polyline·3점법 삭제",
            ha="left", va="center", fontsize=8.2, color="#222")
    ax.text(9, 158.6,
            "② 크기지수 kr = 하이브리드 우도(수식+시뮬레이션 결합) · 최소길이 0.5 m 고정 · 참값 미사용 선택   "
            "③ P32 환산계수 = C(ℓmin) = E[sinφ]·η_det (관측·가상에 같은 최소길이 적용)",
            ha="left", va="center", fontsize=8.2, color="#222")
    ax.text(4, 162.4,
            "용어 |  절리선(trace): 절리가 굴착면과 만나 생기는 선 · 절리군(set): 방향이 비슷한 절리 묶음 · "
            "kr: 절리 반지름 멱법칙 분포의 지수(클수록 작은 절리 비중 증가)",
            ha="left", va="center", fontsize=7.6, color="#555")
    ax.text(4, 165.2,
            "P21: 관측면 단위면적당 절리선 길이(m/m²) · P32: 암반 단위부피당 절리 면적(m²/m³) · "
            "Fisher κ: 방향 집중도(클수록 방향이 평균 주위에 모임)",
            ha="left", va="center", fontsize=7.6, color="#555")
    ax.text(4, 168.4,
            "패키지: handoffv1/dfn_analysis · 상세 근거: docs/제6장_수정안_통합본.md · 작업 지시: docs/제6장_수정작업지시서.md",
            ha="left", va="center", fontsize=7.6, color="#888")

    out_dir = os.path.join("docs", "figures")
    os.makedirs(out_dir, exist_ok=True)
    p = os.path.join(out_dir, "fig_pipeline_overview_v2.png")
    fig.savefig(p, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("[*] written:", p)


if __name__ == "__main__":
    main()
