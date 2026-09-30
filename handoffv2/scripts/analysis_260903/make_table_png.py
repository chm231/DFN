# -*- coding: utf-8 -*-
"""비교표 PNG — 사용자 템플릿 표 테마: 머리글 채움 없음, 굵은 머리글 + 얇은 밑줄, 행 사이 연한 선, 세로선 없음, 전부 왼쪽 정렬."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Malgun Gothic"; plt.rcParams["axes.unicode_minus"] = False
K, LINE_H, LINE_R = "#111111", "#7F7F7F", "#C8C8C8"
header = ["구분", "다면 원판", "막장면 평행 원판 (|nx| ≥ 0.8)", "비율"]
rows = [("이전: 고정 게이트 3.5 m", "171", "12", "7.0 %"),
        ("최종: 적응 게이트", "308", "58", "18.8 %"),
        ("모집단 (전체 절리선)", "3,667", "–", "26.9 %")]
col_w = [2.9, 1.5, 2.9, 1.3]; W = sum(col_w)
h_row = 0.5; pad = 0.10; H = h_row * (len(rows) + 1)
fig = plt.figure(figsize=(W, H), dpi=300); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
xs = [0]
for w in col_w: xs.append(xs[-1] + w)
FS = 12
# 머리글: 채움 없음, 굵게, 아래 얇은 회색 선
for j, t in enumerate(header):
    ax.text(xs[j] + pad, H - h_row / 2, t, ha="left", va="center", fontsize=FS, fontweight="bold", color=K)
ax.plot([0, W], [H - h_row, H - h_row], color=LINE_H, lw=0.9)
# 본문: 왼쪽 정렬, 행 사이 연한 선
for i, cells in enumerate(rows):
    y1 = H - h_row * (i + 2); yc = y1 + h_row / 2
    for j, v in enumerate(cells):
        ax.text(xs[j] + pad, yc, v, ha="left", va="center", fontsize=FS, color=K)
    if i < len(rows) - 1:
        ax.plot([0, W], [y1, y1], color=LINE_R, lw=0.6)
for o in [r"C:\Users\user\OneDrive\2026-2\현대건설\표_적응게이트_비교.png",
          r"C:\Users\user\OneDrive\2026-1\3D DFN modeling\docs\figures\table_adaptive_gate_vs_fixed.png"]:
    fig.savefig(o, dpi=300, bbox_inches="tight", pad_inches=0.04, facecolor="white")
print("saved")
