# -*- coding: utf-8 -*-
"""260910 공유자료 v3 — 10분 발표용 11장. 흑백 테마 규약(CLAUDE.md §18).
   삭제(사용자 지시): B-1 방식 비교, C-2 결과, C-3 블라인드, 뷰어 목록, 한계·다음 단계.
   A-1/A-2 병합, B-1 요지는 edge-cut 원리 슬라이드에 한 줄, C-2/C-3 핵심 수치는 마지막 뷰어 슬라이드에."""
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE, MSO_CONNECTOR
from pptx.oxml.ns import qn
from lxml import etree
import os, re, sys

SC = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(SC)                 # deliverables/260909_...
PV = os.path.join(PKG, "renders")         # pvsm 렌더 PNG (영구 보관)
FIGDIR = os.path.join(PKG, "figures")     # 개념도 (영구 보관)
REPO = r"C:\Users\user\OneDrive\2026-1\3D DFN modeling"
FIG = os.path.join(REPO, "docs", "figures")
OUT = sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\user\OneDrive\2026-2\현대건설\260910 공유자료.pptx"

# 흑백(그레이스케일) 테마 규약 — 모식도·표·상자는 색 대신 명도·선 굵기로 구분 (데이터 그림·렌더는 원색 유지)
NAVY = RGBColor(0x1A, 0x1A, 0x1A); ACC = RGBColor(0x40, 0x40, 0x40); GREY = RGBColor(0x59, 0x59, 0x59)
LIGHT = RGBColor(0xF2, 0xF2, 0xF2); WHITE = RGBColor(0xFF, 0xFF, 0xFF); BLACK = RGBColor(0x22, 0x22, 0x22)
F1 = RGBColor(0xF7, 0xF7, 0xF7); F2 = RGBColor(0xEB, 0xEB, 0xEB); F3 = RGBColor(0xDF, 0xDF, 0xDF); MIDGREY = RGBColor(0x8C, 0x8C, 0x8C)
FONT = "맑은 고딕"
prs = Presentation(); prs.slide_width = Inches(13.333); prs.slide_height = Inches(7.5)
BLANK = prs.slide_layouts[6]; _n = [0]
FOOT = "지난 일주일 개발 공유 — 다면 복원 max_sep · edge-cut 블록 판정 · 확률 컨투어   |   2026-09-10"


def _run(p, text, size, bold=False, color=BLACK, italic=False):
    r = p.add_run(); r.text = text; f = r.font
    f.name = FONT; f.size = Pt(size); f.bold = bold; f.italic = italic; f.color.rgb = color
    return r


def _rich(p, text, size, color=BLACK, bold_all=False):
    for part in re.split(r"(\*\*[^*]+\*\*)", text):
        if not part: continue
        if part.startswith("**") and part.endswith("**"): _run(p, part[2:-2], size, bold=True, color=color)
        else: _run(p, part, size, bold=bold_all, color=color)


def slide(title, subtitle=None, tag=None):
    _n[0] += 1
    s = prs.slides.add_slide(BLANK)
    tb = s.shapes.add_textbox(Inches(0.5), Inches(0.28), Inches(12.3), Inches(0.6)); tf = tb.text_frame; tf.word_wrap = True
    _run(tf.paragraphs[0], title, 24, bold=True, color=NAVY)
    if subtitle: _run(tf.add_paragraph(), subtitle, 12, color=GREY)
    ln = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.5), Inches(0.98), Inches(12.3), Pt(2.5))
    ln.fill.solid(); ln.fill.fore_color.rgb = NAVY; ln.line.fill.background()
    if tag:
        t = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(11.3), Inches(0.05), Inches(1.5), Inches(0.28))
        t.fill.solid(); t.fill.fore_color.rgb = ACC; t.line.fill.background()
        p = t.text_frame.paragraphs[0]; p.alignment = PP_ALIGN.CENTER; _run(p, tag, 9, bold=True, color=WHITE)
    ft = s.shapes.add_textbox(Inches(0.5), Inches(7.05), Inches(12.3), Inches(0.3)); p = ft.text_frame.paragraphs[0]
    _run(p, f"{FOOT}   |   {_n[0]}", 9, color=GREY); p.alignment = PP_ALIGN.RIGHT
    return s


def text(s, x, y, w, h, items, size=13, spacing=4):
    tb = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h)); tf = tb.text_frame; tf.word_wrap = True; first = True
    for it in items:
        lvl, t = (it if isinstance(it, tuple) else (0, it))
        p = tf.paragraphs[0] if first else tf.add_paragraph(); first = False; p.space_after = Pt(spacing)
        if lvl == -1: _rich(p, t, size + 1, color=NAVY, bold_all=True)
        elif lvl == 0: _rich(p, "•  " + t, size)
        elif lvl == 2: _rich(p, t, size, color=GREY)
        else: _rich(p, "      –  " + t, size - 1, color=GREY)
    return tb


def note(s, script, src=""):
    body = "[발표 스크립트]\n" + script.strip()
    if src: body += "\n\n[출처·메모]\n" + src.strip()
    s.notes_slide.notes_text_frame.text = body


def table(s, x, y, w, rows, col_w=None, size=11, hdr=True, row_h=0.32):
    nr, nc = len(rows), len(rows[0])
    shp = s.shapes.add_table(nr, nc, Inches(x), Inches(y), Inches(w), Inches(row_h * nr)); tbl = shp.table
    if col_w:
        for i, cw in enumerate(col_w): tbl.columns[i].width = Inches(cw)
    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            c = tbl.cell(i, j); c.margin_left = c.margin_right = Inches(0.06); c.margin_top = c.margin_bottom = Inches(0.03)
            tf = c.text_frame; tf.word_wrap = True; p = tf.paragraphs[0]; p.text = ""
            if hdr and i == 0:
                _run(p, str(val), size, bold=True, color=WHITE); c.fill.solid(); c.fill.fore_color.rgb = NAVY
            else:
                _rich(p, str(val), size); c.fill.solid(); c.fill.fore_color.rgb = (LIGHT if i % 2 == 0 else WHITE)
            c.vertical_anchor = MSO_ANCHOR.MIDDLE
    return tbl


def image(s, path, x, y, w=None, h=None):
    if not os.path.exists(path):
        tb = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w or 4), Inches(h or 1))
        _run(tb.text_frame.paragraphs[0], f"[그림 없음] {os.path.basename(path)}", 10, color=ACC); return tb
    kw = {}
    if w: kw["width"] = Inches(w)
    if h: kw["height"] = Inches(h)
    return s.shapes.add_picture(path, Inches(x), Inches(y), **kw)


def caption(s, x, y, w, txt, size=9):
    tb = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(0.4)); tb.text_frame.word_wrap = True
    _rich(tb.text_frame.paragraphs[0], txt, size, color=GREY)


def box(s, x, y, w, h, title, body, fill=LIGHT, size=11):
    shp = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    shp.adjustments[0] = 0.06; shp.fill.solid(); shp.fill.fore_color.rgb = fill; shp.line.color.rgb = NAVY; shp.line.width = Pt(1)
    tf = shp.text_frame; tf.word_wrap = True; tf.margin_left = tf.margin_right = Inches(0.1); tf.margin_top = Inches(0.06)
    tf.vertical_anchor = MSO_ANCHOR.TOP; p = tf.paragraphs[0]; p.alignment = PP_ALIGN.LEFT
    _run(p, title, size + 1, bold=True, color=NAVY)
    for line in body:
        q = tf.add_paragraph(); q.alignment = PP_ALIGN.LEFT; q.space_after = Pt(2); _rich(q, line, size)
    return shp


def arrow(s, x1, y1, x2, y2):
    c = s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    c.line.color.rgb = NAVY; c.line.width = Pt(2)
    ln = c.line._get_or_add_ln(); tail = etree.SubElement(ln, qn("a:tailEnd")); tail.set("type", "triangle")
    return c


def pv(name, var):
    return os.path.join(PV, f"{name}_{var}.png")


# ============================================================ 1 표지
s = prs.slides.add_slide(BLANK); _n[0] += 1
bg = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, Inches(7.5)); bg.fill.solid(); bg.fill.fore_color.rgb = RGBColor(0x26, 0x26, 0x26); bg.line.fill.background()
SUB = RGBColor(0xD9, 0xD9, 0xD9)
tb = s.shapes.add_textbox(Inches(0.9), Inches(1.8), Inches(11.5), Inches(4)); tf = tb.text_frame; tf.word_wrap = True
_run(tf.paragraphs[0], "지난 일주일 개발 내용 공유", 40, bold=True, color=WHITE)
p = tf.add_paragraph(); p.space_before = Pt(10)
_run(p, "A.  다면 복원 — 중심거리 게이트(max_sep) 고정값을 적응형으로", 18, color=SUB)
p = tf.add_paragraph(); _run(p, "B.  edge-cut 복셀 CCA 블록 판정과 면(다면체) 복원", 18, color=SUB)
p = tf.add_paragraph(); _run(p, "C.  몬테카를로 블록 존재확률 컨투어", 18, color=SUB)
p = tf.add_paragraph(); p.space_before = Pt(30)
_run(p, "기준 데이터: DFM 실측 12면 데모 · 1~6면 시나리오 · 슬라이딩/누적 윈도우  |  결과 뷰어: handoffv2/example_io/paraview", 13, color=SUB)
p = tf.add_paragraph(); _run(p, "2026-09-10  |  서울대 에너지자원공학과 이창무", 13, color=SUB)
note(s, """안녕하세요. 지난 일주일에 개발한 세 가지를 10분 안에 공유드리겠습니다.

A는 절리선을 원판으로 복원할 때 여러 막장면의 절리선을 하나로 잇는 중심거리 게이트 max_sep을 고정값에서 적응형으로 바꾼 것입니다. B는 균열을 두께 0의 분리면으로 다루는 edge-cut 복셀 CCA 블록 판정과, 그 결과를 정확한 다면체로 되돌리는 면 복원입니다. C는 그 판정을 시드 50개에 반복해 만든 블록 존재확률 컨투어입니다.

먼저 세 항목이 파이프라인 어디에 있는지 한 장으로 보겠습니다.""")

# ============================================================ 2 개요
s = slide("0. 세 항목이 파이프라인에서 차지하는 위치", "handoffv2 파이프라인(관측 → 역산 → 복원 → 조건부 실현) 뒤에 블록 판정과 확률장을 붙였고, 복원 단계 안의 매칭 규칙을 고침")
steps = [("관측 절리선", "DFM 12면\n6,447개", False), ("역산", "κ · kr · P32\n(lmin 0.5)", False), ("A. 원판 복원", "다면 매칭\n적응 max_sep", True),
         ("조건부 실현", "부재 조건화\n시드별 JSON", False), ("B. 블록 판정", "edge-cut CCA\n+ 다면체 복원", True), ("C. 확률 컨투어", "MC 50 실현\nP(블록) 복셀장", True),
         ("ParaView", "pvsm + bat\n뷰어 11종", False)]
x0, y0, bw, bh, gap = 0.5, 1.3, 1.55, 1.25, 0.24
for i, (t, body, hi) in enumerate(steps):
    x = x0 + i * (bw + gap)
    b = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y0), Inches(bw), Inches(bh))
    b.adjustments[0] = 0.08; b.fill.solid(); b.fill.fore_color.rgb = (ACC if hi else WHITE)
    b.line.color.rgb = (NAVY if hi else MIDGREY); b.line.width = Pt(1.5 if hi else 1)
    tf = b.text_frame; tf.word_wrap = True; tf.vertical_anchor = MSO_ANCHOR.MIDDLE; tf.margin_left = tf.margin_right = Inches(0.05)
    p = tf.paragraphs[0]; p.alignment = PP_ALIGN.CENTER; _run(p, t, 11.5, bold=True, color=(WHITE if hi else NAVY))
    q = tf.add_paragraph(); q.alignment = PP_ALIGN.CENTER; _run(q, body, 9, color=(WHITE if hi else BLACK))
    if i < len(steps) - 1: arrow(s, x + bw + 0.02, y0 + bh / 2, x + bw + gap - 0.02, y0 + bh / 2)
table(s, 0.5, 2.85, 12.3, [
    ["항목", "이전", "이후 (이번 주)", "왜 중요한가", "관련 뷰어"],
    ["A. 다면 복원 max_sep", "고정 3.5 m (CLI 기본은 2.0 — 다면 0개 함정)", "면 간격·원판 방향에 따라 변하는 **적응 게이트** 1.4·Δx/sinθ (인접면만)", "다면 원판 = 크기가 확정된 원판 = 블록 판정의 앵커. 1~6면 171 → **308**개, 막장면 평행 절리 편향 완화", "열기_다면매칭.bat, 슬라이딩윈도우, 윈도우 1-6/1-7"],
    ["B. edge-cut 블록 판정 + 면 복원", "레거시: 균열 복셀(두께 0.6·복셀) 제거 후 CCA", "균열을 **두께 0 분리면**으로, 이웃 연결선만 절단 → 6-이웃 CCA → 규칙 3종 → **경계 평면 역추적 다면체**", "두께 아티팩트 제거, 부피 손실 없음. 12면 터널접촉 168/168 다면체 복원, 부피비 0.90", "열기_블록뷰(_6면/_고정부피)"],
    ["C. 확률 컨투어", "실현 1개의 블록 목록", "시드 50개 × 블록 판정 → 복셀별 P(블록), 정의 3종(굴착가정/제자리/전체)", "붕락을 '확률'로 답하는 형식. 관측 정보 도달거리 ≈1.5 m 정량화, 블라인드 검증 통과", "열기_블록확률_*.bat 4종 + 재보정판"],
], col_w=[2.0, 2.3, 3.1, 3.2, 1.7], size=9.5, row_h=0.72)
text(s, 0.5, 5.85, 12.3, 1.2, [
    "실행 시간(노트북 GPU 1대): 역산·복원 15초~1분(현장당 1회) → 조건부 실현 4~12초/시드 → edge-cut 판정 8초/실현 → MC 50 실현 8병렬 2~4분. 확률장까지 현장당 수 분",
    "모든 뷰어는 pvpython 스크립트로 레이어·색·카메라를 코드로 고정해 생성 — 데이터가 바뀌면 재생성 가능(handoffv2/scripts/analysis_260903)",
], size=10.5)
note(s, """진회색 상자 세 개가 이번 주 작업입니다. A는 원판 복원 단계 안의 매칭 규칙이고, B와 C는 조건부 실현 뒤에 새로 붙인 후처리입니다.

표의 이전과 이후를 보시면, A는 고정 3.5 m 게이트였고 CLI 기본값은 2.0이라 그대로 쓰면 다면 원판이 0개가 되는 함정이 있었습니다. 이후는 면 간격과 원판 방향에 따라 변하는 적응 게이트입니다. B는 레거시의 균열 복셀 제거 방식을 두께 0 분리면으로 바꾸고 다면체 복원까지 붙였습니다. C는 실현 하나의 블록 목록 대신 복셀별 확률로 답합니다.

실행 시간은 확률장까지 현장당 수 분이라 실무 제약이 없고, 뷰어는 전부 코드로 생성되므로 데이터가 바뀌어도 같은 장면을 다시 만들 수 있습니다.""",
"worklog 09-01~09-04, handoffv2/scripts/analysis_260903/README.md")

# ============================================================ 3 A-1 (배경 + 스윕 병합)
s = slide("A-1. 다면 복원 — 고정 max_sep의 함정과 스윕 (3.5 → 4.5)", "reconstruct_discs_from_traces.py: 검증형 응집 연계 → 반지름 결정. 1~6면(win_f01-06_l05) 기준", tag="A. 다면 복원")
text(s, 0.5, 1.15, 6.0, 5.9, [
    (-1, "복원 절차"),
    "후보 게이트: 같은 절리군, 법선 축각 ≤15°, 공면거리 ≤0.15 m, **중심거리 ≤ max_sep**",
    "병합 확정: 결합 평면 SVD 잔차 ≤ 8 cm ∧ 면당 chord 1개 (게이트는 문, 확정은 잔차)",
    "반지름: 절리선 ≥2개 → 원적합(determined) / 1개 → kr 분포 축소추정(shrinkage)",
    (-1, "왜 다면 원판이 중요한가"),
    "단일면 원판은 반지름이 통계 추정치(0.5~0.6 m). 다면 원판은 두 면을 실제로 가로지른 증거라 **크기가 확정**되고 전방 도메인에 걸쳐 블록 판정의 **앵커**가 됨",
    (-1, "발견 1 — 기본값 함정"),
    "CLI 기본 max_sep **2.0 m** vs 정품 파이프라인 **3.5 m**. 면 간격 1.86~2.83 m라 2.0으로는 다면 원판 **0개** → 앞선 LOFO·span 실험이 오염. 3.5로 수정 후 기존 CSV 정확 재현",
    (-1, "발견 2 — 면쌍별 매칭 수 (3.5 m)"),
    "면 1-2 70 / 2-3 32 / 3-4 60 / 4-5 28 / **5-6 5개** — 면 5-6 간격 2.83 m(최대)가 게이트에 근접. 버그가 아니라 굴착 스텝이 뒤로 갈수록 넓어지는 기하",
    (-1, "발견 3 — LOFO 무검정력"),
    "중간 면을 빼면 양옆 간격이 2배(4.4 m)라 재매칭 실패, 단일면 소형 원판은 2 m 옆 면에 안 닿음 → 이 데이터로는 면 간 예측력 검정 불가",
], size=10.5)
table(s, 6.8, 1.2, 6.0, [
    ["max_sep [m]", "다면 원판", "비인접 면 매칭", "판단"],
    ["3.5 (종전)", "171", "0", "면 5-6 급감"],
    ["**4.5**", "**280**", "**0**", "인접면만, 과병합 없음"],
    ["6.0", "359", "19", "면 건너뜀 시작"],
    ["8.0", "404", "51", "과병합"],
], col_w=[1.4, 1.3, 1.5, 1.8], size=10.5, row_h=0.34)
table(s, 6.8, 3.15, 6.0, [
    ["지표 (1~6면)", "3.5", "4.5"],
    ["다면 원판", "171", "**280**"],
    ["면 5-6 매칭", "5", "**41**"],
    ["원적합(determined)", "174", "**196**"],
    ["면 06 전방 도메인에 걸치는 다면 원판", "**1**", "**24**"],
    ["다면 반지름 중앙 / p90 [m]", "1.51 / 1.85", "1.48 / 2.28"],
], col_w=[3.2, 1.3, 1.5], size=10, row_h=0.33)
text(s, 6.8, 5.3, 6.0, 1.7, [
    "게이트를 키워도 **잔차는 나빠지지 않음**(병합 확정은 잔차·chord 검증). 4.5까지 전부 인접면 매칭, 6.0부터 면 건너뜀 → 4.5 채택. CLI 기본값·파이프라인 모두 4.5로",
    "확률장(w16 MC50): P≥0.5 부피 1.79 → 1.37 m³ — 앵커 1→24로 블록 경계가 **재편**된 것이지 '블록이 덜 생김'이 아님. 트레이드오프로 기록",
], size=10)
note(s, """복원은 두 단계입니다. 같은 절리군, 법선 축각 15° 이내, 공면거리 15 cm 이내, 그리고 중심거리가 max_sep 이내인 절리선 쌍을 후보로 뽑고, 병합할 때마다 평면 잔차 8 cm와 면당 chord 하나를 재검사합니다. 게이트는 문이고 확정은 잔차가 합니다.

다면 원판이 중요한 이유는, 두 막장면을 실제로 가로질렀다는 증거라 크기가 확정되고 전방 도메인에 걸쳐 블록 판정의 앵커가 되기 때문입니다.

이번 주에 CLI 기본값이 2.0 m라 면 간격 1.9~2.8 m에서는 다면 원판이 0개가 되는 함정을 발견해 고쳤습니다. 제 앞선 LOFO 실험이 이 값으로 돌아 오염돼 있었습니다. 면쌍별로 세어 보니 간격이 가장 넓은 면 5-6이 5개뿐이어서 게이트를 스윕했습니다. 오른쪽 표처럼 4.5까지는 비인접 매칭 0개로 다면 원판이 171에서 280으로 늘고, 6.0부터 면을 건너뛰는 과병합이 생깁니다. 4.5를 채택했고, 면 06 전방 도메인에 걸치는 다면 원판이 1개에서 24개가 됐습니다. 확률장은 앵커가 늘면서 재분포돼 P 0.5 이상 부피가 줄었는데, 블록이 덜 생긴 게 아니라 경계가 재편된 것이라 트레이드오프로 기록했습니다.""",
"worklog 09-04 §7-2, §7-3")

# ============================================================ 4 A-2 적응
s = slide("A-2. 고정값 → 적응형 max_sep — 막장면 평행 절리의 방향 편향을 줄임", "회의 중 통찰: 막장면과 평행한 절리일수록 두 면 사이 현이 멀리 떨어져 나타난다 → 게이트가 방향에 따라 달라야 함", tag="A. 다면 복원")
box(s, 0.5, 1.2, 4.0, 2.3, "이론", [
    "두 막장면(간격 Δx)에 남는 현의 중심거리 = **Δx / sinθ** = Δx / √(1−nx²)",
    "θ = 원판 법선과 굴진축(x) 사이 각. nx→1(막장면 평행)에서 발산",
    "실측: 4.5 m가 놓친 다면 원판의 |nx| 평균 0.74 > 잡은 것 0.70",
], fill=F1, size=10.5)
box(s, 0.5, 3.65, 4.0, 2.0, "구현 (--adaptive-sep → 파이프라인 기본값)", [
    "게이트 = clamp(**1.4 · Δface_x / sinθ**, same_face, 8.0)",
    "Δface_x > 3.2 m(비인접)면 후보 제외 — 면 건너뜀 원천 차단",
    "병합 확정(잔차 8 cm·면당 chord 1개)은 불변",
], fill=F2, size=10.5)
table(s, 0.5, 5.8, 4.0, [
    ["1~6면", "다면", "비인접", "면5-6", "막장면평행"],
    ["고정 4.5", "280", "0", "41", "36"],
    ["**적응 k1.4**", "**308**", "**0**", "60", "**58**"],
    ["적응 k1.6", "340", "0", "64", "67"],
], col_w=[0.95, 0.7, 0.75, 0.75, 0.85], size=9.5, row_h=0.3)
image(s, os.path.join(FIG, "adaptive_sep_direction_bias.png"), 4.7, 1.2, w=8.1)
text(s, 4.7, 5.15, 8.1, 1.9, [
    "(a) 다면 원판의 |nx| 분포: 회색 = 전체 절리 모집단(3,667), 파랑 = 고정 4.5(280), 빨강 = 적응(308). 적응이 막장면 평행(|nx|→1) 쪽 꼬리를 모집단 형상에 가깝게 채움",
    "(b) 막장면 평행 원판 비율: 모집단 **26.9 %** / 고정 **12.9 %** / 적응 **18.8 %** — 편향이 절반 이상 해소. 비인접 과병합은 0 유지",
    "현재 데이터는 세트 3개 방향이 비슷(|nx| 0.68~0.74)해 편향이 크지 않지만, 막장면 평행 세트가 지배적인 현장에서는 적응이 필수",
], size=10)
note(s, """한 걸음 더 나간 것이 적응형 게이트입니다. 회의 중 나온 통찰대로, 두 면에 남는 현의 중심거리는 면 간격을 sinθ로 나눈 값이라 막장면과 평행한 절리일수록 멀어집니다. 고정 게이트는 그런 원판을 체계적으로 놓칩니다. 실제로 4.5 m가 놓친 다면 원판의 |nx|가 잡은 것보다 컸습니다.

구현은 게이트를 1.4 곱하기 면 간격 나누기 sinθ로 두고, 비인접 쌍은 후보에서 빼며, 병합 확정 조건은 그대로입니다. 결과는 다면 원판 280에서 308, 막장면 평행 원판 36에서 58, 비인접 과병합은 여전히 0입니다.

오른쪽 그림에서 막장면 평행 원판 비율이 모집단 26.9%에 대해 고정 12.9%, 적응 18.8%로 편향이 절반 이상 해소됐습니다. 지금 데이터는 세 절리군 방향이 비슷해 효과가 크지 않지만, 막장면 평행 절리군이 지배적인 현장에서는 이 보정이 없으면 다면 원판을 체계적으로 놓칩니다.""",
"worklog 09-04 §7-4. 그림 docs/figures/adaptive_sep_direction_bias.png")

# ============================================================ 5 A-3 전면 적용
s = slide("A-3. 적응 max_sep 전면 적용 결과 — 슬라이딩 3세트 · 누적 1~6면 · 1~7면", "열기_다면매칭.bat / 열기_슬라이딩윈도우.bat / 열기_윈도우_1-6면·1-7면.bat 이 모두 적응본으로 재생성됨 (파일명 동일)", tag="A. 다면 복원")
image(s, pv("dfn_multiface_matching", "fit"), 0.5, 1.15, w=5.6)
caption(s, 0.5, 4.35, 5.6, "dfn_multiface_matching.pvsm — 1~6면 복원 원판(주황 = 다면 원판 강조). 도메인 상자 안 전방까지 걸치는 큰 원판이 앵커")
table(s, 6.4, 1.2, 6.4, [
    ["윈도우 (관측면 → 대상면)", "다면 원판 (고정 → 적응)", "원적합 (고정 → 적응)", "비인접 과병합", "확률장 max P / P≥0.5 [m³]"],
    ["w14 (1-4 → 5)", "144 → **197**", "118 → 137", "0", "1.0 / 3.44"],
    ["w25 (2-5 → 6)", "106 → **219**", "111 → 161", "0", "1.0 / 1.50"],
    ["w36 (3-6 → 7)", "82 → **169**", "89 → 124", "0", "1.0 / 0.56"],
    ["w16 (1-6 → 7)", "280 → **308**", "196 → 219", "0", "1.0 / 1.05"],
    ["w17 (1-7 → 8)", "180 → **345**", "212 → 279", "0", "0.92 / 0.02"],
], col_w=[1.6, 1.4, 1.3, 0.9, 1.2], size=9.5, row_h=0.42)
text(s, 6.4, 4.0, 6.4, 3.0, [
    "**게이트의 영향 범위는 관측 원판뿐.** 대상면 절리선 예측치(C-2)는 50개 실현 모두 **관측 원판 기여 0개** = 확률 균열 100 % → 그 값은 적응 게이트의 성패 지표가 아니라 강도 보정(P32·lmin)의 지표",
    "w17 증가폭이 큰 이유: 3.5 잔존 상태였고 면 07 포함 창이라 방향 편향이 컸음. 비인접 과병합 0은 Δface_x > 3.2 m 후보 제외로 구조적으로 보장",
    "절차(각 창): reconstruct --adaptive-sep → CSV 교체 → canonical 재export → observed_discs 재생성 → MC 50 재실행 → 확률장 집계 → pvsm (MC 150 실현 약 10분). 파이프라인 기본값도 `--adaptive-sep`으로 승격",
], size=10)
text(s, 0.5, 4.9, 5.6, 2.1, [
    (-1, "부수 분석 — 같은 면 병합 효과 (1~6면, 세트 1~3)"),
    "조각 1,173개(32 %)가 463개 그룹으로 병합 → 절리선 3,667 → 2,957(−19 %). ≥0.5 m 절리선 794 → 901(+13 %): 짧은 조각이 병합으로 검출하한을 넘어옴",
    "원판 복원은 병합 후 분포를, kr·P32는 원시 h5를 보므로 층별 정의만 일관하면 자기보정 — 'relink' 기각 논리 재확인",
], size=10)
note(s, """적응 게이트를 모든 창에 적용한 결과입니다. 왼쪽이 다면 매칭 뷰어이고 주황 원판이 다면 원판입니다. 도메인 상자 안 전방까지 걸치는 큰 원판이 블록 판정의 앵커가 됩니다.

표에서 슬라이딩 세 창의 다면 원판이 각각 197, 219, 169개로 늘었고, 1~7면 창은 180에서 345로 가장 많이 늘었습니다. 3.5가 남아 있던 데다 면 07이 들어가 방향 편향이 컸기 때문인데, 검출하한 재보정과 합쳐 면 08 예측이 125 대 139로 밴드 안에 들어왔습니다. 확률장 열은 슬라이딩 세 창 모두 최대 P 1.0, 즉 모든 실현에서 반복되는 블록이 있습니다.

재생성 절차는 창마다 같고, 파일명을 유지했기 때문에 기존 bat을 열면 적응본이 보입니다. 윈도우 파이프라인의 기본값도 적응 게이트로 올려서 이후 만드는 창은 자동으로 적용됩니다.""",
"worklog 09-04 §6, §7-5, §9")

# ============================================================ 6 B-1 edge-cut 원리
s = slide("B-1. edge-cut 원리와 구현 — 균열을 '두께 0의 분리면'으로", "compute_edge_cuts → run_cca_edgecut (dfn_analysis/detect_blocks_from_domain_json.py). 레거시 균열복셀 방식을 대체", tag="B. edge-cut")
image(s, os.path.join(FIGDIR, "fig_edgecut_concept.png"), 0.5, 1.2, w=8.3)
text(s, 8.9, 1.15, 3.9, 5.9, [
    (-1, "절단 판정 (원판마다, AABB 안 복셀만)"),
    "복셀 중심 p₁, p₂의 평면 부호거리 d₁ = (p₁−c)·n, d₂ = (p₂−c)·n",
    "부호가 다르면 연결선이 평면을 가로지름. 교차점 q = p₁ + t(p₂−p₁), t = d₁/(d₁−d₂)",
    "**|q − c|² ≤ r²** 일 때만 절단 (교차점이 평면 위이므로 3D 거리 = 면내 거리) — 유한 원판 처리의 핵심",
    "축별 절단 배열 cutx/cuty/cutz에 OR 누적 → 원판 2.5만 개, 복셀 500만 개에서 8초",
    (-1, "연결성분"),
    "살아있는 6-이웃 연결선으로 희소 그래프 → connected_components → 복셀 라벨 3D 배열",
    (-1, "왜 6-이웃인가"),
    "26-이웃이면 모서리·꼭짓점 접촉으로 분리면을 '건너뛰어' 블록이 합쳐질 수 있음 → 보수적 6-이웃 고정",
], size=10)
text(s, 0.5, 5.6, 8.3, 1.45, [
    "**왜 바꿨나**: 레거시는 원판 근방 복셀을 두께 0.6·복셀로 잘라내 암반에서 제거 → 격자를 키우면 균열면에 구멍(블록 못 닫음), 줄이면 부피 손실. 복셀 0.2에서 percolation 이상치 59 m³. edge-cut은 복셀을 빼지 않고 연결선만 끊어 두께 파라미터가 없음",
    "(c) 원판 끝(crack tip) 너머는 연결이 유지 → 유한 균열 하나로는 블록이 닫히지 않고 **여러 균열이 협력해야 닫힘**. 참고: 상위 30개 균열만 넣으면 edge-cut 0개 vs 3DEC식 완전 절단 62개 — 부분 관통 균열 취급 차이(하한 vs 상한)",
], size=10)
note(s, """B 파트입니다. 출발점은 레거시 복셀 판정기였는데, 균열 두께를 복셀의 0.6배로 잡아 격자를 키우면 균열면에 구멍이 나고 줄이면 부피가 깎였습니다. 그래서 균열을 두께 0의 분리면으로 다루는 edge-cut으로 바꿨습니다.

(a)가 종전, (b)가 edge-cut입니다. 복셀은 하나도 빼지 않고, 두 복셀 중심의 평면 부호거리가 다르고 교차점이 원판 반지름 안에 있을 때만 그 연결선을 끊습니다. 원판 2만 5천 개, 복셀 500만 개에서 8초입니다. (c)처럼 남은 6-이웃 연결선의 연결성분이 블록 후보인데, 원판 끝 너머는 연결이 살아 있어 유한 균열 하나로는 블록이 닫히지 않고 여러 균열이 협력해야 닫힙니다. 6-이웃을 쓰는 이유는 26-이웃이면 모서리 접촉으로 분리면을 건너뛸 수 있기 때문입니다.

참고로 상위 30개 균열만 넣으면 edge-cut은 블록 0개, 3DEC식 완전 절단은 62개로 정반대인데, 부분 관통 균열 취급 차이입니다. edge-cut을 기하 하한, 3DEC식을 상한으로 둡니다.""",
"코드: compute_edge_cuts(), run_cca_edgecut(). worklog 09-01(복셀 해상도 스윕·edge-cut 도입)")

# ============================================================ 7 B-2 규칙
s = slide("B-2. 연결성분에서 '블록'을 고르는 규칙 3종", "같은 edge-cut 라벨에서 문제 설정(굴착 전/후, 자유면)에 따라 다른 규칙을 적용", tag="B. edge-cut")
table(s, 0.5, 1.15, 12.3, [
    ["정의", "암반(rock)", "채택 조건", "의미 / 용도", "스크립트"],
    ["① 굴착 가정 · 터널 접촉", "도메인 − 터널 프리즘(단면 폴리곤을 x 전 구간으로 연장)", "터널 복셀과 6-이웃 접촉 ∧ 도메인 6면 외곽 비접촉 ∧ 복셀 수 ≥ min", "전방을 다 팠을 때 벽에 드러날 블록(overbreak 후보). 레거시와 동일 기준", "detect_blocks_from_domain_json --method edgecut"],
    ["② 제자리(in-situ) 닫힌 블록", "도메인 전체 (터널 없음)", "도메인 6면 어디에도 닿지 않는 컴포넌트 ∧ 복셀 수 ≥ min", "굴착 전 암반 안에 '블록인 채' 존재하는 것 — 전방 예측·확률장의 기본 정의", "detect_interior_labels.py / make_unexcavated_blocks.py"],
    ["③ 막장면 쐐기", "도메인 전체 (미굴착)", "자유면 패치(막장면 × 터널 단면 내부)에 접촉 ∧ 막장면의 단면 밖 부분에는 비접촉 ∧ 다른 외곽 비접촉", "지금 막장면에 드러난 쐐기. 관측 균열만으로도 판정 가능(ε 0.05 권장)", "detect_face_blocks --labels observed|all"],
    ["(보조) 기굴착 벽면 블록", "굴착 구간(x 뒤쪽) JSON에 ①", "①과 동일", "이미 판 터널 벽에 있는 블록(실측 지배) — 현재상태 뷰의 색 블록", "export --x-range 로 굴착 구간 JSON 생성 후 ①"],
], col_w=[2.0, 2.4, 3.2, 2.9, 1.8], size=9, row_h=0.85)
text(s, 0.5, 5.6, 12.3, 1.45, [
    (-1, "고정 파라미터"),
    "복셀 0.1 m (=닫힘 판정 스케일 ε), 6-이웃, 최소 크기: 8복셀(0.008 m³) 기본 / **고정 부피 0.05 m³**(=50복셀) 비교 뷰. 균열 두께 파라미터 없음(edge-cut). 외곽 접촉 컴포넌트는 항상 제외(도메인 바깥으로 이어진 암반)",
    "출력: 라벨 npz(origin, voxel_size, labels), summary.csv(label, n_voxels, volume_m3, centroid, contact_area_m2, rank) — MC는 라벨만 저장하고 다면체는 대표 실현에서만",
], size=10)
note(s, """연결성분 중 무엇을 블록이라 부를지 규칙 세 가지입니다.

①은 터널을 판 것으로 보고 터널에 접하면서 외곽에는 안 닿는 컴포넌트입니다. overbreak 후보이고 레거시와 같은 기준이라 이전 결과와 비교됩니다. ②는 터널 없이 도메인 여섯 면 어디에도 안 닿는 컴포넌트, 즉 굴착 전 암반에 이미 블록인 채 있는 것으로 전방 예측과 확률장의 기본 정의입니다. 회의에서 미래 터널을 미리 뚫어 판정하는 건 예측이 아니라는 지적을 받고 이렇게 바꿨습니다. ③은 막장면 자유면 패치에 접한 쐐기로, 관측 균열만으로도 판정됩니다.

파라미터는 복셀 0.1 m, 6-이웃, 최소 8복셀 또는 고정 부피 0.05 m³이고 두께 파라미터는 없습니다. MC에서는 라벨만 저장하고 다면체는 대표 실현에서만 만듭니다.""",
"detect_blocks_from_domain_json.py filter_blocks_edgecut(), detect_face_blocks.py, analysis_260903/detect_interior_labels.py")

# ============================================================ 8 B-3 면 복원
s = slide("B-3. 면(다면체) 복원 — 계단형 복셀 블록을 경계 평면으로 되깎기", "reconstruct_block_polyhedra.py: edge-cut이 찾은 '진짜 닫힌 블록'의 경계 원판을 역추적해 볼록 다면체 복원", tag="B. edge-cut")
image(s, os.path.join(FIGDIR, "fig_polyhedra_concept.png"), 0.5, 1.15, w=8.2)
text(s, 8.9, 1.15, 3.9, 5.9, [
    (-1, "경계 원판 판별 (블록 복셀 중심 집합 P, 후보 원판 j)"),
    "부호거리 dᵢⱼ = (pᵢ − cⱼ)·nⱼ 에서 **95 % 이상이 한쪽**(min(양,음 비율) ≤ 0.05) — 블록을 관통하는 슬릿은 제외",
    "**min|dᵢⱼ| ≤ 1.2h** (경계에 실제로 접함, h = 복셀)",
    "근접 복셀의 면내 거리² ≤ (rⱼ + h)² (접촉부가 원판 반경 안)",
    (-1, "클리핑 순서"),
    "복셀 bbox(+0.75h) 볼록체 → ① 경계 원판 반공간 → ② 인접 터널 벽 변 평면(암반 쪽 유지) → ③ 도메인 6면. 각 단계는 hull 모서리-평면 교점으로 ConvexHull 재구성(clip_hull)",
    (-1, "검증"),
    "R_V = V_poly / V_voxel. **R_V > 2 → flag 'open'**(경계면 누락 의심, 표시만 하고 제외 안 함)",
    "12면 터널접촉 168/168 성공, 부피비 중앙값 **0.90**; 닫힌 블록 전체 1,891 중 1,880 ok; 6면 고정부피 171/174; 막장면 쐐기 7/7",
], size=10)
text(s, 0.5, 5.75, 8.2, 1.3, [
    "한계: 볼록 재구성이라 블록 안으로 파고들다 끝나는 슬릿과 비볼록 형상은 외곽 형상으로 근사. 터널 벽은 폴리곤 변 평면으로 국소 근사. 큰 쐐기(한양대 top-20 케이스)는 연장 평면 과절단으로 부피 과소 → 그 경우 복셀 부피(vol_voxel)를 기준값으로 보고. 산출 vtp의 셀 배열: block_id, tunnel_contact, flag_open, rgb(고정색)",
], size=10)
note(s, """edge-cut 결과는 계단형이라 역학해석 입력으로도, 보기에도 좋지 않습니다. 그래서 닫힌 블록으로 판정된 것만 골라 경계 평면을 역추적해 볼록 다면체로 되돌립니다.

경계 원판 조건은 세 개입니다. 블록 복셀의 95% 이상이 그 평면의 한쪽에 있어야 해서 블록을 관통하는 슬릿은 제외되고, 가장 가까운 복셀이 1.2 복셀 이내여서 실제로 접해야 하며, 접촉부가 원판 반경 안에 있어야 합니다. 그 다음 복셀 경계상자로 만든 볼록체를 경계 원판, 인접 터널 벽, 도메인 면 순서로 잘라 나갑니다.

검증은 다면체 부피를 복셀 부피로 나눈 비율로 하고, 2를 넘으면 경계면 누락을 의심해 open 표시만 합니다. 12면 터널 접촉 168개 전부 성공, 부피비 중앙값 0.90, 닫힌 블록 1,891개 중 1,880개 정상입니다. 한계는 볼록 복원이라 슬릿과 비볼록 형상은 근사이고, 큰 쐐기는 과절단으로 부피가 작게 나와 복셀 부피를 같이 보고합니다.""",
"blocks_edgecut/poly_tunnel_summary.csv(168), poly_all_summary.csv(1,891/1,880), f06_minvol05/unexcavated_interior_summary.csv(174/171)")

# ============================================================ 9 B-4 검증·뷰어
s = slide("B-4. 검증 실험과 결과 뷰어 — 블록뷰(1~6면, 고정 부피 0.05 m³)", "블록 통계는 파라미터에 민감하다 — 어떤 값이 물리적이고 어떤 값이 이산화 아티팩트인지 실험으로 분리", tag="B. edge-cut")
table(s, 0.5, 1.15, 6.6, [
    ["실험", "결과", "함의"],
    ["복셀 0.3 → 0.05 m (12면)", "블록 29 → 254개, 총부피 22.9 → 3.7 m³ 단조 감소", "복셀 = ε. 절대값 보고 금지, 0.1 m 고정 상대 비교"],
    ["최소 8복셀 → 고정 0.05 m³ (6면)", "내부 2,067개/51.9 m³ → 171개/19.2 m³, 벽면 145 → 11", "미세조각 제거. 0.2 m 격자면 1,004개 → 분할의 ε 의존은 잔존"],
    ["상위 30개 대형 균열만", "edge-cut 블록 **0개** vs 3DEC식 62개", "블록을 닫는 것은 소형 균열의 교차. top-N 선별은 위험"],
    ["rmax 25 → 10 m", "DFN 통계 동일이나 블록 총부피 −49 %, 최대 −81 %", "크기 절단은 블록 결과를 바꾸는 물리 파라미터"],
    ["시드 12개 반복", "블록 수 CV 11 %, 총부피 CV 21 %, 위치 일치 8~9 %", "재현되는 것은 통계량뿐 → 확률장(C)"],
], col_w=[2.0, 2.5, 2.1], size=9, row_h=0.62)
image(s, pv("dfn_blockview_f06_minvol05", "fit"), 7.3, 1.15, w=5.5)
caption(s, 7.3, 4.3, 5.5, "열기_블록뷰_6면_고정부피.bat — 기굴착(회색 터널 + 벽면 블록 고정색) / 막장면(캡 + 관측 쐐기 빨강) / 미굴착 전방(내부 닫힌 블록 청회색)")
text(s, 0.5, 5.0, 6.6, 2.0, [
    (-1, "원칙"),
    "복셀 0.1 m · 6-이웃 · edge-cut 고정, 최소 크기는 용도별 명시",
    "블록 개수·부피는 실현 분포와 확률장으로 보고, 대표 실현은 다면체로 시각화",
    "역학해석 입력은 '블록 + 경계평면 목록'(다면체 복원 부산물)으로 넘길 수 있게 유지",
], size=10)
text(s, 7.3, 4.75, 5.5, 2.3, [
    "내부 닫힌 블록 171개 / 19.2 m³ (최대 2.10, >0.1 m³ 54개), 벽면 11개, 쐐기 6개. 같은 장면 min 8복셀 판(열기_블록뷰_6면.bat)은 내부 2,067개 — 나란히 열어 비교",
    "12면 데모: 현재상태 뷰(벽면 84 + 쐐기 7 + 내부 2,144), 매트릭스 뷰(닫힌 블록 1,891개 = 암반의 1.5 %) — bat 없이 pvsm 직접 열기",
    "장면은 make_scene_geometry.py(터널·캡·박스) → bake_block_rgb.py → make_pvsm_*.py(pvpython)로 코드 생성",
], size=9.5)
note(s, """수치를 어디까지 믿을지 확인한 실험입니다. 복셀을 0.3에서 0.05 m로 바꾸면 블록 수가 29에서 254로 늘고 수렴하지 않습니다. 복셀이 곧 닫힘 판정 스케일이라 절대값은 보고하지 않고 0.1 m 고정 상대 비교만 합니다. 최소 크기를 고정 부피 0.05 m³로 두면 6면 내부 블록이 2,067에서 171개로 줄어 미세조각이 사라집니다. 상위 30개 균열만으로는 블록이 0개인데, 블록을 닫는 것은 소형 균열의 교차라 top-N 선별이 위험하다는 뜻입니다. 시드 12개에서 블록 수는 안정적이지만 위치 일치는 8~9%뿐이라 C 파트의 확률장이 필요합니다.

오른쪽이 1~6면 블록뷰입니다. 기굴착 구간의 벽면 블록, 막장면 캡의 관측 쐐기, 전방 미굴착 암반의 내부 블록을 한 장면에 놓았고, 내부 171개, 19.2 m³입니다. 8복셀 기준의 원래 뷰와 나란히 열면 미세조각 차이가 바로 보입니다.""",
"docs/블록판정_시드민감도_실험보고_260901.md, worklog 09-01~09-04 §10")

# ============================================================ 10 C-1 정의·절차
s = slide("C-1. 블록 존재확률 컨투어 — 정의와 산출 절차", "'붕락을 확률로 표시할 것인가'(9/4)에 대한 서울대 쪽 답: 기하 블록의 존재확률장은 이미 산출 가능", tag="C. 확률 컨투어")
box(s, 0.5, 1.2, 3.9, 2.5, "정의", [
    "복셀별 **P(블록) = 그 복셀이 닫힌 블록에 속한 실현 수 / 전체 실현 수**",
    "실현 = 조건부 DFN(관측 복원 균열 고정 + 확률 균열 시드별 생성) 1개",
    "블록 = edge-cut CCA 컴포넌트 중 규칙 ①/②/③을 만족하는 것 (B 파트)",
], fill=F1, size=10.5)
box(s, 4.6, 1.2, 3.9, 2.5, "블록 정의 — 제자리(in-situ)를 대표로 채택", [
    "**제자리(in-situ)** [채택]: 터널을 파지 않은 암반 안에서 균열면으로 완전히 둘러싸여 도메인 외곽에 닿지 않는 덩어리 — '굴착 전 암반에 블록인 채 있는가'",
    "굴착 가정·터널 접촉 [보조]: 전 구간 굴착 시 벽에 닿는 블록(overbreak 후보) — 구보정 기반, 재생성 예정",
    "전체 도메인 [보조]: 기굴착 + 전방 한 장면",
], fill=F2, size=10)
box(s, 8.7, 1.2, 4.1, 2.5, "실행 시간 (노트북 GPU 1대)", [
    "실현 1개 생성(조건화 포함): **4~12초**",
    "기하 블록 판정(균열 2.5만, 복셀 500만): **8초**",
    "MC 50 실현(8병렬): 6면 전방 **~2분**, 전체 도메인 **~4분**",
    "→ 몬테카를로는 실무 시간 제약이 사실상 없음",
], fill=F3, size=10.5)
text(s, 0.5, 3.95, 12.3, 3.1, [
    (-1, "절차"),
    "① 관측 막장면(예: 1~6면) 절리선 → 역산(절리군 재분류, κ, kr, P32) → 관측 절리선 복원 원판(결정론, A 파트) — 현장당 1회, 15초~1분",
    "② 시드 k = 1..50 (k×101): 확률 균열 생성 → **부재 조건화**(관측면에 0.5 m 이상 절리선을 남겼을 균열 제거) → 도메인 절단 (export_domain_dfn_json)",
    "③ 실현마다 복셀 0.1 m 격자 edge-cut 판정 → 라벨 npz만 저장(detect_interior_labels.py) → 8병렬 워커(xargs -P 8, 완료 시드 skip)",
    "④ 집계(aggregate_block_prob_*.py): 복셀별 블록 소속 횟수/50 = p_block, 가우시안 σ=3 평활 p_smooth → vti + 2D 투영 png + pvsm(등치면 0.02/0.1/0.3/0.5, 범위 0~0.5)",
    (-1, "해석 시 명시할 것"),
    "P는 **기하 존재확률**이지 거동 확률이 아님 (운동학·역학은 한양대 단계). 블록 닫힘 판정 스케일 ε = 복셀 0.1 m — 절대값이 아니라 **상대 비교**용. 수치 인용 시 버전(재보정 여부·적응 매칭 여부) 명시",
    "제자리 정의를 택한 이유: 미굴착 암반에 미래 터널을 미리 뚫어 '터널 접촉'을 묻는 것은 예측이 아니라 굴착 결과의 가정(회의 지적). 제자리 블록은 도메인 깊은 곳까지 세므로 **계획 터널 윤곽과 겹쳐** 읽음",
], size=11)
note(s, """C 파트, 확률 컨투어입니다. 9/4의 '붕락을 확률로 표시할 것인가'에 대한 저희 쪽 답입니다.

정의는 복셀마다 닫힌 블록이었던 실현 수를 50으로 나눈 것이고, 용도별로 제자리, 굴착 가정, 전체 도메인 세 가지입니다. 제자리가 전방 예측의 기본이고, 굴착 가정은 overbreak 후보에 해당합니다.

절차는 시드마다 확률 균열을 생성해 부재 조건화하고, edge-cut 판정으로 라벨만 저장하며, 8병렬로 돌린 뒤 집계해 vti와 등치면 상태 파일을 만듭니다. 실현 하나 4~12초, 판정 8초, 50개가 2~4분이라 실무 제약이 없습니다.

해석할 때 두 가지를 명시합니다. 이 P는 기하 존재확률이지 거동 확률이 아니고, 복셀 0.1 m 기준이라 상대 비교용입니다. 그리고 재보정과 적응 매칭 버전에 따라 수치가 달라지므로 버전을 같이 적습니다.""",
"analysis_260903/aggregate_block_prob_insitu_l05.py, mc_worker_insitu_l05.sh, worklog 09-02/09-03")

# ============================================================ 11 C-2 뷰어 + 핵심 수치
s = slide("C-2. 확률장 뷰어와 핵심 수치 — 윈도우 뷰에서 예측과 실측을 한 화면에", "열기_슬라이딩윈도우.bat (w14/w25/w36 레이어 토글) · 열기_윈도우_1-6면/1-7면.bat · 열기_블록확률_*.bat", tag="C. 확률 컨투어")
image(s, pv("dfn_sliding_windows", "fit"), 0.5, 1.15, w=6.0)
caption(s, 0.5, 4.55, 6.0, "dfn_sliding_windows.pvsm (w14): 실측면 4 + 대상면 평면, 관측 원판(단일/다면), 제자리 확률장 등치면, **파란 선 = 대상면 실측 절리선**")
image(s, pv("dfn_block_probability_insitu_lmin05", "fit"), 6.8, 1.15, w=6.0)
caption(s, 6.8, 4.55, 6.0, "dfn_block_probability_insitu_lmin05.pvsm: 1~6면 **제자리** 확률장(재보정) — 막장면 캡 직후 천장부에 P≥0.5 등치면, 그 너머 소멸. 굴착가정 확률장(구보정)은 보조")
text(s, 0.5, 5.0, 6.0, 2.1, [
    (-1, "핵심 수치 (1~6면 제자리, lmin 0.5 재보정, MC50)"),
    "실현별 블록 265~402개 · 6.1~11.3 m³. **P≥0.5 부피 1.79 m³**, P=1.0 복셀 396개 — 전부 **막장면 직후 천장부**(z 2.5~5 m), x > 13 m에서 소멸 → 관측 정보 **도달거리 ≈ 1.5 m**",
    "재보정으로 P32 합 −43 %인데 블록 부피는 −85 %: 강도의 비선형 증폭. 적응 매칭 재생성본은 P≥0.5 ≈ 1.05 m³로 재분포(버전 명시)",
], size=10)
text(s, 6.8, 4.95, 6.0, 2.15, [
    (-1, "블라인드 검증 — 대상면 절리선(≥0.5 m) 예측 vs 실측"),
    "w14 122±12 vs 169 / w25 129±11 vs 168 / w17 125±11 vs 139(90백분위) / w36·w16 105·122 vs 73(면07 결손). **관측 원판 기여 0** → 전부 확률 균열 몫",
    "비교는 관측·예측 모두 ≥ 0.5 m 기준(모델 생성 하한 r ≥ 0.5 m). 면 08: lmin 0 → +56 % 과대, **lmin 0.5 → P21 1.98±0.21 vs 2.18 밴드 안(80백분위)**. 개수 114 vs 139 과소는 모델 절리선이 적고 긴 형상 차이",
    "**lmin < 0.5 재보정 기각(09-09)**: 같은 하한으로 대조하면 낮출수록 과소가 커짐(0.4 → 2.15 vs 2.49 / 0.3 → 2.48 vs 2.85) — 모델이 0.5 m 미만 절리선을 만들지 못해 관측에만 짧은 절리선이 늘기 때문 → **검출하한 0.5 m 고정**",
    "블록 위치: 막장면 1.5 m 너머는 복셀 단위 예측력 없음(lift 1.2). 남은 것: 확률장 버전 통일, 한양대 거동 판정과 결합한 '거동 확률장'",
], size=9.5)
note(s, """결과 뷰어와 핵심 수치입니다. 왼쪽 슬라이딩 윈도우 뷰는 실측면과 대상면, 관측 원판, 제자리 확률장 등치면, 그리고 파란 선으로 대상면 실측 절리선을 한 화면에 놓아 예측과 실측을 눈으로 대조합니다. 오른쪽은 전체 도메인 굴착 가정 확률장인데, 기굴착 구간은 관측 벽면 블록, 전방은 막장면 원판 기반 천장 블록이 고확률입니다.

핵심 수치만 말씀드리면, 1~6면 제자리 확률장은 P 0.5 이상이 1.79 m³로 막장면 직후 천장부에만 있고 x 13 m 너머에서 사라집니다. 관측 정보 도달거리가 약 1.5 m라는 뜻입니다. 블라인드 검증에서도 그 너머는 위치 예측력이 없었고, 대신 면 08 절리선 통계는 검출하한 재보정 후 실측이 예측 밴드 안에 들어왔습니다. 재보정 전에는 56% 과대였습니다.

남은 일은 굴착 가정 확률장의 버전 통일, 한양대 거동 판정과 실현 단위로 결합한 거동 확률장, 그리고 MC 횟수 합의입니다. 이상입니다. 질문 주시면 해당 뷰어를 열어 보면서 설명드리겠습니다.""",
"worklog 09-02/09-03 블라인드 검증·재보정 절, 09-04 §2~§5. 그림 출처: docs/figures/block_probability_insitu_lmin05_mc50.png, blind_face08_lmin05.png(삭제한 C-2/C-3 슬라이드에 있던 자료)")

# 10분용 축약 스크립트가 있으면 노트의 스크립트 부분만 교체 (출처·메모는 유지)
try:
    sys.path.insert(0, SC)
    from scripts_v3_short import SHORT
    for i, s_ in enumerate(prs.slides, 1):
        if i in SHORT:
            old = s_.notes_slide.notes_text_frame.text
            src = old.split("[출처·메모]\n")[-1] if "[출처·메모]" in old else ""
            s_.notes_slide.notes_text_frame.text = "[발표 스크립트]\n" + SHORT[i].strip() + ("\n\n[출처·메모]\n" + src if src else "")
    print("short scripts applied:", len(SHORT))
except ImportError:
    pass

prs.save(OUT)
print("saved", OUT, "slides", len(prs.slides))
