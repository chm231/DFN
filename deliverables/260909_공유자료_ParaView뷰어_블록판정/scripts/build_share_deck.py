# -*- coding: utf-8 -*-
"""공유자료: ParaView 뷰어 · 실현 과정 · 블록 판별 알고리즘 (python-pptx)"""
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE, MSO_CONNECTOR
from pptx.oxml.ns import qn
from lxml import etree
import os, re, sys

SC = os.path.dirname(os.path.abspath(__file__))
PV = os.path.join(SC, "pv")
REPO = r"C:\Users\user\OneDrive\2026-1\3D DFN modeling"
FIG = os.path.join(REPO, "docs", "figures")
H2FIG = os.path.join(REPO, "handoffv2", "figures")
OUT_DIR = os.path.join(REPO, "deliverables", "260909_공유자료_ParaView뷰어_블록판정")
OUT = os.path.join(OUT_DIR, "공유자료_ParaView뷰어_블록판정_260909.pptx")
os.makedirs(OUT_DIR, exist_ok=True)

# 렌더 변형 선택 (state → suffix)
VAR = {
    "dfn_blockview_f06_minvol05": "fit", "dfn_blockview_f06": "fit", "dfn_blockview": "fit",
    "dfn_block_probability_insitu_lmin05": "fit", "dfn_block_probability_full": "fit",
    "dfn_sliding_windows": "saved", "dfn_window_f01-06": "saved", "dfn_window_f01-07": "saved",
    "dfn_current_state": "fit", "dfn_face_blocks": "close", "dfn_face_mc": "close",
    "dfn_matrix_blocks": "fit", "dfn_multiface_matching": "saved", "dfn_full": "fit",
    "dfn_top30": "fit", "dfn_top30_3dec": "fit", "dfn_windows": "fit",
}
for k, v in list(VAR.items()):
    if len(sys.argv) > 1:  # override: name=variant
        for a in sys.argv[1:]:
            if "=" in a:
                kk, vv = a.split("="); VAR[kk] = vv
def pv(name):
    return os.path.join(PV, f"{name}_{VAR[name]}.png")

NAVY = RGBColor(0x1F, 0x3A, 0x5F); ACC = RGBColor(0xC0, 0x39, 0x2B); GREY = RGBColor(0x59, 0x59, 0x59)
LIGHT = RGBColor(0xEE, 0xF1, 0xF5); WHITE = RGBColor(0xFF, 0xFF, 0xFF); BLACK = RGBColor(0x22, 0x22, 0x22)
FONT = "맑은 고딕"
prs = Presentation(); prs.slide_width = Inches(13.333); prs.slide_height = Inches(7.5)
BLANK = prs.slide_layouts[6]; _n = [0]


def _run(p, text, size, bold=False, color=BLACK, italic=False):
    r = p.add_run(); r.text = text; f = r.font
    f.name = FONT; f.size = Pt(size); f.bold = bold; f.italic = italic; f.color.rgb = color
    return r


def _rich(p, text, size, color=BLACK, bold_all=False):
    for part in re.split(r"(\*\*[^*]+\*\*)", text):
        if not part: continue
        if part.startswith("**") and part.endswith("**"):
            _run(p, part[2:-2], size, bold=True, color=color)
        else:
            _run(p, part, size, bold=bold_all, color=color)


def slide(title, subtitle=None):
    _n[0] += 1
    s = prs.slides.add_slide(BLANK)
    tb = s.shapes.add_textbox(Inches(0.5), Inches(0.28), Inches(12.3), Inches(0.6))
    tf = tb.text_frame; tf.word_wrap = True
    _run(tf.paragraphs[0], title, 24, bold=True, color=NAVY)
    if subtitle:
        _run(tf.add_paragraph(), subtitle, 12, color=GREY)
    ln = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.5), Inches(0.98), Inches(12.3), Pt(2.5))
    ln.fill.solid(); ln.fill.fore_color.rgb = NAVY; ln.line.fill.background()
    ft = s.shapes.add_textbox(Inches(0.5), Inches(7.05), Inches(12.3), Inches(0.3))
    p = ft.text_frame.paragraphs[0]
    _run(p, f"DFN 블록 판정 공유자료 (handoffv2/example_io)   |   2026-09-09   |   {_n[0]}", 9, color=GREY)
    p.alignment = PP_ALIGN.RIGHT
    return s


def text(s, x, y, w, h, items, size=13, spacing=4):
    tb = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame; tf.word_wrap = True; first = True
    for it in items:
        lvl, t = (it if isinstance(it, tuple) else (0, it))
        p = tf.paragraphs[0] if first else tf.add_paragraph(); first = False
        p.space_after = Pt(spacing)
        if lvl == -1: _rich(p, t, size + 1, color=NAVY, bold_all=True)
        elif lvl == 0: _rich(p, "•  " + t, size)
        elif lvl == 2: _rich(p, t, size, color=GREY)
        else: _rich(p, "      –  " + t, size - 1, color=GREY)
    return tb


def note(s, txt): s.notes_slide.notes_text_frame.text = txt


def table(s, x, y, w, rows, col_w=None, size=11, hdr=True, row_h=0.32, fills=None):
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
                _rich(p, str(val), size); c.fill.solid()
                c.fill.fore_color.rgb = (fills[i][j] if fills and fills[i] and fills[i][j] else (LIGHT if i % 2 == 0 else WHITE))
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


def box(s, x, y, w, h, title, body, fill=LIGHT, title_color=NAVY, size=11):
    shp = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    shp.adjustments[0] = 0.06; shp.fill.solid(); shp.fill.fore_color.rgb = fill; shp.line.color.rgb = NAVY; shp.line.width = Pt(1)
    tf = shp.text_frame; tf.word_wrap = True; tf.margin_left = tf.margin_right = Inches(0.1); tf.margin_top = Inches(0.06)
    tf.vertical_anchor = MSO_ANCHOR.TOP
    p = tf.paragraphs[0]; p.alignment = PP_ALIGN.LEFT; _run(p, title, size + 1, bold=True, color=title_color)
    for line in body:
        q = tf.add_paragraph(); q.alignment = PP_ALIGN.LEFT; q.space_after = Pt(2); _rich(q, line, size)
    return shp


def arrow(s, x1, y1, x2, y2):
    c = s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    c.line.color.rgb = NAVY; c.line.width = Pt(2)
    ln = c.line._get_or_add_ln(); tail = etree.SubElement(ln, qn("a:tailEnd")); tail.set("type", "triangle")
    return c


# ============================================================ 1 표지
s = prs.slides.add_slide(BLANK); _n[0] += 1
bg = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, Inches(7.5)); bg.fill.solid(); bg.fill.fore_color.rgb = NAVY; bg.line.fill.background()
tb = s.shapes.add_textbox(Inches(0.9), Inches(2.0), Inches(11.5), Inches(3.5)); tf = tb.text_frame; tf.word_wrap = True
_run(tf.paragraphs[0], "DFN 블록 판정 결과 공유", 40, bold=True, color=WHITE)
p = tf.add_paragraph(); _run(p, "ParaView 뷰어 소개  ·  결과를 만든 과정  ·  블록 판별 알고리즘(복셀 edge-cut + 다면체 재구성 + 몬테카를로 확률장)", 16, color=RGBColor(0xC9, 0xD6, 0xE8))
p = tf.add_paragraph(); p.space_before = Pt(30); _run(p, "기준 데이터: handoffv2/example_io (DFM 실측 12면 데모, 1~6면·슬라이딩 윈도우 시나리오)", 14, color=RGBColor(0xC9, 0xD6, 0xE8))
p = tf.add_paragraph(); _run(p, "2026-09-09  |  서울대 에너지자원공학과 이창무", 14, color=RGBColor(0xC9, 0xD6, 0xE8))
note(s, "뷰어는 ParaView 6.1.1 기준. 모든 bat는 example_io/paraview/ 에서 더블클릭으로 실행.")

# ============================================================ 2 뷰어 목록
s = slide("1. 뷰어 목록 — example_io/paraview/ 의 상태 파일(pvsm)과 실행 bat", "bat 더블클릭 → ParaView 6.1.1 이 상태 파일을 열어 아래 장면을 그대로 복원 (경로는 절대경로로 저장됨)")
table(s, 0.5, 1.15, 12.3, [
    ["실행 bat", "상태 파일", "보여주는 것", "데이터 근거"],
    ["열기_블록뷰.bat", "dfn_blockview.pvsm", "12면 데모(막장면 x=22.2): 기굴착 벽면 블록 84 + 막장면 쐐기 7 + 전방 내부 닫힌 블록 3,017개(72.7 m³)", "dfn_domain_x22.204-32.204_halo5_hz7.48.json (seed 2026)"],
    ["열기_블록뷰_6면.bat", "dfn_blockview_f06.pvsm", "1~6면 시나리오(막장면 x=11.37): 벽면 145 + 쐐기 6 + 내부 2,067개(51.9 m³), 최소 8복셀", "f06/dfn_domain_f06_forward·excavated.json"],
    ["열기_블록뷰_6면_고정부피.bat", "dfn_blockview_f06_minvol05.pvsm", "같은 장면에 **최소 부피 0.05 m³** 적용: 내부 171개(19.2 m³), 벽면 11개 — 미세조각 제거 비교용", "blocks_edgecut/f06_minvol05/"],
    ["열기_블록확률.bat / _6면 / _전체 / _제자리", "dfn_block_probability*.pvsm", "몬테카를로 50 실현 **블록 존재확률장** 4종: 12면 전방 / 6면 전방 / 전체 도메인 / 제자리(미굴착)", "blocks_edgecut/*.vti (p_block, p_smooth)"],
    ["(bat 없음)", "dfn_block_probability_insitu_lmin05.pvsm", "제자리 확률장 **검출하한 통일 재보정판(lmin 0.5)** — 현행 대표", "f06/block_prob_insitu_lmin05_mc50.vti"],
    ["열기_슬라이딩윈도우.bat", "dfn_sliding_windows.pvsm", "1-4→5 / 2-5→6 / 3-6→7 예측 뷰(레이어 그룹 토글): 관측 원판·확률장·대상면 실측 절리선", "f_slide/w14, w25, w36 (_ref 프레임)"],
    ["열기_윈도우_1-6면.bat / 1-7면.bat", "dfn_window_f01-06 / f01-07.pvsm", "누적 윈도우 → 다음 면 예측 뷰 (같은 포맷)", "f_slide/w16, w17"],
    ["열기_다면매칭.bat", "dfn_multiface_matching.pvsm", "적응 max_sep 다면 매칭으로 복원된 원판(1~6면) — 단일면 vs 다면 원판 구분", "f_slide/w16_multiface"],
    ["(bat 없음)", "dfn_current_state / face_blocks / face_mc / matrix_blocks / full / top30 / top30_3dec / windows", "12면 현재상태 뷰, 막장면 쐐기(+MC), 닫힌 블록 전체 1,891개, 전체 DFN·상위 30·3DEC식 절단 62블록, 누적 윈도우 4종 전달본", "example_io/paraview/{full,top30,f06,windows}, blocks3dec/"],
], col_w=[2.3, 2.6, 4.9, 2.5], size=9, row_h=0.5)
note(s, "bat는 ASCII만(한글 rem 금지). pvsm은 절대경로를 담고 있어 폴더를 옮기면 pvsm 안 경로를 바꿔야 함(또는 ParaView의 데이터 위치 지정 대화상자 이용).")

# ============================================================ 3 실현 과정
s = slide("2. 결과를 만든 과정 — 실측 절리선에서 ParaView 장면까지", "handoffv2 파이프라인(1~3단계) + 블록 판정·시각화 후처리(4~6단계). 좌표: x = 굴진방향, 막장면 = x·const, z 상방, 단위 m")
steps = [
    ("① 입력 변환", "DFM Export 12면\n점군 절리선 6,447개\n→ 터널축 정렬, 전역 절리군\n재분류(30° 군집), 관측면적", "convert_dfm_export_to_trace_dataset.py"),
    ("② 역산", "절리군별 평균방향·κ,\n반지름 멱법칙 kr,\nP32 (검출하한 lmin 0.5)", "estimate_kr / build_dataset_config /\nestimate_p32"),
    ("③ 원판 복원", "절리선 → 원판 (다면 매칭:\n적응 max_sep, 평면 잔차 8 cm,\n면당 chord 1개)", "reconstruct_discs_from_traces.py"),
    ("④ 조건부 실현", "관측 원판 고정 + 확률 균열\n생성 → 부재 조건화 →\n도메인 절단 JSON (시드별)", "export_domain_dfn_json.py\n(generate_conditional_hidden_dfn)"),
    ("⑤ 블록 판정", "복셀 0.1 m edge-cut CCA →\n닫힌/벽접촉/쐐기 블록 →\n다면체 재구성, MC 확률장", "detect_blocks_from_domain_json /\nreconstruct_block_polyhedra /\ndetect_face_blocks / aggregate_*"),
    ("⑥ 시각화", "vtp/vti 내보내기 + 장면 기하\n(tunnel_behind, face_cap,\ndomain_box) → pvpython으로\npvsm 생성 → bat", "export_domain_dfn_vtk /\nmake_scene_geometry /\nmake_pvsm_*.py"),
]
x0, y0, bw, bh, gap = 0.5, 1.35, 1.85, 1.95, 0.24
cols = [RGBColor(0xE3, 0xEC, 0xF7), RGBColor(0xE3, 0xEC, 0xF7), RGBColor(0xE3, 0xEC, 0xF7), RGBColor(0xD6, 0xE9, 0xD6), RGBColor(0xFA, 0xE5, 0xD3), RGBColor(0xEE, 0xEE, 0xEE)]
for i, (t, body, scr) in enumerate(steps):
    x = x0 + i * (bw + gap)
    b = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y0), Inches(bw), Inches(bh))
    b.adjustments[0] = 0.07; b.fill.solid(); b.fill.fore_color.rgb = cols[i]; b.line.color.rgb = NAVY
    tf = b.text_frame; tf.word_wrap = True; tf.vertical_anchor = MSO_ANCHOR.TOP; tf.margin_left = tf.margin_right = Inches(0.06)
    p = tf.paragraphs[0]; p.alignment = PP_ALIGN.CENTER; _run(p, t, 12, bold=True, color=NAVY)
    q = tf.add_paragraph(); q.alignment = PP_ALIGN.CENTER; _run(q, body, 8.5, color=BLACK)
    tb = s.shapes.add_textbox(Inches(x - 0.05), Inches(y0 + bh + 0.03), Inches(bw + 0.1), Inches(0.6)); tb.text_frame.word_wrap = True
    pp = tb.text_frame.paragraphs[0]; pp.alignment = PP_ALIGN.CENTER; _run(pp, scr, 7.5, color=GREY, italic=True)
    if i < len(steps) - 1: arrow(s, x + bw + 0.02, y0 + bh / 2, x + bw + gap - 0.02, y0 + bh / 2)
text(s, 0.5, 4.15, 12.3, 2.9, [
    (-1, "각 단계의 산출물과 시간"),
    "①~③: 현장(윈도우)당 1회, 15초~1분. 산출: trace_dataset_3d.h5, kr_summary_by_set.csv, p32_summary.csv, reconstructed_discs.csv",
    "④: 실현(시드)당 4~12초. 산출: dfn_domain_*.json — 균열 목록(center_xyz_m, normal_xyz, radius_m, set_id, label observed/unobserved) + meta(좌표계·도메인·절리군 파라미터·시드)",
    "⑤: 실현당 8초(GPU, 균열 2.5만·복셀 500만). 산출: *_labels.npz(복셀 라벨), *_summary.csv, *_polyhedra.vtp, block_prob_*.vti(MC 집계)",
    "⑥: 장면 기하 vtp 3종 + 블록별 고정색(rgb 셀 배열, bake_block_rgb.py) + pvpython 스크립트로 레이어·색·카메라·범례를 코드로 고정해 pvsm 저장 → 누구나 같은 장면",
    "⑤~⑥의 스크립트는 handoffv2/dfn_analysis/(배포용)와 handoffv2/scripts/analysis_260903/(분석·MC·pvsm 생성기 아카이브, README 있음)에 있음",
], size=10.5)
note(s, "출처: handoffv2/README.md 파이프라인 실행 순서, scripts/analysis_260903/README.md, worklog 09-01~09-04.")

# ============================================================ 4 입력·시나리오
s = slide("3. 입력 데이터와 시나리오 — 12면 실측 데모, 1~6면, 슬라이딩 윈도우", "막장면 x 위치와 절리선 수(위), 관측면적(아래). 면 07(x=12.63)은 데이터 품질 이상 면(RGB 없음·정확도 낮음, 송재준 교수님 측 확인)")
image(s, os.path.join(H2FIG, "fig_face_layout.png"), 0.5, 1.2, w=6.3)
table(s, 7.0, 1.2, 5.8, [
    ["시나리오", "관측 면", "막장면 x / 전방 도메인", "용도"],
    ["12면 데모", "01~12", "22.20 / x 22.2~32.2 (halo 5, z +7.48)", "블록뷰·현재상태·매트릭스·top30"],
    ["1~6면", "01~06", "11.37 / x 11.4~21.4", "블록뷰 6면, 확률장 4종, 블라인드 검증(면 07~12)"],
    ["1~7면 (w17)", "01~07", "12.63 / 면 08 예측", "윈도우 뷰, 면07 결손 영향 진단"],
    ["슬라이딩 w14/w25/w36", "1-4 / 2-5 / 3-6", "면 5 / 6 / 7 예측", "실무형 4면 윈도우 검증"],
    ["누적 4종 (한양대 전달)", "1-7 / 2-7 / 3-7 / 4-7", "면 07 전방 동일 구간 × 시드 5", "막장 수에 따른 예측 품질 비교"],
], col_w=[1.5, 1.2, 1.9, 1.2], size=9, row_h=0.45)
text(s, 7.0, 4.0, 5.8, 3.0, [
    (-1, "도메인 정의 (export_domain_dfn_json)"),
    "도메인 = 마지막 막장면 전방 10 m × (터널 단면 bbox + halo 5 m, z는 +7.48 m) 직육면체",
    "포함 규칙: 원판의 경계구(중심, 반지름)가 도메인과 만나면 포함 — 밖에 중심이 있어도 도메인을 자르면 들어감",
    "확률 균열: 절리군별 kr·P32로 rmin 0.5~rmax 25 m 생성 후 **부재 조건화** — 관측 막장면에 0.5 m 이상 절리선을 남겼을 균열은 제거(그 자리는 관측 원판이 설명)",
    "예: 12면 도메인 균열 24,802개(관측 복원 포함), 6면 도메인 12,535개(관측 434)",
], size=10)
note(s, "그림: handoffv2/figures/fig_face_layout.png. 면 x: -0.20, 1.63, 4.07, 6.16, 8.50, 11.31, 12.63, 15.43, 16.91, 18.23, 20.93, 22.20.")

# ============================================================ 5 역산·복원·조건부
s = slide("4. 역산 → 원판 복원 → 조건부 실현 (블록 판정의 입력이 만들어지는 곳)", "관측된 것은 그대로(결정론), 관측되지 않은 곳만 통계로(확률) — 두 부류를 label로 구분해 JSON에 담음")
image(s, os.path.join(SC, "crop_reconstruction_3d.png"), 0.5, 1.15, h=3.4)
caption(s, 0.5, 4.6, 4.3, "12면 복원 원판 4,601개(회색 shrinkage·초록 원적합) vs 관측 절리선(파랑)")
image(s, os.path.join(SC, "crop_observed_vs_conditioned_traces.png"), 5.0, 1.15, h=2.55)
caption(s, 5.0, 3.72, 4.9, "관측 절리선(파랑) vs 조건부 DFN이 만드는 절리선(빨강) — 관측면에서 정합")
table(s, 10.35, 1.15, 2.5, [
    ["절리군", "κ", "kr", "P32"],
    ["1", "13.1", "3.95", "2.01"],
    ["2", "15.0", "3.95", "1.50"],
    ["3", "11.0", "3.85", "1.45"],
    ["4", "12.5", "3.85", "1.19"],
], col_w=[0.6, 0.6, 0.6, 0.7], size=9, row_h=0.3)
caption(s, 10.35, 2.7, 2.5, "12면 데모 추정치 (P32: r≥0.5 m, m²/m³)")
text(s, 5.0, 4.15, 7.8, 2.9, [
    (-1, "복원(③)에서 내가 정한 것"),
    "절리선 매칭 = 같은 절리군·법선 축각 ≤15°·공면거리 ≤0.15 m·중심거리 게이트 → 병합마다 평면 잔차(RMS ≤ 8 cm)와 '면당 chord 1개' 재검사",
    "중심거리 게이트는 고정 4.5 m 대신 **적응형**(Δx/sinθ × 1.4, 인접면만): 막장면과 평행한 원판이 덜 매칭되는 방향 편향을 줄임 (슬라이드 16)",
    "반지름: 절리선 2개 이상이면 원적합(determined), 아니면 kr 분포 기반 축소추정(shrinkage)",
    (-1, "조건부 실현(④)에서 내가 정한 것"),
    "6면 재보정(lmin 0.5): kr 4.0/3.8/3.65, P32 1.39/0.96/0.74 — 검출하한을 관측·모델에 똑같이 적용해야 면 08 블라인드 예측이 밴드 안에 들어옴(+56% 과대 해소)",
    "부재 조건화의 파이썬 이중루프(4분 20초)를 면별 벡터화로 4초로 — 같은 시드에서 출력 동일 검증",
], size=10)
note(s, "출처: example_io/dataset_config.json, p32_summary.csv, kr_summary_by_set.csv, demo_output/win_f01-06_l05/p32. 그림: handoffv2/figures.")

# ============================================================ 6 알고리즘 ① 선택
s = slide("5. 블록 판별 알고리즘 ① — 세 가지 방식을 비교하고 edge-cut을 택한 이유", "출발점은 레거시 GPU 복셀 판정기(_archive/block_detection). 그 위에 edge-cut 절단 · 다면체 재구성 · 막장면 쐐기 · MC 확률장을 내가 추가")
table(s, 0.5, 1.15, 12.3, [
    ["", "A. 종전: 균열 복셀 (레거시)", "B. edge-cut CCA (채택)", "C. 3DEC식 완전 절단 (비교용 프로토타입)"],
    ["균열 표현", "원판 근방 복셀을 FRACTURE로 분류(두께 = 0.6×복셀), 암반에서 제거", "두께 0의 분리면 — 원판을 가로지르는 **이웃 복셀 연결선만 절단**", "무한 평면으로 블록 다면체를 완전 분할(원판이 닿은 블록만)"],
    ["블록", "남은 ROCK 복셀의 연결성분 (26/6-이웃)", "절단되지 않은 6-이웃 그래프의 연결성분", "반공간 교집합 볼록 다면체"],
    ["장점", "구현 단순, GPU 빠름", "암반 부피 손실 없음, 두께 설정 불필요, 원판 끝(crack tip)이 자연스럽게 열려 있음", "기하 정확, 해상도 무관, 부피 보존 0.000%"],
    ["문제", "두께 < 복셀이면 균열면에 구멍 → 블록 못 닫음(0.1 격자+6 cm 두께에서 블록 3개). 복셀 0.2에서 percolation 이상치(59 m³)", "닫힘 판정 스케일 ε = 복셀 크기에 의존 (0.1 m로 고정·명시). 계단형 표면 → 다면체 재구성으로 해결", "부분 관통 균열을 **과절단**(작은 원판이 셀 모서리만 스쳐도 셀 전체 절단). 비용 초선형: 균열 500개 219 s, 700개 timeout"],
    ["결과 예 (12면 top-30)", "—", "닫힌 블록 **0개** (소형 균열이 없으면 블록이 안 닫힘)", "블록 **62개**, 벽 걸침 20 — 정반대 결과 = 부분 균열 취급 차이"],
], col_w=[1.5, 3.4, 3.7, 3.7], size=9.5, row_h=0.75)
text(s, 0.5, 5.75, 12.3, 1.3, [
    "판단: 이 데이터는 **수천~수만 개의 소형 유한 원판**(반지름 중앙값 0.6 m)이라 '전체 균열 다면체 절단'은 실용 불가. 복셀 edge-cut은 기하 하한(보수적), 3DEC식은 상한 — 역할 분담으로 두 방식을 모두 유지",
    "edge-cut의 ε 의존은 없앨 수 없는 성질이므로 '복셀 0.1 m 기준 블록'으로 명시하고 **절대 블록 수가 아니라 상대 비교·확률**로 쓰는 것을 원칙으로 함",
], size=10.5)
note(s, "출처: worklog 09-01(복셀 해상도 스윕, edge-cut 도입), cut_blocks_polyhedral.py docstring, 260904.pptx 정리(메쉬 절단 비용).")

# ============================================================ 7 알고리즘 ② edge-cut 원리
s = slide("6. 블록 판별 알고리즘 ② — edge-cut 원리와 구현", "compute_edge_cuts → run_cca_edgecut (dfn_analysis/detect_blocks_from_domain_json.py)")
image(s, os.path.join(SC, "fig_edgecut_concept.png"), 0.5, 1.2, w=8.3)
text(s, 8.9, 1.15, 3.9, 5.9, [
    (-1, "절단 판정 (원판마다, AABB 안 복셀만)"),
    "복셀 중심 p₁, p₂의 평면 부호거리 d₁ = (p₁−c)·n, d₂ = (p₂−c)·n",
    "부호가 다르면 연결선이 평면을 가로지름. 교차점 q = p₁ + t(p₂−p₁), t = d₁/(d₁−d₂)",
    "**|q − c|² ≤ r²** 일 때만 절단 (교차점이 평면 위이므로 3D 거리 = 면내 거리) — 유한 원판 처리의 핵심",
    "축별(x, y, z) 절단 배열 cutx/cuty/cutz(bool)에 OR 누적 → 원판 2.5만 개, 복셀 500만 개에서 8초",
    (-1, "연결성분"),
    "살아있는 6-이웃 연결선으로 희소 그래프(scipy coo_matrix) → connected_components → 복셀 라벨 3D 배열",
    (-1, "왜 6-이웃인가"),
    "26-이웃이면 모서리·꼭짓점 접촉으로 분리면을 '건너뛰어' 블록이 합쳐질 수 있음 → 보수적 6-이웃 고정",
], size=10)
text(s, 0.5, 5.75, 8.2, 1.3, [
    "(a) 종전 방식은 균열 복셀을 암반에서 빼므로 부피가 줄고, 두께가 복셀보다 얇으면 균열면에 구멍이 남. (b) edge-cut은 복셀을 빼지 않고 연결선만 끊음. (c) 원판 밖(crack tip 너머)은 연결이 유지되므로 유한 균열 하나로는 블록이 닫히지 않고, **여러 균열이 협력해야 닫힘** — 실제 블록 형성 논리와 일치",
], size=10)
note(s, "그림: 2D 개념도(scratchpad hdec/fig_edgecut_concept.png). 코드: compute_edge_cuts(), run_cca_edgecut().")

# ============================================================ 8 알고리즘 ③ 판정 규칙
s = slide("7. 블록 판별 알고리즘 ③ — 연결성분에서 '블록'을 고르는 규칙 3종", "같은 edge-cut 라벨에서 문제 설정(굴착 전/후, 자유면)에 따라 다른 규칙을 적용")
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
note(s, "출처: detect_blocks_from_domain_json.py filter_blocks_edgecut(), detect_face_blocks.py, scripts/analysis_260903/detect_interior_labels.py.")

# ============================================================ 9 알고리즘 ④ 다면체
s = slide("8. 블록 판별 알고리즘 ④ — 계단형 복셀 블록을 정확한 다면체로 되돌리기", "reconstruct_block_polyhedra.py: edge-cut이 찾은 '진짜 닫힌 블록'의 경계 평면을 역추적해 볼록 다면체 복원")
image(s, os.path.join(SC, "fig_polyhedra_concept.png"), 0.5, 1.15, w=8.2)
text(s, 8.9, 1.15, 3.9, 5.9, [
    (-1, "경계 원판 판별 (블록 복셀 중심 집합 P, 후보 원판 j)"),
    "부호거리 dᵢⱼ = (pᵢ − cⱼ)·nⱼ 에서 **95 % 이상이 한쪽**(min(양,음 비율) ≤ 0.05) — 블록을 관통하는 슬릿은 제외",
    "**min|dᵢⱼ| ≤ 1.2h** (경계에 실제로 접함, h = 복셀)",
    "근접 복셀의 면내 거리² ≤ (rⱼ + h)² (접촉부가 원판 반경 안)",
    (-1, "클리핑 순서"),
    "복셀 bbox(+0.75h) 볼록체 → ① 경계 원판 반공간 → ② 인접 터널 벽 변 평면(암반 쪽 유지) → ③ 도메인 6면. 각 단계는 hull 모서리-평면 교점으로 ConvexHull 재구성(clip_hull)",
    (-1, "검증"),
    "R_V = V_poly / V_voxel. **R_V > 2 → flag 'open'**(경계면 누락 의심, 표시만 하고 제외 안 함)",
    "12면 터널접촉 168/168 성공, 부피비 중앙값 **0.90**; 닫힌 블록 전체 1,891 중 1,880 ok; 6면 고정부피 171/174",
], size=10)
text(s, 0.5, 5.75, 8.2, 1.3, [
    "한계: 볼록 재구성이라 블록 안으로 파고들다 끝나는 슬릿과 비볼록 형상은 외곽 형상으로 근사. 터널 벽은 폴리곤 변 평면으로 국소 근사. 큰 쐐기(한양대 top-20 케이스)는 연장 평면 과절단으로 부피 과소 → 그 경우 복셀 부피(vol_voxel)를 기준값으로 보고",
], size=10)
note(s, "그림: fig_polyhedra_concept.png(2D 개념도). 수치: blocks_edgecut/poly_tunnel_summary.csv(168), poly_all_summary.csv(1,891/1,880), f06_minvol05/unexcavated_interior_summary.csv(174/171).")

# ============================================================ 10 알고리즘 ⑤ 검증·한계
s = slide("9. 블록 판별 알고리즘 ⑤ — 검증 실험과 한계 (내가 확인한 것)", "블록 통계는 파라미터에 민감하다 — 어떤 값이 물리적이고 어떤 값이 이산화 아티팩트인지 실험으로 분리")
table(s, 0.5, 1.15, 7.6, [
    ["실험", "결과", "함의"],
    ["복셀 해상도 스윕 0.3 → 0.05 m (12면)", "블록 수 29 → 254, 총부피 22.9 → 3.7 m³ 단조 감소. 종전 두께 방식은 0.2에서 percolation 이상치 59 m³", "복셀 = 닫힘 판정 스케일 ε. 절대값 보고 금지, 0.1 m 고정 상대 비교"],
    ["최소 크기: 8복셀 → 고정 0.05 m³ (6면)", "내부 2,067개/51.9 m³ → 171개/19.2 m³, 벽면 145 → 11", "미세조각 제거. 단 0.2 m 격자에 같은 하한을 주면 1,004개 → 분할 자체의 ε 의존은 잔존"],
    ["상위 30개 대형 균열만 (12면)", "edge-cut 블록 **0개** vs 3DEC식 완전 절단 62개", "블록을 닫는 것은 소형 균열의 교차. top-N 선별은 블록 관점에서 위험"],
    ["rmax 25 → 10 m", "DFN 통계 동일(19,617→19,594)이나 블록 총부피 −49 %, 최대 블록 −81 %", "크기 절단은 블록 결과를 바꾸는 물리 파라미터"],
    ["시드 12개 반복", "블록 수 CV 11 %, 총부피 CV 21 %, 최대 블록 CV 48 %, 개별 블록 위치 일치 8~9 %", "재현되는 것은 통계량뿐 → 확률장으로 보고"],
    ["관측 균열만 vs 전체 DFN (막장면 쐐기)", "동일 7개(0.055 m³) — 부재 조건화가 막장면 근방 확률 균열을 지운 결과(구성상 동어반복 일부)", "검출한계 이하 잔여 불확실성은 MC로 정량화(추가 쐐기 0~5개)"],
], col_w=[2.2, 3.0, 2.4], size=9, row_h=0.7)
image(s, pv("dfn_top30_3dec"), 8.3, 1.15, w=4.5)
caption(s, 8.3, 3.75, 4.5, "dfn_top30_3dec.pvsm — 상위 30개 무한평면 완전 절단(3DEC식) 62블록. 같은 입력의 복셀 edge-cut은 0개")
text(s, 8.3, 4.3, 4.5, 2.7, [
    (-1, "결론으로 삼은 원칙"),
    "복셀 0.1 m·6-이웃·edge-cut 고정, 최소 크기는 용도별 명시",
    "블록 개수·부피는 **실현 분포와 확률장**으로 보고, 대표 실현은 다면체로 시각화",
    "역학해석 입력은 '블록 + 경계평면 목록'(다면체 재구성 부산물)으로 넘길 수 있게 유지",
], size=10)
note(s, "출처: docs/블록판정_시드민감도_실험보고_260901.md, worklog 09-01~09-04 §10.")

# ============================================================ 11 뷰어: 블록뷰 6면
s = slide("10. 뷰어 — 블록뷰(1~6면, 고정 부피 0.05 m³)", "열기_블록뷰_6면_고정부피.bat  |  굴착 구간과 생성 도메인을 겹치지 않게 배치한 '현재 상태' 구성")
image(s, pv("dfn_blockview_f06_minvol05"), 0.5, 1.15, w=8.0)
text(s, 8.7, 1.15, 4.1, 5.9, [
    (-1, "장면 구성 (x 기준)"),
    "x < 11.37 **기굴착**: 반투명 회색 터널(tunnel_behind) + 벽면 접촉 블록(블록별 고정색 rgb) — 굴착 구간 JSON(--x-range)에 규칙 ① 적용",
    "x = 11.37 **막장면**: 베이지 face_cap + 관측 균열 기반 쐐기(빨강, 규칙 ③)",
    "x > 11.37 **미굴착 전방**: 청회색 내부 닫힌 블록(규칙 ②) + 검정 도메인 박스",
    (-1, "수치"),
    "내부 닫힌 블록 171개 / 19.2 m³ (최대 2.10, >0.1 m³ 54개, >0.5 m³ 4개), 벽면 11개 / 0.6 m³, 쐐기 6개 / 0.025 m³",
    "같은 장면 min 8복셀 판(열기_블록뷰_6면.bat): 내부 2,067개 / 51.9 m³, 벽면 145 — 나란히 열어 비교",
    (-1, "만든 방법"),
    "make_scene_geometry.py(터널·캡·박스 vtp) → bake_block_rgb.py(tab20 고정색) → make_pvsm_blockview_f06_minvol05.py(pvpython: 레이어·색·불투명도·카메라 코드로 고정)",
], size=10)
note(s, "렌더: pvpython 오프스크린(흰 배경). GUI에서 열면 저장된 카메라 그대로.")

# ============================================================ 12 뷰어: 현재상태 12면 + 매트릭스
s = slide("11. 뷰어 — 12면 현재상태 뷰와 닫힌 블록 전체(매트릭스) 뷰", "dfn_current_state.pvsm (열기_블록뷰.bat 계열)  |  dfn_matrix_blocks.pvsm")
image(s, pv("dfn_current_state"), 0.5, 1.15, w=6.1)
caption(s, 0.5, 4.6, 6.1, "현재상태 뷰: 기굴착 터널(막장면 뒤 8 m, 벽면 블록 84개 색) + 막장면 쐐기 7 + 전방 미굴착 내부 블록(2,144개)")
image(s, pv("dfn_matrix_blocks"), 6.7, 1.15, w=6.1)
caption(s, 6.7, 4.6, 6.1, "매트릭스 뷰: 도메인 전체 균열 원판(반투명) + 완전히 둘러싸인 블록 1,891개 다면체(터널접촉 168 + 내부 1,723)")
text(s, 0.5, 5.05, 12.3, 2.0, [
    "**사용자 지적으로 고친 표현**: 처음엔 전방 도메인(막장면 +10 m)에 미래 터널을 미리 뚫어 보여줬음 → 미굴착 꽉 찬 암반 + 막장면 벽 + 기굴착 터널(뒤 8 m)로 재구성. 도메인 x 범위(22.204~32.204)는 처음부터 정확했고 문제는 표현이었음",
    "닫힌 블록 1,891개 합 48.5 m³ ≈ 도메인 암반의 1.5 % — 완전 절단 관례(3DEC)와 달리 유한 원판 액면으로는 '소수 닫힌 블록 + 연결된 모암 1개'가 정상",
    "매트릭스 뷰는 blocks_polyhedra 층이 기본 표시, 복셀층(vti threshold)은 기본 숨김 — Pipeline Browser 눈 아이콘으로 전환",
], size=10)
note(s, "출처: worklog 09-01/09-02 현재상태 뷰·하이브리드 재구성 절, blocks_edgecut/poly_all_summary.csv.")

# ============================================================ 13 뷰어: 막장면 쐐기
s = slide("12. 뷰어 — 막장면 쐐기 판정과 검출한계 이하 불확실성", "dfn_face_blocks.pvsm (관측 균열만)  |  dfn_face_mc.pvsm (12 실현 MC)")
image(s, pv("dfn_face_blocks"), 0.5, 1.15, w=6.1)
caption(s, 0.5, 4.6, 6.1, "막장면(x=22.2) 자유면 패치에 접한 쐐기 7개(0.055 m³, 최대 0.02) — 전부 관측 절리선에서 복원한 원판이 만든 것")
image(s, pv("dfn_face_mc"), 6.7, 1.15, w=6.1)
caption(s, 6.7, 4.6, 6.1, "MC 12 실현: 확률 균열 추가로 생기는 쐐기 0~5개(평균 1.2), 부피 +14 % 평균, 최대 37 L(seed 808)")
text(s, 0.5, 5.05, 12.3, 2.0, [
    "규칙 ③(슬라이드 7)로 판정. 관측 균열만 넣은 결과와 전체 DFN 결과가 **동일** — 부재 조건화가 막장면 관측창에 0.5 m 이상 절리선을 남길 확률 균열을 지우므로 구성상 당연한 부분이 있음(순환성, 사용자 지적으로 확정)",
    "그래서 '검출한계 이상 = 실측으로 결정(조건화 보장), 이하 = 통계 불확실성 잔존'으로 서술하고, 이하 부분을 MC로 따로 정량화",
    "ε = 0.05 m 사용(관측 원판이 소형이라 0.1에서는 닫힘이 거칠어짐). 다면체 재구성 7/7",
], size=10)
note(s, "출처: worklog 09-01 막장면 쐐기 판정 절. 6면 시나리오는 6개/0.025 m³.")

# ============================================================ 14 확률 컨투어
s = slide("13. 블록 존재확률 컨투어 — 몬테카를로 50 실현을 복셀 단위로 겹치기", "열기_블록확률_*.bat 4종 + 재보정판. P(블록) = 그 복셀이 닫힌 블록에 속한 실현 수 / 50")
image(s, pv("dfn_block_probability_insitu_lmin05"), 0.5, 1.15, w=6.1)
caption(s, 0.5, 4.6, 6.1, "제자리 확률장(6면, lmin 0.5 재보정): 막장면 직후 천장부에 P≥0.5 등치면(빨강). 그 너머는 소멸")
image(s, pv("dfn_block_probability_full"), 6.7, 1.15, w=6.1)
caption(s, 6.7, 4.6, 6.1, "전체 도메인 굴착가정 확률장: 기굴착 구간(관측 벽면 블록) + 전방(막장면 원판 기반 천장 블록)")
text(s, 0.5, 5.05, 12.3, 2.0, [
    "**절차**: 시드 k×101 (k=1..50) → export_domain_dfn_json(실현 4~12초) → 라벨만 저장(detect_interior_labels / edge-cut, 8초) → 8병렬(xargs -P 8) 완주 2~4분 → aggregate_*.py: 복셀별 소속 횟수/50 = p_block, 가우시안 σ=3 평활 p_smooth → vti + 2D 투영 png + pvsm(등치면 0.02/0.1/0.3/0.5, 흰→노랑→빨강 커스텀 LUT)",
    "**수치(제자리, 6면, 재보정)**: 실현별 265~402개·6.1~11.3 m³, ≥1회 5.0 %, P≥0.5 1.79 m³, P=1.0 396복셀 — 관측 원판 기반 천장 클러스터는 준결정론적, 배경은 얇은 구름. 관측 정보 도달거리 ≈ 1.5 m (블라인드 검증: 막장면 1.5 m 너머 위치 예측력 없음, 절리선 통계는 밴드 안)",
    "범례 서식 주의: ParaView 6.1은 '{:.2f}' 브레이스 스타일만 유효. 다면 매칭 개선(9/4) 후 재생성본은 P≥0.5 ≈1.05 m³로 재분포",
], size=9.5)
note(s, "출처: scripts/analysis_260903/aggregate_block_prob_insitu_l05.py, mc_worker_insitu_l05.sh, worklog 09-02/09-03.")

# ============================================================ 15 슬라이딩 윈도우
s = slide("14. 뷰어 — 슬라이딩 윈도우 (1-4→5, 2-5→6, 3-6→7)", "열기_슬라이딩윈도우.bat  |  한 상태 파일에 3세트를 레이어 그룹(w14_/w25_/w36_)으로 담고 눈 아이콘으로 전환 (기본 w14)")
image(s, pv("dfn_sliding_windows"), 0.5, 1.15, w=7.6)
text(s, 8.3, 1.15, 4.5, 5.9, [
    (-1, "레이어 (각 세트 동일)"),
    "faceplanes(실측면 4 + 대상면 1), tunnel_behind(실측면 구간), tunnel_to_target(유령), face_cap",
    "observed_discs_single(단일면 원판) / multi(다면 원판, 강조색), domain_box",
    "block_prob_mc50(제자리 확률장 등치면), **target_face_traces(파란 선 = 대상면 실측 절리선)** — 예측과 실측을 같은 화면에서",
    (-1, "프레임"),
    "세 창의 국소 좌표가 조금씩 달라(면 중심 궤적 재적합) 공통 참조 프레임(_ref)으로 회전·정렬. 파일 규약: 국소 / _world / _ref 3종",
    (-1, "검증 수치 (적응 매칭 재생성본)"),
    "다면 원판 w14 197 / w25 219 / w36 169. 대상면 절리선 예측 vs 실측: 122±12 vs 169, 129±11 vs 168, 105±9 vs 73(면07 결손)",
    "4면 윈도우는 P21 −18~−27 % 하향(대상면 5·6이 고강도 면 + 4면 추정 분산). 6면→면08의 −9 %보다 큼",
], size=9.5)
note(s, "출처: worklog 09-03 슬라이딩 윈도우 절, 09-04 §7-5. 생성: make_pvsm_slide.py, slide_worker.sh, aggregate_slide.py.")

# ============================================================ 16 윈도우 1-6 / 1-7
s = slide("15. 뷰어 — 누적 윈도우 1~6면 / 1~7면 → 다음 면 예측", "열기_윈도우_1-6면.bat / 1-7면.bat  |  같은 포맷. 1~7면은 면 07 결손이 결과에 어떻게 번지는지 보여주는 사례")
image(s, pv("dfn_window_f01-06"), 0.5, 1.15, w=6.1)
caption(s, 0.5, 4.6, 6.1, "w16: 관측 원판 434, 확률장 max P 1.0 — 막장면 직후 천장 클러스터 반복 출현")
image(s, pv("dfn_window_f01-07"), 6.7, 1.15, w=6.1)
caption(s, 6.7, 4.6, 6.1, "w17: 면 07(절리선 79개) 포함. 관측 원판 108개뿐 → 앵커 부재로 확률장이 흐림")
text(s, 0.5, 5.05, 12.3, 2.0, [
    "**w17 부실 원인 분리**: ① 면 07 검출 결손(0.5 m 미만 절리선 6개, 타면 400~700) → 관측 원판 108(w16 434) → max P 0.30~0.44, 데이터 문제라 재보정으로 안 고쳐짐. ② 면 08 +51 % 과대예측은 별개 원인 = 구버전 P32 보정(observed_P21을 전체 길이로 계산) → lmin 0.5 재보정으로 P32 합 5.97→3.56, 면 08 122±10 vs 139(96백분위)로 복귀",
    "적응 매칭 재생성 후: w16 → 면 07 예측 122±10 vs 실측 73(실측 결손), w17 → 면 08 125±11 vs 139(90백분위, 밴드 안), max P 0.92",
    "한양대 쐐기 전달본(260903)이 구버전 win_f01-07 기반(강도 약 1.7배 과생성)이라 재전달 여부 판단 필요",
], size=9.5)
note(s, "출처: worklog 09-04 §2~§5, §9.")

# ============================================================ 17 다면 매칭
s = slide("16. 뷰어 — 다면 매칭(적응 max_sep)으로 복원된 원판", "열기_다면매칭.bat  |  막장면과 평행한 절리일수록 두 면 사이 현이 멀리 떨어져 나타난다는 사용자 통찰을 게이트에 반영")
image(s, pv("dfn_multiface_matching"), 0.5, 1.15, w=6.4)
image(s, os.path.join(FIG, "adaptive_sep_direction_bias.png"), 7.1, 1.15, w=5.7)
text(s, 0.5, 4.85, 12.3, 2.2, [
    "이론: 두 현 중심거리 = Δx / sinθ = Δx / √(1−nx²) (θ = 원판 법선과 x축 각). nx→1(막장면 평행)에서 발산 → 고정 4.5 m 게이트는 막장면 평행 원판을 과소 매칭",
    "구현: 게이트 = clamp(1.4 · Δface_x / sinθ, same_face, 8.0), 단 Δface_x > 3.2 m(비인접)면 후보에서 제외. 병합 확정은 여전히 평면 잔차 8 cm + 면당 chord 1개 검증",
    "1~6면 결과: 다면 원판 280 → **308**, 비인접 과병합 **0** 유지, 면 5-6 매칭 41 → 60, 막장면 평행(|nx|≥0.8) 36 → 58. 막장면 평행 비율: 모집단 26.9 % / 고정 12.9 % / 적응 **18.8 %** (모집단에 근접)",
    "이후 모든 윈도우 파이프라인 기본값으로 승격(run_window_pipeline.py --adaptive-sep). 확률장은 앵커 원판이 늘며 재분포(P≥0.5 부피 1.37→1.05 m³) — 개선 단정이 아니라 트레이드오프로 기록",
], size=9.5)
note(s, "출처: worklog 09-04 §7-3~§7-5, §9. 그림: docs/figures/adaptive_sep_direction_bias.png.")

# ============================================================ 18 사용법·재현
s = slide("17. 뷰어 사용법 · 파일 규약 · 재현 방법", "다른 PC에서 열려면 ParaView 6.1.1 설치 후 pvsm 안의 절대경로만 바꾸면 됨")
text(s, 0.5, 1.15, 6.0, 5.9, [
    (-1, "사용법"),
    "bat 더블클릭 (또는 ParaView → File → Load State → pvsm). 데이터 위치를 묻는 대화상자가 뜨면 'Search files under specified directory'로 example_io 지정",
    "Pipeline Browser의 **눈 아이콘**으로 레이어 토글 — 슬라이딩 윈도우는 w14_/w25_/w36_ 접두사 그룹 단위로 전환",
    "확률장: Contour 필터의 Isosurfaces 값(0.02/0.1/0.3/0.5)을 바꾸면 등치면 재생성. 색 범위는 0~0.5 고정",
    "블록 다면체: Coloring = rgb(고정색) 또는 block_id / tunnel_contact / flag_open 셀 배열로 전환 가능",
    (-1, "파일 규약"),
    "국소 = 파이프라인 프레임(x 굴진). _world = 라이다 world 좌표(JSON meta의 R·t로 역변환). _ref = 여러 창을 겹치기 위한 공통 참조 프레임",
    "vtp = 표면/선(원판·블록·터널·절리선), vti = 정규 격자 스칼라(확률장·라벨), vts = 회전된 격자(확률장 _ref/_world)",
    "bat는 ASCII만(한글 rem 금지: cp949 파싱 깨짐)",
], size=10)
text(s, 6.8, 1.15, 6.0, 5.9, [
    (-1, "재현 명령 (handoffv2 루트, PYTHONPATH=.)"),
    (2, "python scripts/analysis_260903/run_window_pipeline.py f01-06 01 02 03 04 05 06"),
    (2, "python -m dfn_analysis.export_domain_dfn_json --pipeline-dir demo_output/win_f01-06_l05 --sets 1 2 3 --rmax-local 25 --lmin-det 0.5 --seed 2026 --halo 5 --halo-z 7.48 --out example_io/f06/dfn_domain_f06_forward.json"),
    (2, "python -m dfn_analysis.detect_blocks_from_domain_json --json <json> --voxel 0.1 --connectivity 6 --method edgecut --min-voxels 50 --out-prefix <out>"),
    (2, "python -m dfn_analysis.reconstruct_block_polyhedra --json <json> --voxel 0.1 --include-interior --out-prefix <out>"),
    (2, "python -m dfn_analysis.detect_face_blocks --json <json> --voxel 0.05 --labels observed --out-prefix <out>"),
    (2, "bash mc_worker_insitu_l05.sh k  (k=1..50, xargs -P 8) → python aggregate_block_prob_insitu_l05.py"),
    (2, "python make_scene_geometry.py <json> <x_back0> <out_dir>; python bake_block_rgb.py in.vtp out.vtp"),
    (2, "pvpython make_pvsm_blockview_f06_minvol05.py <handoffv2> <out.pvsm>"),
    (-1, "주의"),
    "analysis_260903 스크립트는 상단 SP/H2 경로 상수가 하드코딩 — 재사용 시 그 둘만 수정",
    "detect_blocks_from_domain_json은 리포의 _archive/block_detection/code(tunnel_geometry, block_detector)에 의존 — 배포 시 동봉",
], size=9.5)
note(s, "출처: scripts/analysis_260903/README.md, 주요함수_사용법.md.")

# ============================================================ 19 한계·다음
s = slide("18. 한계와 다음 단계", "결과를 읽을 때 같이 봐야 할 것")
box(s, 0.5, 1.2, 6.0, 4.3, "한계 (알고리즘)", [
    "**ε 의존**: 복셀 0.1 m가 곧 닫힘 판정 스케일. 절대 블록 수·부피는 보고하지 않고 상대 비교·확률로만",
    "**볼록 재구성**: 슬릿·비볼록 블록은 외곽 형상 근사. 큰 쐐기는 연장 평면 과절단 → vol_voxel 병기",
    "**기하 판정만**: 노출(daylighting)은 굴착 가정 정의에서 '터널 접촉'으로 부분 대응, 운동학·역학(마찰·점착·안전율)은 미포함(한양대 단계)",
    "**위치 예측력 한계**: 막장면 1.5 m 너머는 통계로만 — 순수 블라인드에서 복셀 단위 lift 1.2배, P≥0.5 영역과 실측 블록 겹침 0 %",
    "**입력 품질 의존**: 면 07 결손처럼 추출 단계 문제는 재보정으로 못 고침. 검출하한(lmin)은 입력 면 품질에 따라 0.3~0.5",
], size=10.5)
box(s, 6.8, 1.2, 6.0, 4.3, "다음 단계 (제안)", [
    "굴착가정 확률장(6면/전체)도 lmin 0.5 재보정·적응 매칭 기준으로 재생성해 버전 통일 (제자리는 완료)",
    "한양대 입력용 '블록 + 경계평면 목록' 내보내기 정식화 (wedge_bounding_planes.csv 형식을 일반화)",
    "복셀 ε 수렴 검토 또는 대형 균열만 메쉬 절단과 결합한 하이브리드 (역할 분담 확정)",
    "lmin 자동 선택(면별 rolloff + LOFO-CV) 구현 — 현재는 수동 0.3/0.5",
    "재보정본 기반 쐐기 케이스 재전달 여부 결정, 실측 overbreak 지점 대조(CloudCompare bin 수령 후)",
    "world 좌표 정본 + 모듈별 변환으로 좌표 규약 통일 (9/10 회의 의제)",
], fill=RGBColor(0xE8, 0xF3, 0xE8), size=10.5)
note(s, "9/10 연계협의 자료(현대건설 폴더)의 가능/불가/시간부족 표와 정합.")

prs.save(OUT)
print("saved", OUT, "slides", len(prs.slides))
