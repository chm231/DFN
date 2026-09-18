# -*- coding: utf-8 -*-
"""블록 시드 민감도 아티팩트 HTML 생성 (그림 base64 임베드)."""
import base64
from pathlib import Path

SC = Path(__file__).parent
figs = {}
for name in ["blocks_3d_by_seed", "block_volume_ccdf_by_seed", "block_counts_by_seed", "block_ccdf_mc_band"]:
    b = (SC / f"block_figs/{name}.png").read_bytes()
    figs[name] = base64.b64encode(b).decode()

html = """<title>블록 시드 민감도</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+KR:wght@400;500;700&family=IBM+Plex+Mono:wght@400;600&display=swap">
<style>
:root{
  --ground:#f4f6f8; --panel:#ffffff; --ink:#1f2933; --ink-soft:#52606d;
  --line:#d9e0e6; --accent:#2b6777; --accent-soft:#e3edf0;
  --warn:#b5762a; --warn-soft:#f6ede0;
}
@media (prefers-color-scheme: dark){
  :root:not([data-theme="light"]){
    --ground:#16191d; --panel:#1e2329; --ink:#e6eaee; --ink-soft:#9aa7b2;
    --line:#333c45; --accent:#6fb3c4; --accent-soft:#22333a;
    --warn:#d9a05b; --warn-soft:#3a2f1e;
  }
}
:root[data-theme="dark"]{
  --ground:#16191d; --panel:#1e2329; --ink:#e6eaee; --ink-soft:#9aa7b2;
  --line:#333c45; --accent:#6fb3c4; --accent-soft:#22333a;
  --warn:#d9a05b; --warn-soft:#3a2f1e;
}
body{background:var(--ground); color:var(--ink);
  font-family:"IBM Plex Sans KR","Malgun Gothic",sans-serif;
  line-height:1.65; margin:0; padding:2.5rem 1.25rem 4rem;}
main{max-width:860px; margin:0 auto;}
.eyebrow{font-size:.78rem; letter-spacing:.14em; color:var(--accent);
  font-weight:700; text-transform:uppercase; margin-bottom:.4rem;}
h1{font-size:1.9rem; font-weight:700; line-height:1.25; margin:0 0 .4rem;
  text-wrap:balance;}
.sub{color:var(--ink-soft); margin:0 0 2rem; font-size:.95rem;}
h2{font-size:1.2rem; font-weight:700; margin:2.6rem 0 .9rem;
  padding-top:1.4rem; border-top:1px solid var(--line);}
.verdicts{display:grid; grid-template-columns:repeat(auto-fit,minmax(240px,1fr));
  gap:.9rem; margin:1.4rem 0;}
.verdict{background:var(--panel); border:1px solid var(--line);
  border-radius:6px; padding:1rem 1.1rem;}
.verdict .tag{display:inline-block; font-size:.72rem; font-weight:700;
  letter-spacing:.08em; padding:.15rem .55rem; border-radius:999px; margin-bottom:.55rem;}
.tag.stable{background:var(--accent-soft); color:var(--accent);}
.tag.varies{background:var(--warn-soft); color:var(--warn);}
.verdict p{margin:0; font-size:.92rem;}
table{border-collapse:collapse; width:100%; font-size:.9rem;
  font-variant-numeric:tabular-nums; background:var(--panel);}
.tblwrap{overflow-x:auto; border:1px solid var(--line); border-radius:6px; margin:1rem 0;}
th{font-size:.78rem; letter-spacing:.05em; text-align:left; color:var(--ink-soft);
  border-bottom:1.5px solid var(--line); padding:.55rem .8rem; white-space:nowrap;}
td{padding:.5rem .8rem; border-bottom:1px solid var(--line); white-space:nowrap;}
tr:last-child td{border-bottom:none;}
td.num, th.num{text-align:right; font-family:"IBM Plex Mono",monospace; font-size:.85rem;}
.hl{color:var(--warn); font-weight:600;}
.zero{color:var(--warn); font-weight:700;}
figure{margin:1.4rem 0;}
figure img{max-width:100%; border:1px solid var(--line); border-radius:6px;
  background:#fff;}
figcaption{font-size:.82rem; color:var(--ink-soft); margin-top:.45rem;}
.callout{border-left:3px solid var(--warn); background:var(--warn-soft);
  padding:.9rem 1.1rem; border-radius:0 6px 6px 0; margin:1.2rem 0; font-size:.92rem;}
.callout b{color:var(--warn);}
.method{font-size:.88rem; color:var(--ink-soft);}
code{font-family:"IBM Plex Mono",monospace; font-size:.84em;
  background:var(--accent-soft); padding:.08em .35em; border-radius:4px;}
pre{background:var(--panel); border:1px solid var(--line); border-radius:6px;
  padding:1rem; overflow-x:auto; font-size:.8rem; line-height:1.5;}
pre code{background:none; padding:0;}
ul{padding-left:1.2rem;}
li{margin:.35rem 0;}
</style>
<main>
<p class="eyebrow">DFN &rarr; 블록 판정 &middot; 야간 자동 실험 &middot; 2026-09-01</p>
<h1>랜덤 시드가 블록 판정 결과를 얼마나 바꾸는가</h1>
<p class="sub">클라이언트 재질문 ⑤의 블록 수준 완성판 &mdash; 몬테카를로 12개 실현 &times; 복셀 블록 판정(0.1 m, 6-connectivity), 도메인: 마지막 막장면 전방 10 m + halo 5 m.</p>

<div class="verdicts">
  <div class="verdict"><span class="tag varies">중간 변동</span>
    <p><strong>블록 통계는 총부피 ±21%, 최대 블록 ±48% 변동</strong> (12개 실현). 블록 수는 CV 11%로 비교적 안정, 극값일수록 시드 지배.</p></div>
  <div class="verdict"><span class="tag varies">변동</span>
    <p><strong>개별 블록은 전부 다름.</strong> 시드 간 위치&middot;부피 매칭 8~9%, 복셀 Jaccard 0.02~0.04. 최대 블록 범위 0.44~2.32 m³ (5배).</p></div>
  <div class="verdict"><span class="tag varies">주의</span>
    <p><strong>상위 30개 균열만으로는 블록 0개.</strong> 블록 경계를 닫는 것은 소형 균열 &mdash; 반지름 상위 N개 필터링은 블록 형성 자체를 제거.</p></div>
</div>

<h2>결과 요약</h2>
<div class="tblwrap"><table>
<thead><tr><th>run</th><th class="num">균열 수</th><th class="num">블록 수</th><th class="num">총부피 m³</th><th class="num">최대 m³</th><th class="num">중앙값 m³</th></tr></thead>
<tbody>
<tr><td>seed 2026 (full)</td><td class="num">19,617</td><td class="num">132</td><td class="num">8.92</td><td class="num">1.406</td><td class="num">0.023</td></tr>
<tr><td>seed 7 (full)</td><td class="num">19,430</td><td class="num">149</td><td class="num">9.60</td><td class="num"><span class="hl">1.890</span></td><td class="num">0.026</td></tr>
<tr><td>seed 123 (full)</td><td class="num">19,393</td><td class="num">142</td><td class="num">8.92</td><td class="num"><span class="hl">0.654</span></td><td class="num">0.029</td></tr>
<tr><td>seed 2026 &middot; rmax 10 m</td><td class="num">19,594</td><td class="num">108</td><td class="num"><span class="hl">4.53</span></td><td class="num"><span class="hl">0.263</span></td><td class="num">0.022</td></tr>
<tr><td>모든 run &middot; 상위 30개만</td><td class="num">30</td><td class="num"><span class="zero">0</span></td><td class="num">0</td><td class="num">&mdash;</td><td class="num">&mdash;</td></tr>
</tbody></table></div>

<div class="tblwrap"><table>
<thead><tr><th>교차 시드 비교 (full)</th><th class="num">블록 매칭*</th><th class="num">복셀 Jaccard</th></tr></thead>
<tbody>
<tr><td>seed 2026 vs 7</td><td class="num">12 / 132 (9%)</td><td class="num">0.039</td></tr>
<tr><td>seed 2026 vs 123</td><td class="num">11 / 132 (8%)</td><td class="num">0.017</td></tr>
<tr><td>seed 7 vs 123</td><td class="num">14 / 149 (9%)</td><td class="num">0.015</td></tr>
</tbody></table></div>
<p class="method">* 매칭 = 중심거리 0.5 m 이내이고 부피비 0.5~2배인 블록 쌍.</p>

<figure><img src="data:image/png;base64,__FIG3D__" alt="시드별 블록 3D 비교">
<figcaption>시드 3종의 블록 3D 분포(동일 카메라). 터널 주변 형성 밀도는 유사하나 개별 블록의 위치&middot;형상은 전혀 다르다.</figcaption></figure>

<figure><img src="data:image/png;base64,__FIGCCDF__" alt="블록 부피 CCDF">
<figcaption>블록 부피 상보누적분포. full DFN에서는 세 시드의 곡선 형태가 유사하고 오른쪽 꼬리(최대급 블록)에서 벌어진다. 변동 폭의 정량화는 아래 몬테카를로 12개 실현 참조. top-30 패널은 블록이 없어 비어 있다.</figcaption></figure>

<figure><img src="data:image/png;base64,__FIGBAR__" alt="블록 수와 총부피">
<figcaption>run별 블록 수&middot;총부피. rmax_local 10 m에서는 총부피가 절반으로 감소 &mdash; 대형 균열 꼬리가 블록 형성을 지배한다.</figcaption></figure>

<h2>몬테카를로 12개 실현</h2>
<p class="method">시드 3개는 우연히 총부피가 비슷해(8.9~9.6 m³) 변동을 과소평가할 수 있어, 시드 9개를 추가해 12개 실현으로 확장했다.</p>
<div class="tblwrap"><table>
<thead><tr><th>지표</th><th class="num">평균 &plusmn; 표준편차</th><th class="num">CV</th><th class="num">범위</th></tr></thead>
<tbody>
<tr><td>블록 수</td><td class="num">137.8 &plusmn; 15.3</td><td class="num">11%</td><td class="num">102 ~ 163</td></tr>
<tr><td>총 블록 부피 [m³]</td><td class="num">7.95 &plusmn; 1.67</td><td class="num"><span class="hl">21%</span></td><td class="num">4.16 ~ 10.08</td></tr>
<tr><td>최대 블록 부피 [m³]</td><td class="num">1.13 &plusmn; 0.54</td><td class="num"><span class="hl">48%</span></td><td class="num">0.44 ~ 2.32</td></tr>
</tbody></table></div>
<figure><img src="data:image/png;base64,__FIGMC__" alt="몬테카를로 CCDF 밴드">
<figcaption>12개 실현의 블록 부피 CCDF와 5&ndash;95 백분위 밴드. 분포의 형태는 실현 간 일관되나 총량과 꼬리는 실현마다 흔들린다.</figcaption></figure>

<h2>이전 회신 정정 1건</h2>
<div class="callout"><b>rmax_local은 블록 결과를 바꾸는 물리 파라미터다.</b>
R2 회신에서 &ldquo;rmax-local 25&rarr;10은 결과 영향 미미&rdquo;라 했으나 이는 DFN 반지름 통계 기준이었다.
블록 수준에서는 총부피 8.92&rarr;4.53 m³(&minus;49%), 최대 블록 1.41&rarr;0.26 m³(&minus;81%)로 크게 달라진다.
클라이언트에게 rmax 축소를 권할 때 이 단서를 반드시 붙일 것.</div>

<h2>클라이언트 질문에 대한 최종 답</h2>
<ul>
<li><strong>&ldquo;시드 바꿔도 안 달라졌으면&rdquo;</strong> &mdash; 바란 대로는 되지 않는다. 개별 블록은 실현마다 전부 다르고, 블록 통계도 총부피 &plusmn;21%&middot;최대 블록 &plusmn;48% 변동한다. 블록 수준 결론은 <em>몬테카를로(10개 이상 실현)의 평균&plusmn;CI</em>로 보고하는 것이 맞다.</li>
<li><strong>부담은 크지 않다</strong> &mdash; 이번 12개 실현이 이미 그 몬테카를로다. 실현당 약 5분이라 20회도 2시간 이내. 산출물은 평균&plusmn;CI와 공간 셀별 블록 발생 빈도(위험도 맵) 권장.</li>
<li><strong>&ldquo;상위 20~30개만 입력&rdquo;은 재고 필요</strong> &mdash; 반지름 기준 상위 30개로는 어떤 시드에서도 닫힌 블록이 형성되지 않았다. 균열 수를 줄이려면 블록 판정 후 <em>위험 블록을 만드는 균열 부분집합</em>을 남기는 쪽이 타당하다.</li>
</ul>

<h2>방법&middot;재현</h2>
<p class="method">신규 스크립트 <code>handoffv1/dfn_analysis/detect_blocks_from_domain_json.py</code>가
[7] 도메인 JSON을 읽어 레거시 블록 탐지(<code>_archive/block_detection</code>)를 재사용한다.
가정: 도메인 x 전 구간 터널 연장(전방 굴착), 균열 두께 = 복셀&times;0.6, ROCK CCA 6-connectivity,
블록 = 터널 접촉 &and; 도메인 외곽 비접촉(min 8 복셀). 상세와 원자료:
<code>docs/블록판정_시드민감도_실험보고_260901.md</code>, <code>docs/data/block_seed/</code>.</p>
<pre><code>PYTHONPATH=. python -m dfn_analysis.export_domain_dfn_json --pipeline-dir demo_output/dfm_demo \\
  --sets 1 2 3 4 --rmax-local 25 --lmin-det 0.5 --seed &lt;SEED&gt;
PYTHONPATH=. python -m dfn_analysis.detect_blocks_from_domain_json \\
  --json &lt;도메인 JSON&gt; --voxel 0.1 --connectivity 6 --out-prefix &lt;프리픽스&gt; [--top-n 30]</code></pre>
</main>
"""
html = html.replace("__FIG3D__", figs["blocks_3d_by_seed"])
html = html.replace("__FIGCCDF__", figs["block_volume_ccdf_by_seed"])
html = html.replace("__FIGBAR__", figs["block_counts_by_seed"])
html = html.replace("__FIGMC__", figs["block_ccdf_mc_band"])
(SC / "block_seed_report.html").write_text(html, encoding="utf-8")
print("written", len(html) // 1024, "KB")
