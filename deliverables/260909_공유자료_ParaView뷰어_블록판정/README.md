# 260909 공유자료 — ParaView 뷰어 · 실현 과정 · 블록 판별 알고리즘

- `공유자료_ParaView뷰어_블록판정_260909.pptx` — 19장. handoffv2/example_io/paraview 의 상태 파일(pvsm)을
  기준으로 (1) 어떤 뷰어가 있고 무엇을 보여주는지, (2) 실측 절리선 → 조건부 DFN → 블록 판정 → pvsm 까지
  어떻게 만들었는지, (3) 블록 판별 알고리즘(복셀 edge-cut CCA + 규칙 3종 + 다면체 재구성 + MC 확률장)을
  정리. 각 슬라이드 노트에 수치 출처(worklog·csv) 기재.
- `renders/` — pvsm 을 pvpython 오프스크린으로 렌더한 PNG (흰 배경). 접미사:
  `_fit` = 저장 카메라 방향으로 데이터에 맞춤, `_nodisc` = 균열 원판 층 숨김, `_blocksonly` = 블록·터널 층만,
  `_close` = 근접. 원본 pvsm 은 절대경로를 담고 있어 GUI 에서는 bat 로 여는 것이 정확.
- `figures/` — 2D 개념도 2장 (edge-cut 원리, 다면체 재구성).
- `scripts/` — 슬라이드 생성기(python-pptx)와 pvpython 렌더 스크립트. 경로 상수만 바꾸면 재생성 가능.

수치·정의의 근거: `docs/worklog/2026-09-01~04.md`, `docs/블록판정_시드민감도_실험보고_260901.md`,
`handoffv2/example_io/blocks_edgecut/*_summary.csv`, `handoffv2/scripts/analysis_260903/README.md`.
- 2026-09-09: 발표 스크립트를 각 슬라이드 노트에 추가 (add_scripts.py). 사용자 사본: 현대건설/260910 공유자료.pptx
