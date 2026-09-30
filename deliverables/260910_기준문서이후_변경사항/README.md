# 260910 변경사항 보고 — 기준문서 이후 변경·추가 사항

정본: `C:\Users\user\OneDrive\2026-2\현대건설\260910 변경사항 보고.pptx` (11장 / 10분)

- **기준선(청중이 이미 아는 것)**: 260714 중간 보고서 · 260821 실측 데이터 기반 단계별 설명 · 260904 회의
- **청중**: 현대건설 중심 + 전 연구진(한양대·송재준 연구실·BieL) 배석
- **성격**: `260910 공유자료.pptx`(지난 일주일 개발 3건 상세, 11장)와 **별도**. 이 자료는 기준문서 대비 델타 요약이고, 공유자료는 A/B/C 개발 내용의 상세판.

## 구성

| # | 슬라이드 | 핵심 |
|---|---|---|
| 1 | 표지 | 정정 1 · 변경 1 · 신규 2단계 + 검증·규칙 |
| 2 | 무엇이 바뀌고 무엇이 붙었나 | [0]~[7] 흐름도에 ▲변경 2곳 / ◆신규 3단계 표시 + 변경 요약표 |
| 3 | [정정] P32 검출하한 통일 | 5.45→3.09(−43%), 5.97→3.56. 12면 데모 6.16은 재산정 예정 |
| 4 | [변경] 다면 매칭 적응 게이트 | 171→280→308, 막장면 평행 비율 12.9→18.8%(모집단 26.9%) |
| 5 | [신규] edge-cut 블록 판정 + 다면체 복원 | 두께 0 분리면, 168/168, 부피비 0.90, 고정부피 하한 0.05 m³ |
| 6 | [신규] 블록 존재확률 컨투어 | MC 50, 대표 정의 = 제자리, maxP 1.00 / P≥0.5 1.05 m³ |
| 7 | [검증] 블라인드 3종 | P21 80백분위 통과, 위치 도달거리 ≈1.5 m, 4면 −18~27% vs 6면 −9%, LOFO 무검정력 원인 |
| 8 | [규칙] 결과를 읽는 4원칙 | 절대값 금지 · 상위 N개 위험(edge-cut 0 vs 3DEC 62) · 확률 보고 · 면 07 데이터 이상 |
| 9 | [연계] 좌표계·전달본 | world/REF 이중기록, 쐐기 100%/92%, 4~8평면 협력 → 3-절리 도구 0건 정상, **재전달 판단 요청** |
| 10 | 9/4 요구 5항목 대비 현 위치 | 제공 가능 / 남은 것 + 실행시간(조건화 4분20초→4초) |
| 11 | 진행 중·보류 + 확인 요청 | 결정 필요 3건: 재전달 여부 · MC N · 한양대 포맷·좌표 |

발표 스크립트는 각 장 **노트**에 내장(총 ~5천자).

## 재생성

```bash
cd deliverables/260910_기준문서이후_변경사항/scripts
python build_change_deck.py ["출력경로.pptx"]
```

- 스타일: 흑백 테마 규약(CLAUDE.md §18). 데이터 그림·ParaView 렌더만 원색 유지.
- 그림 출처: `docs/figures/`(adaptive_sep_direction_bias, block_probability_insitu_lmin05_mc50,
  blind_validation_f06_lmin05_mc50, block_seed/voxel_vs_edgecut_convergence),
  `deliverables/260909_공유자료_ParaView뷰어_블록판정/`(figures/fig_edgecut_concept, renders/dfn_blockview_f06_minvol05_fit),
  `deliverables/260903_한양대_쐐기케이스_top20/case1_f01-07_seed101/wedge_view_iso.png`.
- `figures/blind_face08_lmin05_2panel.png` = `docs/figures/blind_face08_lmin05.png`에서 앞 두 패널만 잘라
  가독성을 높인 사본(생성: PIL crop `(0, 88, 0.665·W, H)`).

## 수치 검증 메모

- P32 합은 `handoffv2/demo_output/*/p32/p32_summary.csv`의 `P32_hat` 합으로 직접 확인:
  dfm_demo 6.16(analytic_esinphi) / win_f01-06 5.45 → _l05 3.09 / win_f01-07 5.97 → _l05 3.56.
- 면 08 블라인드 백분위: worklog 09-03 §3(1~6면 P21 80백분위), 09-04 §4(개수 114±10, 100백분위),
  09-04 §5(1~7면 재보정 122±10 / 96백분위, P21 72백분위), 09-04 §9(적응 후 w17 125±11 / 90백분위).
  → 슬라이드 7은 "강도 통과, 개수 −18%"로 구분해 서술함(과대 주장 금지).
- 3DEC식 62블록 vs edge-cut 0블록: `handoffv2/dfn_analysis/cut_blocks_polyhedral.py` 프로토타입 실행분,
  `deliverables/260909.../scripts/add_scripts.py`에 서술 보존.
