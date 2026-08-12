# 최종보고서(제6장) 그림·코드 아카이브 — 2026-08

「터널 3D 매핑을 통한 암반특성 평가기법 불확실성 평가 용역 최종 보고서」(0807 final)
제6장 작성에 사용된 그림 원본과 이를 생성한 코드의 스냅샷이다. 보고서 수치의 기준은
**해석식 P32 (analytic_esinphi, 2026-08-04 재확정) + 하이브리드 k_r + 평면 막장면·터널
단면 다각형 관측 기준(v2)** 이다.

## 구성

- `figures/` — docs/figures 전체 (보고서 수록 그림 + 미사용 예비 그림 포함)
- `scripts/` — 그림 생성·민감도·비교 스크립트
- `dfn_analysis/` — 역산 파이프라인 모듈 (보고서 수치 산출 시점 스냅샷)
- `dfn generator v1/python/generate_dfn.py` — 합성 DFN 생성기 (벤치마크 시드 42)
- `storage/data/단면_폴리곤.dat` — 터널 단면 다각형

## 보고서 그림 번호 ↔ 파일 대응 (최종본 기준)

| 그림 | 파일 | 생성 스크립트 |
|---|---|---|
| 6-1 파이프라인 개요 | fig_pipeline_overview.png | make_pipeline_overview_figure.py |
| 6-2 조건화 개요 | fig_block2_inversion.png 계열 | make_report_figures.py |
| 6-3 단일 vs 연속 막장면 | fig_block3_generation.png 계열 | make_report_figures.py |
| 6-4 전처리 4패널 | fig_block1_preprocessing.png | make_report_figures.py |
| 6-5 전처리 흐름 | fig_preprocess_flow.png | make_fig_preprocess_flow.py |
| 6-6 정방향·역방향 역할 | fig_trace_dual_role.png | make_fig_trace_dual_role.py |
| 6-7 터널 단면 다각형 | tunnel_polygon_yz.png | make_report_figures.py |
| 6-8 경계 접촉 유형 | (전처리 계열) | make_fig_supplement.py / make_report_figures.py |
| 6-9 P32 보정계수 (해석식) | fig_p32_calibration_schematic.png | make_fig_p32_schematic.py |
| 6-10 복원 균열 3D | fig_recon_discs_set4_v2.png / set5_v2.png | make_fig_recon_discs.py |
| 6-11 조건부 생성 예시 | (conditional 계열) | dfn_analysis/generate_conditional_hidden_dfn.py |
| 6-12 벤치마크–역산 비교 | fig_validation_bars_v2.png | make_fig_validation_bars.py |
| 6-13 k_r 프로파일 우도 | fig_kr_profile_likelihood_v2.png | (kr 산출물 기반) |
| 6-14 시드 민감도 | fig_sensitivity_seed_v2.png | plot_seed_sensitivity.py |
| 6-15 굴착 간격 민감도 | fig_sensitivity_face_interval_v2.png | sensitivity_face_interval.py |
| 6-16 방위각 P32 | fig_sensitivity_angle_p32_v2.png | sensitivity_tunnel_angle.py |
| 6-17 방위각 k_r | fig_sensitivity_angle_kr_v2.png | sensitivity_tunnel_angle.py |
| 6-18 방위각 κ | fig_sensitivity_angle_kappa_v2.png | sensitivity_tunnel_angle.py |

미사용 예비: fig_kr_hybrid_likelihood.png (하이브리드 우도 모식도 — 미수록 결정),
fig_p32_clmin_schematic.png (C_ℓmin 구판 — 폐기), fig_grid_discretization_v2.png ·
fig_km_diagnostic_v2.png (보충원고용), fig_pipeline_overview_v2.png ·
fig_preprocess_flow_v2.png (C_ℓmin 시절 구판 — 폐기).

## 수치 재현

- 벤치마크(표 6-5): 배포 패키지 `handoffv1/`에서 `run_demo.py --seed 42`
  (시드 42 = 보고서 벤치마크 DFN과 동일; `--p32-fisher preset` 기본값이 표 6-5 재현)
- 민감도(표 6-7~6-9): scripts/sensitivity_tunnel_angle.py(0–180°/12°, 30실현),
  sensitivity_face_interval.py({0.5,1,2,3,6} m, 20실현), plot_seed_sensitivity.py.
  원자료 CSV는 저장소 docs/sensitivity_*_v2_analytic_results.csv

## 주의

- 표 6-5의 환산계수 C는 `--config` 미지정 실행 경로(사이트 방향 프리셋 표)에서 계산된
  값이다. 역산 방향분포로 계산하면 C·P32가 달라진다 — 상세:
  메모리/문서 "P32 C 방향분포 소스 불일치" 항목 참조.
