# analysis_260903 — 세션 분석 스크립트 아카이브 (2026-08-31 ~ 09-03)

세션 scratchpad(임시 폴더)에서 사용한 분석·MC·시각화 스크립트를 유실 방지용으로
그대로 이관한 것. **경로 상수 주의**: 대부분 스크립트 상단의 `SP`(scratchpad 절대경로)와
`H2`(handoffv2 절대경로)가 하드코딩되어 있음 — 재사용 시 이 두 상수만 수정하면 됨.
실행 규약: handoffv2 루트에서 `PYTHONPATH=. python -X utf8 ...`, pvsm 생성은
`pvpython`(ParaView 6.1.1). 배경·산출물 맥락은 `docs/worklog/2026-09-03.md` 참조.

## 그룹별 안내

### 윈도우 파이프라인 / export
- `run_window_pipeline.py` — 가변 면 수 윈도우 A단계(변환→kr→config→P32→복원)
- `run_window_export.sh`, `run_faces4_seeds.sh` — 윈도우 export 배치

### 블록 판정·MC 워커 (시드 = k×101, xargs -P 8 병렬용)
- `detect_interior_labels.py` — 제자리(미굴착·터널없음) 닫힌 블록 라벨만 저장 (MC 핵심)
- `mc_worker_f06.sh` / `mc_worker_full.sh` — 굴착가정 터널접촉 블록 MC (6면 전방 / 전체 도메인)
- `mc_worker_insitu.sh` / `_l05.sh` — 제자리 블록 MC (구보정 / lmin0.5 재보정)
- `slide_worker.sh` — 슬라이딩 윈도우(1-4→5 등) 블록+대상면 절리선 MC
- `mc_worker.sh`, `run_mc_batch.sh`, `run_face_mc.sh` — 이전(12면) MC 워커
- `make_unexcavated_blocks.py` — 내부 닫힌 블록 + 다면체 재구성 (단일 실행용)

### 확률장 집계
- `aggregate_block_prob*.py` — 라벨 npz → 복셀 확률장 vti + 컨투어 png
  (무印=12면 mc, `_f06`/`_full`/`_insitu`/`_insitu_l05` 변형은 경로·제목만 상이)
- `aggregate_slide.py` — 슬라이딩 윈도우 3세트 집계 + 장면 vtp + 대상면 대조

### 블라인드 검증
- `compute_face_plane_traces.py` — 도메인 JSON을 면 평면으로 절단해 관측창 내 절리선 통계
- `mc_worker_face08.sh` / `_l05.sh`, `lmin_trace_worker*.sh` — 면 08/05 예측 MC 워커
- `compare_face08.py` / `_l05.py` — 절리선 통계 대조(개수·P21·길이CCDF·방향)
- `validate_blind_f06.py` / `_l05.py` — 블록 위치 블라인드 (면 07~12 증거 vs 확률장)

### 한양대 쐐기 (top-N 무한평면)
- `select_top.py` — 무한평면 확장 대상 N개 선별 (**관측 우선 할당**, 기본 n_obs=10).
  아래 세 스크립트가 공유한다 — 전달 CSV 의 rank 와 실제 판정 평면이 일치해야 하므로
  선별 로직은 여기 한 곳에만 둔다. `n_obs=0` 이면 260903 전달본의 순수 반지름 순위.
- `wedge_top20.py` — 상위 N 무한평면 → 제거가능 쐐기 판정 (+SUM 요약 라인).
  `<json> [n_top] [out_npz] [n_obs]`
- `wedge_seed_worker.sh` — 추가 시드 발생확률 전수용
- `build_wedge_case.py` — 케이스 패키지(표+쐐기 메쉬: 볼록체×터널 clip_surface,
  슬리버는 평활복셀, implicit distance 최종 절단)
- `make_bounding_planes.py` — 쐐기별 경계평면 rank 목록 (3-절리 도구 대조용)
- `build_case_all.sh` — 케이스 1개 전체 빌드(판정→CSV→색→pvsm→PNG→zip)

### ParaView 상태/렌더 (pvpython)
- `make_pvsm_blockview*.py` — 블록뷰 (12면 / f06)
- `make_pvsm_blockprob*.py` — 확률장 컨투어 뷰 (범례 라벨은 `{:.2f}` 브레이스 서식 필수)
- `make_pvsm_slide.py` — 슬라이딩 윈도우 단일 상태(w14/w25/w36 레이어 토글 + Axes Grid)
- `make_pvsm_wedgecase.py`, `render_wedge_views.py` — 쐐기 케이스 뷰/3방향 PNG
- `make_pvsm.py`, `make_pvsm_current.py`, `make_pvsm_face*.py`, `make_pvsm_matrix.py`,
  `make_pvsm_windows.py` — 이전 뷰 생성기
- `make_scene_geometry.py` — tunnel_behind/face_cap/domain_box vtp
- `bake_block_rgb.py` — 블록별 tab20 RGB 셀 배열 베이크 (MapScalars=0 용)

### 기타
- `fig_facewise_0902.py` — 면별 절리선 수 그림 / `profile_export.py`,
  `bench_conditioning.py` — [7] 병목 프로파일·벡터화 벤치 /
  `aggregate_mc.py`, `compare_blocks.py`, `check_windows_state.py`,
  `build_report_html.py` — 이전 실험 보조

## 알려진 규칙 (이 폴더 스크립트 공통)
- bat 파일은 ASCII만 (한글 rem 금지 — cp949 파싱 깨짐)
- 콘솔 한글: `python -X utf8`, matplotlib 폰트 `Malgun Gothic`
- 방위각 규약: trend = atan2(ny, nx) (+x=굴진방향 기준, 지리방위 아님)
