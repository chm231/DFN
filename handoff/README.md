# DFN 역산 파이프라인 — 핸드오프 (최종 버전)

터널 막장면의 2D 절리선(trace) 관측으로부터 3D DFN(Discrete Fracture Network) 파라미터를
역산하고, 관측을 조건으로 하는 3D DFN을 생성·시각화하는 **최종 파이프라인 코드**입니다.

이 폴더(`handoff/dfn_analysis/`)에는 **최종 버전 파이프라인 파일**이 포함됩니다(관측 역산 → 복원 → 조건부 생성).
진단(diagnose_*)·대체(estimate_p32_combined_bootstrap, v3 radius)·보조 시각화(plot_*)는 제외했습니다.
이번 버전에서 **복원(reconstruct)·복원 검증(visualize/LOFO)** 모듈 3종이 추가되었습니다.

모든 파일은 각 코드 블록마다 한글 주석이 달려 있습니다.

---

## 좌표 규약
- **x** = East = 터널 진행축, **y** = North, **z** = Up
- 막장면(관측면)은 x = 상수 평면 (예: x = 0, 1, 2, 3 m)

---

## 파이프라인 실행 순서

```
[1] 입력 생성 (합성 벤치마크)
    generate_synthetic_rough_face_mesh.py   # 거친 막장면 mesh 생성
    export_setwise_3d_traces.py             # DFN + mesh → 3D 막장면 trace 추출

[2] set별 파라미터 역산
    estimate_mean_orientation.py            # 평균 방향(축성 평균)  ← 기반 모듈
    estimate_fisher_kappa.py                # Fisher κ (집중도)      ← 기반 모듈
    estimate_radius_powerlaw_window_mc.py   # 반지름/kr window MC (v4.1)  ← 핵심 우도
    estimate_kr.py                          # kr 멱법칙 지수 추정 (진입점)
    summarize_setwise_trace_statistics.py   # set별 trace 통계
    build_p32_pilot_summary.py              # SITE_SET_CONFIG · 지지구간 P32 스케일
    estimate_p32_mc_calibrated.py           # P32 (analytic_esinphi 수식 보정, 최종)

[2.5] 관측 절리 복원 (trace → 원판 disc)   ← 이번 버전에서 handoff 에 포함
    reconstruct_discs_from_traces.py        # 연결(검증형 응집) + 반지름(원적합/kr 축소추정)
    visualize_reconstruction.py             # 복원 검증: 3D + 면별(관측 vs 복원 chord)
    validate_reconstruction_lofo.py         # LOFO 외삽검증(참값 불필요, 정합성 지표)

[3] 조건부 DFN 생성 + 시각화
    generate_conditional_hidden_dfn.py      # 관측 복원 disc(visible) + 확률 hidden disc 결합
                                            #   (set별 P21 진단표 포함; Set4=visible-only)
    visualize_conditional_dfn_3d.py         # PyVista 3D 시각화 (disc + 막장면 + trace)
```

---

## 파일 역할 요약

| 파일 | 역할 |
|------|------|
| `estimate_mean_orientation.py` | 절리 normal의 축성(axial) 평균 방향, trend/plunge(NED) 변환 |
| `estimate_fisher_kappa.py` | Fisher 분포 집중도 κ 추정, 3점법 trace normal 추정 |
| `estimate_radius_powerlaw_window_mc.py` | 반지름 멱법칙 우도 + 유한 창 MC 보정 (v4.1, kr/P32 공통 핵심) |
| `estimate_kr.py` | set별 반지름 멱법칙 지수 kr 추정 진입점. `--likelihood-mode {window_mc(기본·**최종 추정용**)|hybrid(보조)}` — hybrid는 참 현길이 분포를 닫힌형(해석식)으로, 창·절단 변환은 kr불변 MC 커널 1회로 분해. 프로파일 평활·lmin 불변·2.7× 속도가 강점이라 **프로파일 해석·민감도 스캔용**이며, 20시드 검증에서 최종 추정 효율(RMSE)은 window_mc+per-set 선택이 우위라 기본값을 유지한다. hybrid는 lmin_fit 미지정 시 전 set 공통 0.5 m 고정. 원리·검증·판정: docs/제6장_수정안_통합본.md 제3부 §4 |
| `build_p32_pilot_summary.py` | 사이트/set 설정, 지지구간 스케일 P32 헬퍼 |
| `summarize_setwise_trace_statistics.py` | set별 trace 길이/개수 통계 |
| `estimate_p32_mc_calibrated.py` | **최종 P32 추정** (observed_P21 / 보정계수 C). `--calibration-factor-mode`: `analytic_esinphi`(**수식 기반** C=E[sinφ] 결정론적 구적, **기본·최종**, ~2초) 또는 `unit_p32_forward_mc`(legacy 순방향 MC). 두 모드의 2~4% 차이는 unit-MC의 면적 기준 불일치(분자=다각형 클리핑, 분모=mesh 면적이 −3.85% 작음) 편향으로 판명 — 관측 정의와 정합하는 쪽은 해석식(유도·검증·결정실험: docs/제6장_수정안_통합본.md 제3부 §2.3–2.5) |
| `export_setwise_3d_traces.py` | DFN(h5) + 거친 막장면 mesh → 3D trace 데이터셋 순방향 생성. `--trace-normal-source external`(기본)=외부 제공 3D 방향 사용(2026-07 결정; 벤치마크는 fracture 참값 법선이 그 역할), `3pt`=legacy polyline 3점법 |
| `generate_synthetic_rough_face_mesh.py` | 합성 거친 막장면 mesh 생성 (검증용 입력) |
| `reconstruct_discs_from_traces.py` | **관측 trace → 복원 원판**. 연결=검증형 응집(결합평면 잔차+면당1chord, oracle 대비 순도 95%), 반지름=경계 원적합 / kr 축소추정(empirical-Bayes, 참R 오차 26→17%) |
| `visualize_reconstruction.py` | 복원 검증 시각화: 3D 개요 + 면별(관측 trace vs 복원 원판 chord, 미재현 강조) |
| `validate_reconstruction_lofo.py` | Leave-One-Face-Out 외삽검증. 참값 없이 recall/precision 로 정합성 측정(관측 P21 대비 오차의 대체 지표) |
| `generate_conditional_hidden_dfn.py` | 조건부 hidden DFN 생성 (remove-and-resample) + **set별 P21 진단표** |
| `visualize_conditional_dfn_3d.py` | PyVista 3D 시각화 (disc + 막장면 + 관측/조건화 trace) |

---

## 중요 정책 / 주의사항
- **GT(ground truth)는 검증 전용**입니다. 추정기 설계·보정에 truth를 사용하지 않습니다 (정책 D013).
- **하나의 parsimonious 추정기**를 모든 set에 동일 적용합니다. set별 보정계수·fudge factor 금지.
- **Laxemar Set 4는 지수분포**라 멱법칙 역산·조건화에서 제외됩니다 (D002).
- 최종 P32 보정계수는 `analytic_esinphi`(수식 기반 C=E[sinφ], 기본값) 방식입니다 —
  2026-07-29 결정으로 종전 `unit_p32_forward_mc`(D004)를 대체. unit-MC는 legacy/진단용으로
  유지되며, 면적 기준 불일치로 C가 +2~4% 과대(P32 과소)한 알려진 편향이 있습니다
  (근거: docs/제6장_수정안_통합본.md 제3부 §2.5). combined_bootstrap은 대체 방식으로 제외.
- 반지름/kr은 `estimate_radius_powerlaw_window_mc.py`(v4.1 창 MC)가 최종입니다. v3(from_traces)는 제외.

## 경로 의존성 (수정 필요)
- `reconstruct_discs_from_traces.py`는 **자립적**입니다(numpy/scipy/h5py만; 생성기·경로 의존 없음).
- `visualize_reconstruction.py` / `validate_reconstruction_lofo.py`는 `generate_conditional_hidden_dfn`의
  기하 함수(`visible_trace_on_face` 등)를 import 하므로, 그것과 **동일한 리포 루트 경로 의존**을 물려받습니다
  (리포 루트 `PYTHONPATH="."`에서 실행 필요).
- `generate_conditional_hidden_dfn.py` / `visualize_conditional_dfn_3d.py`는 원본 리포지토리 기준
  상대경로(`dfn generator v1/`, `storage/output/...`)를 참조합니다. 이 폴더만 단독 이전 시
  해당 경로를 받는 팀 환경에 맞게 조정해야 합니다.
- 나머지 모듈은 `from dfn_analysis.<module> import ...` 형태로 서로만 참조하므로, 이 폴더를
  `dfn_analysis` 패키지로 유지하면 상호 import는 정상 동작합니다.

## 임의(arbitrary) 데이터셋으로 실행 — 일반화

내장 preset(forsmark/laxemar) 외의 데이터셋에도 돌릴 수 있습니다.

- **방향/κ 추정**: normal 기반이라 site 무관, 그대로 동작.
- **kr 추정 (`estimate_kr`)**: 이미 일반화됨. `--site` 없이 `--dfn-model <라벨> --generation-rmin <값>`
  만으로 임의 trace 데이터셋에 동작 (기본 `empirical_trace` 모드).
- **P32 추정 (`estimate_p32_mc_calibrated`)**: set별 크기분포·방향이 방법론에 필수이므로,
  외부 JSON config로 주입한다. `dataset_config.py` 참조.

### P32용 데이터셋 config (JSON)
`handoff/configs/example_dataset_forsmark.json` 참고:
```json
{
  "dataset_name": "my_dataset",
  "sets": {
    "1": {"p32_base": 0.602, "dist_type": "powerlaw", "r0": 0.28,
          "trend": 182.8, "plunge": -1.7, "kappa": 22.1},
    ...
  }
}
```
- `p32_base / dist_type / r0` : 크기분포 (모집단 반지름 샘플링 + support 스케일링)
- `trend / plunge / kappa`    : Fisher 방향 (교차확률 forward MC)

실행 예:
```
python -m dfn_analysis.estimate_p32_mc_calibrated \
  --trace-h5 <trace.h5> --config configs/my_dataset.json --site my_dataset \
  --target-set 1 2 5 --kr-summary-csv <kr_summary.csv> --dfn-h5 <dfn.h5> \
  --calibration-factor-mode analytic_esinphi --outcsv <out.csv>
```
(`--site`는 config의 `dataset_name`과 일치해야 하며, 내장 preset이 아니면 `--dfn-h5` 명시 필수)

### config를 trace에서 자동 생성 (자기완결적 추정)
방향(trend/plunge/kappa)은 참값 대신 **trace에서 추정**할 수 있다. `build_dataset_config_from_traces.py`가
trace h5의 per-trace 법선(`trace_normal_xyz`)으로 set별 평균 방향+κ를 추정하고, kr_hat과 함께 config JSON을 만든다:
```
python -m dfn_analysis.build_dataset_config_from_traces \
  --trace-h5 <trace.h5> --kr-summary-csv <kr_summary.csv> \
  --dataset-name my_dataset --target-set 1 2 5 --out configs/my_dataset.json
```
→ 이 config를 P32에 넘기면 크기(kr)·방향이 **모두 trace에서 추정된** 값으로 P32가 계산된다.
(powerlaw 세트는 P32 추정에 `dist_type`+방향만 필요하고 `r0`/`p32_base`는 불필요 → 추정 config에서 생략 가능;
생략 시 검증용 `P32_reference`만 NaN이 되고 추정 `P32_hat`에는 영향 없음.)

### 검증 (Forsmark)
- kr: `--site` 없이 generic 실행 → 내장 preset 경로와 동일 kr_hat (Set1 2.80/2.88, Set5 3.10/2.92).
- P32(참값 방향 config): forsmark preset을 JSON으로 추출 → 내장 preset과 **결정론 수치 완전 동일**
  (support_scaled_p32/radius_moments/Fisher 일치). MC 부분만 시드 차이.
- P32(**추정 방향** config, 자기완결): trace 추정 방향으로 P32 → 참값 방향 대비 Set1 +1.7% / Set2 +17% /
  Set5 −20%. 집중 셋(κ 큰 Set1)은 거의 일치, 등방 셋(κ≈0.9 Set5)은 방향 추정 불안정으로 편차 큼(예상된 한계).

### 아직 종속인 부분
- **조건화**(`generate_conditional_hidden_dfn`)의 `POWERLAW_SETS=(1,2,3,5)`는 여전히 고정. 임의
  데이터셋의 set 구조에 맞게 조정 필요.
- **복원 단계**(reconstructed_discs.csv 생성)는 이제 `reconstruct_discs_from_traces.py`로 **handoff 에 포함**됨.
  실행: `python -m dfn_analysis.reconstruct_discs_from_traces --trace-h5 <trace.h5> --out-csv <pdir>/reconstruct/reconstructed_discs.csv --target-set 1 2 3 5 --kr-summary-csv <pdir>/kr/kr_summary_by_set.csv`
  - 연결/반지름 방법 요약: 검증형 응집 연결(순도 ~95%) + 경계 원적합/kr 축소추정 반지름.
  - **정합성 주의**: 축소추정은 참 DFN(un-censoring)을 목표로 하므로 **관측 P21 대비 과대**가 정상이다.
    검증은 관측 P21 오차가 아니라 **참값(합성 GT: 반지름 오차 ~17%) 또는 LOFO 외삽(recall/precision)**으로 한다.
  - Laxemar 예시 LOFO: 축소추정이 하한보다 recall↑(14→20%)·precision 동일 → 예측력 개선 확인. 희소 set(2)은 저신뢰.

## 실행 환경
- Python 3.9+ / numpy, scipy, h5py, matplotlib, pyvista
- (선택) `visualize_conditional_dfn_3d.py --html` 로 대화형 HTML(vtk.js) 뷰를 만들려면
  trame 필요: `pip install "pyvista[jupyter]"` (trame, trame-vtk, trame-vuetify).
  미설치 시 HTML export만 건너뛰고 PNG/창 출력은 정상 동작.
