# DFN 역산 파이프라인 — 핸드오프 (최종 버전)

터널 막장면의 2D 절리선(trace) 관측으로부터 3D DFN(Discrete Fracture Network) 파라미터를
역산하고, 관측을 조건으로 하는 3D DFN을 생성·시각화하는 **최종 파이프라인 코드**입니다.

이 폴더(`handoff/dfn_analysis/`)에는 **최종 버전으로 분류된 11개 파일만** 포함됩니다.
진단(diagnose_*)·대체(estimate_p32_combined_bootstrap, v3 radius)·보조 시각화(plot_*)는 제외했습니다.

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
    estimate_p32_mc_calibrated.py           # P32 (unit_p32_forward_mc 보정, 최종)

[3] 조건부 DFN 생성 + 시각화
    generate_conditional_hidden_dfn.py      # 관측 복원 disc + 확률 hidden disc 결합
    visualize_conditional_dfn_3d.py         # PyVista 3D 시각화 (disc + 막장면 + trace)
```

---

## 파일 역할 요약

| 파일 | 역할 |
|------|------|
| `estimate_mean_orientation.py` | 절리 normal의 축성(axial) 평균 방향, trend/plunge(NED) 변환 |
| `estimate_fisher_kappa.py` | Fisher 분포 집중도 κ 추정, 3점법 trace normal 추정 |
| `estimate_radius_powerlaw_window_mc.py` | 반지름 멱법칙 우도 + 유한 창 MC 보정 (v4.1, kr/P32 공통 핵심) |
| `estimate_kr.py` | set별 반지름 멱법칙 지수 kr 추정 진입점 |
| `build_p32_pilot_summary.py` | 사이트/set 설정, 지지구간 스케일 P32 헬퍼 |
| `summarize_setwise_trace_statistics.py` | set별 trace 길이/개수 통계 |
| `estimate_p32_mc_calibrated.py` | **최종 P32 추정** (observed_P21 / unit_p32_forward_mc 보정계수) |
| `export_setwise_3d_traces.py` | DFN(h5) + 거친 막장면 mesh → 3D trace 데이터셋 순방향 생성 |
| `generate_synthetic_rough_face_mesh.py` | 합성 거친 막장면 mesh 생성 (검증용 입력) |
| `generate_conditional_hidden_dfn.py` | 조건부 hidden DFN 생성 (remove-and-resample) |
| `visualize_conditional_dfn_3d.py` | PyVista 3D 시각화 (disc + 막장면 + 관측/조건화 trace) |

---

## 중요 정책 / 주의사항
- **GT(ground truth)는 검증 전용**입니다. 추정기 설계·보정에 truth를 사용하지 않습니다 (정책 D013).
- **하나의 parsimonious 추정기**를 모든 set에 동일 적용합니다. set별 보정계수·fudge factor 금지.
- **Laxemar Set 4는 지수분포**라 멱법칙 역산·조건화에서 제외됩니다 (D002).
- 최종 P32는 `unit_p32_forward_mc` 방식입니다 (D004). combined_bootstrap은 대체 방식으로 제외.
- 반지름/kr은 `estimate_radius_powerlaw_window_mc.py`(v4.1 창 MC)가 최종입니다. v3(from_traces)는 제외.

## 경로 의존성 (수정 필요)
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
  --calibration-factor-mode unit_p32_forward_mc --outcsv <out.csv>
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
- **복원 단계**(reconstructed_discs.csv 생성)는 이 핸드오프에 없음 → 조건화 입력을 별도 확보해야 함.

## 실행 환경
- Python 3.9+ / numpy, scipy, h5py, matplotlib, pyvista
