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

## 실행 환경
- Python 3.9+ / numpy, scipy, h5py, matplotlib, pyvista
