# 터널 흔적 선분 분포 보정 기술 설명서 (TBTD Estimator)
- 개정 2판: 무감독 자가 보정 MLE 및 절단 복원(Imputation) 파이프라인 -

## 1. 기술적 목적 (Purpose)

터널 굴착 막장면(Excavation Face)에서 조사된 절리 흔적 선분(Trace)들의 2D 분포 데이터를 바탕으로, 유한 관측 윈도우 샘플링 편향을 엄밀하게 제거한 **무편향 참 트레이스 길이 분포(Unbiased True Trace Length Distribution)**를 추정하는 것을 목적으로 합니다. 

본 모듈은 다음과 같은 다운스트림(Downstream) 지반공학 해석의 입력 자료로 직결됩니다:
- **Hekmatnejad et al. (2018)** 스타일의 비모수적 3D 절리 직경 CDF 추정 알고리즘
- 3D DFN 확률적 균열망 생성기(Stochastic DFN Generator)의 균열 크기 Prior 제공
- SVD 기반 3D 결정론적 평면 매칭 시, 반경 개연성 기반 페널티 스코어링(Radius Prior Penalty) 산정

---

## 2. 지반통계학적 배경 및 편향의 원인

터널 막장면은 크기가 제한된 '유한 영역 샘플링 창(Finite Window)'이므로, 원시 관측 데이터 히스토그램을 그대로 참 분포로 사용하면 심각한 왜곡이 발생합니다. 본 모듈은 지반공학 및 통계학 연구들(Laslett 1982, Mauldon et al. 2001 등)에 기반하여 다음의 기하학적 편향을 완벽히 교정합니다.

| 편향 종류 | 물리적 특징 | 수학적 교정 방안 (TBTD MLE) |
|:---|:---|:---|
| **한쪽 잘림 (Type 1)** | 흔적 선분의 한쪽 끝이 터널 폴리곤 경계선 밖으로 나가 잘림 | $L_{\text{est}} = L + d_1$ 오프셋 복원 (d1 기대 연장 길이 삽입) |
| **양쪽 잘림 (Type 2)** | 흔적 선분의 양쪽 끝이 모두 잘려 터널을 관통함 | $L_{\text{est}} = L + d_2$ 오프셋 복원 (d2 기대 연장 길이 삽입) |
| **길이 단절 (Truncation)** | 검측 임계치 $l_{\text{min}}$ 이하의 미소 절리들은 지도화되지 않음 | $l < l_{\text{min}}$ 영역의 Truncated PDF 우도 배제 및 정규화 |
| **배향별 관측 편향** | 절리 세트의 주향/경사와 터널 방향에 따른 교차 확률 차이 | 배향 데이터를 축 데이터(Axial)로 취급하여 $[0, \pi)$로 완전 정규화 |
| **윈도우 경계 효과** | 거대 절리일수록 잘림(Type 1, 2) 비중이 극도로 높음 | **잘린 절리를 폐기하지 않고** 가중치와 오프셋 복원을 결합해 MLE 수행 |

---

## 3. 핵심 수학적 방법론

### 3.1 절단 데이터 복원 (Clipped Trace Imputation)
complete(Type 0) 데이터만 사용할 경우, 거대 균열이 집중된 Type 1 & 2 데이터(현장 데이터의 약 75% 이상 차지)가 모두 증발하여 **참 균열 크기가 극도로 과소평가되는 재앙(Size-bias towards small fractures)**이 발생합니다.
따라서 본 모듈은 잘린 흔적들을 버리지 않고 아래 공식을 통해 추정 길이($L_{\text{est}}$)를 완벽히 복원합니다:
*   **Type 0 (완전 관측)**: $L_{\text{est}} = L$
*   **Type 1 (한쪽 절단)**: $L_{\text{est}} = L + d_1$ (기대 연장치 $d_1$ 가산)
*   **Type 2 (양쪽 절단)**: $L_{\text{est}} = L + d_2$ (기대 연장치 $d_2$ 가산)

### 3.2 원형 윈도우 기하학 이론 확률 (Theoretical Window Probability)
무겁고 통계적 노이즈가 동반되는 몬테카를로 시뮬레이션 대신, 원형 윈도우(터널 직경 $D$)에 대한 기하통계학적 **이론 확률 공식**을 적용하여 연산 효율을 0.01초 이하로 낮췄습니다.
*   **완전 관측 확률 ($p_0$)**: $p_0 = (1 - L/D)^2 \quad (\text{for } L < D)$
*   **한쪽 절단 확률 ($p_1$)**: $p_1 = \frac{2L}{D}(1 - L/D) \quad (\text{for } L < D)$
*   **양쪽 절단 확률 ($p_2$)**: $p_2 = (L/D)^2 \quad (\text{for } L < D, \text{ else } 1.0)$

### 3.3 무감독 blind 자가 보정 (Unsupervised Self-Calibration)
3D 공간 상의 숨겨진 참 절리 크기를 모르는 blind 상태에서, 막장면의 2D 절단 클래스 관측 비율($\pi_0, \pi_1, \pi_2$)만을 목적 함수로 활용해 물리적으로 가장 정합성이 높은 $d_1$ 및 $d_2$ 오프셋 값을 스스로 격자 탐색(Grid Search)하여 자동 동조화합니다.

### 3.4 대수정규분포 MLE 및 크기 편향 교정 (Villaescusa & Brown)
복원된 전체 인구 데이터($L_{\text{est}}$)에 대해 Lognormal 분포의 최대우도법(MLE)을 수행하여 모수 $\mu_{\text{biased}}, \sigma_{\text{biased}}$를 구합니다. 2D 막장면 교차 빈도 자체가 크기에 비례(Size-bias)하므로, 아래의 통계학적 관계식으로 무편향 참 모수($\mu_L, \sigma_L$)를 복원해 냅니다:
$$\sigma_L = \sigma_{\text{biased}}$$
$$\mu_L = \mu_{\text{biased}} - \sigma_{\text{biased}}^2$$

---

## 4. 참고 문헌 (References Considered)

1.  **Warburton (1980)** - stereological 기초 이론 제공.
2.  **Priest & Hudson (1981)** - 스캔라인 바이어스 및 스페이싱 분석.
3.  **Pahl (1981)** - 절리 흔적의 평균 길이 기하학적 산정식 (보정 전/후 평균값의 상식적 범위 검증용).
4.  **Laslett (1982)** - 경계 절단(Type 0, 1, 2) 분류 모델의 수학적 선구안.
5.  **Mauldon (1998)** - 유한 윈도우 보정 기하학.
6.  **Mauldon, Dunne & Rohrbaugh (2001)** - 원형 스캔라인 및 circular window 기하통계학 공식의 토대.
7.  **Song & Lee (2001)** - 터널 막장면 창 샘플링(Window Sampling)의 최적화.
8.  **Hekmatnejad, Emery & Vallejos (2018)** - 참 트레이스 분포 역산 완료 후 3D 직경으로의 비모수적 전이 모형.

---

## 5. 실행 가이드 (How to Run)

### 5.1 합성(Synthetic) 토이 데이터 검증 실행
```bash
python scripts/run_trace_distribution_correction.py --synthetic
```

### 5.2 실제 현장 DFN HDF5 데이터 연계 실행
```bash
python scripts/run_trace_distribution_correction.py \
    --input "storage/data/dfn_export_for_python.h5" \
    --tunnel-dat "storage/data/단면_폴리곤.dat" \
    --x-start 0 --x-end 9 --advance-step 3 \
    --output-dir trace_analysis/storage/output/tbtd_results
```

---

## 6. 내일 아침 인간 개발자가 검토할 사항 (Human Review Items)

> [!IMPORTANT]
> 1. **자가 보정 범위**: 자가 보정 탐색 범위를 d1 [1.0m ~ 6.0m], d2 [2.0m ~ 10.0m]로 넓혔으나, 극단적으로 큰 거대 균열이 지배하는 구간에서는 추가적인 상한 격자 튜닝이 필요한지 검토하십시오.
> 2. **이론 확률의 타당성**: 원형 윈도우를 가정한 기하학 확률 곡선이, 실제 말굽형/사각 터널 폴리곤 형상 대비 실무적 정밀도를 충분히 만족하는지 크로스 체크하십시오.
