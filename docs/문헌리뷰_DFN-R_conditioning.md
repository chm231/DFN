# 문헌 리뷰: DFN-R 조건화(Conditioning) 방법론과 ONKALO 적용

**대상 문헌**

| 약칭                | 서지 정보                                                                                                                                                                                                                                                                                               | 역할                                                       |
| ------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------- |
| **R-17-11**   | Appleyard P, Jackson P, Joyce S, Hartley L, 2018.*Conditioning discrete fracture network models on intersection, connectivity and flow data.* SKB R-17-11, Svensk Kärnbränslehantering AB. ISSN 1402-3091, February 2018.                                                                           | **방법론 원전** (ConnectFlow/Amec Foster Wheeler 팀) |
| **Report 07** | Baxter S, Appleyard P, Hartley L, Hoek J, Williams T, 2018.*Exploring conditioned simulations of discrete fracture networks in support of hydraulic acceptance of deposition holes — Application to the ONKALO demonstration area.* Posiva SKB Report 07, SKB & Posiva Oy. ISSN 2489-2742, May 2018. | **실적용/검증** (ONKALO DT2)                         |

> **두 문헌의 관계.** R-17-11이 조건화 *방법*을 개발·검증한 원전이고, Report 07이 그 방법을 ONKALO 실증터널 2(DT2)의 실측 데이터에 *적용*한 후속 보고서다. 실제로 Report 07 본문(43쪽 인근)은 방법 출처를 **"Appleyard et al. 2018, Bym and Hermanson 2018"** 으로 인용하는데, 여기서 *Appleyard et al. 2018 = R-17-11*이다. (Bym & Hermanson 2018은 동일 DFN-R 프로젝트의 FracMan/Golder 편 결론 보고서로, R-17-11과 짝을 이루는 두 결론 보고서 중 하나다.)

---

## 1. R-17-11 — 조건화 방법론 (ConnectFlow)

### 1.1 문제 정의: 관측의 비유일성(non-uniqueness)

DFN-R("R"=Repository) 프로젝트의 목표는 지표·시추공 기반의 **비조건화(unconditioned)** 확률 DFN을, 터널·파일럿공·처분공에서 얻은 **국소 관측에 정합**시키는 것이다. 핵심 난점은 명확하다 (R-17-11 §1.1):

> 하나의 교차(intersection) 관측은 균열의 **방향(orientation)은 결정**하지만, **크기와 중심 위치는 유일하게 결정하지 못한다.** 동일한 트레이스가 "중심이 가까운 작은 균열"에서도, "중심이 먼 큰 균열"에서도 나올 수 있다.

→ **이는 우리 프로젝트의 disc 복원이 마주한 문제와 정확히 동일하다.** (경계점이 원호를 충분히 덮지 못하면 반지름이 하한만 결정됨 → 우리의 `arc-span gate`/`lower_bound` 처리와 같은 인식.)

### 1.2 방법: 라이브러리(library) + 근접도(closeness-of-fit)

저자들은 세 가지 후보 접근을 비교한 뒤 세 번째를 채택한다 (§1.4):

1. **해석적(analytical)** — 관측에 대응하는 균열 속성 분포를 수학적으로 유도 → 원형-원통 등 이상화된 특수 경우만 가능, 일반화 불가.
2. **무차별 대입(brute force)** — 관측마다 분포에서 균열을 재표집해 정합될 때까지 반복 → 계산비 과다.
3. **라이브러리(채택)** — 확률 DFN을 다수 실현하여 **터널을 교차하는 균열들의 경험적 분포를 라이브러리로 1회 구축**, 이후 각 관측에 대해 라이브러리에서 "충분히 정합하는" 균열을 **유사도 가중으로 추출**해 삽입.

핵심 절차 (§2):

- **2단계 조건화**: (1) 관측 개구부를 교차하는 기존 확률 균열 제거 → (2) 각 관측 트레이스에 대해 라이브러리 검색·삽입.
- **정상성(stationarity) 가정**: 도메인 내 균열 분포가 정상적이고 속성이 상호 독립이면, 라이브러리를 재사용해 동일 통계의 어떤 모델에도 조건화 가능.
- **불편성(unbiasedness) 검증**: 조건화가 사전(a priori) 분포(강도·크기·방향)를 유의하게 왜곡하지 않아야 한다 → 라이브러리가 원 DFN recipe와 같은 통계에서 나오므로 국소 정합을 하면서도 전역 통계를 보존.
- **유량(flow) 조건화**: 기하만이 아니라 유입량/주입시험(specific capacity)까지 정합 → 처분공 폐쇄 후 유동 예측의 불확실성을 추가로 감소.

### 1.3 검증: HypoSite BM-1b (합성 진실)

SKB의 합성 "가상 현실" **HypoSite**(모든 균열이 알려진 DFN 실현)로 검증. 관측 가능한 데이터(교차·유입량)만 "관측"으로 쓰고 나머지는 미지로 가정 → 예측-실측 비교가 가능.

- P21, 균열 위치·방향, P32, 크기 분포 일치성 검사 통과.
- 유량 데이터를 포함하면 처분공 주변 **폐쇄 후 유동률(U), 유동관련 이동저항(F)** 예측이 비조건화 대비 크게 정확. 기하+유량 조건화 > 기하만 조건화 > 비조건화 순.

### 1.4 한계 / 미해결 (§4.6, §5)

- **"drill/use 판단으로의 연결"** 미해결: 조건화는 분포를 좁혀 유용한 입력을 주지만, 처분공 채택 여부에 대한 명확한 yes/no를 주지 않음.
- **실현 선별(screening)** 방법의 최적성 불명확: 관측과 유사한 실현만 골라 예측력을 높이는 방식(§4.3)이 최선인지 미확정.
- **모델 규모 한계**: 소규모 배열이라 정상성 가정의 이점이 충분히 발현되지 않음; 여러 터널을 관통하는 큰 균열의 기하가 완전히 확정되지 못함.
- 실제 현장 데이터를 이상화 모델로 변환하는 단계는 **범위 밖**(별도 연구, Baxter et al. 2016).

---

## 2. Report 07 — ONKALO Demonstration Area 적용

### 2.1 목적

R-17-11 방법을 **ONKALO 실증터널 2(DT2) 하부 6개 실증 처분공**에 적용하여:

1. 처분공 위치별 **폐쇄 후 유동·이동 특성 예측**과 수리적 **수용기준(acceptance criteria)** 정의 지원.
2. Demonstration Area의 개념/수치 DFN 모델(**DADFN**, Hartley et al. 2017)을 **예측-실측(prediction-outcome)** 방식으로 검증.

### 2.2 방법상 추가된 것 (R-17-11 대비)

- **실측 데이터 적용**: DT2/DT1 터널면 디지털 트레이스, DT2 바닥 파일럿공(ONK-PP379~384) 균열 로그, 주입시험·유입량, 실증 처분공(ONK-EH) 프로파일 등 — R-17-11이 "범위 밖"으로 남긴 실데이터→모델 변환을 실제로 수행.
- **근접도 측정식(closeness-of-fit, Eq 4-1) 명시**: 개별 측정치 $M_i$와 계수 $a_i$의 유클리드 노름

$$
M = \sqrt{\textstyle\sum_i a_i^2 M_i^2}
$$

  트레이스용 최대 10개 지표(R-17-11), 시추공용 4개 중, 본 연구는 wall angle·relative length·trace angle·(flow)·... 을 사용. **$M>5$ 이거나 개별 $M_i$가 상한 초과면 후보에서 제외.** 길이·유량 지표에 높은 가중(유동 예측의 지배 인자).

- **5개 모델 비교**: 비조건화(UC) 1개 + 조건화 4형(파일럿공/처분공 × 구조/수리).

### 2.3 결과

- **비조건화 DADFN도 상당히 신뢰**: P21의 공간 경향(BFZ 근접 고강도, 특정 도메인 저강도)을 사전지식 없이 재현; 주입 specific capacity의 기하평균이 실측과 반차수(half-order) 이내, 검출한계 이하 구간도 정확히 예측.
- **조건화가 국소 불확실성 감소**: 파일럿공·처분공 관측에 조건화하면 인근 처분공 유동 예측 산포가 UC 대비 축소; 유량 포함 조건화가 기하만보다 우수.
- **주입시험 vs 유입량 비대칭**: 주입 specific capacity는 유동망의 *최대* 용량(보수적 상한), 개방 처분공 유입량은 다른 개구부로의 배출로 축소될 수 있음 → 폐쇄 후 유동은 거의 항상 주입 기반 추정 *아래*(보수적 예측자).
- 폐쇄 후 유동은 처분공 간 3~4 차수 변동, 개방시 유동 측정치와는 약한 상관.

### 2.4 한계

- BFZ(취성단층대) 근접 처분공 예측 신뢰도는 **손상대(damage zone) 기하 모델링에 강하게 의존** (EH15처럼 조건화가 오히려 과대예측을 유발하는 경우 존재 → 국소 손상대가 비전형적일 때).
- 소규모 실증 배열이라 R-17-11과 동일하게 정상성 이점이 제한적.

---

## 3. 우리 프로젝트(2D 트레이스 → 3D DFN)와의 연결

| 항목                       | DFN-R (R-17-11 / Report 07)                                                            | 본 프로젝트                                                                      | 관계                                                                   |
| -------------------------- | -------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------- | ---------------------------------------------------------------------- |
| **비유일성 문제**    | 교차 관측이 방향은 정하나 크기·중심 미결정                                            | arc-span 부족 시 반지름 하한만 결정                                              | **동일 인식**                                                    |
| **정합 척도**        | closeness-of-fit$M=\sqrt{\sum a_i^2 M_i^2}$ (wall angle, 길이, trace angle, flow …) | 매칭 cost $=\Delta\theta + w_{off}\cdot\text{offset} + w_{lnr}\cdot|\ln(L_i/L_j) | $                                                                      |
| **미관측 균열 처리** | 라이브러리(경험적 분포)에서 유사도 가중 추출 → 통계 보존                              | kr-조건부 posterior로 잔여 강도 확률 생성                                        | **동일 철학**(사전 통계 보존하며 관측 정합)                      |
| **결정론/확률 전이** | 관측 근방 결정론↑, 원거리 확률론 유지                                                 | 결정론 disc 복원 +`orientation_only` 반결정론 + stochastic 보완 3계층          | **직접 대응** (`orientation_only` = semi-deterministic)        |
| **관측 형상**        | 단일 터널면/시추공 교차 (점·선)                                                       | **연속 4개 막장면**(트레이스 스택)                                         | **우리의 신규 기여점** — DFN-R은 다중 연속면 정합을 다루지 않음 |

**핵심 시사점**

1. **우리 방향은 SKB 표준 방법론과 같은 계열**임이 확인된다. closeness-of-fit(가중 유사도), library(통계 보존형 확률 보완), deterministic–stochastic transition이 모두 우리 3계층 복원과 대응한다.
2. **DFN-R의 미해결 지점이 곧 우리 차별점**: 이들은 단일 개구부 교차에서 크기·중심의 비유일성을 "라이브러리 확률 추출"로 *회피*할 뿐 개별 균열을 유일하게 복원하지 못한다. 우리는 **연속 4개 막장면의 chain 정합(offset 선형성)** 으로 그 비유일성을 실제로 *축소*한다(순수 track 잔차 0.005 m vs 오연결 0.293 m, 60배 분리 — 별도 정량화 결과). 이것이 SSTI 워크플로우의 독창성 근거다.
3. **적용 시 유의**: DFN-R은 유동(flow) 정합을 중심에 두지만 본 프로젝트는 기하 복원이 1차 목표다. 유량 데이터가 없으므로 closeness-of-fit에서 flow 항을 제외한 기하 전용(geometric conditioning) 형태에 대응한다.

---

## 4. 참고문헌 (보고서 삽입용)

- Appleyard, P., Jackson, P., Joyce, S., & Hartley, L. (2018). *Conditioning discrete fracture network models on intersection, connectivity and flow data.* SKB R-17-11. Svensk Kärnbränslehantering AB.
- Baxter, S., Appleyard, P., Hartley, L., Hoek, J., & Williams, T. (2018). *Exploring conditioned simulations of discrete fracture networks in support of hydraulic acceptance of deposition holes — Application to the ONKALO demonstration area.* Posiva SKB Report 07. SKB & Posiva Oy.
- (짝 문헌) Bym, T., & Hermanson, J. (2018). *[DFN-R FracMan/Golder 편 결론 보고서].* — 정확한 서지는 원문 표지에서 확인 요망.
