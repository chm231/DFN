# DFN 역산 파이프라인 데모 — 설치·실행 안내

이 패키지는 보고서 제6장의 전체 파이프라인(합성 DFN 생성 → 막장면 절리선 →
파라미터 역산 → 관측 균열 복원 → 조건부 암반균열망 → 안정성 해석 입력 JSON)을
한 명령으로 시연하기 위한 것입니다.

## 1. 요구 환경

- Windows 10/11, Python 3.10 이상 (3.13까지 확인)
- 인터넷 연결 (최초 설치 시 라이브러리 다운로드)
- 필요 라이브러리: numpy, scipy, h5py, matplotlib, pyvista(3차원 시각화용, 선택)

## 2. 설치

압축을 푼 폴더에서 `install_demo.bat` 을 더블클릭(또는 명령창에서 실행)합니다.
가상환경(.venv)을 만들고 requirements.txt 의 라이브러리를 설치합니다.

인터넷이 안 되는 환경이라면, 인터넷이 되는 PC에서
`pip download -r requirements.txt -d wheels` 로 wheels 폴더를 만들어 함께 복사한 뒤
`.venv\Scripts\pip install --no-index --find-links wheels -r requirements.txt` 로
설치할 수 있습니다.

## 3. 실행

```
run_demo.bat --seed 2026
```

명령창에서 직접 실행할 경우:

```
.venv\Scripts\python.exe run_demo.py --seed 2026
```

주요 옵션:

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--seed` | 2026 | DFN 생성 난수 시드. 바꾸면 다른 합성 암반이 생성됩니다 |
| `--face-x` | 0 1 2 3 | 관측 막장면 x 위치 [m] |
| `--n-bootstrap` | 100 | k_r 신뢰구간용 bootstrap 반복 수. 0이면 생략(실행 단축) |
| `--rmax-local` | 25 | 조건부 생성의 국소 반지름 상한 [m] |
| `--skip-3d` | - | 3차원 시각화 생략 |
| `--p32-fisher` | preset | P32 환산계수의 방향분포 소스. `preset` = 사이트 방향 모델 표(보고서 표 6-5 수치 재현), `estimated` = 이번 실행에서 역산한 방향분포 |

**추천 시드: 42** — 보고서 제6장 검증(표 6-5)에 사용된 벤치마크 DFN과 동일한
암반이 생성되어, 절리선 수(170/39/132/171)·k_r·kappa·방향 결과가 보고서와 그대로
일치합니다. 다른 시드는 새로운 합성 암반이며, 단일 실현의 역산 산포(보고서 제7절의
시드 민감도 수준)가 결과에 그대로 나타납니다.

기본 설정 실행 시간은 수 분 이내입니다(대부분 k_r bootstrap).
`--n-bootstrap 0 --skip-3d` 로 줄이면 1분 안팎입니다.

## 4. 단계 구성 (보고서 제6장 대응)

| 단계 | 내용 | 보고서 |
|---|---|---|
| 1 | 합성 DFN 생성 (참값 파라미터, 시드 지정) | 제1절 2.1 |
| 2 | 평면 막장면 × 터널 단면 다각형 → 절리선 데이터셋 | 제2절 |
| 3 | k_r 역산 (하이브리드 우도, lmin_fit 0.5 m 고정) | 제3절 2 |
| 4 | 절리군별 방향·kappa 정리 → dataset config | 제3절 1 |
| 5 | P32 역산 (해석식 C = E[sin phi]) | 제3절 3 |
| 6 | 관측 균열 복원 (하드 게이트 연결 + 원판 복원) | 제4절 |
| 7 | 조건부 암반균열망 생성 (관측 모순 균열 제거) | 제5절 |
| 8 | 안정성 해석 입력 JSON 추출 (표 6-4 형식) | 제5절 3 |
| 9 | 3차원 시각화 (관측/미관측 균열 구분) | - |

참값(벤치마크 파라미터)은 마지막 비교표 출력에만 사용되며, 역산 과정에는 입력되지
않습니다(보고서 제6절 1의 설계 원칙).

## 5. 출력물 (demo_output/seed<시드>/ 아래)

- `dfn_export_for_python.h5` — 생성된 합성 DFN
- `trace_dataset/` — 막장면별 절리선 데이터셋 (h5/csv)
- `kr/` — k_r 프로파일 우도, 요약표, bootstrap 신뢰구간
- `p32/p32_summary.csv` — P32 역산 결과 (환산계수 C 포함)
- `reconstruct/reconstructed_discs.csv` — 복원 원판
  (radius_status: determined / shrinkage / lower_bound)
- `conditional_hidden/` — 조건부 암반균열망 (관측 ∪ 미관측) csv + 진단 그림
- `export/dfn_domain_*.json` — 안정성 해석 모듈 입력 JSON
- `export/domain_dfn_3d_*.png` — 3차원 뷰 (pyvista 설치 시)
- `comparison_vs_laxemar.csv` — 참값 대비 역산 비교표

## 6. 문제 해결

- **pyvista 설치·실행 실패**: 3차원 시각화(9단계)만 생략되고 나머지는 정상
  동작합니다. `--skip-3d` 로 명시적으로 끌 수도 있습니다.
- **한글 폰트 경고**: 그림의 한글 라벨은 Windows 기본 폰트(Malgun Gothic)를
  사용합니다. 다른 OS에서는 폰트 경고가 나올 수 있으나 실행에는 지장이 없습니다.
- **같은 시드 재실행**: 출력 폴더를 삭제하거나 `--outdir` 로 다른 폴더를 지정하세요.
