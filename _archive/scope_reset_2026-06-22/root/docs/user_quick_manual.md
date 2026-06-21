# 3D DFN Modeling 사용자용 간단 매뉴얼

## 1. 이 문서로 할 수 있는 것

이 문서는 처음 사용하는 사람이 바로 실행할 수 있도록 가장 자주 쓰는 명령어만 짧게 정리한 빠른 시작 가이드다.

주요 작업은 아래 4가지다.

1. 3D DFN에서 바로 블록 검출
2. 3D 터널-절리 관계 시각화
3. 2D trace 맵 확인
4. trace 기반 3D DFN 재구성

---

## 2. 시작하기

PowerShell에서 프로젝트 폴더로 이동한다.

```powershell
Set-Location "C:\Users\user\OneDrive\2026-1\3D DFN modeling"
```

가상환경이 있으면 활성화한다.

```powershell
.\.venv\Scripts\Activate.ps1
```

활성화가 안 되면 아래처럼 Python 경로를 직접 써도 된다.

```powershell
& ".\.venv\Scripts\python.exe" --version
```

---

## 3. 기본 입력 파일

보통 아래 파일들을 사용한다.

- DFN HDF5: `storage\data\dfn_export_for_python.h5`
- 터널 경계 DAT: `storage\data\tunnel_boundary.dat`

주의:

- 블록 검출과 3D 시각화는 HDF5 안에 `/tunnel/poly_YZ` 정보가 있어야 한다.
- 좌표계는 `x=터널 진행 방향`, 터널 단면은 `YZ` 평면이다.

---

## 4. 가장 많이 쓰는 명령어

### 4.1 블록 검출

가장 기본적인 실행 명령이다.

```powershell
& ".\.venv\Scripts\python.exe" "dfn_analysis\run_dfn_pipeline.py" `
  --input "storage\data\dfn_export_for_python.h5"
```

조금 더 자주 쓰는 옵션까지 포함하면 아래와 같다.

```powershell
& ".\.venv\Scripts\python.exe" "dfn_analysis\run_dfn_pipeline.py" `
  --input "storage\data\dfn_export_for_python.h5" `
  --voxel_size 0.5 `
  --tol_factor 0.6 `
  --min_voxels 8 `
  --connectivity 6 `
  --outdir "storage\output\results"
```

GPU를 쓰지 않고 CPU로만 돌리려면:

```powershell
& ".\.venv\Scripts\python.exe" "dfn_analysis\run_dfn_pipeline.py" `
  --input "storage\data\dfn_export_for_python.h5" `
  --no_gpu
```

결과는 보통 아래 폴더에 저장된다.

- `storage\output\results`

대표 결과 파일:

- `block_summary.json`
- `block_results.h5`
- `block_labels.npy`
- `block_overview_*.png`

---

### 4.2 3D 터널-절리 시각화

터널과 절리의 공간 관계를 먼저 보고 싶을 때 쓴다.

```powershell
& ".\.venv\Scripts\python.exe" "dfn_analysis\plot_3d_tunnel_fractures.py" `
  --input "storage\data\dfn_export_for_python.h5" `
  --mode intersect
```

`--mode` 옵션:

- `all`: 터널과 관련된 절리 전체
- `intersect`: 터널 경계와 교차하는 절리만
- `inside`: 터널 내부에 완전히 포함되는 절리만

---

### 4.3 2D trace 맵 보기

특정 `x` 위치에서 `YZ` 평면 trace를 슬라이더로 확인한다.

```powershell
& ".\.venv\Scripts\python.exe" "trace_analysis\plot_2d_trace_map.py" `
  --input "storage\data\dfn_export_for_python.h5"
```

이 창에서는 다음을 빠르게 확인할 수 있다.

- trace 개수
- trace 총 길이
- `x` 위치별 trace 분포 변화

---

### 4.4 trace 기반 3D DFN 재구성

2D trace 정보를 바탕으로 3D DFN을 재구성하는 파이프라인이다.

```powershell
& ".\.venv\Scripts\python.exe" "trace_analysis\run_trace_to_dfn.py" `
  --input "storage\data\dfn_export_for_python.h5" `
  --tunnel-dat "storage\data\tunnel_boundary.dat"
```

블록 검출까지 이어서 하고 싶으면:

```powershell
& ".\.venv\Scripts\python.exe" "trace_analysis\run_trace_to_dfn.py" `
  --input "storage\data\dfn_export_for_python.h5" `
  --tunnel-dat "storage\data\tunnel_boundary.dat" `
  --x-start 0.0 `
  --x-end 6.0 `
  --advance-step 3.0 `
  --sa-iterations 150 `
  --output-dir "storage\output\reconstruction_results" `
  --run-block-detector
```

대표 결과 폴더:

- `storage\output\reconstruction_results`

대표 결과 파일:

- `reconstructed_dfn.h5`
- `trace_side_by_side_comparison.png`
- `trace_overlay_comparison.png`

---

## 5. 자주 조절하는 옵션

### 블록 검출에서

- `--voxel_size`
작을수록 정밀하지만 느려진다.

- `--tol_factor`
절리 voxel 두께 판정에 영향이 있다.

- `--min_voxels`
너무 작은 블록을 제거할 때 쓴다.

- `--connectivity`
현재 기본은 `6`이다. 결과 블록 수가 달라질 수 있다.

### 재구성에서

- `--x-start`, `--x-end`
분석할 터널 구간

- `--advance-step`
굴착면 간 거리

- `--sa-iterations`
최적화 반복 수

---

## 6. 문제가 생기면 먼저 확인할 것

### `python` 명령이 안 먹는 경우

아래처럼 직접 실행한다.

```powershell
& ".\.venv\Scripts\python.exe" "dfn_analysis\run_dfn_pipeline.py" --input "storage\data\dfn_export_for_python.h5"
```

### GPU 오류가 나는 경우

CPU 모드로 먼저 확인한다.

```powershell
& ".\.venv\Scripts\python.exe" "dfn_analysis\run_dfn_pipeline.py" `
  --input "storage\data\dfn_export_for_python.h5" `
  --no_gpu
```

### 블록 수가 너무 이상한 경우

우선 아래 4개를 점검한다.

- `voxel_size`
- `tol_factor`
- `min_voxels`
- `connectivity`

### 3D 창이 안 뜨는 경우

다음을 확인한다.

- `pyvista` 설치 여부
- 그래픽 환경 사용 가능 여부

---

## 7. 가장 빠른 추천 사용 순서

처음이라면 아래 순서가 가장 안전하다.

1. `plot_3d_tunnel_fractures.py`로 기하 관계 확인
2. `plot_2d_trace_map.py`로 trace 분포 확인
3. `run_dfn_pipeline.py`로 블록 검출 실행
4. 필요하면 `run_trace_to_dfn.py`로 재구성 수행

---

## 8. 한 줄 요약

블록 검출만 빨리 하려면 아래 명령부터 실행하면 된다.

```powershell
& ".\.venv\Scripts\python.exe" "dfn_analysis\run_dfn_pipeline.py" `
  --input "storage\data\dfn_export_for_python.h5" `
  --voxel_size 0.5 `
  --tol_factor 0.6 `
  --min_voxels 8 `
  --connectivity 6 `
  --outdir "storage\output\results"
```
