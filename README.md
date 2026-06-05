# Tunnel-DFN Analysis & Block Detection Suite

본 프로젝트는 3D DFN(Discrete Fracture Network)과 터널 기하구조를 결합하여 암반 블록의 안정성을 분석하는 통합 파이프라인입니다. 2D Trace 기반의 역산(Direction B)과 직접적인 DFN 분석(Direction A)을 모두 지원합니다.

## 📂 디렉토리 구조 (3-Part Architecture)

### 1. `dfn_analysis/` (Direct Analysis & Core Engine)
**DFN 생성 데이터를 바탕으로 직접 블록을 탐지하고 시각화하는 패키지**입니다. 공통으로 사용되는 핵심 엔진 모듈들이 포함되어 있습니다.
- `run_dfn_pipeline.py`: 메인 실행 스크립트.
- `block_detector.py`: GPU 가속 복셀 분류 및 CCA 알고리즘 코어.
- `tunnel_geometry.py`: 터널 3D 복셀화 모듈.
- `visualize_blocks.py`: PyVista 및 Matplotlib 기반 시각화 엔진.
- `export_blocks.py`: 분석 결과(CSV, JSON) 내보내기.

### 2. `trace_analysis/` (Inverse Reconstruction Pipeline)
**막장면 2D Trace 데이터를 역산하여 3D 절리면을 복원하고 블록을 분석하는 패키지**입니다.
- `run_trace_pipeline.py`: Trace 기반 역산 메인 실행 스크립트.
- `load_tunnel_dat.py`: 터널 설계 데이터(.dat) 로더.
- `slab_reconstruction/`: [NEW] Slab 기반 평면 복원 알고리즘 테스트 베드.
  - `run_slab_pipeline.py`: Slab 방식 평면 복원 및 정밀도 검증 스크립트.
- `trace_reconstruction/`: 추적 매칭, SVD 기반 평면 복원 알고리즘 서브패키지.
- `plot_2d_trace_map.py`: 2D Trace 분포 인터랙티브 뷰어.

### 3. `storage/` (Data & Results Archive)
**모든 입력 데이터와 출력 결과물을 일괄 관리하는 폴더**입니다.
- `data/`: `.h5` (DFN 데이터), `.dat` (터널 설계), `.csv` (입력 Trace) 등.
- `output/`: 각 파이프라인 실행 결과(이미지, 분석 리포트, 통계 데이터) 저장.

---

## 🚀 실행 가이드

### Package A: 직접 DFN 블록 탐지
```powershell
$env:PYTHONPATH = "."
& python dfn_analysis/run_dfn_pipeline.py --input storage/data/dfn_export_for_python.h5
```

### Package C: Slab 기반 평면 복원 검증 (Direction B 전용)
```powershell
$env:PYTHONPATH = "."
& python trace_analysis/slab_reconstruction/run_slab_pipeline.py --spacing 3.0 --thickness 0.2
```

---

## 💡 개발 및 활용 원칙
1. **코드 상속**: `trace_analysis`는 핵심 연산을 위해 `dfn_analysis` 내부의 Core 모듈들을 참조합니다.
2. **데이터 격리**: 모든 원본 데이터는 `storage/data`에 위치하며, 소스 코드 내부에서는 상대 경로를 통해 접근합니다.
3. **GPU 활용**: `cupy` 환경이 설정된 경우 Voxel 연산 시 GPU 가속이 자동으로 적용됩니다.
