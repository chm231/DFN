# GPU Accelerated 3D DFN Block Detection – Python Architecture Summary

이 문서는 AI가 매호출마다 무거운 파이썬 파일 전체를 읽지 않고도, 코드베이스의 아키텍처와 흐름을 즉시 파악(크레딧 및 메모리 최적화)하도록 설계된 **핵심 요약 가이드**입니다.

## 📂 파일별 핵심 역할 (Python Modules)

### 1. `detect_blocks_gpu.py` (Main Pipeline)
- **역할:** 터널/DFN 블록 탐지 메인 실행부 (CLI Entry point).
- **흐름:** 
  1. HDF5 데이터 로드 (`dfn_export_for_python.h5`)
  2. `tunnel_geometry.py` 호출 → 터널 복셀 마스크 생성
  3. `block_detector.py` 호출 → 복셀 분류(ROCK, FRACTURE, TUNNEL) 및 CCA 실행
  4. 결과(블록 라벨 및 메타)를 HDF5/JSON에 기록
  5. `visualize_blocks.py` 호출 → 요약 이미지 저장 및 PyVista 3D 인터랙티브 창 실행

### 2. `block_detector.py` (Core Algorithm)
- **역할:** GPU 가속 복셀 연산 & CCA 알고리즘 코어.
- **주요 함수:**
  - `classify_voxels`: 균열 원판의 방정식과 AABB(Bounding Box)를 이용해 각 복셀의 상태 할당. (CuPy 활용 병렬화)
  - `run_cca`: SciPy/CuPy의 `ndimage.label`을 이용하여 서로 끊어진 암반 덩어리에 고유 식별 번호 부여 (`--connectivity 6` 혹은 `26` 지원).
  - `filter_and_stat_blocks`: 팽창(Dilation) 연산을 통해 **터널과 닿아 있으면서 외곽 경계에는 닿지 않은** 고립된 '위험 블록'들만 필터링.

### 3. `tunnel_geometry.py` (Tunnel Voxelizer)
- **역할:** 2D 말굽형 터널 폴리곤(Y-Z) 데이터를 3D 복셀 메쉬 공간(X축 돌출)으로 변환.
- **주요 특징:** 포인트-인-폴리곤(Ray-casting) 알고리즘을 최적화하여 도메인 내 어느 복셀이 터널 내부 허공(`TUNNEL(2)`)인지 `True/False` 마스크 행렬을 생성함.

### 4. `visualize_blocks.py` (Result Rendering)
- **역할:** 탐지된 암반 블록 데이터의 다양한 시각화 출력 모듈.
- **주요 함수:**
  - `plot_block_overview`: `block_overview.png` 생성 (2D 단면도, 볼륨 분포 히스토그램, 데이터 표 4-패널 대시보드).
  - `plot_block_3d_scatter`: `block_3d_scatter.png` 생성 (Matplotlib 기반 정적 3D 점묘법 & 터널 와이어프레임 플롯).
  - `plot_block_3d_pyvista`: 실시간 3D 인터랙티브 뷰어 (Marching Cubes 기술로 곡상 단면 시각화).

### 5. `plot_3d_tunnel_fractures.py` (DFN/Tunnel topological Viz)
- **역할:** 블록 탐지 전, 터널을 튜브로 가정하여 단층(Disc)들과의 위상 관계(Intersect / Inside) 필터링 및 3D 시각화 전문 도구.

### 6. `plot_2d_trace_map.py` (2D Trace Map Viewer)
- **역할:** 특정 X 좌표 평면과 DFN 교차 흔적(Trace Line)을 나타내는 GUI 인터랙티브 플롯.
- **기능:** PyVista/GPU 없이 해석학적 교차 방정식 구현. P21 Intensity 계산 지원.

---

## 💡 성능 및 메모리 최적화 컨벤션 (AI Guidelines)
이후 추가적인 수정 작업 시, AI는 전체 `.py` 파일 소스를 읽어올 필요가 없습니다. 
현재 이 레포지토리의 아키텍처는 명확히 분리(Decoupled)되어 있으므로 다음 원칙을 따릅니다:
1. **신규 기능 추가/수정** 시 이 `README.md` 요약을 먼저 참조합니다.
2. 연산 로직 변경은 `block_detector.py`, 시각화나 그래프 변경은 `visualize_blocks.py`처럼 **정확히 대상이 되는 하나의 파이썬 파일만 `view_file`로 열어 수정**합니다.
3. 이를 통해 불필요한 토큰 낭비(크레딧) 및 컨텍스트 초과 오류를 완벽하게 방지할 수 있습니다.
