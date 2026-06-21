# 3D DFN Modeling

현재 채택 범위는 아래 두 단계만 유지합니다.

1. DFN 생성
2. 터널 폴리곤과 만나는 절리군별 3D trace 데이터셋 생성

그 외의 역추정, 검증, 블록 검출, 실험 스크립트와 결과물은 `_archive/`로 이동합니다.

## Active Scripts

### 1. DFN 생성
파일: [dfn generator v1/python/generate_dfn.py](dfn%20generator%20v1/python/generate_dfn.py)

예시:

```powershell
$env:PYTHONPATH="."
python "dfn generator v1/python/generate_dfn.py"
```

이 스크립트는 HDF5 DFN과 터널 폴리곤(`/tunnel/poly_YZ`)을 함께 저장할 수 있습니다.

### 2. 절리군별 3D Trace 데이터셋 생성
파일: [dfn_analysis/export_setwise_3d_traces.py](dfn_analysis/export_setwise_3d_traces.py)

예시:

```powershell
$env:PYTHONPATH="."
python dfn_analysis/export_setwise_3d_traces.py `
  --input storage/data/dfn_export_for_python.h5 `
  --face-step 3.0 `
  --outdir storage/output/trace_dataset
```

출력:

- `trace_dataset_3d.csv`
- `trace_dataset_3d.h5`

각 trace 레코드는 아래 정보를 포함합니다.

- `set_id`
- `fracture_id`
- `face_id`
- `face_x_m`
- clipped 3D endpoints `p0_xyz`, `p1_xyz`
- unclipped 3D endpoints `full_p0_xyz`, `full_p1_xyz`
- `observed_length_m`
- `full_length_m`
- `censoring_class`

## Dependencies

최소 의존성은 [requirements.txt](requirements.txt)에 정리되어 있습니다.
