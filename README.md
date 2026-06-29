# 3D DFN Modeling

현재 채택 범위는 아래 세 단계만 유지합니다.

1. DFN 생성
2. 터널 단면 polygon 내부 synthetic rough face mesh 생성
3. 터널 폴리곤과 만나는 절리군별 3D trace 데이터셋 생성

그 외의 역추정, 검증, 블록 검출, 실험 스크립트와 결과물은 `_archive/`로 이동합니다.

## Active Scripts

### 1. DFN 생성
파일: [dfn generator v1/python/generate_dfn.py](dfn%20generator%20v1/python/generate_dfn.py)

예시:

```powershell
$env:PYTHONPATH="."
python "dfn generator v1/python/generate_dfn.py"
```

### 2. Synthetic Rough Face Mesh 생성
파일: [dfn_analysis/generate_synthetic_rough_face_mesh.py](dfn_analysis/generate_synthetic_rough_face_mesh.py)

예시:

```powershell
$env:PYTHONPATH="."
python dfn_analysis/generate_synthetic_rough_face_mesh.py `
  --tunnel-dat storage/data/단면_폴리곤.dat `
  --outdir storage/output/rough_face_mesh `
  --grid-step 0.2 `
  --amplitude 0.05 `
  --corr-length 1.0 `
  --base-x 0.0 `
  --merge-into-hdf5 storage/data/dfn_export_for_python.h5 `
  --seed 42
```

출력:

- `synthetic_rough_face_mesh.h5`
- `synthetic_rough_face_mesh_preview.png`

미리보기 이미지에는 아래 중간 확인 결과가 함께 들어갑니다.

- polygon 내부 mask
- roughness field
- 최종 3D mesh

`--merge-into-hdf5`를 주면 생성한 rough face mesh가 기존 DFN HDF5 내부 `/rough_face/...` 그룹으로 같이 저장됩니다.

### 3. 절리군별 3D Trace 데이터셋 생성
파일: [dfn_analysis/export_setwise_3d_traces.py](dfn_analysis/export_setwise_3d_traces.py)

예시:

```powershell
$env:PYTHONPATH="."
python dfn_analysis/export_setwise_3d_traces.py `
  --input storage/data/dfn_export_for_python.h5 `
  --face-step 3.0 `
  --outdir storage/output/trace_dataset_collection
```

출력:

- `trace_dataset_3d.csv`
- `trace_dataset_3d.h5`

## Dependencies

최소 의존성은 [requirements.txt](requirements.txt)에 정리되어 있습니다.
