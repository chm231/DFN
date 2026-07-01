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
  --rough-mesh-h5 storage/output/rough_face_mesh_collection/synthetic_rough_face_collection.h5 `
  --outdir storage/output/trace_dataset_collection
```

출력:

- `trace_dataset_3d.csv`
- `trace_dataset_3d.h5`

### 4. Radius power-law estimation from censored trace lengths
Script: [dfn_analysis/estimate_radius_powerlaw_from_traces.py](dfn_analysis/estimate_radius_powerlaw_from_traces.py)

Purpose: Estimate set-wise radius power-law candidate `kr` using a size-biased radius-to-chord likelihood.

Status: active experimental estimator. `P32` is not estimated in this step.

Radius lower bound is a modeling choice. Current benchmarks distinguish `r >= 1.0 m` and `r >= 0.5 m` populations. P32 estimates and kr recovery should be interpreted only within the declared radius range.

Previous direct trace-length Pareto fitting experiments were removed from the active tree because they treated observed trace length as Pareto-distributed directly and produced boundary-sensitive fits.

### 5. Window clipping diagnostics
Script: [dfn_analysis/diagnose_window_clipping_effects.py](dfn_analysis/diagnose_window_clipping_effects.py)

Purpose: Diagnose whether rejected radius-to-chord fits are caused by finite face/window clipping and censoring structure.

Status: diagnostic step before window-aware likelihood and P32 estimation.

### 6. Window-aware Monte Carlo radius likelihood
Script: [dfn_analysis/estimate_radius_powerlaw_window_mc.py](dfn_analysis/estimate_radius_powerlaw_window_mc.py)

Purpose: Estimate set-wise radius power-law candidates with finite face/window clipping and censoring classes included through forward Monte Carlo simulation.

Status: active experimental estimator. `P32` is not estimated in this step.

Finite-window MC likelihood should use `proposal_area` center weighting for parameter recovery.
Unweighted center sampling is retained only as a legacy diagnostic comparison mode.

## Dependencies

최소 의존성은 [requirements.txt](requirements.txt)에 정리되어 있습니다.

## Default Radius Population (r >= 0.5 m)
The default radius population is now defined as r >= 0.5 m.
Accordingly, generation_rmin, estimation_rmin, likelihood_rmin, and P32 labels default to 0.5 m and P32_r_ge_0p5m.
Previous r >= 1.0 m outputs are retained only as legacy comparison benchmarks and should not be mixed with r >= 0.5 m results.

본 프로젝트의 기본 DFN 반경 population은 r >= 0.5 m로 통일한다.
P32 및 kr 추정값은 명시된 radius range 안에서만 해석해야 하며,
P32_r_ge_0p5m와 P32_r_ge_1m는 직접 같은 값으로 비교하지 않는다.
