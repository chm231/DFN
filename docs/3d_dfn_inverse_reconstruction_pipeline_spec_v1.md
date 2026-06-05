# 3D DFN Inverse Reconstruction Pipeline Specification v1

## 1. Purpose

This document defines the baseline implementation specification for reconstructing a 3D DFN from tunnel-face trace observations and passing the reconstructed DFN to the downstream block detection pipeline.

The intended workflow is:

1. Ingest observed trace endpoints and `set_id` on excavation faces.
2. Correct set-wise geometric sampling bias and estimate true 3D fracture statistics.
3. Reconstruct deterministic 3D fractures where observation evidence is strong.
4. Generate stochastic residual fractures only for unexplained intensity.
5. Export a bounded-disc DFN in a schema compatible with downstream block detection.

This specification is written to resolve the main design ambiguities raised during review and to serve as the approval-ready baseline for implementation.

## 2. Coordinate and Geometry Premises

### 2.1 Global Coordinate System

- All trace endpoints are treated as global 3D points `[x, y, z]`.
- `x` is the tunnel advance direction.
- Each excavation face is a rough surface centered around a nominal face position `x_face`.
- `y` and `z` are cross-sectional tunnel coordinates.

### 2.2 Input Trace Representation

- Baseline input is a trace segment with two global 3D endpoints:
  - `p0_xyz = [p0_x, p0_y, p0_z]`
  - `p1_xyz = [p1_x, p1_y, p1_z]`
- Required metadata:
  - `trace_id`
  - `face_id`
  - `x_face`
  - `set_id`
- Optional metadata:
  - `parent_fracture_id` for synthetic validation only
  - polyline samples on the rough face, if available

If only local 2D face coordinates `[y, z]` are available, they must be converted to global `[x, y, z]` before entering the inverse reconstruction pipeline. The inverse pipeline must not silently mix 2D and 3D coordinates.

### 2.3 Fracture Geometry Model

- Each reconstructed fracture is represented as a bounded disc.
- Required disc parameters:
  - `center_xyz = [cx, cy, cz]`
  - `normal_xyz = [nx, ny, nz]`
  - `radius`
  - `set_id`

Infinite planes may be used as an intermediate fitting model, but the final export object must be a bounded disc because:

- `P32` is defined on fracture area, not infinite plane count.
- downstream block detection requires finite fracture extents
- deterministic and stochastic fractures must share the same output schema

## 3. Observation and Bias-Correction Premises

### 3.1 Set-Wise Processing

Bias correction is performed separately for each `set_id`. The pipeline must keep the following categories separate in logs and outputs:

- observed traces
- deterministic reconstructed fractures
- stochastic residual fractures
- total reconstructed fractures

### 3.2 Censoring and Truncation

Each observed trace must be classified before size inference:

- fully observed: both endpoints visible within the face observation window
- singly censored: one endpoint truncated by the tunnel boundary
- doubly censored: both ends truncated or incompletely observed

The baseline estimator may use a parametric MLE-style correction model, but the implementation must emit counts for each censoring class per set. No estimator may assume all traces are fully observed.

### 3.3 Size Bias

Larger 3D fractures have higher probability of intersecting a face and appearing as traces. The inverse pipeline must therefore estimate a true 3D radius distribution, not directly reuse the observed trace-length distribution.

The output of this stage is:

- corrected trace-length model per set
- inferred true 3D radius model per set
- uncertainty summary for radius estimation

Radius must be treated as a weakly identified quantity when only short chords are observed. The baseline implementation must prefer a regularized or posterior-style estimate over an unconstrained direct radius solve.

### 3.4 Orientation Bias

Orientation bias correction follows a Terzaghi-style weighting principle. The baseline uses a set-level orientation target rather than a per-trace fully identified 3D normal.

For each set, the pipeline must maintain:

- target set orientation prior or reference mean normal
- orientation spread parameter if available
- orientation mismatch between reconstructed discs and the target set orientation

If the source data do not provide a reliable set orientation prior, the pipeline must explicitly flag that orientation correction is underconstrained.

## 4. Deterministic Reconstruction Rules

### 4.1 Matching Target

The primary deterministic target is a multi-face track across consecutive excavation faces.

- high-confidence deterministic candidate: observed on `3` or more consecutive faces
- conditional deterministic candidate: observed on exactly `2` consecutive faces

This two-tier rule replaces a hard `3-face only` rule so that large fractures visible on only two faces are not automatically forced into the stochastic pool.

### 4.2 Matching Evidence

The face-matching stage must expose an explicit Bayesian or likelihood-style linkage score. Baseline threshold:

- accept high-confidence track when `ln(BF) >= 2.0`

For `2-face` conditional candidates, deterministic promotion is allowed only if all of the following hold:

- `ln(BF) >= 2.5`
- fitted plane residual `<= 0.10 m`
- predicted radius remains within the allowed radius bounds
- orientation is consistent with the target set prior

If these conditions are not met, the traces stay outside the deterministic pool and contribute to residual stochastic intensity.

### 4.3 Plane and Disc Fitting

Deterministic fitting proceeds in two steps:

1. fit a supporting 3D plane from matched multi-face trace evidence
2. derive a bounded disc center and radius under geometric and prior constraints

The fitted disc must satisfy:

- point-to-plane residual `<= 0.15 m`
- radius `>= 0.5 m`
- radius `<= 15.0 m`

The implementation must log:

- number of supporting traces
- number of supporting faces
- plane residual
- estimated center
- estimated radius
- set consistency score

### 4.4 Deterministic Failure Modes

Deterministic failures must not be collapsed into a single generic fallback. The baseline requires separate failure labels:

- `matching_failure`
- `insufficient_face_support`
- `plane_fit_failure`
- `radius_bound_failure`
- `orientation_inconsistency`

Fallback handling:

- `matching_failure`: keep traces ungrouped for residual modeling
- `insufficient_face_support`: allow consideration as conditional deterministic or residual stochastic
- `plane_fit_failure`: break the track and return traces to residual modeling
- `radius_bound_failure`: reject deterministic disc export and return traces to residual modeling
- `orientation_inconsistency`: reject deterministic promotion unless manually overridden

## 5. Stochastic Residual DFN Rules

### 5.1 Allowance Policy

Stochastic fractures are allowed, but only as residual correction.

Residual intensity is defined per set after subtracting the forward-simulated contribution of accepted deterministic discs from the target observation metrics.

### 5.2 Hard Constraint

If deterministic fractures alone already exceed the target `P21` for a set or face group, the stochastic generator must not add new fractures for that set in that window. The excess must be reported as deterministic over-generation.

### 5.3 Sampling Targets

Stochastic fracture synthesis uses:

- corrected radius distribution per set
- corrected orientation prior per set
- residual `P21` or equivalent intensity target per set
- reconstruction domain bounds around the tunnel

All stochastic fractures must still be exported as bounded discs in the same schema as deterministic ones.

## 6. Optimization and Validation Objective

### 6.1 Baseline Loss Schema

The baseline global loss is a weighted sum:

- `P21` total intensity error: `0.50`
- trace count relative error: `0.20`
- trace-length distribution RMSE: `0.20`
- orientation mismatch: `0.10`

These are baseline weights, not fixed constants for all datasets. The implementation must keep them configurable and log the active values used in a run.

### 6.2 Metric Definitions

- `P21 error`: mismatch between observed and simulated trace intensity on faces
- `trace count error`: relative mismatch of trace counts by set and total
- `trace-length RMSE`: RMSE between observed and simulated trace-length ECDF or CDF summary
- `orientation mismatch`: mismatch between reconstructed disc normals and the target set orientation prior

The optimizer must report each term separately before reporting the weighted total loss.

### 6.3 Required Diagnostics

For each run, emit at minimum:

- observed trace count by set
- simulated trace count by set
- deterministic fracture count by set
- stochastic fracture count by set
- observed `P21` by set
- simulated `P21` by set
- `P21` error by set
- trace-length mismatch by set
- orientation mismatch by set
- total loss

## 7. Output DFN Schema

The reconstructed DFN export must remain compatible with downstream readers.

Required HDF5 fields:

- `/fractures/centers`
- `/fractures/normals`
- `/fractures/radii`
- `/fractures/set_id`

Recommended additional fields:

- `/fractures/source_type`
  - `0 = deterministic`
  - `1 = stochastic`
- `/fractures/support_face_count`
- `/fractures/fit_residual`
- `/fractures/reconstruction_score`
- `/meta/grid_resolution`
- `/meta/input_trace_path`
- `/meta/connectivity`
- `/meta/reconstruction_version`
- `/meta/loss_weights`

Downstream block detection must consume the same bounded-disc representation without requiring schema translation.

## 8. Component-Level Baseline Changes

### 8.1 `trace_analysis/trace_reconstruction_unified.py`

Required implementation direction:

- update the constrained plane fitting path to accept and preserve global 3D endpoint coordinates
- derive bounded-disc center and radius explicitly from fitted geometry and priors
- expose deterministic failure labels
- parameterize the linkage threshold such as `ln_bf_threshold`

### 8.2 `trace_analysis/run_3d_trace_to_dfn.py`

Required implementation direction:

- define the multi-term loss schema
- log each loss component and the total loss
- separate deterministic and stochastic counts in run summaries
- optionally chain the reconstructed DFN export into the block detector

## 9. Verification Plan

### 9.1 Function-Level Verification

- generate synthetic ground-truth traces with global 3D endpoints
- confirm that exported CSV or HDF5 traces contain `p0_x, p0_y, p0_z, p1_x, p1_y, p1_z`
- confirm that deterministic fitting accepts valid multi-face synthetic tracks and rejects intentionally unstable tracks

### 9.2 Pipeline-Level Verification

- run the inverse reconstruction pipeline on a known synthetic dataset
- export `reconstructed_dfn.h5`
- confirm that the exported DFN includes bounded-disc fields required by block detection
- run the downstream block detector without schema exceptions

### 9.3 Manual Geometry Verification

- inspect 3D visualizations of rough faces, reconstructed deterministic discs, and stochastic residual discs
- verify that disc centers, normals, and radii are physically plausible around the tunnel
- compare observed and simulated trace maps on selected faces

## 10. Approval Items

The following baseline settings require explicit project approval before they are treated as locked defaults:

- global input coordinates use `[x, y, z]`
- final fracture representation is bounded disc
- deterministic reconstruction uses a two-tier rule:
  - `3+ faces = high-confidence deterministic`
  - `2 faces = conditional deterministic`
- baseline matching threshold uses `ln(BF) >= 2.0`
- deterministic acceptance uses residual `<= 0.15 m` and radius bounds `0.5 m` to `15.0 m`
- baseline loss weights are `0.50 / 0.20 / 0.20 / 0.10`
- stochastic generation is residual-only and forbidden when deterministic output already exceeds target `P21`

Until these items are approved, they should be treated as proposed defaults rather than immutable rules.
