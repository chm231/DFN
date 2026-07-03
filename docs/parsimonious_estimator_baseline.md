# Parsimonious Estimator Baseline

## Scope
This benchmark evaluates how stably `kr` and `P32` can be estimated from 2D tunnel face trace observations.
The goal is not to force numerical agreement with benchmark truth values.

## Common Estimator
- Estimation targets:
  - fracture-set-wise radius power-law exponent `kr`
  - fracture-set-wise `P32` over the declared radius support
- Observations:
  - 2D tunnel face trace geometry
  - observed trace-length statistics
  - finite face/window clipping and censoring
- Shared estimator components:
  1. trace observation model
  2. power-law radius likelihood
  3. finite window / face clipping treatment
  4. size-biased observation effect
  5. `unit_p32_forward_mc` calibration for `P32`

## Truth Policy
Benchmark truth values are used only for validation, not for estimator design or calibration.
A single parsimonious estimator is applied consistently to all fracture sets.
Remaining discrepancies are interpreted through observation bias, finite-window clipping, censoring, finite-sample variability, orientation-dependent intersection probability, and limited identifiability of 3D DFN parameters from 2D trace observations.

## Explicit Non-Goals
- No set-specific correction factors
- No empirical fudge factors
- No post-hoc rescaling to match GT
- No truth-tuned model selection across multiple estimators

## Output Structure
The `benchmark1` baseline report writes:
- `storage/output/benchmark1_parsimonious/common_estimator_results.csv`
- `storage/output/benchmark1_parsimonious/gt_comparison.csv`
- `storage/output/benchmark1_parsimonious/mismatch_diagnostics.csv`
- `storage/output/benchmark1_parsimonious/methodology_summary.md`
