# Current Task

## Goal
Completed: benchmark1 parsimonious baseline report and legacy smoke/tmp archive cleanup.

## Status
- `docs/parsimonious_estimator_baseline.md` added to state the common estimator and validation-only GT policy.
- `scripts/run_parsimonious_baseline_report.py` added as a wrapper/reporting script; it does not modify the estimator.
- Final report outputs created under `storage/output/benchmark1_parsimonious/`:
  - `common_estimator_results.csv`
  - `gt_comparison.csv`
  - `mismatch_diagnostics.csv`
  - `methodology_summary.md`
- Legacy smoke/tmp artifacts were moved out of the active tree into `_archive/benchmark1_legacy_cleanup_2026-07-01/`.

## Key interpretation
- The benchmark objective is framework validation, not truth-tuned matching.
- One parsimonious estimator is applied across sets.
- Remaining mismatch should be discussed through observation limits, not patched with set-specific corrections.

## Next step
- Paper (SCI, English): integrated framework trace -> inversion -> conditional DFN -> CCA block formation.
- Done: steps 1-2 (block detection wired to export JSON; analytic verification). See D015/D016.
  - `handoffv1/dfn_analysis/detect_blocks.py`, `block_detector.py`, `block_stability.py`, `tunnel_geometry.py`
  - `handoffv1/scripts/verify_block_detection.py`
- Done: step 3 conditional-DFN ensemble (`handoffv1/scripts/run_block_ensemble.py`, D017).
- Next: step 4 conditioned vs unconditioned vs true DFN (value of observation); step 5 field data (pending from user).

## Do not change
- Do not change the `kr` estimator.
- Do not change `effective_rmin` logic.
- Do not change the `unit_p32_forward_mc` estimator.
- Do not add set-specific corrections, fudge factors, or GT-tuned rescaling.
- Do not use benchmark truth inside estimator fitting logic.
- Do not reinterpret Laxemar Set 4 as a power-law inversion target.
