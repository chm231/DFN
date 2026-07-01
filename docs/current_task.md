# Current Task

## Goal
Completed: MC KM-emulated survival export and KM vs MC comparison refresh.

## Status
- `diagnose_trace_length_km.py` implemented and smoke-tested.
- Full KM runs completed for:
  - Laxemar Sets 1, 2, 3, 5
  - Forsmark Sets 1, 2, 5
- Final merged outputs created:
  - `storage/output/trace_length_km_diagnostics/km_final_diagnostic_summary.csv`
  - `storage/output/trace_length_km_diagnostics/km_final_diagnostic_report.md`
- `window_mc_predicted_survival_curve.csv` exported for:
  - `storage/output/window_mc_predicted_survival/laxemar/`
  - `storage/output/window_mc_predicted_survival/forsmark/`
- KM vs MC comparison outputs created:
  - `storage/output/trace_length_km_diagnostics/laxemar_mc_comparison/`
  - `storage/output/trace_length_km_diagnostics/forsmark_mc_comparison/`
  - `storage/output/trace_length_km_diagnostics/km_mc_final_comparison_summary.csv`
  - `storage/output/trace_length_km_diagnostics/km_mc_final_comparison_report.md`
- `window_mc_predicted_survival_curve.csv` now exports:
  - `mc_observed_visible_survival`
  - `mc_km_emulated_survival`
  - `mc_true_chord_survival`
- KM vs MC consistency was re-evaluated using `mc_km_emulated_survival`.

## Key interpretation
- KM is diagnostic-only.
- It does not replace final `kr_hat`, `P32_hat`, or adoption status.
- The earlier all-set mismatch was partly a comparison-definition problem.
- Direct `observed KM` vs `MC visible survival` comparison remains diagnostic-only.
- Primary consistency check now uses `observed KM` vs `mc_km_emulated_survival`.
- Current best-lmin results:
  - `mc_consistent_with_km`: Forsmark Sets 2, 5; Laxemar Set 5
  - `mc_km_tail_mismatch`: Forsmark Set 1; Laxemar Sets 1, 2, 3

## Next step
- Diagnose the remaining KM-emulated mismatches for:
  - Forsmark Set 1
  - Laxemar Sets 1, 2, 3
- Candidate causes:
  - polygon clipping mismatch
  - censoring-class handling difference
  - lmin-fit interaction
  - trace filtering mismatch
  - face-level sampling variability

## Do not change
- Do not change the `kr` estimator.
- Do not change `effective_rmin` logic.
- Do not change the `unit_p32_forward_mc` estimator.
- Do not replace accepted/provisional/rejected statuses using KM alone.
- Do not use Kaplan-Meier outputs as direct `P32_hat` estimates.
- Do not include Laxemar Set 4 in the power-law KM/kr recovery interpretation.
