# Archive Manifest — benchmark1_legacy_cleanup_2026-07-01

## Date
2026-07-01

## Purpose
Reduce the `benchmark1` active tree to only what the parsimonious paper baseline
needs: DFN / trace input generation → one common `kr`/`P32` estimator →
validation-only GT comparison → mismatch diagnostics. Everything not on that path
(GT-tuned calibration, oracle checks, empirical-consistency audits, intermediate
summary builders, dense tables, sensitivity sweeps, decomposition experiments,
old diagnostic runs, and plotting-only summaries) was moved here.

Nothing was deleted. Tracked scripts were moved with `git mv`; untracked
experiment outputs under `storage/output/` (git-ignored) were moved with `mv`.
All moves are reversible.

## How the active baseline stays reproducible
`scripts/run_parsimonious_baseline_report.py` was made self-contained before the
cleanup: the exact CSVs it consumes were copied into
`storage/output/benchmark1_parsimonious/inputs/`, and the script now reads that
curated snapshot first (falling back to the original experiment path only if a
curated file is still present). After moving the 50 experiment output entries
below, the report still runs and produces byte-identical
`common_estimator_results.csv`, `gt_comparison.csv`, and `mismatch_diagnostics.csv`.

## Moved scripts (dfn_analysis/ → dfn_analysis/, via git mv)
GT / empirical-consistency experiments:
- audit_laxemar_set2_empirical_consistency.py
- diagnose_p32_calibration_factor.py
- diagnose_unit_p32_importance_oracle.py

Intermediate summary builders / dense table:
- build_dense_ckr_table.py
- build_final_kr_bootstrap_summary.py
- build_final_kr_recovery_summary.py
- build_km_diagnostic_summary.py
- build_km_mc_comparison_summary.py
- build_p32_full_unit_summary.py

Diagnostic-only experiments not on the baseline path:
- diagnose_observed_radius_mixture.py
- diagnose_powerlaw_convention.py
- diagnose_radius_conditioned_visibility.py
- diagnose_trace_censoring.py
- diagnose_tunnel_direction_bias.py

(Additionally, `check_dense_c_interpolation.py` was moved here by an earlier
cleanup pass on the same day.)

## Moved outputs (storage/output/ → storage/output/, via mv)
Curated into `benchmark1_parsimonious/inputs/` first, then archived:
- final_kr_recovery_summary_effective_rmin.csv
- final_kr_bootstrap_effective_rmin/
- p32_mc_calibrated_effective_rmin/  (moved as `p32_mc_calibrated_effective_rmin__from_active`; an earlier partial copy already existed here)
- trace_length_km_diagnostics/       (moved as `trace_length_km_diagnostics__from_active`; an earlier partial copy already existed here)
- trace_censoring_diagnostics/, trace_censoring_diagnostics_forsmark/
- window_clipping_diagnostics/, window_clipping_diagnostics_forsmark/
- tunnel_direction_bias_rmin0p5/

Other experiment outputs (sensitivity sweeps, decomposition, old window-MC runs,
pilots, diagnostics, smoke runs, visualization):
- estimate_kr_smoke/, estimate_kr_smoke2/
- forsmark_rmin0p5_default_window_mc/, laxemar_rmin0p5_default_window_mc_sets_1_2_3_5/
- forsmark_rmin0p5_direction_mode_comparison/, laxemar_rmin0p5_direction_mode_comparison/
- forsmark_rmin0p5_effective_rmin_window_mc_sets_1_2_3_4_5/, laxemar_rmin0p5_effective_rmin_window_mc_sets_1_2_3_5/
- forsmark_rmin0p5_radius_powerlaw_window_mc/, laxemar_rmin0p5_radius_powerlaw_window_mc_sets_1_2_3_5/
- forsmark_rmin0p5_trace_censoring_diagnostics/, laxemar_rmin0p5_trace_censoring_diagnostics/
- forsmark_rmin_sensitivity_summary.csv, laxemar_rmin0p5_sensitivity_summary_sets_1_2_3_5.csv
- forsmark_set2_center_weighted_window_mc/, laxemar_set2_center_weighted_window_mc/, laxemar_set3_center_weighted_window_mc/
- forsmark_set2_decomposition_full/, laxemar_sets23_decomposition_full/
- forsmark_set2_effective_rmin_window_mc/, laxemar_sets23_effective_rmin_window_mc/
- observed_radius_mixture_diagnostics/
- p32_pilot_effective_rmin/, p32_pilot_candidate_sets_effective_rmin.csv
- powerlaw_convention_diagnostics/
- radius_conditioned_visibility_diagnostics/
- radius_powerlaw_fit_v3/, radius_powerlaw_fit_v3_forsmark/, radius_powerlaw_fit_v3_set_4_provisional/, radius_powerlaw_fit_v3_sets_1_2_5/
- radius_powerlaw_window_mc/, radius_powerlaw_window_mc_polygon/, radius_powerlaw_window_mc_polygon_forsmark/, radius_powerlaw_window_mc_polygon_set3/, radius_powerlaw_window_mc_polygon_set4/, radius_powerlaw_window_mc_polygon_sets_1_2/, radius_powerlaw_window_mc_set_3_reference/
- trace_visualization_collection/
- tunnel_direction_bias/
- window_mc_decomposition_summary_full.csv
- window_mc_predicted_survival/

(Additionally, `global_rmin_legacy_superseded_cases.csv` and `tmp/` were moved
here by an earlier cleanup pass on the same day.)

## Kept active
Scripts (dfn_analysis/): __init__, generate_synthetic_rough_face_mesh,
export_setwise_3d_traces, estimate_fisher_kappa, estimate_mean_orientation,
estimate_radius_powerlaw_from_traces, diagnose_window_clipping_effects,
estimate_radius_powerlaw_window_mc, estimate_kr, estimate_p32_mc_calibrated,
estimate_p32_combined_bootstrap, build_p32_pilot_summary,
summarize_setwise_trace_statistics, diagnose_trace_length_km,
plot_setwise_trace_length_distribution, plot_3d_traces_on_rough_faces.
(Several of these are kept because active README steps, the baseline report, or
the retained tests import them — verified with ripgrep before moving.)

Wrapper: scripts/run_parsimonious_baseline_report.py
Tests: all six under tests/.

storage/output/ kept (canonical inputs + final baseline bundle):
- benchmark1_parsimonious/  (+ curated inputs/)
- rough_face_mesh_collection/
- trace_dataset_collection/, trace_dataset_collection_forsmark/
- laxemar_rmin0p5_trace_dataset_collection/, forsmark_rmin0p5_trace_dataset_collection/
- dfn_forsmark/, dfn_forsmark_rmin0p5/, dfn_laxemar_rmin0p5/
- trace_dataset_multi_face/

storage/data/ (DFN inputs) was not touched.

## Reproducing archived experiments
Each moved script still imports from the active `dfn_analysis` package, so it can
be re-run from its archived location with `PYTHONPATH=.`. The archived output
folders hold the original experiment products for inspection; the baseline itself
does not depend on them (it uses the curated `inputs/` snapshot).
