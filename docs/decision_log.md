# Decision Log

## D001 - Use effective_generation rmin
Global `rmin=0.5` caused support mismatch for Laxemar Sets 2/3.
Estimator must use `set_likelihood_rmin = set_effective_generation_rmin`.

## D002 - Laxemar Set 4 excluded
Laxemar Set 4 follows exponential radius distribution, not power-law.

## D003 - proposal_area center weighting
Unweighted center proposal underestimates `kr` for biased sets.
`proposal_area` is preferred for window MC.

## D004 - P32 proxy is not final
`conditional_visible_trace_proxy` is scaffold only.
Final pilot P32 must use `unit_p32_forward_mc`.

## D005 - unit-P32 IS oracle passed
Importance sampling estimator is unbiased relative to brute-force oracle.
Remaining P32 issues are not IS estimator bugs.

## D006 - Laxemar Set 2 marginal cause
Laxemar Set 2 P32 discrepancy is likely due to support-label ambiguity and high face-level sampling variability, not estimator failure.

## D007 - Forsmark Set 5 dense C(kr) interpolation check
Sparse 3-point `log_linear_3point` interpolation caused a small CI miss for Forsmark Set 5.
Dense `log_linear_dense` interpolation removed `C_extrapolation_fraction` and brought `P32_reference` back inside the bootstrap CI.

## D008 - Forsmark Set 5 promoted after dense C(kr) check
Forsmark Set 5 should be treated as `p32_final_pilot_candidate` after the dense C(kr) interpolation audit.
The earlier CI miss is interpreted as a sparse interpolation artifact rather than a residual estimator bias.

## D009 - Kaplan-Meier is auxiliary only
Kaplan-Meier trace-length correction is introduced only as a non-parametric diagnostic for censoring effects.
It must not replace the final effective-rmin window-aware MC kr inversion or the unit-P32 forward-MC pilot estimator.

## D010 - Full KM diagnostic completed for accepted/provisional sets
Full Kaplan-Meier diagnostics were run for Laxemar Sets 1, 2, 3, 5 and Forsmark Sets 1, 2, 5.
The outputs are used as explanatory diagnostics for censoring and tail sensitivity only.

## D011 - KM vs MC comparison indicates systematic tail mismatch
The current window-aware MC predicted survival curves are consistently shorter-tailed than the KM-adjusted empirical survival curves for the accepted/provisional sets.
This is treated as a diagnostic of the observation model, not as a reason to overwrite the existing kr or P32 statuses.

## D012 - Use MC KM-emulated survival for primary KM-MC consistency checks
Direct comparison between observed KM survival and MC visible-length survival is diagnostic only because they represent different censoring treatments.
The primary KM-MC consistency check should use observed KM survival versus `mc_km_emulated_survival`, where the same Kaplan-Meier procedure is applied to simulated observed lengths and simulated censoring classes.
