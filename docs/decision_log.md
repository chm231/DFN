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

## D013 - Keep benchmark1 on a single parsimonious estimator
Benchmark truth must be used only for validation, not for estimator design, fitting, or post-hoc correction.
`benchmark1` should report one common estimator across fracture sets and interpret residual mismatch diagnostically rather than adding set-specific tuning.

## D014 - Polygon clipping ruled out as KM-MC tail-mismatch cause
Diagnostic re-clip test on the observed traces (Laxemar Sets 1/2/3, Forsmark Set 1 vs consistent sets) rules out polygon clipping as the tail-mismatch cause:
- `meta/tunnel_poly_yz` is convex for both sites, so the MC `clip_segments_to_convex_polygon_vectorized` path is exact (max|dLen|=0, 0 class diffs vs the general `clip_segment_to_polygon` loop on the same segments).
- Observed traces already use the same flat polygon window: 0% of observed endpoints fall outside the polygon and stored `observed_length_m` matches the flat-polygon re-clip to within ~0.1% (median stored/reclip ratio 1.000) for every set.
Mismatch and consistent sets are indistinguishable on all clipping axes; the distinguishing signal remains `observed_radius_mixture_still_larger_than_mc`. Remaining KM-MC tail mismatch should therefore be pursued as a radius-mixture / lmin-fit issue (candidate 3), not polygon clipping or censoring-class geometry.

## D015 - Block detection uses 6-connectivity CCA
Paper scope extended to CCA block formation (`handoffv1/dfn_analysis/detect_blocks.py`).
With fracture slab half-thickness `tol = 0.6*vs`, 6-connectivity always separates rock across a single plane,
while 26-connectivity leaks through every oblique plane (Test A) and loses all three analytic wedges (Test B).
26 is kept only as a comparison option.

## D016 - Removability + simple limit-equilibrium FS
Blocks = ROCK components touching the tunnel and not the domain boundary.
Removability is tested by a voxel sweep (other ROCK blocks; out-of-grid treated as passable).
Modes: fall (FS=0) / single-plane / two-plane sliding (Goodman-Shi, Hoek-Bray), gravity only, no support/water; min FS among admissible modes.
Weight uses `V_corr = V_voxel + tol*sum(A_j)` (returns the block's half of each fracture slab).
Analytic wedge check at vs=0.05 m: failure mode matches in all cases; V_corr error <=0.5%; FS exact for c=0, -3 to -4% for c>0 (O(vs) face-area bias).
On the seed-42 conditional DFN, use vs<=0.05 m and a physical `--min-volume` threshold; coarser grids (0.1-0.2 m) are not converged.
