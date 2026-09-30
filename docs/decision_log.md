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

## D017 - Probabilistic block assessment by conditional-DFN ensemble
One inversion (reconstructed discs + inverted params) is held fixed; only the unobserved stochastic discs are resampled
(`export_domain_dfn_json --seed s`), so the ensemble spread is the uncertainty of the unobserved rock mass given the observations.
kr/P32 parameter uncertainty is not yet propagated.
Script: `handoffv1/scripts/run_block_ensemble.py` (vs=0.05 m, 6-conn, min_volume=0.01 m3).
Demo seed 42, N=50: P(>=1 FS<1 block)=0.98, n(FS<1)=4.4+/-2.9, V(FS<1)=0.34+/-0.38 m3; running means stable from N~25.
Unstable blocks occur at the crown/upper walls only (P~0 at the invert), consistent with gravity-driven modes.

## D018 - Step 4 result: block prediction is biased low; conditioning adds no local skill (demo seed 42, vs=0.05 m)
Truth vs ensembles (N=50 each): truth 391 blocks / 43 FS<1; conditioned 16 / 4.4; unconditioned 27 / 6.6.
Voxel-level wall scores: BSS(cond vs uncond) ~ 0, AUC 0.5-0.6 for both at every 2 m band -> no local predictive skill from conditioning.
Attribution:
- Set 4 (exponential, visible-only in pipeline) is 35% of true P32 in the domain (2.06 of 5.85 /m) and absent ahead of the face;
  removing it from truth: 391 -> 64 blocks. Dominant cause.
- Truth seed 42 is typical (true-param DFNs, 10 seeds, no Set 4: 63 +/- 16 blocks, 15 +/- 5 FS<1).
- Pipeline generator ablation (no Set 4, 10 seeds): current 25 blocks; kappa from config 21 (not the cause);
  true params 46 -> ~2/3 of the gap is parameter error, ~1/3 is generator-vs-truth-generator difference (unexplained).
- rmax_local=25 m cap has no effect (truth with r>25 removed: identical blocks).
Implication: block formation is a strongly non-linear functional of DFN params; inversion errors acceptable for kr/P32
validation give ~2x fewer blocks. Set 4 must be generated (blind distribution-family fit) before block results are credible.

## D019 - Reconstruction normal bug fixed; radius cap and small reconstructed radii explain the block deficit (corrects D018)
Bug: on flat faces (v2) a single-face cluster lies in the face plane, so the SVD plane fit returned the face normal (1,0,0).
Laxemar 307/418 and Forsmark 182/222 reconstructed discs lay flat in the face; the stochastic generator's orientation
(estimated from these discs) was corrupted (Forsmark Set 1 kappa=1e6, Laxemar Set 2 kappa=252).
Fix (`reconstruct_discs_from_traces.py`): single-face / degenerate clusters use the axial mean of the member trace 3D normals.
After fix: 0 face-normal discs; disc-based orientation matches trace-based config (kappa close, mean within 3 deg).
Radius tiers unchanged (Laxemar 48/329/41). The 0807 report reconstruction figures (fig_recon_discs_*_v2) show the bug.
Optional: `--config dataset_config.json` now sets stochastic orientation from trace-based trend/plunge/kappa.
Block deficit attribution after fix (no Set 4, vs=0.05 m, 6-conn):
- Laxemar: truth 64 blocks / 14 FS<1; cond 28 / 8.1; uncond 52 / 13.7 (before fix 16 / 4.4 and 27 / 6.6).
  Per-band truth FS<1 counts now inside the ensemble 5-95% ranges; wall AUC(unstable) 0.71 cond / 0.73 uncond, BSS ~ 0.
- Radius cap: rmax_local=25 m halves block counts. Truth generator with --rmax 25 vs 250 (Forsmark, 10 seeds, no Set 4):
  3.0 / 1.1 vs 5.3 / 1.8 blocks / FS<1; pipeline generator with true params 2.7 / 0.8 -> generator gap fully explained.
  D018's "cap has no effect" was a single-realization artifact and is withdrawn.
- Conditioning lowers blocks (cond < uncond at both sites): face-crossing large stochastic discs are removed
  (~6-7 discs with r>5 m per realization, -13 to -16% of large-fracture area) and replaced by reconstructed discs with r <= 5 m.
- Forsmark truth 14 / 3 is in the upper tail of true-param DFNs (5.9 +/- 3.0 / 2.1); observed discs bound 0 of 126 blocks.

## D020 - Two-tier stochastic generation removes the radius-cap bias
`generate_hidden_discs(..., rmax_far=R)` / `export_domain_dfn_json --rmax-far R` / `run_block_ensemble --rmax-far R`:
discs with r in (rmax_local, R] are generated in the box widened by (R - rmax_local); set P32 is split between tiers by the
area moment of the [rmin, R] truncated distribution (sum unchanged). Default (no --rmax-far) is byte-identical to before.
Check, pipeline generator with true params vs truth generator (rmax 250, no Set 4):
Laxemar 58.2 / 15.7 vs 63 / 15.1 blocks / FS<1 (was 46 / 12.6 with the 25 m cap);
Forsmark (n=20 vs 30 seeds) 3.80+/-0.63 / 1.15 vs 3.97+/-0.55 / 1.10.
Ensembles (N=50, vs=0.05 m, config orientation, fixed reconstruction, rmax_far=250):
- Laxemar (truth no Set 4: 64 / 14): uncond 60 / 15.3 (p5-p95 33-91 / 7.5-25.6) -> truth reproduced; cond 33 / 9.2.
  Largest block volume mean 5.6 m3 uncond (2.9 before). Wall AUC(FS<1) cond 0.69 / uncond 0.78; BSS ~ 0.
- Forsmark (truth 14 / 3, upper tail of true-param DFNs 4.0 / 1.1): uncond 5.6 / 1.34 (p95 15 / 5), cond 2.5 / 0.6.
Remaining bias is conditioning-specific (cond ~ half of uncond at both sites; near-face 0-2 m P(>=1 FS<1) 0.02-0.26 vs uncond 0.22-0.86),
consistent with face-crossing large discs being replaced by small reconstructed discs (D019).

## D021 - Censoring-aware posterior radius sampling for observed discs (partial fix of the conditioning deficit)
Diagnosis (truth used only for checking): reconstructed radius / true radius median 0.99 (r<1 m) -> 0.58 (2-5) -> 0.35 (5-10)
-> 0.12 (10-25 m); large fractures' traces are 90-100% window-censored, and the shrinkage posterior treated censored chords
as complete chords (posterior mean = 1.14-1.22 x half-chord a for any a).
Change (ensemble option only; point estimates / report values unchanged):
- reconstructed_discs.csv gains a_half, censored, chord_m*, chord_u*, faces (existing columns byte-identical).
- `sample_visible_discs` (+ `export_domain_dfn_json --sample-visible-radius`, `run_block_ensemble --sample-visible-radius`):
  per realization, shrinkage discs get R from prior R^-kr (R >= set rmin) x likelihood
  [uncensored: density of half-chord = a; censored: P(half-chord >= a) = sqrt(1 - a^2/R^2)], center placed to contain the chord,
  and samples that leave >= lmin_det traces on non-observed faces (or none on observed faces) are rejected (fallback: point estimate).
Calibration (200 draws): posterior median / true 0.99-1.02 overall, 90% interval coverage 78% (target 90%; misses symmetric);
>10 m fractures still under-covered (single trace cannot identify them). Sum r^2 Laxemar true 756 / point 277 / posterior 949.
Ensembles (rmax_far=250, config orientation): cond blocks / FS<1 Laxemar 33 / 9.2 -> 37.4 / 10.0, Forsmark 2.5 / 0.6 -> 3.7 / 0.9
(uncond 60 / 15.3 and 5.6 / 1.34; truth 64 / 14 and 14 / 3). Near-face 0-2 m P(>=1 FS<1): Laxemar 0.26 -> 0.56, Forsmark 0.02 -> 0.08.
Large discs (r>5) near the domain: cond 64 -> 66.5 vs uncond 71.3 (Laxemar). The remaining cond < uncond gap (~35-40%) is unexplained.

## D022 - Cause of the conditioned < unconditioned block gap (synthetic benchmark)
With the same seed, cond and uncond share identical stochastic discs; the difference is exactly
R (stochastic discs removed because they leave >= 0.5 m traces on observed faces) vs V (reconstructed discs added).
Reference T = truth discs that leave >= 0.5 m traces (same criterion). In-domain P32 (10 seeds, Set 4 excluded for Laxemar):
- Laxemar: T 0.208 (r>5: 0.090) | R 0.303 (0.157) | V 0.120 (0.029); disc counts 176 / 177 / 145.
- Forsmark: T 0.143 (0.093) | R 0.249 (0.185) | V 0.101 (0.064); counts 97 / 85 / 78.
Net V - R = -0.18 / -0.15 per m, ~70-80% of it in r > 5 m discs -> this is the cond < uncond gap.
Two parts of similar size: (i) V < T: reconstructed discs still carry ~40-50% too little in-domain intensity
(large fractures seen only as short censored chords are not identifiable; posterior under-covers r > 10 m, D021);
(ii) R > T: the model removes ~1.5-1.7x the truth's face-crossing intensity; the removal rule matches T's criterion,
so this reflects parameter tail differences and single-truth variance (large discs are few), not a rule flaw.
The removal rule itself is consistent; the fixable part is (i).

## D023 - First field inversion (DFM export, 12 faces, converter output; sets 1-4)
Converter reproduced (6,447 traces, 6 global sets at 30 deg cut, observation area 795.43 m2). Sets 5/6 (<100 traces, 1-2 faces) not inverted.
Orientation (trace-based): kappa 11-15. Hybrid kr with generation rmin 0.5 m:
- lmin 0.5: kr 3.85-3.95; model median too long (1.03 vs 0.82 m) and q90 too short (1.66 vs 1.94-2.13 m); class L1 0.22-0.27 (sets 2-4).
- kr is lmin-dependent: 0.75 m -> 3.15-3.30, 1.0 m -> 2.85-3.20, 1.5 m -> 3.05-3.35 (median fit improves above 0.75 m).
- Blind lmin rule picks 0.5 (set 1) and 1.0 (sets 2-4).
P32 (forward_mc_lmin, same lmin on observed and simulated): total sets 1-4 = 3.98 / 3.28 / 2.95 m2/m3 at lmin 0.5 / 0.75 / 1.0.
P32(r >= 0.5) should be lmin-invariant; the 26% drift means the truncated power-law disc model does not describe the
0.5-1 m trace band. The excess of 0.5-0.75 m traces is not a face-resolution artifact (n(>=0.5)/n(>=0.75) = 1.70 high-res vs 1.77 low-res).
Open: C_lmin > E[sin phi] (~5%) in forward_mc_lmin (eta > 1 is unphysical). Field results are provisional.

## D024 - Field lmin-drift diagnosis (hypotheses 1-3)
Scratch diagnostics (field/diag_family.py, diag_termination.py; converter re-runs with --edge-tol).
H1 radius family (trace-length MLE, size-biased discs, uniform offset, right-censoring): simplified model reproduces the
pipeline's power-law drift (kr 3.8 / 3.1 / 2.6 at lmin 0.5 / 0.75 / 1.0). Power law is rejected at lmin 0.5 (dAIC 170-410 vs lognormal),
but the best lognormal has an unphysical cm-scale median and unstable parameters across lmin -> no smooth radius family fixes it;
the trace-length distribution has more short traces than any "trace = full disc chord" kernel allows.
H2 T-termination: uncensored endpoints lie within 5 cm of another-set trace 32.9% vs 26.1% for midpoints (2 cm: 13.1% vs 9.4%)
-> real but modest (~7% of endpoints).
H3 censoring tolerance: edge_tol 0.05 / 0.15 / 0.30 m -> censored share 2 / 8 / 16%, kr(lmin 0.5) ~4.1-4.35 / 3.85-3.95 / 3.5-3.65;
shifts kr but the ~1.0 drop between lmin 0.5 and 1.0 remains at every tolerance.
Leading open hypothesis H4 (needs the DFM team): DFM traces are "planar polygonal patch" based (patch radius = half trace length by
construction), i.e. trace length may measure the exposed planar patch extent on the rough face, not the fracture-face intersection chord.
If so, the observation model (not the radius family) must change.

## D025 - Trace length-angle dependence: (b) patch-boundary mechanism NOT confirmed
The dependence survives set and censoring control (rho -0.21 to -0.41), but above the 0.5 m detection floor it is not significant
for set 1 (rho -0.097, p 0.078). Short traces have significantly larger normal deviation and their angle alpha regresses to 45 deg,
so the patch-boundary mechanism and the normal-noise mechanism cannot be separated with the current data. Confirmation of (b) is withheld
(the earlier "(b) supported" reading overstated the evidence).
Separately, patch normals violate perpendicularity to their own traces: median 8.26 deg, 43.4% > 10 deg (other session, 3D endpoints);
cross-check here with flattened endpoints: median 9.16 deg, 46.4% > 10 deg, decreasing with length (10.8 deg for L < 0.3 m -> 5.4 deg for L >= 1 m).
This is promoted to the main item of the DFM-team query. It propagates into the converter and reconstruction (patch normal used as
fracture orientation); impact not yet measured.
Also recorded: DFM disc radius = half trace length by construction ((L/2)/R median 1.008, centre at trace midpoint) -> no independent size information.
Next: normal-independent plane refit from 3D_tunnel_face_point_cloud.ply around each trace (separates the two mechanisms; also feeds kappa
correction and association tolerance). Spacing-based P32: auxiliary cross-check only (gives no kr; Terzaghi still uses orientation).
