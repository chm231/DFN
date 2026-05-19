# Trace Distribution Correction — Technical Notes

## 1. Purpose

This module estimates a **bias-corrected true trace length distribution** from observed tunnel-face trace data. Raw observed trace lengths are biased by finite-window sampling effects and should not be treated as the true trace length distribution.

The corrected output is later usable as input for:
- Hekmatnejad et al. (2018) distribution-free fracture diameter CDF estimation
- Fracture radius prior estimation
- SVD-based trace matching radius plausibility scoring

## 2. Scientific Motivation

When fractures intersect a tunnel excavation face, the resulting 2D traces are observed through a finite polygonal observation window. This introduces several systematic biases:

| Bias Type | Description |
|-----------|-------------|
| **Edge censoring** | Trace endpoints cut by window boundary — observed length < true length |
| **One-end clipping** | One endpoint inside window, other clipped by boundary |
| **Both-end clipping** | Both endpoints clipped — strongly biases length downward |
| **Truncation** | Traces shorter than detection threshold `l_min` are not mapped |
| **Orientation bias** | Observation probability depends on trace angle relative to window geometry |
| **Finite-window geometry** | The tunnel face is finite, not an infinite plane |

The module explicitly handles these biases rather than treating the raw histogram as unbiased.

## 3. References Considered

1. **Warburton (1980)** — "A stereological interpretation of joint trace data". Foundation for stereological correction of trace data.
2. **Priest & Hudson (1981)** — "Estimation of discontinuity spacing and trace length using scanline surveys". Terminology and bias awareness for scanline sampling.
3. **Pahl (1981)** — "Estimating the mean length of discontinuity traces". Sanity check for corrected mean trace length.
4. **Laslett (1982)** — "Censoring and edge effects in areal and line transect sampling of rock joint traces". Core reference for censoring classification and edge effects.
5. **Mauldon (1998)** — "Estimating mean fracture trace length and density from observations in convex windows". Finite-window correction reasoning.
6. **Mauldon, Dunne & Rohrbaugh (2001)** — "Circular scanlines and circular windows". Window geometry affects trace statistics.
7. **Song & Lee (2001)** — "Estimation of joint length distribution using window sampling". Conceptual support for window-based estimation.
8. **Zhang & Einstein (2000)** — "Estimating the intensity of rock discontinuities". Future connection to P21/P32 calibration.
9. **Jimenez-Rodriguez & Sitar (2006)** — "Inference of discontinuity trace length distributions using statistical graphical models". Future probabilistic extension.
10. **Hekmatnejad, Emery & Vallejos (2018)** — "Robust estimation of the fracture diameter distribution from the true trace length distribution in the Poisson-disc DFN model". Next step after trace distribution correction.

## 4. Current Implementation

### Method: Inverse Probability Weighting (IPW)

The implementation uses a two-stage approach:

**Stage 1 — Monte Carlo observation probability estimation:**
For each (true_length, angle) bin, synthetic line segments are randomly placed in an expanded region around the tunnel window polygon. Each segment is clipped to the polygon. The fraction of segments that produce an observable trace (length ≥ `l_min`) gives the observation probability `p_obs(L, θ)`.

**Stage 2 — IPW correction:**
```
corrected_count(L, θ) = observed_count(L, θ) / p_obs(L, θ)
```

The corrected 2D histogram is marginalized over angle to produce the 1D corrected trace length distribution.

### Key functions

| Function | Purpose |
|----------|---------|
| `compute_trace_length(trace)` | 2D Euclidean trace length from endpoints |
| `compute_trace_angle(trace)` | Axial orientation angle in [0, π) |
| `classify_censoring(trace, window_polygon, eps)` | Classify as complete / one_end_clipped / both_end_clipped |
| `build_observed_trace_histogram(...)` | Raw observed histogram with optional censoring/angle breakdown |
| `estimate_observation_probability_mc(...)` | Monte Carlo p_obs estimation |
| `build_bias_corrected_trace_distribution(...)` | IPW-corrected distribution |
| `plot_trace_distribution_comparison(...)` | Raw vs corrected validation plots |
| `export_corrected_distribution(...)` | CSV export of corrected distribution |
| `run_tbtd_pipeline(...)` | End-to-end convenience function |

## 5. Assumptions

- The tunnel face is treated as a finite polygonal observation window (not an infinite plane).
- Trace orientation is axial: θ and θ + π are equivalent.
- A detection threshold `l_min` exists (default: 0.15 m). Traces shorter than this are considered unobservable.
- The first implementation uses **complete traces only** for IPW correction. Clipped traces are reported separately but excluded from the correction to avoid treating clipped observed length as true length.
- The sampling region for Monte Carlo includes points outside the window boundary, since a trace whose midpoint lies outside the window can still intersect it.

## 6. Limitations

- The corrected distribution is an **estimate**, not exact truth.
- IPW correction is a first-order method, less rigorous than full kernel inversion or maximum likelihood deconvolution.
- **Clipped traces are excluded** from IPW correction in the current implementation. They are treated as censored lower-bound observations and their counts are reported. This means the correction may underweight long traces that are frequently clipped.
- Small sample sizes may lead to unstable correction (high variance in bins with low p_obs).
- Fracture set separation is not performed within this module. If sets have very different length distributions, per-set correction may be needed.
- The Monte Carlo p_obs estimation quality depends on `n_mc`. Default is 5000 samples per cell; increase for production runs.

## 7. How to Run

### Synthetic example (no data needed):
```bash
cd "c:\Users\user\OneDrive\2026-1\3D DFN modeling"
python scripts/run_trace_distribution_correction.py --synthetic \
    --output-dir trace_analysis/storage/output/tbtd_results
```

### With real DFN data:
```bash
python scripts/run_trace_distribution_correction.py \
    --input "storage/data/dfn_export_for_python.h5" \
    --tunnel-dat "storage/data/단면_폴리곤.dat" \
    --x-start 0 --x-end 9 --advance-step 3 \
    --output-dir trace_analysis/storage/output/tbtd_results \
    --n-mc 10000
```

### Programmatic usage:
```python
from trace_distribution_correction import run_tbtd_pipeline

result = run_tbtd_pipeline(
    traces=obs_traces,          # list of FaceTrace or TraceRecord
    window_polygon=poly_yz,     # shape (N, 2)
    output_dir='output/',
    l_min=0.15,
    n_mc=5000,
)
```

## 8. What Needs Human Review

> [!IMPORTANT]
> The following items should be reviewed before relying on the corrected distribution for downstream analysis.

1. **Censoring classification correctness** — Verify that the boundary tolerance `eps=0.10 m` correctly identifies clipped endpoints for the specific tunnel polygon geometry.

2. **p_obs geometric reasonableness** — Check the Monte Carlo observation probability matrix. Values near 0 or 1 at unexpected bins may indicate issues with the expanded sampling region or polygon clipping.

3. **Clipped trace handling** — Currently, clipped traces are excluded from IPW. Decide whether future versions should include them with conservative Kaplan-Meier-style treatment.

4. **Existing SVD reconstruction code** — Confirm that no existing files in `trace_reconstruction/` were modified. This module is standalone.

5. **Output CSV/plots** — Verify that `corrected_trace_distribution.csv` and comparison plots are generated correctly and contain reasonable values.

6. **Per-set correction** — If fracture sets have very different size distributions, the module should be run per-set. This is not automated yet.
