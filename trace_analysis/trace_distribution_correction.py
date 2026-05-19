"""
Tunnel-window Bias-corrected Trace Distribution Estimator (TBTD Estimator)

Estimates the bias-corrected true trace length distribution from observed
tunnel-face trace data. Handles edge censoring, truncation, orientation bias,
and finite-window geometry effects.

This module is standalone and optional. Existing SVD-based reconstruction
pipelines continue to work without it.

References:
    Warburton (1980), Priest & Hudson (1981), Pahl (1981), Laslett (1982),
    Mauldon (1998), Mauldon et al. (2001), Song & Lee (2001),
    Zhang & Einstein (2000), Jimenez-Rodriguez & Sitar (2006),
    Hekmatnejad et al. (2018)
"""
import os
import json
import warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class TraceRecord:
    """Lightweight trace record for distribution correction.
    Compatible with FaceTrace from trace_types but does not require it.
    """
    p0: np.ndarray          # shape (2,) -- endpoint 0 in (y, z)
    p1: np.ndarray          # shape (2,) -- endpoint 1 in (y, z)
    observed_length: float  # Euclidean length of observed (clipped) segment
    observed_angle: float   # axial orientation in [0, pi)
    censoring_type: str     # 'complete', 'one_end_clipped', 'both_end_clipped'
    face_id: int = 0
    set_id: Optional[int] = None


# ---------------------------------------------------------------------------
# 1. compute_trace_length
# ---------------------------------------------------------------------------
def compute_trace_length(trace) -> float:
    """Compute 2D Euclidean trace length from endpoints.

    Accepts:
        - TraceRecord with .p0, .p1 arrays
        - FaceTrace with .p0_y, .p0_z, .p1_y, .p1_z attributes
        - dict with keys 'p0' and 'p1' (each a 2-element sequence)
    """
    if hasattr(trace, 'p0') and hasattr(trace, 'p1'):
        p0, p1 = np.asarray(trace.p0), np.asarray(trace.p1)
    elif hasattr(trace, 'p0_y'):
        p0 = np.array([trace.p0_y, trace.p0_z])
        p1 = np.array([trace.p1_y, trace.p1_z])
    elif isinstance(trace, dict):
        p0, p1 = np.asarray(trace['p0']), np.asarray(trace['p1'])
    else:
        raise TypeError(f"Unsupported trace type: {type(trace)}")
    return float(np.linalg.norm(p1 - p0))


# ---------------------------------------------------------------------------
# 2. compute_trace_angle
# ---------------------------------------------------------------------------
def compute_trace_angle(trace) -> float:
    """Compute axial orientation angle in [0, pi).

    Trace orientation is axial: theta and theta+pi are equivalent.
    """
    if hasattr(trace, 'p0') and hasattr(trace, 'p1'):
        p0, p1 = np.asarray(trace.p0), np.asarray(trace.p1)
    elif hasattr(trace, 'p0_y'):
        p0 = np.array([trace.p0_y, trace.p0_z])
        p1 = np.array([trace.p1_y, trace.p1_z])
    elif isinstance(trace, dict):
        p0, p1 = np.asarray(trace['p0']), np.asarray(trace['p1'])
    else:
        raise TypeError(f"Unsupported trace type: {type(trace)}")

    d = p1 - p0
    angle = np.arctan2(d[1], d[0])
    # Normalize to [0, pi)
    angle = angle % np.pi
    return float(angle)


# ---------------------------------------------------------------------------
# 3. classify_censoring
# ---------------------------------------------------------------------------
def _point_to_polygon_distance(pt: np.ndarray, poly: np.ndarray) -> float:
    """Minimum distance from a 2D point to a closed polygon boundary."""
    min_dist = float('inf')
    n = len(poly)
    for i in range(n):
        v0 = poly[i]
        v1 = poly[(i + 1) % n]
        d = v1 - v0
        len_sq = np.dot(d, d)
        if len_sq < 1e-12:
            dist = np.linalg.norm(pt - v0)
        else:
            t = np.clip(np.dot(pt - v0, d) / len_sq, 0.0, 1.0)
            closest = v0 + t * d
            dist = np.linalg.norm(pt - closest)
        if dist < min_dist:
            min_dist = dist
    return float(min_dist)


def classify_censoring(trace, window_polygon: np.ndarray,
                       eps: float = 0.10) -> str:
    """Classify a trace as 'complete', 'one_end_clipped', or 'both_end_clipped'.

    A trace endpoint is considered clipped if it lies within *eps* metres
    of the window polygon boundary.

    Parameters
    ----------
    trace : TraceRecord, FaceTrace, or dict
    window_polygon : np.ndarray, shape (N, 2)
    eps : float, tolerance in metres

    Returns
    -------
    str : censoring type label
    """
    if hasattr(trace, 'p0') and hasattr(trace, 'p1'):
        p0, p1 = np.asarray(trace.p0), np.asarray(trace.p1)
    elif hasattr(trace, 'p0_y'):
        p0 = np.array([trace.p0_y, trace.p0_z])
        p1 = np.array([trace.p1_y, trace.p1_z])
    elif isinstance(trace, dict):
        p0, p1 = np.asarray(trace['p0']), np.asarray(trace['p1'])
    else:
        raise TypeError(f"Unsupported trace type: {type(trace)}")

    poly = np.asarray(window_polygon)
    touch0 = _point_to_polygon_distance(p0, poly) <= eps
    touch1 = _point_to_polygon_distance(p1, poly) <= eps

    if touch0 and touch1:
        return 'both_end_clipped'
    elif touch0 or touch1:
        return 'one_end_clipped'
    else:
        return 'complete'


# ---------------------------------------------------------------------------
# 4. build_observed_trace_histogram
# ---------------------------------------------------------------------------
def build_observed_trace_histogram(
    traces: list,
    length_bins: np.ndarray,
    angle_bins: Optional[np.ndarray] = None,
    include_censoring: bool = True
) -> Dict:
    """Build raw observed trace length histogram.

    Parameters
    ----------
    traces : list of TraceRecord (or compatible)
    length_bins : 1-D array of bin edges for trace length
    angle_bins : optional 1-D array of bin edges for orientation angle
    include_censoring : if True, separate counts by censoring type

    Returns
    -------
    dict with keys:
        'length_bin_edges', 'length_bin_centers',
        'raw_counts', 'normalized_histogram',
        and optionally 'counts_by_censoring' and 'counts_by_angle'
    """
    lengths = np.array([_get_length(t) for t in traces])
    raw_counts, _ = np.histogram(lengths, bins=length_bins)
    total = max(raw_counts.sum(), 1)
    centers = 0.5 * (length_bins[:-1] + length_bins[1:])

    result = {
        'length_bin_edges': length_bins,
        'length_bin_centers': centers,
        'raw_counts': raw_counts,
        'normalized_histogram': raw_counts / total,
    }

    if include_censoring:
        cens_types = ['complete', 'one_end_clipped', 'both_end_clipped']
        counts_by_c = {}
        for ct in cens_types:
            sub = [_get_length(t) for t in traces if _get_censoring(t) == ct]
            c, _ = np.histogram(sub, bins=length_bins)
            counts_by_c[ct] = c
        result['counts_by_censoring'] = counts_by_c

    if angle_bins is not None:
        angles = np.array([_get_angle(t) for t in traces])
        counts_by_a = {}
        for j in range(len(angle_bins) - 1):
            mask = (angles >= angle_bins[j]) & (angles < angle_bins[j + 1])
            sub = lengths[mask]
            c, _ = np.histogram(sub, bins=length_bins)
            a_label = f"{np.degrees(angle_bins[j]):.0f}-{np.degrees(angle_bins[j+1]):.0f}deg"
            counts_by_a[a_label] = c
        result['counts_by_angle'] = counts_by_a

    return result


# ---------------------------------------------------------------------------
# 5. estimate_observation_probability_mc
# ---------------------------------------------------------------------------
def estimate_observation_probability_mc(
    window_polygon: np.ndarray,
    length_bins: np.ndarray,
    angle_bins: np.ndarray,
    l_min: float = 0.15,
    n_mc: int = 10000,
    seed: int = 42
) -> np.ndarray:
    """Monte Carlo estimate of observation probability p_obs(L, theta).

    For each (length_bin, angle_bin) pair, randomly places synthetic true
    line segments in an expanded region around the window, clips them to
    the polygon, and counts observable fraction.

    Parameters
    ----------
    window_polygon : shape (N, 2) -- closed polygon vertices
    length_bins : 1-D bin edges for true length
    angle_bins : 1-D bin edges for orientation angle in [0, pi)
    l_min : minimum detectable trace length (truncation threshold)
    n_mc : Monte Carlo samples per (L, theta) cell
    seed : random seed

    Returns
    -------
    p_obs : shape (n_length_bins, n_angle_bins) -- observation probabilities
    """
    from trace_reconstruction.forward_simulator import (
        clip_line_segment_to_polygon, is_point_inside_polygon
    )

    rng = np.random.default_rng(seed)
    poly = np.asarray(window_polygon)

    # Bounding box of polygon with buffer
    ymin, zmin = poly.min(axis=0)
    ymax, zmax = poly.max(axis=0)

    n_lb = len(length_bins) - 1
    n_ab = len(angle_bins) - 1
    p_obs = np.zeros((n_lb, n_ab))

    for i in range(n_lb):
        L = 0.5 * (length_bins[i] + length_bins[i + 1])
        half_L = L / 2.0
        # Expand sampling region so midpoints outside window are included
        buf = half_L + 0.5
        ey_min, ey_max = ymin - buf, ymax + buf
        ez_min, ez_max = zmin - buf, zmax + buf

        for j in range(n_ab):
            theta = 0.5 * (angle_bins[j] + angle_bins[j + 1])
            dy = np.cos(theta) * half_L
            dz = np.sin(theta) * half_L

            n_observed = 0
            for _ in range(n_mc):
                # Random midpoint in expanded region
                my = rng.uniform(ey_min, ey_max)
                mz = rng.uniform(ez_min, ez_max)

                p0 = np.array([my - dy, mz - dz])
                p1 = np.array([my + dy, mz + dz])

                clipped = clip_line_segment_to_polygon(p0, p1, poly)
                if clipped:
                    for cp0, cp1 in clipped:
                        obs_len = np.linalg.norm(cp1 - cp0)
                        if obs_len >= l_min:
                            n_observed += 1
                            break  # count once per synthetic trace

            p_obs[i, j] = n_observed / n_mc

    return p_obs


# ---------------------------------------------------------------------------
# 6. build_bias_corrected_trace_distribution
# ---------------------------------------------------------------------------
def build_bias_corrected_trace_distribution(
    traces: list,
    length_bins: np.ndarray,
    angle_bins: np.ndarray,
    p_obs: np.ndarray,
    l_min: float = 0.15,
    use_complete_only: bool = True,
    clipped_treatment: str = 'report'
) -> Dict:
    """Estimate bias-corrected true trace length distribution via IPW.

    Uses inverse probability weighting (IPW):
        corrected_count = observed_count / p_obs

    Parameters
    ----------
    traces : list of TraceRecord or compatible
    length_bins, angle_bins : bin edges
    p_obs : observation probability matrix from MC estimation
    l_min : detection threshold
    use_complete_only : if True, only use complete traces for correction
    clipped_treatment : 'report' (default) -- report clipped counts separately;
                        'include' -- include clipped traces with same IPW;
                        'exclude' -- silently drop clipped traces

    Returns
    -------
    dict with corrected distribution arrays
    """
    n_lb = len(length_bins) - 1
    n_ab = len(angle_bins) - 1
    centers = 0.5 * (length_bins[:-1] + length_bins[1:])

    # Filter traces
    if use_complete_only:
        selected = [t for t in traces if _get_censoring(t) == 'complete']
        clipped = [t for t in traces if _get_censoring(t) != 'complete']
    elif clipped_treatment == 'exclude':
        selected = [t for t in traces if _get_censoring(t) == 'complete']
        clipped = [t for t in traces if _get_censoring(t) != 'complete']
    else:
        selected = list(traces)
        clipped = []

    if use_complete_only and len(clipped) > 0:
        n_1clip = sum(1 for t in clipped if _get_censoring(t) == 'one_end_clipped')
        n_2clip = sum(1 for t in clipped if _get_censoring(t) == 'both_end_clipped')
        warnings.warn(
            f"TBTD Estimator: {n_1clip} one-end-clipped and {n_2clip} both-end-clipped "
            f"traces excluded from IPW correction. Their observed lengths are lower bounds "
            f"of the true lengths. Total complete traces used: {len(selected)}."
        )

    # Build observed histogram for selected traces
    lengths_s = np.array([_get_length(t) for t in selected])
    angles_s = np.array([_get_angle(t) for t in selected])

    obs_2d = np.zeros((n_lb, n_ab))
    for k in range(len(selected)):
        li = np.searchsorted(length_bins, lengths_s[k], side='right') - 1
        ai = np.searchsorted(angle_bins, angles_s[k], side='right') - 1
        li = np.clip(li, 0, n_lb - 1)
        ai = np.clip(ai, 0, n_ab - 1)
        obs_2d[li, ai] += 1

    # IPW correction
    corrected_2d = np.zeros_like(obs_2d)
    for i in range(n_lb):
        for j in range(n_ab):
            if p_obs[i, j] > 1e-6:
                corrected_2d[i, j] = obs_2d[i, j] / p_obs[i, j]
            else:
                corrected_2d[i, j] = 0.0  # unobservable bin

    # Marginalize over angle to get 1-D length distribution
    raw_counts_1d = obs_2d.sum(axis=1)
    corrected_counts_1d = corrected_2d.sum(axis=1)

    total_corr = max(corrected_counts_1d.sum(), 1e-12)
    corrected_prob = corrected_counts_1d / total_corr
    corrected_cdf = np.cumsum(corrected_prob)

    total_raw = max(raw_counts_1d.sum(), 1)
    raw_prob = raw_counts_1d / total_raw
    raw_cdf = np.cumsum(raw_prob)

    # Clipped trace summary
    clipped_summary = {}
    if clipped:
        for ct in ['one_end_clipped', 'both_end_clipped']:
            sub = [_get_length(t) for t in clipped if _get_censoring(t) == ct]
            clipped_summary[ct] = {
                'count': len(sub),
                'mean_observed_length': float(np.mean(sub)) if sub else 0.0,
            }

    result = {
        'length_bin_edges': length_bins,
        'length_bin_centers': centers,
        'raw_counts': raw_counts_1d,
        'corrected_counts': corrected_counts_1d,
        'corrected_probability': corrected_prob,
        'corrected_cdf': corrected_cdf,
        'raw_probability': raw_prob,
        'raw_cdf': raw_cdf,
        'n_complete_used': len(selected),
        'clipped_summary': clipped_summary,
        'corrected_2d': corrected_2d,
        'obs_2d': obs_2d,
        'angle_bins': angle_bins,
    }

    # Sanity check: compare raw vs corrected mean
    raw_mean = float(np.sum(centers * raw_prob))
    corr_mean = float(np.sum(centers * corrected_prob))
    result['raw_mean_length'] = raw_mean
    result['corrected_mean_length'] = corr_mean
    print(f"  [TBTD] Raw mean trace length     : {raw_mean:.3f} m")
    print(f"  [TBTD] Corrected mean trace length: {corr_mean:.3f} m  (Pahl 1981 sanity check)")

    return result


# ---------------------------------------------------------------------------
# 7. plot_trace_distribution_comparison
# ---------------------------------------------------------------------------
def plot_trace_distribution_comparison(result: Dict, output_dir: str,
                                       prefix: str = '') -> List[str]:
    """Generate validation plots comparing raw vs corrected distributions.

    Saves:
        - raw_vs_corrected_trace_histogram.png
        - raw_vs_corrected_trace_cdf.png

    Returns list of saved file paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    saved = []
    centers = result['length_bin_centers']
    edges = result['length_bin_edges']
    widths = np.diff(edges)

    # --- Histogram ---
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(centers - widths * 0.2, result['raw_counts'], width=widths * 0.4,
           alpha=0.7, label='Raw observed', color='#5B9BD5', edgecolor='white')
    ax.bar(centers + widths * 0.2, result['corrected_counts'], width=widths * 0.4,
           alpha=0.7, label='Bias-corrected (IPW)', color='#ED7D31', edgecolor='white')
    ax.set_xlabel('Trace length (m)', fontsize=12)
    ax.set_ylabel('Count (weighted)', fontsize=12)
    ax.set_title('Raw Observed vs Estimated Bias-Corrected Trace Length Distribution',
                 fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Annotate means
    ax.axvline(result['raw_mean_length'], color='#5B9BD5', ls='--', lw=1.5,
               label=f"Raw mean = {result['raw_mean_length']:.2f} m")
    ax.axvline(result['corrected_mean_length'], color='#ED7D31', ls='--', lw=1.5,
               label=f"Corrected mean = {result['corrected_mean_length']:.2f} m")
    ax.legend(fontsize=10)

    p = os.path.join(output_dir, f'{prefix}raw_vs_corrected_trace_histogram.png')
    fig.savefig(p, dpi=150, bbox_inches='tight')
    plt.close(fig)
    saved.append(p)

    # --- CDF ---
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.step(centers, result['raw_cdf'], where='mid', lw=2,
            label='Raw observed CDF', color='#5B9BD5')
    ax.step(centers, result['corrected_cdf'], where='mid', lw=2,
            label='Bias-corrected CDF (IPW)', color='#ED7D31')
    ax.set_xlabel('Trace length (m)', fontsize=12)
    ax.set_ylabel('Cumulative probability', fontsize=12)
    ax.set_title('Raw vs Estimated Bias-Corrected Trace Length CDF', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    p = os.path.join(output_dir, f'{prefix}raw_vs_corrected_trace_cdf.png')
    fig.savefig(p, dpi=150, bbox_inches='tight')
    plt.close(fig)
    saved.append(p)

    return saved


# ---------------------------------------------------------------------------
# 8. export_corrected_distribution
# ---------------------------------------------------------------------------
def export_corrected_distribution(result: Dict, output_path: str) -> str:
    """Save corrected trace length distribution as CSV.

    Fields: length_bin_left, length_bin_right, length_bin_center,
            raw_count, corrected_count, corrected_probability, corrected_cdf
    """
    edges = result['length_bin_edges']
    centers = result['length_bin_centers']
    n = len(centers)

    lines = ['length_bin_left,length_bin_right,length_bin_center,'
             'raw_count,corrected_count,corrected_probability,corrected_cdf']
    for i in range(n):
        lines.append(
            f"{edges[i]:.4f},{edges[i+1]:.4f},{centers[i]:.4f},"
            f"{result['raw_counts'][i]:.2f},{result['corrected_counts'][i]:.4f},"
            f"{result['corrected_probability'][i]:.6f},{result['corrected_cdf'][i]:.6f}"
        )

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')

    return output_path


# ---------------------------------------------------------------------------
# Convenience: convert FaceTrace list to TraceRecord list
# ---------------------------------------------------------------------------
def facetrace_to_records(traces, window_polygon: np.ndarray,
                         eps: float = 0.10) -> List[TraceRecord]:
    """Convert a list of FaceTrace objects to TraceRecord list,
    classifying censoring along the way."""
    records = []
    poly = np.asarray(window_polygon)
    for t in traces:
        p0 = np.array([t.p0_y, t.p0_z])
        p1 = np.array([t.p1_y, t.p1_z])
        ct = classify_censoring(t, poly, eps=eps)
        records.append(TraceRecord(
            p0=p0, p1=p1,
            observed_length=compute_trace_length(t),
            observed_angle=compute_trace_angle(t),
            censoring_type=ct,
            face_id=t.face_id,
            set_id=getattr(t, 'set_id', None),
        ))
    return records


# ---------------------------------------------------------------------------
# Full pipeline convenience function
# ---------------------------------------------------------------------------
def run_tbtd_pipeline(
    traces: list,
    window_polygon: np.ndarray,
    output_dir: str,
    l_min: float = 0.15,
    length_bin_max: float = 12.0,
    n_length_bins: int = 24,
    n_angle_bins: int = 6,
    n_mc: int = 5000,
    seed: int = 42,
    prefix: str = '',
    eps: float = 0.10,
) -> Dict:
    """Run the complete TBTD pipeline end-to-end.

    Parameters
    ----------
    traces : list of FaceTrace or TraceRecord objects
    window_polygon : shape (N, 2) tunnel face polygon
    output_dir : where to save outputs
    l_min : minimum detectable trace length
    length_bin_max : upper edge of length bins
    n_length_bins : number of length bins
    n_angle_bins : number of angle bins in [0, pi)
    n_mc : Monte Carlo samples per cell
    seed : random seed
    prefix : filename prefix for outputs
    eps : censoring classification tolerance

    Returns
    -------
    dict with all corrected distribution data
    """
    os.makedirs(output_dir, exist_ok=True)
    poly = np.asarray(window_polygon)

    # Convert to TraceRecord if needed
    if traces and hasattr(traces[0], 'p0_y'):
        records = facetrace_to_records(traces, poly, eps=eps)
    elif traces and isinstance(traces[0], TraceRecord):
        records = traces
    else:
        records = traces  # assume compatible

    print(f"\n{'='*70}")
    print(f" TBTD Estimator -- Tunnel-window Bias-corrected Trace Distribution")
    print(f"{'='*70}")
    print(f"  Total traces         : {len(records)}")

    # Censoring summary
    n_complete = sum(1 for r in records if r.censoring_type == 'complete')
    n_1clip = sum(1 for r in records if r.censoring_type == 'one_end_clipped')
    n_2clip = sum(1 for r in records if r.censoring_type == 'both_end_clipped')
    print(f"  Complete             : {n_complete}")
    print(f"  One-end clipped      : {n_1clip}")
    print(f"  Both-end clipped     : {n_2clip}")
    print(f"  Detection threshold  : {l_min} m")

    # Bin definitions
    length_bins = np.linspace(l_min, length_bin_max, n_length_bins + 1)
    angle_bins = np.linspace(0, np.pi, n_angle_bins + 1)

    # Stage 1: raw histogram
    print(f"\n[Stage 1] Building raw observed histogram...")
    raw_hist = build_observed_trace_histogram(records, length_bins, angle_bins)

    # Stage 2: MC observation probability
    print(f"[Stage 2] Monte Carlo observation probability estimation (n_mc={n_mc})...")
    p_obs = estimate_observation_probability_mc(
        poly, length_bins, angle_bins, l_min=l_min, n_mc=n_mc, seed=seed
    )
    print(f"  p_obs range: [{p_obs.min():.4f}, {p_obs.max():.4f}]")

    # Stage 3: IPW correction
    print(f"[Stage 3] Inverse probability weighting correction...")
    result = build_bias_corrected_trace_distribution(
        records, length_bins, angle_bins, p_obs,
        l_min=l_min, use_complete_only=True
    )

    # Stage 4: save outputs
    print(f"\n[Stage 4] Saving outputs to {output_dir}")
    plots = plot_trace_distribution_comparison(result, output_dir, prefix=prefix)
    for p in plots:
        print(f"  Plot saved: {p}")

    csv_path = os.path.join(output_dir, f'{prefix}corrected_trace_distribution.csv')
    export_corrected_distribution(result, csv_path)
    print(f"  CSV  saved: {csv_path}")

    print(f"\n{'='*70}")
    print(f" TBTD Estimator complete.")
    print(f"{'='*70}\n")

    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _get_length(t) -> float:
    if hasattr(t, 'observed_length'):
        return t.observed_length
    return compute_trace_length(t)

def _get_angle(t) -> float:
    if hasattr(t, 'observed_angle'):
        return t.observed_angle
    return compute_trace_angle(t)

def _get_censoring(t) -> str:
    if hasattr(t, 'censoring_type'):
        return t.censoring_type
    if hasattr(t, 'censoring_class'):
        mapping = {0: 'complete', 1: 'one_end_clipped', 2: 'both_end_clipped'}
        return mapping.get(t.censoring_class, 'complete')
    return 'complete'
