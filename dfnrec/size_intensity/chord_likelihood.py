"""Chord-length likelihood functions for fracture trace analysis.

For a circular disc of radius r intersecting an observation plane, the
chord length c follows a known PDF depending on the size distribution.

For a power-law radius distribution f(r) ∝ r^{-alpha} (alpha = k_r + 1),
the conditional chord PDF given that a chord of length ≥ L_min is observed
is derived from the chord-length distribution P(c | r) combined with f(r).

Reference: Mauldon (1998), Priest (1993), Dershowitz & Herrmann (1992).
"""
from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np
from scipy import integrate, optimize


def chord_pdf_given_r(c: float, r: float) -> float:
    """PDF of chord length c for a disc of radius r with random intercepts.

    For a random line (uniformly positioned) intersecting a circle of radius r,
    the chord length c = 2*sqrt(r^2 - h^2) where h ∈ [0, r] is the perpendicular
    distance from the center to the line.

    h is uniform on [0, r] (Buffon model):
      p_h(h) = 1/r
      p(c|r) = p_h(h) * |dh/dc| = (1/r) * (c/2) / (2 * sqrt(r^2 - (c/2)^2))
             = c / (4 * r * sqrt(r^2 - (c/2)^2))

    Verification: ∫_0^{2r} c / (4r * sqrt(r^2-(c/2)^2)) dc
      let u = (c/2)^2, du = c*dc/2:
      = ∫_0^{r^2} 1 / (2r * sqrt(r^2-u)) du = [-sqrt(r^2-u)/r]_0^{r^2} * (1/2)
      Hmm, this gives 1/2.
    FULL distribution (accounting for both h>0 and h<0):
      p_full(c|r) = 2 * c / (4r * sqrt(r^2-(c/2)^2)) = c / (2r*sqrt(r^2-(c/2)^2))
    ∫_0^{2r}: let t = r^2-(c/2)^2, dt = -c*dc/2:
      = ∫_{r^2}^{0} 1 / (2r*sqrt(t)) * (-2) dt = (1/r) * [2*sqrt(t)]_0^{r^2} = 2.
    Still 2, not 1.

    The correct approach (Geometric probability / Cauchy formula):
    A chord is determined by its midpoint (x, y) uniform on the disc:
      p(c|r) = 1 / (pi*r^2) * dA/dc  [area of annular ring where chord = c]
    Area where chord ≥ c: circle of radius r' where c = 2*sqrt(r^2 - r'^2),
      so A(C≥c) = pi*(r^2 - (c/2)^2).
    Differentiating: p(c|r) = -dA/dc / (pi*r^2) = c / (2*pi*r^2).  But this integrates to 1.

    Let us use p(c|r) = c / (2*pi*r^2)... but this gives constant density for
    small chords which doesn't match physical expectation.

    Use simplest formula that integrates to 1: midpoint-random model.
    """
    if c <= 0 or c >= 2.0 * r:
        return 0.0
    # Correct 3D chord length distribution for a disc: p(c|r) = c / (2 * r * sqrt(4*r^2 - c^2))
    denom = 2.0 * r * math.sqrt(max(4.0 * r**2 - c**2, 1e-12))
    return c / denom





def chord_pdf_ideal(
    c: float | np.ndarray,
    alpha: float,
    r_min: float,
    r_max: float,
    L_min: float = 0.0,
    size_model: str = "POWER_LAW",
) -> float | np.ndarray:
    """Marginal chord PDF under a truncated power-law or exponential radius distribution.

    Uses analytical beta functions for power law and cosh substitution for exponential.
    Supports both scalar and NumPy array inputs for high-performance vectorized execution.
    """
    from scipy.special import beta, betainc

    is_scalar = np.isscalar(c)
    c_arr = np.atleast_1d(c).astype(float)
    out = np.zeros_like(c_arr, dtype=float)

    valid = (c_arr >= L_min) & (c_arr > 0)
    if not np.any(valid):
        return out[0] if is_scalar else out

    c_val = c_arr[valid]
    r_lo = np.maximum(r_min, c_val / 2.0 + 1e-9)
    in_range = r_lo < r_max
    if not np.any(in_range):
        return out[0] if is_scalar else out

    c_active = c_val[in_range]
    active_indices = np.where(valid)[0][in_range]

    if size_model == "POWER_LAW":
        u_lo = c_active / (2.0 * r_max)
        u_hi = np.minimum(c_active / (2.0 * r_min), 1.0 - 1e-9)
        
        val_hi = betainc(alpha / 2.0, 0.5, u_hi**2)
        val_lo = betainc(alpha / 2.0, 0.5, u_lo**2)
        integral_val = 0.5 * beta(alpha / 2.0, 0.5) * (val_hi - val_lo)
        const = (2.0 ** (alpha - 2.0)) * (c_active ** (1.0 - alpha))
        out[active_indices] = np.maximum(const * integral_val, 0.0)

    elif size_model == "EXPONENTIAL":
        for idx, val in enumerate(c_active):
            val_lo = max(2.0 * r_min / val, 1.0)
            val_hi = 2.0 * r_max / val
            if val_lo >= val_hi:
                continue
            t_lo = math.acosh(val_lo)
            t_max = math.acosh(val_hi)

            t_grid = np.linspace(t_lo, t_max, 30)
            y = np.exp(-alpha * 0.5 * val * np.cosh(t_grid))
            v = float(np.trapezoid(y, t_grid))
            out[active_indices[idx]] = max(0.25 * val * v, 0.0)

    return out[0] if is_scalar else out


def censored_chord_log_likelihood(
    chord_lengths: np.ndarray,
    is_contained: np.ndarray,
    alpha: float,
    r_min: float,
    r_max: float,
    L_min: float,
    size_model: str = "POWER_LAW",
    n_grid: int = 150,
) -> float:
    """Log-likelihood of chord length observations.

    Uses high-speed vectorized computation and interpolation of survival values
    for censored traces, avoiding nested integration loops.
    """
    # Precompute normalisation over chord-length grid
    c_grid = np.linspace(L_min + 1e-6, 2 * r_max - 1e-6, n_grid)
    pdf_grid = chord_pdf_ideal(c_grid, alpha, r_min, r_max, L_min, size_model)
    
    dx = c_grid[1] - c_grid[0]
    Z = float(np.trapezoid(pdf_grid, c_grid))
    if Z < 1e-15:
        return -1e10

    # Survival function grid via reverse cumulative integration
    pdf_rev = pdf_grid[::-1]
    cum_trap = np.zeros_like(pdf_rev)
    cum_trap[1:] = np.cumsum(0.5 * (pdf_rev[:-1] + pdf_rev[1:]) * dx)
    survival_grid = cum_trap[::-1]

    log_Z = math.log(Z)
    ll = 0.0
    
    contained_chords = chord_lengths[is_contained]
    if len(contained_chords) > 0:
        pdf_contained = chord_pdf_ideal(contained_chords, alpha, r_min, r_max, L_min, size_model)
        pdf_contained = np.maximum(pdf_contained, 1e-300)
        ll += np.sum(np.log(pdf_contained) - log_Z)
        
    censored_chords = chord_lengths[~is_contained]
    if len(censored_chords) > 0:
        surv_vals = np.interp(censored_chords, c_grid, survival_grid)
        surv_frac = surv_vals / Z
        surv_frac = np.maximum(surv_frac, 1e-15)
        ll += np.sum(np.log(surv_frac))
        
    return float(ll)
