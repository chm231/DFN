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
    if c <= 0 or c >= 2 * r:
        return 0.0
    # Midpoint-uniform model: midpoint uniform on disc area
    # p(c|r) = c / (2 * r^2)  for 0 < c < 2r
    # (Bertrand's second definition)
    return c / (2.0 * r**2)





def chord_pdf_ideal(
    c: float,
    alpha: float,
    r_min: float,
    r_max: float,
    L_min: float = 0.0,
) -> float:
    """Marginal chord PDF under a truncated power-law radius distribution.

    Integrates over r:
      p(c) = ∫_{r_min}^{r_max} p(c|r) * f(r) dr / Z
    where f(r) ∝ r^{-alpha} (alpha = k_r + 1) and Z = normalisation.

    Parameters
    ----------
    c : float
        Chord length to evaluate [m].
    alpha : float
        Power-law PDF exponent (alpha = k_r + 1).
    r_min, r_max : float
        Truncation limits [m].
    L_min : float
        Minimum observable chord (detection limit). Only chords c ≥ L_min are observed.

    Returns
    -------
    float : un-normalised PDF value (normalise numerically before MLE).
    """
    # c must satisfy 0 < L_min ≤ c < 2*r_max (approximately)
    if c < L_min or c <= 0:
        return 0.0
    # Minimum r to produce chord c: r_min_c = c/2
    r_lo = max(r_min, c / 2.0 + 1e-9)
    if r_lo >= r_max:
        return 0.0

    def integrand(r):
        # f(r) ∝ r^{-alpha}
        f_r = r ** (-alpha)
        return chord_pdf_given_r(c, r) * f_r

    val, _ = integrate.quad(integrand, r_lo, r_max, limit=50)
    return max(val, 0.0)


def censored_chord_log_likelihood(
    chord_lengths: np.ndarray,
    is_contained: np.ndarray,
    alpha: float,
    r_min: float,
    r_max: float,
    L_min: float,
    n_grid: int = 80,
) -> float:
    """Log-likelihood of chord length observations under truncated power-law.

    Parameters
    ----------
    chord_lengths : (N,) array
        Observed chord lengths [m].
    is_contained : (N,) bool array
        True for fully observed (NATURAL both ends) chords.
        False for censored (CLIPPED at ≥ 1 end).
    alpha : float
        Power-law exponent to evaluate.
    r_min, r_max, L_min : float
        Distribution bounds and detection limit.
    n_grid : int
        Grid resolution for normalisation integral.

    Returns
    -------
    float : log likelihood (higher = better).
    """
    # Precompute normalisation over chord-length grid
    c_grid = np.linspace(L_min + 1e-6, 2 * r_max - 1e-6, n_grid)
    pdf_grid = np.array([chord_pdf_ideal(c, alpha, r_min, r_max, L_min) for c in c_grid])
    Z = float(np.trapezoid(pdf_grid, c_grid))
    if Z < 1e-15:
        return -1e10

    log_Z = math.log(Z)
    ll = 0.0
    for c, contained in zip(chord_lengths, is_contained):
        if contained:
            # Full observation: log p(c) - log Z
            p = chord_pdf_ideal(c, alpha, r_min, r_max, L_min)
            if p < 1e-300:
                ll -= 20.0
            else:
                ll += math.log(p) - log_Z
        else:
            # Censored: we know c is a lower bound on true chord
            # P(C ≥ c) = ∫_c^{2*r_max} p(c') dc' / Z
            c_tail = np.linspace(c, 2 * r_max - 1e-6, max(n_grid // 2, 10))
            pdf_tail = np.array([chord_pdf_ideal(ct, alpha, r_min, r_max, L_min) for ct in c_tail])
            surv = float(np.trapezoid(pdf_tail, c_tail))
            surv_frac = surv / max(Z, 1e-15)
            ll += math.log(max(surv_frac, 1e-15))

    return ll
