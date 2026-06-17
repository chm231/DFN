"""Fisher MLE for fracture set orientation from reconstructed discs.

Algorithm
---------
1. Collect pole normals from reconstructed discs of a given set_id.
2. Apply orientation bias correction: weight each pole by ||n × m_face||
   where m_face is the mean face normal (sampling area correction).
3. Compute weighted mean resultant direction R̂ and length R.
4. Estimate kappa via MLE: solve A(kappa) = R/N where A(kappa) = coth(kappa) - 1/kappa.
5. Bootstrap CI on kappa.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import brentq

from dfnrec.models import ReconstructedDisc, FractureSetOrientation, Face
from dfnrec.geometry.vector import normalize, axial_angle, trend_plunge_from_normal


def _kappa_mle(R_bar: float) -> float:
    """Solve for Fisher kappa given mean resultant length R_bar = R / N.

    Uses the Langevin approximation A(kappa) = R_bar.
    A(kappa) = coth(kappa) - 1/kappa.

    Special cases:
    - R_bar ≈ 0 → kappa ≈ 0 (isotropic)
    - R_bar ≈ 1 → kappa → ∞ (perfectly concentrated)
    """
    if R_bar < 1e-6:
        return 0.0
    if R_bar > 1.0 - 1e-9:
        return 1e5  # effectively infinite concentration

    def objective(k):
        if k < 1e-6:
            return k / 3.0 - R_bar  # Taylor expansion
        return 1.0 / math.tanh(k) - 1.0 / k - R_bar

    # Bracket: objective(0) = -R_bar < 0, objective(large) → 1 - R_bar > 0
    try:
        kappa = brentq(objective, 1e-9, 1e6, xtol=1e-4)
    except ValueError:
        kappa = 3.0 * R_bar / (1.0 - R_bar**2)  # approximation
    return float(kappa)


def estimate_fisher_orientation(
    discs: List[ReconstructedDisc],
    set_id: str,
    faces: Optional[List[Face]] = None,
    apply_bias_correction: bool = True,
    n_bootstrap: int = 200,
    bootstrap_seed: int = 0,
) -> Optional[FractureSetOrientation]:
    """Estimate Fisher orientation parameters for a fracture set.

    Parameters
    ----------
    discs : list of ReconstructedDisc
        All reconstructed discs (any set_id; filtered internally).
    set_id : str
        Which set to process.
    faces : list of Face or None
        Observation faces. Used for orientation bias correction.
        If None, no bias correction is applied.
    apply_bias_correction : bool
        If True, weight poles by sampling area correction factor.
    n_bootstrap : int
        Number of bootstrap replicates for kappa CI.
    bootstrap_seed : int

    Returns
    -------
    FractureSetOrientation or None if fewer than 2 discs are available.
    """
    # Filter discs by set_id and exclude biased single-face discs
    multi_face_discs = [d for d in discs if d.set_id == set_id and d.n_faces_observed >= 2]
    set_discs = []
    for d in discs:
        if d.set_id != set_id:
            continue
        # If single-face disc and normal_x ≈ 0, it's highly biased due to the default cross product,
        # but only filter it out if we have enough multi-face discs to get a reliable estimate.
        if len(multi_face_discs) >= 3:
            if d.n_faces_observed < 2 and abs(d.normal_xyz[0]) < 1e-6:
                continue
        set_discs.append(d)

    n = len(set_discs)
    if n < 2:
        return None

    normals = np.array([normalize(np.asarray(d.normal_xyz, dtype=float)) for d in set_discs])

    # Orientation bias correction weights
    weights = np.ones(n, dtype=float)
    if apply_bias_correction and faces is not None and len(faces) > 0:
        # Mean face normal (all faces ≈ same direction for tunnel)
        m_face = normalize(np.mean([np.asarray(f.normal_xyz) for f in faces], axis=0))
        for k in range(n):
            # Correction: weight is inverse of visibility probability
            # visibility probability ∝ |n × m_face| = sin(angle)
            cross = np.cross(normals[k], m_face)
            weights[k] = 1.0 / max(np.linalg.norm(cross), 0.05)

    w_sum = weights.sum()
    weights /= w_sum

    # Use Watson second-order tensor to find the mean direction of axial data
    T = np.zeros((3, 3))
    for k in range(n):
        T += weights[k] * np.outer(normals[k], normals[k])
    
    evals, evecs = np.linalg.eigh(T)
    mean_direction = evecs[:, 2]

    # Flip all normals to same hemisphere as mean_direction
    for k in range(n):
        if np.dot(normals[k], mean_direction) < 0:
            normals[k] = -normals[k]

    # Weighted mean resultant length
    R = float(np.sum(weights * np.dot(normals, mean_direction)))
    R_bar = R  # already normalised by w_sum=1

    kappa = _kappa_mle(R_bar)
    trend, plunge = trend_plunge_from_normal(mean_direction)

    # Bootstrap CI on kappa
    rng = np.random.default_rng(bootstrap_seed)
    kappa_boots = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        boot_normals = normals[idx]
        boot_weights = weights[idx]
        boot_w_sum = boot_weights.sum()
        if boot_w_sum < 1e-9:
            continue
        boot_weights /= boot_w_sum
        boot_mean = (boot_normals * boot_weights[:, None]).sum(axis=0)
        R_boot = float(np.linalg.norm(boot_mean))
        R_boot = min(R_boot, 1.0 - 1e-9)
        kappa_boots.append(_kappa_mle(R_boot))

    ci_95 = None
    if kappa_boots:
        ci_95 = [
            float(np.percentile(kappa_boots, 2.5)),
            float(np.percentile(kappa_boots, 97.5)),
        ]

    return FractureSetOrientation(
        set_id=set_id,
        mean_trend_deg=trend,
        mean_plunge_deg=plunge,
        kappa=kappa,
        n_discs_used=n,
        kappa_ci_95=ci_95,
        orientation_bias_corrected=apply_bias_correction and faces is not None,
    )
