import os
import sys
import numpy as np
import pytest
from scipy.optimize import minimize_scalar

# Import from 'dfn generator v1/python'
_here = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_here)
sys.path.append(os.path.join(_project_root, "dfn generator v1", "python"))

from generate_dfn import sample_radius


def log_likelihood_truncated_powerlaw(alpha: float, r: np.ndarray, rmin: float, rmax: float) -> float:
    n = len(r)
    if n == 0:
        return -np.inf
    if abs(alpha - 1.0) < 1e-7:
        log_c = -np.log(rmin) - np.log(np.log(rmax / rmin))
    else:
        denom = rmin**(1.0 - alpha) - rmax**(1.0 - alpha)
        if denom <= 0:
            return -np.inf
        log_c = np.log(abs(alpha - 1.0)) - np.log(denom)
    return n * log_c - alpha * np.sum(np.log(r))


def estimate_alpha_mle(r: np.ndarray, rmin: float, rmax: float) -> float:
    res = minimize_scalar(
        lambda a: -log_likelihood_truncated_powerlaw(a, r, rmin, rmax),
        bounds=(1.01, 15.0),
        method="bounded"
    )
    return float(res.x) if res.success else float("nan")


def test_generator_estimator_convention_consistency() -> None:
    # 1. Setup powerlaw parameters
    kr_input = 3.0
    rmin = 1.0
    rmax = 250.0
    N = 100000
    
    size_dist = {
        'type': 'powerlaw',
        'kr': kr_input,
        'rmin': rmin,
        'rmax': rmax
    }
    
    # 2. Generate large N sample using generator's sampler
    rng = np.random.default_rng(42)
    # generate_dfn's sample_radius doesn't take rng directly, but takes a seed or uses global.
    # Let's pass a seed.
    r_sampled = sample_radius(size_dist, N, seed=42)
    
    # 3. Perform MLE on the raw radii
    alpha_mle = estimate_alpha_mle(r_sampled, rmin, rmax)
    
    # Under Convention A (which the generator uses):
    # f(r) ~ r^-(kr+1) => kr_mle = alpha_mle - 1
    kr_mle_A = alpha_mle - 1.0
    
    # Under Convention B:
    # f(r) ~ r^-kr => kr_mle = alpha_mle
    kr_mle_B = alpha_mle
    
    print(f"\n[Test Diagnostics]")
    print(f"  - Input kr      : {kr_input:.3f}")
    print(f"  - Est alpha_mle : {alpha_mle:.3f}")
    print(f"  - MLE under A   : {kr_mle_A:.3f} (error={kr_mle_A - kr_input:+.3f})")
    print(f"  - MLE under B   : {kr_mle_B:.3f} (error={kr_mle_B - kr_input:+.3f})")
    
    # 4. Verify that Convention A recovers the input kr within 0.1 tolerance
    assert abs(kr_mle_A - kr_input) < 0.1, f"Convention A failed to recover input kr. Est: {kr_mle_A:.3f}, Expected: {kr_input:.3f}"
    
    # 5. Verify that Convention B does NOT match (it should be off by approximately 1.0)
    assert abs(kr_mle_B - kr_input) > 0.5, f"Convention B matched when it shouldn't. Est: {kr_mle_B:.3f}, Expected: {kr_input:.3f}"
