import numpy as np
from scipy.stats import kstest
from dfn_analysis.estimate_radius_powerlaw_window_mc import sample_true_chords

def test_chord_length_distribution_vs_analytical():
    """
    Validate that sample_true_chords matches the analytical 3D disc-plane intersection chord distribution.
    For a fixed radius R, the intersection offset is uniformly distributed from 0 to R.
    The half-chord is sqrt(R^2 - offset^2).
    The analytical CDF of chord length L is:
        F_L(l) = 1 - sqrt(1 - l^2 / (4 * R^2)) for 0 <= l <= 2R.
    """
    rng = np.random.default_rng(42)
    R = 5.0
    n_samples = 100000
    radii = np.full(n_samples, R, dtype=np.float64)
    
    # Generate chords using our function
    chords = sample_true_chords(radii, rng)
    
    # Analytical CDF function
    def analytical_cdf(l):
        l = np.clip(l, 0.0, 2 * R)
        return 1.0 - np.sqrt(1.0 - (l ** 2) / (4.0 * R ** 2))
    
    # Perform Kolmogorov-Smirnov test
    res = kstest(chords, analytical_cdf)
    
    # Assert that they are from the same distribution (p-value > 0.05)
    print(f"\n[KS Test] statistic={res.statistic:.5f}, p-value={res.pvalue:.5f}")
    assert res.pvalue > 0.01, f"KS test failed: p-value={res.pvalue}"
    
    # Check mean value: E[L] = R * pi / 2
    expected_mean = R * np.pi / 2.0
    actual_mean = np.mean(chords)
    relative_err = abs(actual_mean - expected_mean) / expected_mean
    print(f"[Mean Chord] Expected={expected_mean:.5f}, Actual={actual_mean:.5f}, Err={relative_err:.2%}")
    assert relative_err < 0.01, f"Mean chord length mismatch: {actual_mean} vs {expected_mean}"
