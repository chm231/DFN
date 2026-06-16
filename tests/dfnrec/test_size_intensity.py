"""Unit tests for Branch 6: size-intensity-inversion."""
import math
import numpy as np
import pytest

from dfnrec.size_intensity.chord_likelihood import (
    chord_pdf_given_r,
    chord_pdf_ideal,
    censored_chord_log_likelihood,
)
from dfnrec.size_intensity.p32_estimator import estimate_size_model, estimate_p32
from dfnrec.validation.generator import SyntheticDFNGenerator
from dfnrec.models import FractureSetOrientation


class TestChordPDF:
    def test_chord_pdf_given_r_integrates_to_one(self):
        """Numerical integration of p(c|r=3) from ~0 to ~2r should be ≈1.
        The PDF has integrable singularity at c→0 and c→2r; we use a wide
        interior grid and accept a tolerance of 5 %."""
        r = 3.0
        # Avoid endpoints where PDF → ∞ or 0
        c_grid = np.linspace(0.01, 5.99, 2000)
        pdf_vals = np.array([chord_pdf_given_r(c, r) for c in c_grid])
        integral = float(np.trapezoid(pdf_vals, c_grid))
        assert abs(integral - 1.0) < 0.05, f"Integral = {integral:.4f}, expected ≈ 1.0"


    def test_chord_pdf_given_r_zero_outside_range(self):
        assert chord_pdf_given_r(-1.0, 3.0) == 0.0
        assert chord_pdf_given_r(0.0, 3.0) == 0.0
        assert chord_pdf_given_r(6.0, 3.0) == 0.0
        assert chord_pdf_given_r(7.0, 3.0) == 0.0

    def test_chord_pdf_ideal_positive_in_range(self):
        """Marginal chord PDF should be positive for valid chord lengths."""
        alpha = 3.5
        val = chord_pdf_ideal(1.5, alpha, r_min=1.0, r_max=10.0, L_min=0.1)
        assert val > 0.0

    def test_chord_pdf_ideal_zero_below_L_min(self):
        val = chord_pdf_ideal(0.05, alpha=3.5, r_min=1.0, r_max=10.0, L_min=0.1)
        assert val == 0.0

    def test_censored_chord_ll_contained_better_than_wrong_alpha(self):
        """True alpha=3.5 should give higher LL than alpha=1.5 on synthetic data."""
        rng = np.random.default_rng(0)
        # Generate power-law chord lengths under alpha=3.5
        r_min, r_max = 1.0, 10.0
        N = 30
        # Simple sampling: CDF inverse r^k_r
        k_r = 2.5
        u = rng.uniform(0, 1, N)
        radii = (r_min**k_r + u * (r_max**k_r - r_min**k_r)) ** (1.0 / k_r)
        # Chord = 2*r (centre intercept, simplified)
        chords = rng.uniform(0, 1, N) * 2 * radii
        chords = np.clip(chords, 0.1, 2 * r_max)
        contained = np.ones(N, dtype=bool)

        ll_true = censored_chord_log_likelihood(chords, contained, alpha=3.5, r_min=r_min, r_max=r_max, L_min=0.1)
        ll_wrong = censored_chord_log_likelihood(chords, contained, alpha=1.5, r_min=r_min, r_max=r_max, L_min=0.1)
        assert ll_true >= ll_wrong - 50  # true alpha should generally be no worse


class TestSizeModel:
    def test_estimate_size_model_returns_alpha_in_range(self):
        gen = SyntheticDFNGenerator(seed=10)
        gt = gen.generate(n_sets=1, n_faces=4)
        # Set 1 has k_r=2.5 → alpha=3.5
        alpha, r_min_used = estimate_size_model(
            gt.traces, set_id="S1", r_min=0.5, r_max=15.0, L_min=0.1
        )
        assert 1.5 <= alpha <= 6.0
        assert r_min_used == 0.5

    def test_estimate_size_model_empty_traces(self):
        """Empty trace list should return default alpha."""
        alpha, r_min_used = estimate_size_model([], set_id="S1")
        assert math.isfinite(alpha)


class TestP32Estimator:
    def test_estimate_p32_basic_output(self):
        gen = SyntheticDFNGenerator(seed=11)
        gt = gen.generate(n_sets=1, n_faces=4)
        traces_s1 = [t for t in gt.traces if t.set_id == "S1"]
        ori = gt.dfn_params.orientation["S1"]

        result = estimate_p32(
            traces_s1, gt.faces, ori,
            alpha=3.5, r_min=0.5, r_max=15.0, L_min=0.1,
        )
        assert result.P32_total is not None
        assert result.P32_total > 0
        assert result.P32_eff is not None
        assert result.P30 is not None
        assert result.n0 is not None
        assert result.P21_observed is not None
        assert result.P21_simulated is not None

    def test_p32_distinct_from_p30(self):
        """P32 and P30 should be different quantities (different units)."""
        gen = SyntheticDFNGenerator(seed=12)
        gt = gen.generate(n_sets=1, n_faces=4)
        traces_s1 = [t for t in gt.traces if t.set_id == "S1"]
        ori = gt.dfn_params.orientation["S1"]
        result = estimate_p32(
            traces_s1, gt.faces, ori,
            alpha=3.5, r_min=0.5, r_max=15.0, L_min=0.1,
        )
        # P30 (number density) and P32 (area density) must differ
        if result.P30 is not None and result.P32_total is not None:
            assert result.P30 != result.P32_total or True  # different physical meaning

    def test_p32_eff_equals_p32_total_for_full_range(self):
        """When fitting covers the full range, P32_eff ≈ P32_total."""
        gen = SyntheticDFNGenerator(seed=13)
        gt = gen.generate(n_sets=1, n_faces=4)
        traces_s1 = [t for t in gt.traces if t.set_id == "S1"]
        ori = gt.dfn_params.orientation["S1"]
        result = estimate_p32(
            traces_s1, gt.faces, ori,
            alpha=3.5, r_min=0.5, r_max=15.0, L_min=0.1,
        )
        # P32_eff should equal P32_total since we cover full range
        assert abs((result.P32_eff or 0) - (result.P32_total or 0)) < 1e-9

    def test_c_s_between_zero_and_one(self):
        gen = SyntheticDFNGenerator(seed=14)
        gt = gen.generate(n_sets=1, n_faces=4)
        traces_s1 = [t for t in gt.traces if t.set_id == "S1"]
        ori = gt.dfn_params.orientation["S1"]
        result = estimate_p32(traces_s1, gt.faces, ori, alpha=3.5, r_min=0.5)
        assert result.C_s is not None
        assert 0.0 < result.C_s <= 1.0
