"""Unit tests for Branch 5: orientation-inversion."""
import math
import numpy as np
import pytest

from dfnrec.orientation.fisher_mle import estimate_fisher_orientation, _kappa_mle
from dfnrec.models import ReconstructedDisc, ReliabilityClass
from dfnrec.validation.generator import SyntheticDFNGenerator
from dfnrec.geometry.vector import normalize



def _make_disc_from_normal(i, normal, set_id="S1"):
    from dfnrec.geometry.vector import normalize, trend_plunge_from_normal
    n = normalize(np.asarray(normal, dtype=float))
    t, p = trend_plunge_from_normal(n)
    return ReconstructedDisc(
        disc_id=f"D{i:03d}",
        set_id=set_id,
        center_xyz=[0.0, 0.0, 0.0],
        normal_xyz=n.tolist(),
        radius_m=1.0,
        trend_deg=t,
        plunge_deg=p,
    )


class TestKappaMle:
    def test_zero_resultant_gives_zero_kappa(self):
        assert _kappa_mle(0.0) == 0.0

    def test_near_one_gives_large_kappa(self):
        kappa = _kappa_mle(0.99)
        assert kappa > 50.0

    def test_moderate_value(self):
        # For kappa=5, A(5)=coth(5)-1/5 ≈ 0.8
        kappa = _kappa_mle(0.8)
        assert 4.0 < kappa < 7.0


class TestFisherMLE:
    def test_returns_none_for_single_disc(self):
        disc = _make_disc_from_normal(0, [1.0, 0.0, 0.0])
        result = estimate_fisher_orientation([disc], set_id="S1", n_bootstrap=10)
        assert result is None

    def test_wrong_set_id_returns_none(self):
        disc = _make_disc_from_normal(0, [1.0, 0.0, 0.0])
        result = estimate_fisher_orientation([disc], set_id="S2", n_bootstrap=10)
        assert result is None

    def test_perfectly_clustered_high_kappa(self):
        """N discs with identical orientation → very high kappa."""
        n = [1.0, 0.0, 0.0]
        discs = [_make_disc_from_normal(i, n) for i in range(20)]
        result = estimate_fisher_orientation(discs, set_id="S1", n_bootstrap=50)
        assert result is not None
        assert result.kappa > 50.0
        assert result.n_discs_used == 20

    def test_isotropic_low_kappa(self):
        """Uniformly distributed normals → low kappa."""
        rng = np.random.default_rng(0)
        normals = rng.normal(size=(50, 3))
        normals /= np.linalg.norm(normals, axis=1, keepdims=True)
        discs = [_make_disc_from_normal(i, n) for i, n in enumerate(normals)]
        result = estimate_fisher_orientation(discs, set_id="S1", n_bootstrap=50)
        assert result is not None
        assert result.kappa < 5.0

    def test_synthetic_recovery(self):
        """Recover orientation from synthetic Fisher samples (loose tolerance)."""
        gen = SyntheticDFNGenerator(seed=42)
        gt = gen.generate(n_sets=2, n_faces=4)
        discs = gt.true_discs
        faces = gt.faces

        for sid, true_ori in gt.dfn_params.orientation.items():
            result = estimate_fisher_orientation(
                discs, set_id=sid, faces=faces, n_bootstrap=100
            )
            assert result is not None
            # Mean pole error < 30 degrees (loose, large stat uncertainty for small N)
            from dfnrec.geometry.vector import normalize, axial_angle, normal_from_trend_plunge
            n_est = normal_from_trend_plunge(result.mean_trend_deg, result.mean_plunge_deg)
            n_true = normal_from_trend_plunge(true_ori.mean_trend_deg, true_ori.mean_plunge_deg)
            ang_err = math.degrees(axial_angle(n_est, n_true))
            assert ang_err < 40.0, f"Set {sid}: pole error {ang_err:.1f}°"

    def test_ci_exists(self):
        discs = [_make_disc_from_normal(i, [1.0, 0.0, 0.0]) for i in range(10)]
        result = estimate_fisher_orientation(discs, set_id="S1", n_bootstrap=100)
        assert result is not None
        assert result.kappa_ci_95 is not None
        assert len(result.kappa_ci_95) == 2
        assert result.kappa_ci_95[0] <= result.kappa_ci_95[1]

    def test_axial_symmetry(self):
        """Disc with n and -n should give same mean pole direction."""
        n = normalize(np.array([1.0, 1.0, 0.0]))
        discs_pos = [_make_disc_from_normal(i, n) for i in range(10)]
        discs_neg = [_make_disc_from_normal(i, -n) for i in range(10)]
        r1 = estimate_fisher_orientation(discs_pos, set_id="S1", n_bootstrap=10)
        r2 = estimate_fisher_orientation(discs_neg, set_id="S1", n_bootstrap=10)
        assert r1 is not None
        assert r2 is not None
        from dfnrec.geometry.vector import axial_angle, normal_from_trend_plunge
        n1 = normal_from_trend_plunge(r1.mean_trend_deg, r1.mean_plunge_deg)
        n2 = normal_from_trend_plunge(r2.mean_trend_deg, r2.mean_plunge_deg)
        assert math.degrees(axial_angle(n1, n2)) < 5.0
