"""Unit tests for Branch 3: synthetic-validation."""
import math
import pytest
import numpy as np

from dfnrec.validation.generator import SyntheticDFNGenerator
from dfnrec.validation.metrics import (
    association_precision_recall,
    plane_normal_angular_error,
    radius_map_relative_error,
    p32_error,
    compare_dfn_parameters,
)
from dfnrec.models import (
    ReconstructedDisc,
    ReliabilityClass,
    FractureSetSizeIntensity,
    SizeModel,
    DFNParameterSet,
    FractureSetOrientation,
)


class TestSyntheticGenerator:
    """Test that the synthetic generator produces sensible output."""

    def test_generate_basic_structure(self):
        gen = SyntheticDFNGenerator(seed=42)
        gt = gen.generate(n_sets=2, n_faces=4)

        assert len(gt.faces) == 4
        assert len(gt.true_discs) > 0
        assert len(gt.traces) > 0
        assert gt.dfn_params is not None

    def test_face_ids_unique(self):
        gen = SyntheticDFNGenerator(seed=0)
        gt = gen.generate(n_faces=4)
        fids = [f.face_id for f in gt.faces]
        assert len(fids) == len(set(fids))

    def test_face_order_increasing(self):
        gen = SyntheticDFNGenerator(seed=0)
        gt = gen.generate(n_faces=4)
        xs = [f.face_x() for f in gt.faces]
        assert all(xs[i] < xs[i+1] for i in range(len(xs)-1))

    def test_trace_face_ids_valid(self):
        gen = SyntheticDFNGenerator(seed=1)
        gt = gen.generate(n_sets=2, n_faces=4)
        valid_face_ids = {f.face_id for f in gt.faces}
        for t in gt.traces:
            assert t.face_id in valid_face_ids

    def test_trace_set_ids_from_configured_sets(self):
        gen = SyntheticDFNGenerator(seed=2)
        gt = gen.generate(n_sets=2, n_faces=4)
        valid_set_ids = {d.set_id for d in gt.true_discs}
        for t in gt.traces:
            assert t.set_id in valid_set_ids

    def test_disc_source_is_observed_reconstructed(self):
        gen = SyntheticDFNGenerator(seed=3)
        gt = gen.generate(n_sets=1, n_faces=4)
        for d in gt.true_discs:
            assert d.source == "observed_reconstructed"

    def test_trace_lengths_above_L_min(self):
        gen = SyntheticDFNGenerator(seed=4)
        gt = gen.generate(n_sets=2, n_faces=4, L_min=0.1)
        for t in gt.traces:
            assert t.observed_length >= 0.1 - 1e-6

    def test_p32_parameters_exist(self):
        gen = SyntheticDFNGenerator(seed=5)
        gt = gen.generate(n_sets=2, n_faces=4)
        for sid, si in gt.dfn_params.size_intensity.items():
            assert si.P32_total is not None
            assert si.P32_total > 0
            assert si.k_r is not None

    def test_orientation_parameters_exist(self):
        gen = SyntheticDFNGenerator(seed=6)
        gt = gen.generate(n_sets=2, n_faces=4)
        for sid, ori in gt.dfn_params.orientation.items():
            assert ori.kappa > 0
            assert 0 <= ori.mean_plunge_deg <= 90

    def test_reproducibility(self):
        gen1 = SyntheticDFNGenerator(seed=99)
        gen2 = SyntheticDFNGenerator(seed=99)
        gt1 = gen1.generate(n_sets=2, n_faces=4)
        gt2 = gen2.generate(n_sets=2, n_faces=4)
        assert len(gt1.true_discs) == len(gt2.true_discs)
        assert len(gt1.traces) == len(gt2.traces)

    def test_different_seeds_give_different_results(self):
        gen1 = SyntheticDFNGenerator(seed=0)
        gen2 = SyntheticDFNGenerator(seed=1)
        gt1 = gen1.generate(n_sets=2, n_faces=4)
        gt2 = gen2.generate(n_sets=2, n_faces=4)
        # Different seeds should generally give different disc counts
        # (not guaranteed but very likely)
        assert len(gt1.true_discs) != len(gt2.true_discs) or len(gt1.traces) != len(gt2.traces)


class TestMetrics:
    """Test validation metric functions."""

    def test_association_perfect(self):
        pairs = [("T1", "D1"), ("T2", "D2")]
        result = association_precision_recall(pairs, pairs)
        assert result["precision"] == 1.0
        assert result["recall"] == 1.0
        assert result["f1"] == 1.0

    def test_association_no_overlap(self):
        pred = [("T1", "D1")]
        gt = [("T2", "D2")]
        result = association_precision_recall(pred, gt)
        assert result["precision"] == 0.0
        assert result["recall"] == 0.0

    def test_association_partial(self):
        pred = [("T1", "D1"), ("T2", "D3")]
        gt = [("T1", "D1"), ("T2", "D2")]
        result = association_precision_recall(pred, gt)
        assert result["precision"] == 0.5
        assert result["recall"] == 0.5

    def _make_disc(self, disc_id, normal):
        return ReconstructedDisc(
            disc_id=disc_id,
            center_xyz=[0, 0, 0],
            normal_xyz=normal,
            radius_m=1.0,
        )

    def test_plane_normal_angular_error_perfect(self):
        disc = self._make_disc("D1", [1.0, 0.0, 0.0])
        result = plane_normal_angular_error([disc], [disc])
        assert result["mean_deg"] < 1e-9

    def test_plane_normal_angular_error_90deg(self):
        d1 = self._make_disc("D1", [1.0, 0.0, 0.0])
        d2 = self._make_disc("D2", [0.0, 1.0, 0.0])
        result = plane_normal_angular_error([d1], [d2])
        assert abs(result["mean_deg"] - 90.0) < 0.5

    def test_p32_error_same(self):
        si = FractureSetSizeIntensity(set_id="S1", P32_total=0.5, P32_eff=0.4, k_r=2.5)
        result = p32_error(si, si)
        assert abs(result.get("p32_total_error", 0.0)) < 1e-9

    def test_p32_error_difference(self):
        si_true = FractureSetSizeIntensity(set_id="S1", P32_total=0.5, k_r=2.5)
        si_est = FractureSetSizeIntensity(set_id="S1", P32_total=0.6, k_r=2.8)
        result = p32_error(si_est, si_true)
        assert abs(result["p32_total_error"] - 0.1) < 1e-9
        assert abs(result["relative_p32_total_error"] - 0.2) < 1e-9

    def test_compare_dfn_parameters(self):
        ori = FractureSetOrientation(set_id="S1", mean_trend_deg=10, mean_plunge_deg=80, kappa=20)
        si = FractureSetSizeIntensity(set_id="S1", P32_total=0.5, k_r=2.5)
        params = DFNParameterSet(orientation={"S1": ori}, size_intensity={"S1": si})
        report = compare_dfn_parameters(params, params)
        assert "S1" in report
        assert abs(report["S1"].get("mean_pole_error_deg", 0.0)) < 0.1
        assert abs(report["S1"].get("kappa_error", 0.0)) < 1e-9
