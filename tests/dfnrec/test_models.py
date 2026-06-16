"""Unit tests for dfnrec.models (Branch 1: data-contracts)."""
import json
import math
import pytest

from dfnrec.models import (
    Face,
    Trace,
    CensorType,
    ReconstructedDisc,
    ReliabilityClass,
    SizeModel,
    FractureSetOrientation,
    FractureSetSizeIntensity,
    GeneratedHiddenFracture,
    DomainGeometry,
    DomainModel,
    Diagnostics,
)


# ======================================================================
# Helpers
# ======================================================================
def make_face(**kw) -> Face:
    defaults = dict(
        face_id="F001",
        order_index=0,
        origin_xyz=[0.0, 0.0, 0.0],
        normal_xyz=[1.0, 0.0, 0.0],
        axis_u_xyz=[0.0, 1.0, 0.0],
        axis_v_xyz=[0.0, 0.0, 1.0],
        observation_window_polygon_uv=[[-2, -2], [2, -2], [2, 2], [-2, 2]],
    )
    defaults.update(kw)
    return Face(**defaults)


def make_trace(**kw) -> Trace:
    defaults = dict(
        trace_id="F001_T001",
        face_id="F001",
        p0_xyz=[0.0, -1.0, 0.0],
        p1_xyz=[0.0, 1.0, 0.0],
    )
    defaults.update(kw)
    return Trace(**defaults)


def make_disc(**kw) -> ReconstructedDisc:
    defaults = dict(
        disc_id="D001",
        center_xyz=[1.0, 0.0, 0.0],
        normal_xyz=[1.0, 0.0, 0.0],
        radius_m=2.0,
        contributing_trace_ids=["F001_T001"],
        contributing_face_ids=["F001"],
    )
    defaults.update(kw)
    return ReconstructedDisc(**defaults)


# ======================================================================
# Face tests
# ======================================================================
class TestFace:
    def test_construction_basic(self):
        f = make_face()
        assert f.face_id == "F001"
        assert f.L_min == 0.1

    def test_normal_is_normalised(self):
        f = make_face(normal_xyz=[2.0, 0.0, 0.0])
        import math
        assert abs(math.sqrt(sum(x**2 for x in f.normal_xyz)) - 1.0) < 1e-9

    def test_window_area(self):
        f = make_face()  # 4x4 square
        assert abs(f.window_area() - 16.0) < 1e-9

    def test_json_round_trip(self):
        f = make_face(metadata={"date": "2026-06-16"})
        f2 = Face.from_json(f.to_json())
        assert f2.face_id == f.face_id
        assert f2.observation_window_polygon_uv == f.observation_window_polygon_uv
        assert f2.metadata["date"] == "2026-06-16"

    def test_invalid_origin_raises(self):
        with pytest.raises(ValueError, match="origin_xyz"):
            make_face(origin_xyz=[0.0, 0.0])  # only 2 components

    def test_invalid_polygon_raises(self):
        with pytest.raises(ValueError):
            make_face(observation_window_polygon_uv=[[-1, -1], [1, -1]])  # only 2 vertices

    def test_zero_normal_raises(self):
        with pytest.raises(ValueError, match="zero vector"):
            make_face(normal_xyz=[0.0, 0.0, 0.0])

    def test_negative_L_min_raises(self):
        with pytest.raises(ValueError, match="L_min"):
            make_face(L_min=-0.1)

    def test_face_x_nominal(self):
        f = make_face(x_nominal=5.0)
        assert f.face_x() == 5.0

    def test_face_x_from_origin(self):
        f = make_face(origin_xyz=[3.0, 0.0, 0.0])
        assert f.face_x() == 3.0


# ======================================================================
# Trace tests
# ======================================================================
class TestTrace:
    def test_observed_length(self):
        t = make_trace(p0_xyz=[0.0, 0.0, 0.0], p1_xyz=[0.0, 3.0, 4.0])
        assert abs(t.observed_length - 5.0) < 1e-9

    def test_observed_length_polyline(self):
        t = make_trace(
            p0_xyz=[0.0, 0.0, 0.0],
            p1_xyz=[0.0, 0.0, 3.0],
            polyline_xyz=[[0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 0, 3]],
        )
        assert abs(t.observed_length - 3.0) < 1e-9

    def test_midpoint(self):
        t = make_trace(p0_xyz=[0.0, -2.0, 0.0], p1_xyz=[0.0, 2.0, 0.0])
        assert t.midpoint_xyz == [0.0, 0.0, 0.0]

    def test_is_contained_both_natural(self):
        t = make_trace(censor_p0=CensorType.NATURAL, censor_p1=CensorType.NATURAL)
        assert t.is_contained

    def test_is_not_contained_when_clipped(self):
        t = make_trace(censor_p0=CensorType.CLIPPED, censor_p1=CensorType.NATURAL)
        assert not t.is_contained

    def test_n_clipped_endpoints(self):
        t = make_trace(censor_p0=CensorType.CLIPPED, censor_p1=CensorType.CLIPPED)
        assert t.n_clipped_endpoints == 2

    def test_json_round_trip(self):
        t = make_trace(censor_p0=CensorType.NATURAL, censor_p1=CensorType.CLIPPED)
        t2 = Trace.from_json(t.to_json())
        assert t2.censor_p0 == CensorType.NATURAL
        assert t2.censor_p1 == CensorType.CLIPPED
        assert abs(t2.observed_length - t.observed_length) < 1e-9

    def test_zero_length_trace_raises(self):
        with pytest.raises(ValueError, match="zero length"):
            make_trace(p0_xyz=[1.0, 0.0, 0.0], p1_xyz=[1.0, 0.0, 0.0])

    def test_censor_default_unknown(self):
        t = make_trace()
        assert t.censor_p0 == CensorType.UNKNOWN
        assert t.censor_p1 == CensorType.UNKNOWN


# ======================================================================
# ReconstructedDisc tests
# ======================================================================
class TestReconstructedDisc:
    def test_construction(self):
        d = make_disc()
        assert d.source == "observed_reconstructed"
        assert d.reliability_class == ReliabilityClass.C

    def test_normal_normalised(self):
        d = make_disc(normal_xyz=[3.0, 4.0, 0.0])
        n = d.normal_xyz
        assert abs(math.sqrt(sum(x**2 for x in n)) - 1.0) < 1e-9

    def test_area(self):
        d = make_disc(radius_m=2.0)
        assert abs(d.area_m2() - math.pi * 4.0) < 1e-9

    def test_json_round_trip(self):
        d = make_disc(reliability_class=ReliabilityClass.A)
        d2 = ReconstructedDisc.from_json(d.to_json())
        assert d2.reliability_class == ReliabilityClass.A
        assert d2.disc_id == d.disc_id

    def test_wrong_source_raises(self):
        with pytest.raises(ValueError, match="observed_reconstructed"):
            make_disc(source="conditional_stochastic")

    def test_classify_reliability_A(self):
        rc = ReconstructedDisc.classify_reliability(n_faces=3, plane_fit_rms=0.05, censoring_dominance=0.3)
        assert rc == ReliabilityClass.A

    def test_classify_reliability_B(self):
        rc = ReconstructedDisc.classify_reliability(n_faces=2, plane_fit_rms=0.15, censoring_dominance=0.2)
        assert rc == ReliabilityClass.B

    def test_classify_reliability_C(self):
        rc = ReconstructedDisc.classify_reliability(n_faces=1, plane_fit_rms=0.05, censoring_dominance=0.1)
        assert rc == ReliabilityClass.C

    def test_classify_reliability_D(self):
        rc = ReconstructedDisc.classify_reliability(n_faces=3, plane_fit_rms=0.05, censoring_dominance=0.9)
        assert rc == ReliabilityClass.D


# ======================================================================
# FractureSetSizeIntensity tests
# ======================================================================
class TestFractureSetSizeIntensity:
    def test_construction_power_law(self):
        si = FractureSetSizeIntensity(
            set_id="S1",
            size_model=SizeModel.POWER_LAW,
            k_r=2.85,
            r_min=1.0,
        )
        assert si.k_r == 2.85

    def test_json_round_trip(self):
        si = FractureSetSizeIntensity(
            set_id="S2",
            size_model=SizeModel.EXPONENTIAL,
            lambda_exp=0.5,
            P32_total=0.3,
            P32_eff=0.2,
        )
        si2 = FractureSetSizeIntensity.from_json(si.to_json())
        assert si2.size_model == SizeModel.EXPONENTIAL
        assert si2.lambda_exp == 0.5
        assert si2.P32_total == 0.3


# ======================================================================
# GeneratedHiddenFracture tests
# ======================================================================
class TestGeneratedHiddenFracture:
    def test_construction(self):
        gf = GeneratedHiddenFracture(
            disc_id="H001",
            set_id="S1",
            center_xyz=[5.0, 0.0, 0.0],
            normal_xyz=[1.0, 0.0, 0.0],
            radius_m=1.5,
        )
        assert gf.source == "conditional_stochastic"

    def test_wrong_source_raises(self):
        with pytest.raises(ValueError, match="conditional_stochastic"):
            GeneratedHiddenFracture(
                disc_id="H001", set_id="S1", source="observed_reconstructed",
                center_xyz=[0, 0, 0], normal_xyz=[1, 0, 0], radius_m=1.0,
            )

    def test_json_round_trip(self):
        gf = GeneratedHiddenFracture(
            disc_id="H002", set_id="S2",
            center_xyz=[1.0, 2.0, 3.0],
            normal_xyz=[0.0, 0.0, 1.0],
            radius_m=3.0,
            realization_id=7,
        )
        gf2 = GeneratedHiddenFracture.from_json(gf.to_json())
        assert gf2.realization_id == 7
        assert gf2.radius_m == 3.0


# ======================================================================
# DomainModel tests
# ======================================================================
class TestDomainModel:
    def test_construction_empty(self):
        dom = DomainModel()
        assert dom.all_fracture_count() == 0

    def test_source_separation(self):
        obs = make_disc()
        hid = GeneratedHiddenFracture(
            disc_id="H001", set_id="S1",
            center_xyz=[0, 0, 0], normal_xyz=[1, 0, 0], radius_m=1.0,
        )
        dom = DomainModel(observed_discs=[obs], hidden_fractures=[hid])
        assert dom.all_fracture_count() == 2
        assert all(d.source == "observed_reconstructed" for d in dom.observed_discs)
        assert all(f.source == "conditional_stochastic" for f in dom.hidden_fractures)

    def test_json_round_trip(self):
        geom = DomainGeometry(x_min=0, x_max=10, y_min=-5, y_max=5, z_min=-3, z_max=3)
        dom = DomainModel(
            domain_id="test_domain",
            domain_geometry=geom,
            observed_discs=[make_disc()],
        )
        js = dom.to_json()
        d = json.loads(js)
        assert d["domain_id"] == "test_domain"
        assert len(d["observed_discs"]) == 1
        assert d["observed_discs"][0]["source"] == "observed_reconstructed"

    def test_domain_geometry_volume(self):
        geom = DomainGeometry(0, 10, -5, 5, -3, 3)
        assert abs(geom.volume_m3() - 600.0) < 1e-9
