"""Unit tests for Branch 7: conditional-hidden-dfn."""
import math
import numpy as np
import pytest

from dfnrec.models import (
    Face,
    DomainModel,
    DomainGeometry,
    DFNParameterSet,
    FractureSetOrientation,
    FractureSetSizeIntensity,
    SizeModel,
    ReconstructedDisc,
)
from dfnrec.hidden_dfn.conditional_sampler import sample_hidden_dfn
from dfnrec.validation.metrics import non_observation_violation_count


@pytest.fixture
def basic_setup():
    # Setup domain geometry
    geom = DomainGeometry(
        x_min=-2.0, x_max=8.0,
        y_min=-5.0, y_max=5.0,
        z_min=-5.0, z_max=5.0,
    )
    domain = DomainModel(domain_id="domain_test", domain_geometry=geom)

    # Setup faces (2 faces at x=0.0 and x=2.0)
    window = [
        [-3.0, -3.0],
        [3.0, -3.0],
        [3.0, 3.0],
        [-3.0, 3.0],
    ]
    faces = [
        Face(
            face_id="F001",
            order_index=0,
            origin_xyz=[0.0, 0.0, 0.0],
            normal_xyz=[1.0, 0.0, 0.0],
            axis_u_xyz=[0.0, 1.0, 0.0],
            axis_v_xyz=[0.0, 0.0, 1.0],
            observation_window_polygon_uv=window,
            L_min=0.1,
        ),
        Face(
            face_id="F002",
            order_index=1,
            origin_xyz=[2.0, 0.0, 0.0],
            normal_xyz=[1.0, 0.0, 0.0],
            axis_u_xyz=[0.0, 1.0, 0.0],
            axis_v_xyz=[0.0, 0.0, 1.0],
            observation_window_polygon_uv=window,
            L_min=0.1,
        )
    ]

    # Setup DFN parameters (one set S1)
    ori = FractureSetOrientation(
        set_id="S1",
        mean_trend_deg=0.0,
        mean_plunge_deg=90.0,  # normal points straight up [0, 0, 1]
        kappa=15.0,
    )
    si = FractureSetSizeIntensity(
        set_id="S1",
        size_model=SizeModel.POWER_LAW,
        k_r=2.5,
        r_min=0.5,
        r_max=10.0,
        P32_total=0.5,
        P30=0.3,
        n0=0.3,
    )
    dfn_params = DFNParameterSet(
        orientation={"S1": ori},
        size_intensity={"S1": si},
    )

    return domain, dfn_params, faces


def test_sample_hidden_dfn_basic(basic_setup):
    domain, dfn_params, faces = basic_setup
    hidden = sample_hidden_dfn(domain, dfn_params, faces, seed=42, realization_id=1, L_min=0.1)

    assert isinstance(hidden, list)
    # Check realization_id and source
    for h in hidden:
        assert h.realization_id == 1
        assert h.source == "conditional_stochastic"
        assert h.set_id == "S1"
        assert len(h.center_xyz) == 3
        assert len(h.normal_xyz) == 3
        assert h.radius_m >= 0.5

        # Check bounds
        assert domain.domain_geometry.x_min <= h.center_xyz[0] <= domain.domain_geometry.x_max
        assert domain.domain_geometry.y_min <= h.center_xyz[1] <= domain.domain_geometry.y_max
        assert domain.domain_geometry.z_min <= h.center_xyz[2] <= domain.domain_geometry.z_max


def test_sample_hidden_dfn_non_observation_constraint(basic_setup):
    domain, dfn_params, faces = basic_setup
    hidden = sample_hidden_dfn(domain, dfn_params, faces, seed=123, L_min=0.1)

    # Use the validation metric to verify that NO hidden fractures violate non-observation
    violations = non_observation_violation_count(hidden, faces, L_min=0.1)
    assert violations == 0, f"Expected 0 violations, found {violations}"


def test_sample_hidden_dfn_duplicate_detection(basic_setup):
    domain, dfn_params, faces = basic_setup
    # Generate one sampler run to get a candidate
    hidden_first = sample_hidden_dfn(domain, dfn_params, faces, seed=7, L_min=0.1)
    if not hidden_first:
        return

    # Add the first generated fracture as an observed reconstructed disc
    sample_frac = hidden_first[0]
    disc = ReconstructedDisc(
        disc_id="D_GT_0",
        set_id=sample_frac.set_id,
        center_xyz=sample_frac.center_xyz,
        normal_xyz=sample_frac.normal_xyz,
        radius_m=sample_frac.radius_m,
        trend_deg=sample_frac.trend_deg,
        plunge_deg=sample_frac.plunge_deg,
    )
    domain.observed_discs.append(disc)

    # Generate again with the same seed. The duplicate should be rejected!
    hidden_second = sample_hidden_dfn(domain, dfn_params, faces, seed=7, L_min=0.1)

    # The second list should not contain a duplicate of the added observed disc
    for h in hidden_second:
        dist = np.linalg.norm(np.array(h.center_xyz) - np.array(disc.center_xyz))
        if dist < 0.05 * disc.radius_m:
            # Check if normal is also very close
            ang = np.arccos(np.clip(abs(np.dot(np.array(h.normal_xyz), disc.normal_np())), 0.0, 1.0))
            if ang < 0.1:
                assert abs(h.radius_m - disc.radius_m) / disc.radius_m >= 0.1, "Duplicate was not filtered out!"


def test_sample_hidden_dfn_seed_consistency(basic_setup):
    domain, dfn_params, faces = basic_setup
    h1 = sample_hidden_dfn(domain, dfn_params, faces, seed=99, L_min=0.1)
    h2 = sample_hidden_dfn(domain, dfn_params, faces, seed=99, L_min=0.1)
    h3 = sample_hidden_dfn(domain, dfn_params, faces, seed=100, L_min=0.1)

    assert len(h1) == len(h2)
    for f1, f2 in zip(h1, h2):
        assert f1.disc_id == f2.disc_id
        assert np.allclose(f1.center_xyz, f2.center_xyz)
        assert np.allclose(f1.normal_xyz, f2.normal_xyz)
        assert math.isclose(f1.radius_m, f2.radius_m)

    # h3 (different seed) should be different from h1
    if len(h1) > 0 and len(h3) > 0:
        # Check if first center coordinates are different
        assert not np.allclose(h1[0].center_xyz, h3[0].center_xyz)
