"""Unit and integration tests for Branch 8: domain-composer & pipeline."""
import math
import numpy as np
import pytest

from dfnrec.models import (
    DomainGeometry,
    ReconstructedDisc,
    GeneratedHiddenFracture,
    DFNParameterSet,
    FractureSetOrientation,
    FractureSetSizeIntensity,
    SizeModel,
)
from dfnrec.composer import compose_domain
from dfnrec.pipeline import run_pipeline
from dfnrec.validation.generator import SyntheticDFNGenerator
from dfnrec.validation.metrics import compare_dfn_parameters


def test_compose_domain_clipping_and_diagnostics():
    # Domain geometry: 10m x 10m x 10m box centered at origin
    geom = DomainGeometry(
        x_min=-5.0, x_max=5.0,
        y_min=-5.0, y_max=5.0,
        z_min=-5.0, z_max=5.0,
    )

    # 1. Observed disc inside domain
    d_inside = ReconstructedDisc(
        disc_id="D_in", set_id="S1",
        center_xyz=[0.0, 0.0, 0.0],
        normal_xyz=[0.0, 0.0, 1.0],
        radius_m=1.0,
    )
    # 2. Observed disc outside domain
    d_outside = ReconstructedDisc(
        disc_id="D_out", set_id="S1",
        center_xyz=[6.0, 0.0, 0.0],
        normal_xyz=[0.0, 0.0, 1.0],
        radius_m=1.0,
    )

    # 3. Stochastic fracture inside domain
    h_inside = GeneratedHiddenFracture(
        disc_id="H_in", set_id="S1",
        center_xyz=[1.0, 1.0, 1.0],
        normal_xyz=[0.0, 0.0, 1.0],
        radius_m=1.5,
    )
    # 4. Stochastic fracture outside domain
    h_outside = GeneratedHiddenFracture(
        disc_id="H_out", set_id="S1",
        center_xyz=[1.0, 6.0, 1.0],
        normal_xyz=[0.0, 0.0, 1.0],
        radius_m=1.5,
    )

    # 5. Duplicate stochastic fracture inside domain (close to d_inside)
    h_dup = GeneratedHiddenFracture(
        disc_id="H_dup", set_id="S1",
        center_xyz=[0.01, 0.01, 0.01],
        normal_xyz=[0.01, 0.0, 0.99995],
        radius_m=1.01,
    )

    # Setup true DFNParameterSet to evaluate P32 warning
    # Target P32 = 0.05, Target P30 = 0.01
    si = FractureSetSizeIntensity(
        set_id="S1",
        size_model=SizeModel.POWER_LAW,
        k_r=2.5, r_min=0.5, r_max=10.0,
        P32_total=0.05,
    )
    dfn_params = DFNParameterSet(
        orientation={"S1": FractureSetOrientation("S1", 0, 90, 15)},
        size_intensity={"S1": si},
    )

    # Compose domain
    composed = compose_domain(
        reconstructed=[d_inside, d_outside],
        hidden=[h_inside, h_outside, h_dup],
        domain_geom=geom,
        dfn_params=dfn_params,
    )

    # Verify clipping
    # Only inside elements must remain (and h_dup must be filtered out as duplicate)
    assert len(composed.observed_discs) == 1
    assert composed.observed_discs[0].disc_id == "D_in"

    assert len(composed.hidden_fractures) == 1
    assert composed.hidden_fractures[0].disc_id == "H_in"

    # Verify statistics calculation
    # Area of d_inside: pi * 1.0^2 = pi
    # Area of h_inside: pi * 1.5^2 = 2.25 * pi
    # Total area: 3.25 * pi = 10.21
    # Volume: 10 * 10 * 10 = 1000
    # Apparent P32: 10.21 / 1000 = 0.01021
    # Target P32: 0.05
    # Relative error: (0.01021 - 0.05) / 0.05 = -0.7958 -> -79.6%
    assert composed.diagnostics is not None
    assert composed.diagnostics.n_observed_discs == 1
    assert composed.diagnostics.n_hidden_fractures == 1
    assert math.isclose(composed.diagnostics.p32_apparent["S1"], (math.pi + math.pi * 1.5**2) / 1000.0, rel_tol=1e-5)
    assert composed.diagnostics.p32_relative_error["S1"] < -0.70
    assert len(composed.diagnostics.warnings) > 0


def test_run_pipeline_integration():
    # Generate small synthetic dataset
    gen = SyntheticDFNGenerator(seed=8)
    gt = gen.generate(n_sets=1, n_faces=4, face_spacing=2.0, face_half_size=3.0, L_min=0.1)

    # Domain box matching the generator domain bounds
    bounds = gt.domain_bounds
    geom = DomainGeometry(
        x_min=bounds["x_min"], x_max=bounds["x_max"],
        y_min=bounds["y_min"], y_max=bounds["y_max"],
        z_min=bounds["z_min"], z_max=bounds["z_max"],
    )

    # Run the pipeline
    composed_domain = run_pipeline(
        faces=gt.faces,
        traces=gt.traces,
        domain_geom=geom,
        seed=10,
        r_min=0.5,
        r_max=15.0,
    )

    # Check that composed domain has all output structures
    assert composed_domain is not None
    assert composed_domain.domain_geometry == geom
    assert composed_domain.dfn_params is not None
    assert composed_domain.diagnostics is not None

    # Check observed & hidden separation
    assert len(composed_domain.observed_discs) > 0
    assert all(d.source == "observed_reconstructed" for d in composed_domain.observed_discs)
    assert all(h.source == "conditional_stochastic" for h in composed_domain.hidden_fractures)

    # Validate estimated parameters compared to ground truth
    report = compare_dfn_parameters(composed_domain.dfn_params, gt.dfn_params)
    assert "S1" in report
    s1_rep = report["S1"]
    
    # Should have pole orientation error and size_intensity comparison
    assert "mean_pole_error_deg" in s1_rep
    assert "size_intensity" in s1_rep
    assert s1_rep["mean_pole_error_deg"] < 15.0  # reasonable recovery
