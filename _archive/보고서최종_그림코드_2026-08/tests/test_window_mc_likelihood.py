import numpy as np

from dfn_analysis.estimate_radius_powerlaw_window_mc import (
    binned_counts,
    clip_segment_to_polygon,
    empirical_survival_curve,
    fit_set_lmin,
    kaplan_meier_survival_curve,
    probability_table,
    simulate_window_observations,
)


def _square_polygon() -> np.ndarray:
    return np.array(
        [
            [0.0, 0.0],
            [10.0, 0.0],
            [10.0, 10.0],
            [0.0, 10.0],
        ],
        dtype=np.float64,
    )


def test_segment_fully_inside_is_class_zero() -> None:
    length, cls = clip_segment_to_polygon(np.array([2.0, 5.0]), np.array([8.0, 5.0]), _square_polygon())
    assert cls == 0
    assert abs(length - 6.0) < 1e-9


def test_segment_one_endpoint_outside_is_class_one() -> None:
    length, cls = clip_segment_to_polygon(np.array([-2.0, 5.0]), np.array([8.0, 5.0]), _square_polygon())
    assert cls == 1
    assert abs(length - 8.0) < 1e-9


def test_segment_both_endpoints_outside_crossing_is_class_two() -> None:
    length, cls = clip_segment_to_polygon(np.array([-2.0, 5.0]), np.array([12.0, 5.0]), _square_polygon())
    assert cls == 2
    assert abs(length - 10.0) < 1e-9


def test_mc_probability_table_normalizes() -> None:
    lengths = np.array([0.5, 0.8, 1.2, 2.0, 3.0], dtype=np.float64)
    classes = np.array([0, 1, 2, 0, 1], dtype=np.int32)
    edges = np.array([0.5, 1.0, 2.0, 4.0], dtype=np.float64)
    counts = binned_counts(lengths, classes, edges)
    probs, n_used = probability_table(lengths, classes, edges)
    assert n_used == int(np.sum(counts))
    assert abs(float(np.sum(probs)) - 1.0) < 1e-12

    # Test with weights
    weights = np.array([1.0, 2.0, 1.5, 3.0, 0.5], dtype=np.float64)
    probs_w, total_w = probability_table(lengths, classes, edges, weights=weights)
    assert total_w == float(np.sum(weights))
    assert abs(float(np.sum(probs_w)) - 1.0) < 1e-12


def test_simple_synthetic_window_recovers_kr_smoke() -> None:
    rng = np.random.default_rng(1234)
    polygon = _square_polygon()
    directions = np.array([[1.0, 0.0], [0.0, 1.0], [1.0 / np.sqrt(2.0), 1.0 / np.sqrt(2.0)]])
    kr_true = 3.0
    sim_lengths, sim_classes, _ = simulate_window_observations(
        kr_true,
        rmin=1.0,
        rmax=40.0,
        polygon_yz=polygon,
        directions_yz=directions,
        n_samples=12000,
        rng=rng,
    )
    mask = sim_lengths >= 0.5
    sim_lengths = sim_lengths[mask][:450]
    sim_classes = sim_classes[mask][:450]
    rows = []
    for idx, (length, cls) in enumerate(zip(sim_lengths, sim_classes)):
        direction = directions[idx % len(directions)]
        rows.append(
            {
                "set_id": 99,
                "face_id": 1,
                "observed_length_m": float(length),
                "censoring_class": int(cls),
                "p0_y": 0.0,
                "p0_z": 0.0,
                "p1_y": float(direction[0]),
                "p1_z": float(direction[1]),
            }
        )

    kr_grid = np.linspace(1.5, 5.5, 21)
    fit_row, _, _, candidate_rows = fit_set_lmin(
        set_id=99,
        set_rows=rows,
        polygon_yz=polygon,
        kr_grid=kr_grid,
        rmin=1.0,
        rmax=40.0,
        lmin_fit=0.5,
        mc_samples_per_grid=8000,
        bin_count=20,
        bin_mode="log",
        window_mode="polygon",
    )
    assert abs(float(fit_row["kr_window_mc_hat"]) - kr_true) < 0.7
    assert fit_row["center_weighting_status"] == "legacy_diagnostic"
    assert "kr_hat_length_only" in fit_row
    assert "kr_hat_class_only" in fit_row
    assert "kr_hat_joint" in fit_row
    assert len(candidate_rows) == len(kr_grid)


def test_length_only_component_and_weighted_fields_exist() -> None:
    polygon = _square_polygon()
    rows = [
        {
            "set_id": 7,
            "face_id": 1,
            "observed_length_m": 2.0,
            "censoring_class": 0,
            "p0_y": 0.0,
            "p0_z": 0.0,
            "p1_y": 1.0,
            "p1_z": 0.0,
        },
        {
            "set_id": 7,
            "face_id": 1,
            "observed_length_m": 3.0,
            "censoring_class": 1,
            "p0_y": 0.0,
            "p0_z": 0.0,
            "p1_y": 0.0,
            "p1_z": 1.0,
        },
    ]
    kr_grid = np.array([2.0, 3.0], dtype=np.float64)
    fit_row, profile_rows, _, _ = fit_set_lmin(
        set_id=7,
        set_rows=rows,
        polygon_yz=polygon,
        kr_grid=kr_grid,
        rmin=1.0,
        rmax=10.0,
        lmin_fit=0.5,
        mc_samples_per_grid=4000,
        bin_count=5,
        bin_mode="linear",
        window_mode="polygon",
        center_weighting="proposal_area",
        likelihood_component="length_only",
        class_likelihood_weight=0.25,
    )
    assert fit_row["center_weighting_status"] == "preferred_for_window_mc"
    assert fit_row["likelihood_component"] == "length_only"
    assert abs(float(fit_row["class_likelihood_weight"]) - 0.25) < 1e-12
    assert all("loglik_length_only" in row for row in profile_rows)
    assert all("loglik_class_only" in row for row in profile_rows)


def test_oracle_radius_mode_uses_observed_trace_radii() -> None:
    polygon = _square_polygon()
    rows = [
        {
            "set_id": 8,
            "face_id": 1,
            "observed_length_m": 2.0,
            "censoring_class": 0,
            "radius_m": 4.0,
            "p0_y": 0.0,
            "p0_z": 0.0,
            "p1_y": 1.0,
            "p1_z": 0.0,
        },
        {
            "set_id": 8,
            "face_id": 1,
            "observed_length_m": 1.5,
            "censoring_class": 1,
            "radius_m": 6.0,
            "p0_y": 0.0,
            "p0_z": 0.0,
            "p1_y": 0.0,
            "p1_z": 1.0,
        },
    ]
    kr_grid = np.array([2.0, 3.0], dtype=np.float64)
    fit_row, _, _, candidate_rows = fit_set_lmin(
        set_id=8,
        set_rows=rows,
        polygon_yz=polygon,
        kr_grid=kr_grid,
        rmin=1.0,
        rmax=10.0,
        lmin_fit=0.5,
        mc_samples_per_grid=2000,
        bin_count=5,
        bin_mode="linear",
        window_mode="polygon",
        oracle_radius_mode="observed_trace_radii",
    )
    assert fit_row["oracle_radius_mode"] == "observed_trace_radii"
    assert all(row["sim_radii"] is not None for row in candidate_rows)


def test_survival_helpers_accept_scalar_inputs() -> None:
    grid, survival = empirical_survival_curve(np.float64(3.0))
    assert grid.tolist() == [3.0]
    assert survival.tolist() == [0.0]

    km_grid, km_survival = kaplan_meier_survival_curve(np.float64(3.0), np.int32(0))
    assert km_grid.tolist() == [3.0]
    assert km_survival.tolist() == [0.0]
